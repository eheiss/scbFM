from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version.
    StratifiedGroupKFold = None
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from cancerfoundation_backbone import CancerFoundationBackbone
from preprocess import (
    filter_min_genes,
    reindex_adata_genes,
    validate_token_matrix,
)
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COHORTS = ["BRCA", "BLCA", "GBM", "LGG", "LUAD", "UCEC"]
TASK_NAME = "canc_type_class"
CHECKPOINT_MODEL_KEYS = ("pretrain_sc", "pretrain_bulk", "preadapt_sc", "preadapt_bulk")
RANDOM_INIT_MODEL_KEY = "random_init"
MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)


def _digitize_expression(
    values: np.ndarray,
    bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    left_digits = np.digitize(values, bins)
    right_digits = np.digitize(values, bins, right=True)
    random_offsets = rng.random(len(values))
    digits = random_offsets * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def _quantile_bin_expression(
    values: np.ndarray,
    bin_num: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the same per-sequence expression binning used during pretraining."""
    if bin_num < 2:
        raise ValueError("bin_num must be at least 2 because bin 0 is reserved for zero expression.")

    values = np.asarray(values, dtype=np.float32)
    binned = np.zeros(values.shape, dtype=np.int64)
    nonzero = values > 0
    if not nonzero.any():
        return binned

    nonzero_values = values[nonzero]
    bins = np.quantile(nonzero_values, np.linspace(0, 1, bin_num - 1))
    digits = _digitize_expression(nonzero_values, bins, rng)
    binned[nonzero] = np.clip(digits, 1, bin_num - 1)
    return binned


class CancTypePredHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        bottleneck_dim: int = 128,
        use_cls: bool = False,
        pooling: str | None = None,
    ) -> None:
        super().__init__()
        if pooling is None:
            pooling = "cls" if use_cls else "mean"
        pooling = str(pooling).lower()
        if pooling not in {"mean", "cls", "mean_cls"}:
            raise ValueError(
                f"Unsupported classification head pooling '{pooling}'. "
                "Expected one of: mean, cls, mean_cls."
            )
        self.pooling = pooling
        self.use_cls = pooling == "cls"
        input_dim = embedding_dim * 2 if pooling == "mean_cls" else embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SELU(),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Position zero is the <cls> token. "mean" retains the existing
        # BulkRNABert-style mean pooling over gene tokens only.
        if self.pooling == "cls":
            sample_embedding = x[:, 0, :]
        elif self.pooling == "mean":
            sample_embedding = x[:, 1:, :].mean(dim=1)
        else:
            sample_embedding = torch.cat(
                (x[:, 0, :], x[:, 1:, :].mean(dim=1)),
                dim=-1,
            )
        return self.mlp(sample_embedding)


class CancerFoundationCancTypeClassifier(nn.Module):
    def __init__(self, backbone: CancerFoundationBackbone, head: CancTypePredHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.to_out = head

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.backbone(
            batch["gene_ids"],
            batch["expr"],
            src_key_padding_mask=batch.get("attention_key_padding_mask"),
        )
        return self.to_out(hidden)

    def add_adapters(self, **kwargs) -> nn.ModuleList:
        return self.backbone.add_adapters(**kwargs)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return self.backbone.adapter_parameters()

    def enable_grad_checkpoint(self) -> None:
        self.backbone.enable_grad_checkpoint()


class GroupedCosineAnnealingWarmupRestarts:
    """Cosine warmup scheduler that preserves per-parameter-group max LRs."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        max_lrs: list[float],
        min_lr_ratio: float,
        cycle_mult: float = 1.0,
        warmup_steps: int = 0,
        gamma: float = 1.0,
    ) -> None:
        if warmup_steps >= first_cycle_steps:
            raise ValueError("warmup_steps must be smaller than first_cycle_steps.")
        if len(max_lrs) != len(optimizer.param_groups):
            raise ValueError("max_lrs must match optimizer.param_groups.")

        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lrs = [float(lr) for lr in max_lrs]
        self.max_lrs = list(self.base_max_lrs)
        self.min_lrs = [float(lr) * float(min_lr_ratio) for lr in self.base_max_lrs]
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = -1
        self.last_epoch = -1
        self._set_lrs(self.min_lrs)

    def _set_lrs(self, lrs: list[float]) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def get_lr(self) -> list[float]:
        if self.step_in_cycle == -1:
            return self.min_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                min_lr + (max_lr - min_lr) * self.step_in_cycle / self.warmup_steps
                for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)
            ]
        return [
            min_lr
            + (max_lr - min_lr)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.step_in_cycle - self.warmup_steps)
                    / (self.cur_cycle_steps - self.warmup_steps)
                )
            )
            / 2
            for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)
        ]

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    self.cycle = int(
                        math.log(
                            epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1,
                            self.cycle_mult,
                        )
                    )
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult**self.cycle - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult**self.cycle
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [lr * (self.gamma**self.cycle) for lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)
        self._set_lrs(self.get_lr())

    def state_dict(self) -> dict[str, object]:
        return {
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "base_max_lrs": self.base_max_lrs,
            "max_lrs": self.max_lrs,
            "min_lrs": self.min_lrs,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
            "cur_cycle_steps": self.cur_cycle_steps,
            "cycle": self.cycle,
            "step_in_cycle": self.step_in_cycle,
            "last_epoch": self.last_epoch,
        }


class CancTypeClassDataset(Dataset):
    def __init__(
        self,
        data,
        labels: np.ndarray,
        bin_num: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
        selected_gene_count: int,
        seed: int,
        do_binning: bool,
        fixed_gene_indices: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.labels = np.asarray(labels, dtype=np.int64)
        self.bin_num = int(bin_num)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.gene_num = int(data.shape[1])
        self.selected_gene_count = int(selected_gene_count)
        self.seed = int(seed)
        self.do_binning = bool(do_binning)
        self.epoch = 0
        self.fixed_gene_indices = (
            None
            if fixed_gene_indices is None
            else np.asarray(fixed_gene_indices, dtype=np.int64)
        )

        if self.selected_gene_count <= 0 or self.selected_gene_count > self.gene_num:
            raise ValueError(
                f"selected_gene_count must be in [1, {self.gene_num}], "
                f"got {self.selected_gene_count}."
            )
        if (
            self.fixed_gene_indices is not None
            and self.fixed_gene_indices.shape != (self.selected_gene_count,)
        ):
            raise ValueError(
                "fixed_gene_indices must contain exactly "
                f"{self.selected_gene_count} genes."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.data[index]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()

        # Fixed-HVG datasets must produce identical inputs across epochs. The
        # epoch contributes to the RNG only for random gene-subset training.
        rng_epoch = self.epoch if self.fixed_gene_indices is None else 0
        rng_seed = self.seed + rng_epoch * 1_000_003 + index
        rng = np.random.default_rng(rng_seed)
        if self.fixed_gene_indices is None:
            selected_cols = rng.choice(
                self.gene_num,
                size=self.selected_gene_count,
                replace=False,
            )
        else:
            selected_cols = self.fixed_gene_indices

        selected_values = values[selected_cols].astype(np.float32, copy=False)
        if self.do_binning:
            selected_values = _quantile_bin_expression(
                selected_values,
                self.bin_num,
                rng,
            ).astype(np.float32)

        gene_ids = torch.from_numpy(
            selected_cols.astype(np.int64, copy=False) + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        expression = torch.from_numpy(selected_values)
        expression = torch.cat((torch.tensor([self.cls_value]), expression))
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return {
            "gene_ids": gene_ids,
            "expr": expression,
        }, label


class CancTypeClassRunner:
    task_name = TASK_NAME

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.task_cfg = self._resolve_task_cfg(cfg)
        self.model_cfg = self._resolve_model_cfg(cfg)

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        self.cls_gene_id = 0
        self.pad_gene_id = 1
        self.gene_token_offset = 2
        self.num_gene_tokens = int(self.model_cfg.gene_num) + self.gene_token_offset
        self.cls_value = float(getattr(self.model_cfg, "pad_value", -2.0))
        self.selected_gene_count = int(getattr(self.model_cfg, "selected_gene_count", 1199))
        self.max_seq_len = int(
            getattr(self.model_cfg, "max_seq_len", self.selected_gene_count + 1)
        )
        if self.max_seq_len != self.selected_gene_count + 1:
            raise ValueError(
                "Cancer type classification expects max_seq_len to equal "
                "selected_gene_count + 1 for the <cls> token."
            )

        self.label_dict: np.ndarray | None = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size = 0
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.loss_fn: nn.Module | None = None
        self.train_class_weights: torch.Tensor | None = None
        self.backbone_optimizer_enabled = False

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "canc_type_class" in cfg.finetune:
            return cfg.finetune.canc_type_class
        raise ValueError(
            "Could not find a cancer type classification config. "
            "Expected cfg.finetune.canc_type_class."
        )

    @staticmethod
    def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
        if "pretrain" in cfg:
            return cfg.pretrain
        raise ValueError(
            "Could not find model architecture config. Expected cfg.pretrain."
        )

    def _setup_runtime(self) -> None:
        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
            dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + self.rank)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]], comment: str = "") -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        preferred = [
            "model",
            "fold",
            "n_folds",
            "finetune_mode",
            "checkpoint_path",
            "sample_id",
            "patient_id",
            "project",
            "true_label",
        ]
        fieldnames = [
            field
            for field in preferred
            if any(field in row for row in rows)
        ]
        extra_fields = sorted(
            {
                field
                for row in rows
                for field in row
                if field not in fieldnames
            }
        )
        with path.open("w", newline="") as handle:
            if comment:
                handle.write(f"# {comment}\n")
            writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _get_git_commit() -> str | None:
        repo_dir = ROOT / "scbFM"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    def _save_run_metadata(self, checkpoint_paths: dict[str, str]) -> None:
        if not self.is_master:
            return
        out_dir = self._task_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_prefix()
        (out_dir / f"{prefix}_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        self._write_json(
            out_dir / f"{prefix}_run_metadata.json",
            {
                "task": self.task_name,
                "finetune_mode": self._finetune_mode(),
                "output_suffix": self._output_suffix(),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "git_commit": self._get_git_commit(),
                "checkpoint_paths": checkpoint_paths,
            },
        )

    @staticmethod
    def _aggregate_numeric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
        aggregate: dict[str, object] = {"n_folds": len(rows)}
        skip_fields = {"model", "fold", "n_folds", "finetune_mode", "checkpoint_path"}
        numeric_fields = sorted(
            {
                field
                for row in rows
                for field, value in row.items()
                if field not in skip_fields and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in rows if field in row], dtype=float)
            values = values[~np.isnan(values)]
            aggregate[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            aggregate[f"{field}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return aggregate

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                "finetune.canc_type_class.pretrained_model_paths must define "
                f"{', '.join(CHECKPOINT_MODEL_KEYS)}."
            )

        checkpoint_paths: dict[str, str] = {}
        missing = []
        for key in CHECKPOINT_MODEL_KEYS:
            value = paths_cfg.get(key)
            if value:
                checkpoint_paths[key] = str(Path(hydra.utils.to_absolute_path(str(value))))
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                "Missing checkpoint paths in finetune.canc_type_class.pretrained_model_paths: "
                f"{missing}"
            )
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
        return checkpoint_paths

    def _load_tcga(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "tcga_data_dir", None)
        if not configured_path:
            raise ValueError("finetune.canc_type_class.tcga_data_dir must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"TCGA h5ad file not found: {data_path}")

        adata = ad.read_h5ad(data_path)
        required_obs = {"sample_id", "patient_id", "project"}
        missing_obs = sorted(required_obs.difference(adata.obs.columns))
        if missing_obs:
            raise ValueError(f"TCGA AnnData is missing required obs columns: {missing_obs}.")

        adata.obs["project"] = adata.obs["project"].astype(str).str.strip().str.upper()

        cohorts = list(getattr(self.task_cfg, "cohorts", DEFAULT_COHORTS))
        selected_cancer_types = {str(cohort).upper() for cohort in cohorts}
        keep_mask = adata.obs["project"].isin(selected_cancer_types).to_numpy()
        adata = adata[keep_mask].copy()

        if adata.n_obs == 0:
            raise ValueError(
                f"No TCGA samples matched cohorts {cohorts} in the project metadata."
            )

        adata.obs["cancer_type"] = adata.obs["project"].astype(str)
        adata.obs_names = adata.obs["sample_id"].astype(str)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        return adata

    def _load_input_adata(self) -> ad.AnnData:
        log.info("Loading TCGA cohorts for cancer type classification")
        return self._load_tcga()

    def _resolve_gene_list_path(self) -> Path:
        gene_list_path = getattr(self.task_cfg, "gene_list_path", None)
        if gene_list_path:
            return Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        return ROOT / "scbFM" / "data" / "gene_list.txt"

    def _should_preprocess_input(self) -> bool:
        return bool(getattr(self.task_cfg, "preprocess", False))

    def _preprocess_adata(self, adata: ad.AnnData) -> ad.AnnData:
        gene_list_path = self._resolve_gene_list_path()
        if self._should_preprocess_input():
            min_genes = int(getattr(self.task_cfg, "min_genes", 200))
            adata, missing_genes = reindex_adata_genes(adata, gene_list_path=gene_list_path)
            adata = filter_min_genes(adata, min_genes=min_genes)
            log.info(
                "Aligned raw input for on-the-fly sequence binning: "
                "%d target genes missing, output shape %s",
                len(missing_genes),
                adata.shape,
            )
        else:
            adata, missing_genes = reindex_adata_genes(adata, gene_list_path=gene_list_path)
            log.info(
                "Reindexed preprocessed input to gene list: %d target genes missing, output shape %s",
                len(missing_genes),
                adata.shape,
            )

        self._missing_genes_note = (
            f"Model genes missing from TCGA GEX and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
        ) if missing_genes else ""

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes for the scbFM backbone, got {adata.n_vars}. "
                "Provide a matching gene list or aligned input matrix."
            )

        if not self._should_preprocess_input():
            validate_token_matrix(
                adata.X,
                bin_num=int(self.model_cfg.bin_num),
                name="cancer type input data",
            )

        return adata

    def _select_training_hvg_indices(self, adata: ad.AnnData) -> np.ndarray:
        """Fit fold-level gene selection on training data only."""
        if adata.n_vars < self.selected_gene_count:
            raise ValueError(
                f"Cannot select {self.selected_gene_count} genes from only {adata.n_vars} genes."
            )

        selection_method = str(
            getattr(self.task_cfg, "hvg_selection_method", "mad")
        ).strip().lower()
        if selection_method == "mad":
            return self._select_training_mad_indices(adata)
        if selection_method != "scanpy":
            raise ValueError(
                "hvg_selection_method must be one of: mad, scanpy. "
                f"Got '{selection_method}'."
            )

        batch_key = getattr(self.task_cfg, "hvg_batch_key", None)
        if batch_key is not None:
            batch_key = str(batch_key).strip() or None
        if batch_key is not None and batch_key not in adata.obs:
            raise ValueError(f"HVG batch key '{batch_key}' is not present in adata.obs.")

        hvg_stats = sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.selected_gene_count,
            flavor=str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
            batch_key=batch_key,
            inplace=False,
        )
        selected = np.flatnonzero(hvg_stats["highly_variable"].to_numpy())

        # Scanpy can retain extra tied genes. Keep the requested sequence length
        # deterministically while preserving its HVG ranking.
        if selected.size > self.selected_gene_count:
            ranking_column = (
                "highly_variable_rank"
                if "highly_variable_rank" in hvg_stats
                else "dispersions_norm"
            )
            scores = hvg_stats[ranking_column].to_numpy()[selected]
            if ranking_column == "highly_variable_rank":
                order = np.argsort(np.nan_to_num(scores, nan=np.inf), kind="stable")
            else:
                order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
            selected = selected[order[: self.selected_gene_count]]

        if selected.size != self.selected_gene_count:
            raise RuntimeError(
                "Scanpy selected "
                f"{selected.size} HVGs; expected exactly {self.selected_gene_count}."
            )

        log.info(
            "Selected %d training-fold HVGs for training and evaluation "
            "with method=scanpy, flavor=%s, batch_key=%s",
            selected.size,
            str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
            batch_key,
        )
        return selected.astype(np.int64, copy=False)

    def _select_training_mad_indices(self, adata: ad.AnnData) -> np.ndarray:
        """Select genes with highest median absolute deviation on log1p training data."""
        matrix = adata.X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D expression matrix, got shape {matrix.shape}.")

        gene_medians = np.nanmedian(matrix, axis=0)
        mad = np.nanmedian(np.abs(matrix - gene_medians), axis=0)
        scores = np.nan_to_num(mad, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        if not np.any(np.isfinite(scores)):
            raise ValueError("Could not compute finite MAD scores for any genes.")

        ranked = np.lexsort((np.arange(scores.size), -scores))
        selected = np.sort(ranked[: self.selected_gene_count]).astype(np.int64, copy=False)
        if selected.size != self.selected_gene_count:
            raise RuntimeError(
                "MAD selection selected "
                f"{selected.size} genes; expected exactly {self.selected_gene_count}."
            )

        log.info(
            "Selected %d training-fold genes for training and evaluation "
            "with method=mad on log1p expression | selected MAD min=%.6g median=%.6g max=%.6g",
            selected.size,
            float(np.min(scores[selected])),
            float(np.median(scores[selected])),
            float(np.max(scores[selected])),
        )
        return selected

    def _prepare_cv_data(self) -> tuple[ad.AnnData, np.ndarray, np.ndarray | None]:
        adata = self._load_input_adata()
        adata.obs["cancer_type"] = adata.obs["cancer_type"].astype(str)

        if bool(getattr(self.task_cfg, "merge_gbm_lgg", True)):
            adata.obs["cancer_type"] = adata.obs["cancer_type"].replace(
                {"GBM": "GBMLGG", "LGG": "GBMLGG"}
            )

        adata = self._preprocess_adata(adata)
        self.label_dict = np.unique(np.asarray(adata.obs["cancer_type"]).astype(str))
        labels = np.asarray(adata.obs["cancer_type"]).astype(str)
        patient_ids = adata.obs["patient_id"].astype(str).to_numpy()
        groups = None
        if len(np.unique(patient_ids)) < len(patient_ids):
            groups = patient_ids
            log.info("Duplicate TCGA patient_id values detected; using patient-grouped CV.")
        return adata, labels, groups

    def _build_cv_splits(
        self,
        labels: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 5))
        if n_splits < 2:
            raise ValueError("finetune.canc_type_class.cv_folds must be at least 2.")

        _, class_counts = np.unique(labels, return_counts=True)
        min_class_count = int(class_counts.min())
        if n_splits > min_class_count:
            raise ValueError(
                f"cv_folds={n_splits} is larger than the smallest class size ({min_class_count})."
            )

        if groups is not None:
            unique_groups = np.unique(groups)
            if n_splits > unique_groups.size:
                raise ValueError(
                    f"cv_folds={n_splits} is larger than the number of patient_id "
                    f"groups ({unique_groups.size})."
                )
            if StratifiedGroupKFold is not None:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=int(getattr(self.task_cfg, "random_seed", 42)),
                )
                return list(splitter.split(np.zeros(labels.shape[0]), labels, groups))

            log.warning(
                "StratifiedGroupKFold is unavailable in this sklearn version; "
                "falling back to non-stratified GroupKFold."
            )
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(np.zeros(labels.shape[0]), labels, groups))

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        return list(splitter.split(np.zeros(labels.shape[0]), labels))

    def _build_loaders(self, train_adata: ad.AnnData, test_adata: ad.AnnData) -> None:
        if self.label_dict is None:
            self.label_dict = np.unique(np.asarray(train_adata.obs["cancer_type"]).astype(str))
        label_to_idx = {label: idx for idx, label in enumerate(self.label_dict.tolist())}
        train_labels = np.array(
            [label_to_idx[label] for label in np.asarray(train_adata.obs["cancer_type"]).astype(str)],
            dtype=np.int64,
        )
        test_labels = np.array(
            [label_to_idx[label] for label in np.asarray(test_adata.obs["cancer_type"]).astype(str)],
            dtype=np.int64,
        )

        class_weight_power = float(
            getattr(self.task_cfg, "class_weight_power", 0.0)
        )
        if not np.isfinite(class_weight_power) or class_weight_power < 0:
            raise ValueError(
                "class_weight_power must be a finite non-negative number."
            )
        if class_weight_power == 0.0:
            # Preserve the previous behavior exactly rather than passing an
            # all-ones tensor to CrossEntropyLoss.
            self.train_class_weights = None
        else:
            class_counts = np.bincount(
                train_labels,
                minlength=len(self.label_dict),
            ).astype(np.float64)
            if np.any(class_counts == 0):
                missing_labels = self.label_dict[class_counts == 0].tolist()
                raise ValueError(
                    "Cannot compute class weights because the training fold "
                    f"contains no samples for classes {missing_labels}."
                )
            balanced_weights = (
                len(train_labels) / (len(self.label_dict) * class_counts)
            )
            class_weights = np.power(balanced_weights, class_weight_power)
            self.train_class_weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )

        if self.is_master:
            if self.train_class_weights is None:
                log.info("Classification loss: unweighted cross-entropy")
            else:
                log.info(
                    "Classification loss: class-weighted cross-entropy | "
                    "power=%.3f | weight range=[%.4f, %.4f]",
                    class_weight_power,
                    float(self.train_class_weights.min()),
                    float(self.train_class_weights.max()),
                )

        batch_size = int(getattr(self.task_cfg, "batch_size", 2))
        num_workers = int(getattr(self.task_cfg, "num_workers", 0))
        if num_workers < 0:
            raise ValueError("finetune.canc_type_class.num_workers must be non-negative.")

        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            prefetch_factor = int(getattr(self.task_cfg, "prefetch_factor", 2))
            if prefetch_factor <= 0:
                raise ValueError(
                    "finetune.canc_type_class.prefetch_factor must be positive."
                )
            loader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": False,
                }
            )

        random_seed = int(getattr(self.task_cfg, "random_seed", 42))
        do_binning = self._should_preprocess_input()
        # Fit feature selection on the training fold only. Use the resulting
        # fixed vocabulary indices for both training and held-out samples.
        fold_hvg_indices = self._select_training_hvg_indices(train_adata)
        train_dataset = CancTypeClassDataset(
            train_adata.X,
            train_labels,
            bin_num=int(self.model_cfg.bin_num),
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=self.cls_value,
            selected_gene_count=self.selected_gene_count,
            seed=random_seed,
            do_binning=do_binning,
            fixed_gene_indices=fold_hvg_indices,
        )
        test_dataset = CancTypeClassDataset(
            test_adata.X,
            test_labels,
            bin_num=int(self.model_cfg.bin_num),
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=self.cls_value,
            selected_gene_count=self.selected_gene_count,
            seed=random_seed,
            do_binning=do_binning,
            fixed_gene_indices=fold_hvg_indices,
        )
        self.test_dataset_size = len(test_dataset)

        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            test_sampler = SequentialDistributedSampler(
                test_dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,
                **loader_kwargs,
            )
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                **loader_kwargs,
            )
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )

        if self.is_master:
            log.info(
                "DataLoader: num_workers=%d | prefetch_factor=%s | pin_memory=%s | "
                "persistent_workers=false",
                num_workers,
                (
                    int(getattr(self.task_cfg, "prefetch_factor", 2))
                    if num_workers > 0
                    else "disabled"
                ),
                self.device.type == "cuda",
            )

    @staticmethod
    def _strip_module_prefix(state_dict: dict) -> dict:
        if not state_dict:
            return state_dict
        if not all(key.startswith("module.") for key in state_dict):
            return state_dict
        return {key.removeprefix("module."): value for key, value in state_dict.items()}

    def _build_model(self, checkpoint_path: str) -> None:
        finetune_mode = self._finetune_mode()
        valid_modes = {"head_only", "full_ft", "adapters"}
        if finetune_mode not in valid_modes:
            raise ValueError(
                f"Unsupported finetune_mode '{finetune_mode}'. Expected one of {sorted(valid_modes)}."
            )

        backbone = CancerFoundationBackbone(
            num_gene_tokens=self.num_gene_tokens,
            d_model=int(self.model_cfg.embsize),
            nhead=int(self.model_cfg.nheads),
            d_hid=int(self.model_cfg.d_hid),
            nlayers=int(self.model_cfg.nlayers),
            dropout=float(self.model_cfg.dropout),
            pad_gene_id=self.pad_gene_id,
            max_value=int(getattr(self.model_cfg, "value_encoder_max_value", 512)),
        )

        if checkpoint_path:
            resolved_path = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved_path, map_location="cpu")
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            backbone.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved_path)
        else:
            log.info("Using randomly initialized backbone")

        head = CancTypePredHead(
            embedding_dim=int(self.model_cfg.embsize),
            output_dim=len(self.label_dict),
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 256)),
            bottleneck_dim=int(getattr(self.task_cfg, "head_bottleneck_dim", 128)),
            use_cls=bool(getattr(self.task_cfg, "use_cls", False)),
            pooling=getattr(self.task_cfg, "head_pooling", None),
        )
        if self.is_master:
            representation = {
                "cls": "CLS token",
                "mean": "mean-pooled gene tokens",
                "mean_cls": "CLS token + mean-pooled gene tokens",
            }[head.pooling]
            log.info(
                "Classification representation: %s | head dims: %d -> %d -> %d -> %d",
                representation,
                int(self.model_cfg.embsize) * (2 if head.pooling == "mean_cls" else 1),
                int(getattr(self.task_cfg, "head_hidden_dim", 256)),
                int(getattr(self.task_cfg, "head_bottleneck_dim", 128)),
                len(self.label_dict),
            )
        model = CancerFoundationCancTypeClassifier(backbone=backbone, head=head)

        if finetune_mode == "adapters":
            model.add_adapters(
                bottleneck_dim=int(getattr(self.task_cfg, "adapter_bottleneck_dim", 32)),
                dropout=float(getattr(self.task_cfg, "adapter_dropout", 0.0)),
                after_attention=bool(getattr(self.task_cfg, "adapter_after_attention", True)),
                after_ff=bool(getattr(self.task_cfg, "adapter_after_ff", True)),
            )

        if finetune_mode == "head_only":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.to_out.parameters():
                param.requires_grad = True
            self.backbone_optimizer_enabled = False
        elif finetune_mode == "adapters":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.to_out.parameters():
                param.requires_grad = True
            for param in model.adapter_parameters():
                param.requires_grad = True
            self.backbone_optimizer_enabled = False
        elif finetune_mode == "full_ft":
            for param in model.parameters():
                param.requires_grad = True
            self.backbone_optimizer_enabled = (
                int(getattr(self.task_cfg, "burn_in_epochs", 0)) <= 0
            )

        if finetune_mode in ("adapters", "full_ft"):
            model.enable_grad_checkpoint()

        model = model.to(self.device)
        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                model = DDP(model)

        self.model = model

    def _build_optimization(self) -> None:
        if not hasattr(self.task_cfg, "head_learning_rate"):
            raise ValueError("finetune.canc_type_class.head_learning_rate must be set.")
        if not hasattr(self.task_cfg, "backbone_learning_rate"):
            raise ValueError("finetune.canc_type_class.backbone_learning_rate must be set.")
        head_learning_rate = float(self.task_cfg.head_learning_rate)
        backbone_learning_rate = float(self.task_cfg.backbone_learning_rate)
        adapter_learning_rate = float(
            getattr(self.task_cfg, "adapter_learning_rate", head_learning_rate)
        )

        model = self.model.module if isinstance(self.model, DDP) else self.model
        head_params = [param for param in model.to_out.parameters() if param.requires_grad]
        head_param_ids = {id(param) for param in head_params}
        adapter_params = [
            param
            for param in model.adapter_parameters()
            if param.requires_grad
        ]
        adapter_param_ids = {id(param) for param in adapter_params}
        backbone_params = [
            param
            for param in model.parameters()
            if (
                param.requires_grad
                and id(param) not in head_param_ids
                and id(param) not in adapter_param_ids
            )
        ]

        param_groups = []
        if backbone_params and self.backbone_optimizer_enabled:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_learning_rate,
                    "name": "backbone",
                }
            )
        if adapter_params:
            param_groups.append(
                {
                    "params": adapter_params,
                    "lr": adapter_learning_rate,
                    "name": "adapters",
                }
            )
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": head_learning_rate,
                    "name": "head",
                }
            )
        if not param_groups:
            raise ValueError("No trainable parameters found for cancer type classification.")

        max_lrs = [float(group["lr"]) for group in param_groups]
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_min_lr_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = (
            float(configured_min_lr_ratio)
            if configured_min_lr_ratio is not None
            else min_lr / max(head_learning_rate, 1e-12)
        )

        self.optimizer = Adam(param_groups)
        self.scheduler = GroupedCosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 15)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 2)),
            max_lrs=max_lrs,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 5)),
            gamma=float(getattr(self.task_cfg, "gamma", 0.9)),
        )
        loss_weights = (
            None
            if self.train_class_weights is None
            else self.train_class_weights.to(self.device)
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights).to(self.device)

        if self.is_master:
            group_summaries = [
                f"{group.get('name', idx)}: params={sum(p.numel() for p in group['params'])}, "
                f"max_lr={max_lr:.2e}, min_lr={max_lr * min_lr_ratio:.2e}"
                for idx, (group, max_lr) in enumerate(zip(param_groups, max_lrs))
            ]
            log.info("Optimizer parameter groups: %s", "; ".join(group_summaries))

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        finetune_mode = self._finetune_mode()
        burn_in_epochs = int(getattr(self.task_cfg, "burn_in_epochs", 0))
        if (
            finetune_mode != "full_ft"
            or burn_in_epochs <= 0
            or self.backbone_optimizer_enabled
            or epoch <= burn_in_epochs
        ):
            return

        self.backbone_optimizer_enabled = True
        self._build_optimization()
        if self.is_master:
            log.info(
                "Finished %d burn-in epochs; enabled backbone optimization for full fine-tuning.",
                burn_in_epochs,
            )

    def _optimizer_parameters(self) -> list[torch.nn.Parameter]:
        return [
            param
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_enable_backbone_optimizer(epoch)

        self.train_loader.dataset.set_epoch(epoch)

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 1)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss = 0.0
        running_acc = 0.0

        for step_idx, (data, labels) in enumerate(self.train_loader, start=1):
            data = self._move_batch_to_device(data)
            labels = labels.to(self.device, non_blocking=True)

            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and step_idx % grad_acc_steps != 0
            )
            sync_context = self.model.no_sync() if use_no_sync else nullcontext()

            with sync_context:
                logits = self.model(data)
                loss = self.loss_fn(logits, labels)
                (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)

            running_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            running_acc += (predictions == labels).float().mean().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * running_acc / len(self.train_loader)

        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.device, 0, self.world_size)
            epoch_acc = get_reduced(epoch_acc, self.device, 0, self.world_size)

        self.scheduler.step()
        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def _evaluate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        predictions = []
        truths = []

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = self._move_batch_to_device(data)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(data)
                loss = self.loss_fn(logits, labels)
                running_loss += loss.item()
                predictions.append(logits.argmax(dim=-1))
                truths.append(labels)

        predictions = torch.cat(predictions, dim=0)
        truths = torch.cat(truths, dim=0)

        if self.is_distributed:
            predictions = distributed_concat(predictions, self.test_dataset_size, self.world_size)
            truths = distributed_concat(truths, self.test_dataset_size, self.world_size)

        predictions_np = predictions.cpu().numpy()
        truths_np = truths.cpu().numpy()
        precision, recall, fscore, support = precision_recall_fscore_support(
            truths_np,
            predictions_np,
            labels=np.arange(len(self.label_dict)),
            zero_division=0,
        )

        test_loss = running_loss / len(self.test_loader)
        if self.is_distributed:
            test_loss = get_reduced(test_loss, self.device, 0, self.world_size)

        return {
            "loss": float(test_loss),
            "accuracy": float(accuracy_score(truths_np, predictions_np)),
            "f1_macro": float(
                f1_score(
                    truths_np,
                    predictions_np,
                    labels=np.arange(len(self.label_dict)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    truths_np,
                    predictions_np,
                    labels=np.arange(len(self.label_dict)),
                    average="weighted",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                truths_np,
                predictions_np,
                labels=np.arange(len(self.label_dict)),
            ),
            "classification_report": classification_report(
                truths_np,
                predictions_np,
                labels=np.arange(len(self.label_dict)),
                target_names=self.label_dict.tolist(),
                digits=4,
                zero_division=0,
            ),
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": fscore,
            "support_per_class": support,
            "label_dict": self.label_dict.tolist(),
            "n_test_samples": int(len(truths_np)),
            "truth_indices": truths_np,
            "prediction_indices": predictions_np,
        }

    def _flatten_fold_metrics(
        self,
        model_key: str,
        fold: int,
        n_folds: int,
        checkpoint_path: str,
        train_metrics: dict[str, float],
        test_metrics: dict[str, object],
    ) -> dict[str, object]:
        row: dict[str, object] = {
                "model": model_key,
                "fold": fold,
                "n_folds": n_folds,
                "finetune_mode": self._finetune_mode(),
                "checkpoint_path": checkpoint_path,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
            }
        for key, value in test_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[key] = float(value)

        labels = list(test_metrics.get("label_dict", self.label_dict.tolist()))
        for metric_key, prefix in (
            ("precision_per_class", "precision"),
            ("recall_per_class", "recall"),
            ("f1_per_class", "f1"),
            ("support_per_class", "support"),
        ):
            values = test_metrics.get(metric_key)
            if values is None:
                continue
            for label, value in zip(labels, np.asarray(values).tolist()):
                row[f"{prefix}_{label}"] = float(value)
        return row

    def _prediction_rows(
        self,
        model_key: str,
        fold: int,
        checkpoint_path: str,
        test_adata: ad.AnnData,
        test_metrics: dict[str, object],
    ) -> list[dict[str, object]]:
        truth_indices = np.asarray(test_metrics["truth_indices"], dtype=int)
        prediction_indices = np.asarray(test_metrics["prediction_indices"], dtype=int)
        labels = self.label_dict.tolist()
        rows: list[dict[str, object]] = []
        patient_ids = test_adata.obs["patient_id"].astype(str).to_numpy()
        projects = test_adata.obs["project"].astype(str).to_numpy()
        sample_ids = test_adata.obs["sample_id"].astype(str).to_numpy()
        for idx, (truth_idx, pred_idx) in enumerate(zip(truth_indices, prediction_indices)):
            rows.append(
                {
                    "model": model_key,
                    "fold": fold,
                    "finetune_mode": self._finetune_mode(),
                    "checkpoint_path": checkpoint_path,
                    "sample_id": sample_ids[idx],
                    "patient_id": patient_ids[idx],
                    "project": projects[idx],
                    "true_idx": int(truth_idx),
                    "pred_idx": int(pred_idx),
                    "true_label": labels[int(truth_idx)],
                    "pred_label": labels[int(pred_idx)],
                    "correct": int(truth_idx == pred_idx),
                }
            )
        return rows

    def _write_confusion_matrix(
        self,
        path: Path,
        matrix: np.ndarray,
    ) -> None:
        labels = self.label_dict.tolist()
        rows = [
            {
                "true_label": label,
                **{f"pred_{pred_label}": int(matrix[row_idx, col_idx]) for col_idx, pred_label in enumerate(labels)},
            }
            for row_idx, label in enumerate(labels)
        ]
        self._write_csv(path, rows)

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / self.task_name / self._output_variant()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._output_variant()}"

    def _finetune_mode(self) -> str:
        return str(getattr(self.task_cfg, "finetune_mode", "head_only"))

    def _output_suffix(self) -> str:
        configured = getattr(self.task_cfg, "output_suffix", "")
        suffix = "" if configured is None else str(configured).strip()
        if suffix and re.fullmatch(r"[A-Za-z0-9_-]+", suffix) is None:
            raise ValueError(
                "output_suffix may contain only letters, numbers, underscores, and hyphens."
            )
        return suffix

    def _output_variant(self) -> str:
        suffix = self._output_suffix()
        return (
            f"{self._finetune_mode()}_{suffix}"
            if suffix
            else self._finetune_mode()
        )

    def _write_model_results(
        self,
        checkpoint_path: str,
        fold_rows: list[dict[str, object]],
        prediction_rows: list[dict[str, object]],
        confusion_matrices: list[np.ndarray],
    ) -> dict[str, object]:
        aggregate = self._aggregate_numeric_rows(fold_rows)
        aggregate.update(
            {
                "model": fold_rows[0]["model"],
                "finetune_mode": fold_rows[0]["finetune_mode"],
                "checkpoint_path": checkpoint_path,
            }
        )
        out_dir = self._task_output_dir()
        model_key = str(fold_rows[0]["model"])
        prefix = self._output_prefix()
        self._write_csv(out_dir / f"{prefix}_{model_key}_fold_metrics.csv", fold_rows)
        self._write_csv(out_dir / f"{prefix}_{model_key}_evaluation_metrics.csv", [aggregate], comment=getattr(self, "_missing_genes_note", ""))
        self._write_csv(out_dir / f"{prefix}_{model_key}_predictions.csv", prediction_rows)
        if confusion_matrices:
            self._write_confusion_matrix(
                out_dir / f"{prefix}_{model_key}_confusion_matrix.csv",
                np.sum(confusion_matrices, axis=0),
            )
        return aggregate

    def _cleanup_fold_state(self) -> None:
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_class_weights = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> dict:
        try:
            self._setup_runtime()
            adata, labels, groups = self._prepare_cv_data()
            splits = self._build_cv_splits(labels, groups)
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)
            if self.is_master:
                log.info(
                    "Prepared cancer type classification CV data: samples=%d, genes=%d, folds=%d",
                    adata.n_obs,
                    adata.n_vars,
                    len(splits),
                )

            epochs = int(getattr(self.task_cfg, "epochs", 10))
            aggregate_rows: list[dict[str, object]] = []

            for model_key, checkpoint_path in checkpoint_paths.items():
                fold_rows: list[dict[str, object]] = []
                prediction_rows: list[dict[str, object]] = []
                confusion_matrices: list[np.ndarray] = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + fold_idx
                    )
                    train_adata = adata[train_idx].copy()
                    test_adata = adata[test_idx].copy()
                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | train=%d, test=%d",
                            model_key,
                            fold_idx,
                            len(splits),
                            train_adata.n_obs,
                            test_adata.n_obs,
                        )

                    self._build_loaders(train_adata, test_adata)
                    self._build_model(checkpoint_path)
                    self._build_optimization()

                    last_train_metrics = {"loss": float("nan"), "accuracy": float("nan")}
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        if self.is_master:
                            log.info(
                                "Model %s | Fold %d/%d | Epoch %d | Training Loss: %.6f | Accuracy: %.4f%%",
                                model_key,
                                fold_idx,
                                len(splits),
                                epoch,
                                last_train_metrics["loss"],
                                last_train_metrics["accuracy"],
                            )

                    test_metrics = self._evaluate()
                    if self.is_master:
                        fold_rows.append(
                            self._flatten_fold_metrics(
                                model_key=model_key,
                                fold=fold_idx,
                                n_folds=len(splits),
                                checkpoint_path=checkpoint_path,
                                train_metrics=last_train_metrics,
                                test_metrics=test_metrics,
                            )
                        )
                        prediction_rows.extend(
                            self._prediction_rows(
                                model_key=model_key,
                                fold=fold_idx,
                                checkpoint_path=checkpoint_path,
                                test_adata=test_adata,
                                test_metrics=test_metrics,
                            )
                        )
                        confusion_matrices.append(np.asarray(test_metrics["confusion_matrix"]))
                    self._cleanup_fold_state()

                if self.is_master:
                    aggregate_rows.append(
                        self._write_model_results(
                            checkpoint_path,
                            fold_rows,
                            prediction_rows,
                            confusion_matrices,
                        )
                    )
                if self.is_distributed:
                    dist.barrier()

            if self.is_master:
                combined_dir = self._task_output_dir()
                output_path = combined_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
                self._write_csv(output_path, aggregate_rows, comment=getattr(self, "_missing_genes_note", ""))
                return {
                    "results_path": str(output_path),
                    "results": aggregate_rows,
                }
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
