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
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.canc_type_class.runner import _quantile_bin_expression
from preprocess import (
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
TASK_NAME = "gene_essent"
CHECKPOINT_MODEL_KEYS = ("pretrain_sc", "pretrain_bulk", "preadapt_sc", "preadapt_bulk")
RANDOM_INIT_MODEL_KEY = "random_init"
MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)


class GeneEssentPredHead(nn.Module):
    """Shared per-gene MLP mapping selected gene states to essentiality scores."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
        context_pooling: str = "none",
    ) -> None:
        super().__init__()
        valid_pooling = {"none", "cls", "mean", "mean_cls"}
        if context_pooling not in valid_pooling:
            raise ValueError(
                f"Unsupported gene-essentiality context pooling '{context_pooling}'. "
                f"Expected one of {sorted(valid_pooling)}."
            )
        self.context_pooling = context_pooling
        context_multiplier = {
            "none": 0,
            "cls": 1,
            "mean": 1,
            "mean_cls": 2,
        }[context_pooling]
        input_dim = embedding_dim * (1 + context_multiplier)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SELU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class CancerFoundationGeneEssentModel(nn.Module):
    """CancerFoundation backbone with one prediction per selected gene token."""

    def __init__(
        self,
        backbone: CancerFoundationBackbone,
        head: GeneEssentPredHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.to_out = head

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.backbone(
            batch["gene_ids"],
            batch["expr"],
            src_key_padding_mask=batch.get("attention_key_padding_mask"),
        )
        # Position zero is the CancerFoundation <cls> token. Every remaining
        # output stays aligned with the corresponding selected input gene.
        gene_hidden = hidden[:, 1:, :]
        if self.to_out.context_pooling == "none":
            head_input = gene_hidden
        else:
            contexts = []
            if self.to_out.context_pooling in {"cls", "mean_cls"}:
                contexts.append(hidden[:, 0, :])
            if self.to_out.context_pooling in {"mean", "mean_cls"}:
                contexts.append(gene_hidden.mean(dim=1))
            sample_context = torch.cat(contexts, dim=-1)
            sample_context = sample_context[:, None, :].expand(
                -1,
                gene_hidden.shape[1],
                -1,
            )
            head_input = torch.cat((gene_hidden, sample_context), dim=-1)
        return self.to_out(head_input)

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


class GeneEssentDataset(Dataset):
    def __init__(
        self,
        X_expression,
        Y_targets: np.ndarray,
        *,
        fixed_gene_indices: np.ndarray,
        bin_num: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
        seed: int,
        do_binning: bool,
    ) -> None:
        self.X = X_expression
        self.Y = np.asarray(Y_targets, dtype=np.float32)
        self.fixed_gene_indices = np.asarray(fixed_gene_indices, dtype=np.int64)
        self.bin_num = int(bin_num)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.seed = int(seed)
        self.do_binning = bool(do_binning)

        if self.fixed_gene_indices.ndim != 1 or self.fixed_gene_indices.size == 0:
            raise ValueError("fixed_gene_indices must be a non-empty one-dimensional array.")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.X[index]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()

        rng = np.random.default_rng(self.seed + index)
        selected_values = values[self.fixed_gene_indices].astype(np.float32, copy=False)
        if self.do_binning:
            selected_values = _quantile_bin_expression(
                selected_values,
                self.bin_num,
                rng,
            ).astype(np.float32)

        gene_ids = torch.from_numpy(
            self.fixed_gene_indices + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        expression = torch.from_numpy(selected_values)
        expression = torch.cat((torch.tensor([self.cls_value]), expression))
        target = torch.as_tensor(
            self.Y[index, self.fixed_gene_indices],
            dtype=torch.float32,
        )
        return {"gene_ids": gene_ids, "expr": expression}, target


class GeneEssentRunner:
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
        self.selected_gene_count = int(
            getattr(self.model_cfg, "selected_gene_count", 1199)
        )
        self.max_seq_len = int(
            getattr(self.model_cfg, "max_seq_len", self.selected_gene_count + 1)
        )
        if self.max_seq_len != self.selected_gene_count + 1:
            raise ValueError(
                "Gene essentiality expects max_seq_len to equal "
                "selected_gene_count + 1 for the <cls> token."
            )

        # Set at data load time: True for genes with CRISPR data in depmap
        self.valid_gene_mask: np.ndarray | None = None   # shape (gene_num,), bool
        self.n_valid_genes: int = 0
        self.fold_gene_indices: np.ndarray | None = None
        self.fold_valid_gene_mask: np.ndarray | None = None
        self.fold_n_valid_genes: int = 0

        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size: int = 0
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.backbone_optimizer_enabled = False

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "gene_essent" in cfg.finetune:
            return cfg.finetune.gene_essent
        raise ValueError(
            "Could not find gene essentiality config. Expected cfg.finetune.gene_essent."
        )

    @staticmethod
    def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
        if "pretrain" in cfg:
            return cfg.pretrain
        raise ValueError("Could not find model architecture config. Expected cfg.pretrain.")

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

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / TASK_NAME / self._output_variant()

    def _output_prefix(self) -> str:
        return f"{TASK_NAME}_{self._output_variant()}"

    def _resolve_gene_list_path(self) -> Path:
        gene_list_path = getattr(self.task_cfg, "gene_list_path", None)
        if gene_list_path:
            return Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        return ROOT / "scbFM" / "data" / "gene_list.txt"

    # ------------------------------------------------------------------
    # Runtime setup
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

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
            "cell_line_id",
        ]
        fieldnames = [f for f in preferred if any(f in row for row in rows)]
        extra_fields = sorted({f for row in rows for f in row if f not in fieldnames})
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
            OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8"
        )
        self._write_json(
            out_dir / f"{prefix}_run_metadata.json",
            {
                "task": self.task_name,
                "finetune_mode": self._finetune_mode(),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "n_valid_genes": self.n_valid_genes,
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
                if field not in skip_fields
                and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for field in numeric_fields:
            values = np.asarray(
                [float(row[field]) for row in rows if field in row], dtype=float
            )
            values = values[~np.isnan(values)]
            aggregate[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            aggregate[f"{field}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return aggregate

    # ------------------------------------------------------------------
    # Checkpoint paths
    # ------------------------------------------------------------------

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                "finetune.gene_essent.pretrained_model_paths must define "
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
                "Missing checkpoint paths in finetune.gene_essent.pretrained_model_paths: "
                f"{missing}"
            )
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
        return checkpoint_paths

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_depmap_data(self) -> tuple[object, np.ndarray, list[str]]:
        """Load expression + CRISPR data from the combined depmap.h5ad.

        The file must contain layers 'expr_array' (expression) and 'essen_array'
        (CRISPR scores), with ENSG IDs as var_names. Both layers share the same obs.

        Returns (X_expression, Y_targets, cell_ids).
        Y_targets has shape (n_cells, gene_num) — aligned to gene_list.txt.
        Genes absent from depmap are filled with 0; self.valid_gene_mask marks which genes
        have actual CRISPR data.
        """
        depmap_path_cfg = getattr(self.task_cfg, "depmap_data_path", None)
        if not depmap_path_cfg:
            raise ValueError("finetune.gene_essent.depmap_data_path must be set.")

        depmap_path = Path(hydra.utils.to_absolute_path(str(depmap_path_cfg)))
        if not depmap_path.exists():
            raise FileNotFoundError(f"depmap.h5ad not found: {depmap_path}")

        log.info("Loading depmap h5ad from %s", depmap_path)
        combined = ad.read_h5ad(depmap_path)
        log.info("Loaded: %d cell lines × %d genes", combined.n_obs, combined.n_vars)

        if "expr_array" not in combined.layers:
            raise KeyError("depmap.h5ad must contain an 'expr_array' layer for expression data.")
        if "essen_array" not in combined.layers:
            raise KeyError("depmap.h5ad must contain an 'essen_array' layer for CRISPR scores.")

        # Extract CRISPR targets before expression rows are aligned to the model gene list.
        essen_X = combined.layers["essen_array"]
        if sparse.issparse(essen_X):
            essen_X = essen_X.toarray()
        essen_X = np.asarray(essen_X, dtype=np.float32)
        combined_obs_to_idx = {str(obs): i for i, obs in enumerate(combined.obs_names)}

        # Build expression AnnData from the expr_array layer. Preserve sparse
        # storage so each DDP rank does not materialize an unnecessary dense copy.
        expr_X = combined.layers["expr_array"]
        if sparse.issparse(expr_X):
            expr_X = expr_X.tocsr().astype(np.float32, copy=False)
        else:
            expr_X = np.asarray(expr_X, dtype=np.float32)
        expr_adata = ad.AnnData(X=expr_X)
        expr_adata.var_names = combined.var_names.copy()
        expr_adata.obs_names = combined.obs_names.copy()

        # Align raw expression to the backbone vocabulary. Quantile binning is
        # deferred to Dataset.__getitem__ so DataLoader workers can do it.
        gene_list_path = self._resolve_gene_list_path()
        should_preprocess = bool(getattr(self.task_cfg, "preprocess", True))
        expr_adata, missing_genes = reindex_adata_genes(
            expr_adata,
            gene_list_path=gene_list_path,
        )
        if should_preprocess:
            log.info(
                "Aligned raw expression for worker-side sequence binning: "
                "%d target genes missing, shape %s",
                len(missing_genes),
                expr_adata.shape,
            )
        else:
            log.info(
                "Expression reindexed (no preprocessing): %d genes missing, shape %s",
                len(missing_genes),
                expr_adata.shape,
            )

        self._missing_genes_note = (
            f"Model genes missing from DepMap GEX and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
        ) if missing_genes else ""

        expected_gene_num = int(self.model_cfg.gene_num)
        if expr_adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes after preprocessing, got {expr_adata.n_vars}."
            )
        if not should_preprocess:
            validate_token_matrix(
                expr_adata.X,
                bin_num=int(self.model_cfg.bin_num),
                name="gene essentiality expression input",
            )

        # Align CRISPR rows to the expression AnnData obs order.
        remaining_obs = list(expr_adata.obs_names.astype(str))
        essen_row_order = [combined_obs_to_idx[obs] for obs in remaining_obs]
        essen_X = essen_X[essen_row_order]

        # Reindex CRISPR targets to match gene_list (same order as expr_adata.var_names).
        # DepMap screens all protein-coding genes, so coverage should be ~complete.
        # Any gene not found in depmap is filled with 0 and excluded via valid_gene_mask.
        gene_list = list(expr_adata.var_names.astype(str))
        gene_num = len(gene_list)
        depmap_gene_to_idx = {str(g): j for j, g in enumerate(combined.var_names)}

        valid_mask = np.zeros(gene_num, dtype=bool)
        depmap_src_cols: list[int] = []
        gene_list_tgt_cols: list[int] = []
        for i, g in enumerate(gene_list):
            if g in depmap_gene_to_idx:
                depmap_src_cols.append(depmap_gene_to_idx[g])
                gene_list_tgt_cols.append(i)
                valid_mask[i] = True

        if valid_mask.sum() == 0:
            raise ValueError(
                "No genes matched between the scbFM gene list and DepMap CRISPR var_names. "
                "Check that depmap.h5ad var_names use the same ID format as data/gene_list.txt."
            )

        n_missing = gene_num - int(valid_mask.sum())
        if n_missing > 0:
            log.warning(
                "%d / %d gene list genes have no CRISPR data in depmap.h5ad and will be "
                "excluded from the loss. Check gene ID formats if this number is unexpectedly large.",
                n_missing,
                gene_num,
            )
        log.info(
            "DepMap reindexed to gene list: %d / %d genes have CRISPR scores",
            int(valid_mask.sum()),
            gene_num,
        )

        self.valid_gene_mask = valid_mask
        self.n_valid_genes = int(valid_mask.sum())

        # Build full (n_cells, gene_num) target matrix aligned to gene_list
        n_cells = len(remaining_obs)
        Y_targets = np.zeros((n_cells, gene_num), dtype=np.float32)
        Y_targets[:, gene_list_tgt_cols] = essen_X[:, depmap_src_cols]

        cell_ids = remaining_obs
        return expr_adata.X, Y_targets, cell_ids

    def _select_training_hvg_indices(
        self,
        X_expression,
        train_idx: np.ndarray,
    ) -> np.ndarray:
        """Fit the 1,199-gene input vocabulary using training cell lines only."""
        if X_expression.shape[1] < self.selected_gene_count:
            raise ValueError(
                f"Cannot select {self.selected_gene_count} HVGs from "
                f"{X_expression.shape[1]} genes."
            )
        hvg_adata = ad.AnnData(X=X_expression[train_idx])
        hvg_stats = sc.pp.highly_variable_genes(
            hvg_adata,
            n_top_genes=self.selected_gene_count,
            flavor=str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
            inplace=False,
        )
        selected = np.flatnonzero(hvg_stats["highly_variable"].to_numpy())
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
                order = np.argsort(
                    -np.nan_to_num(scores, nan=-np.inf),
                    kind="stable",
                )
            selected = selected[order[: self.selected_gene_count]]
        if selected.size != self.selected_gene_count:
            raise RuntimeError(
                f"Scanpy selected {selected.size} HVGs; expected exactly "
                f"{self.selected_gene_count}."
            )
        log.info(
            "Selected %d HVGs from %d training cell lines with flavor=%s",
            selected.size,
            len(train_idx),
            str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
        )
        return selected.astype(np.int64, copy=False)

    # ------------------------------------------------------------------
    # CV splits
    # ------------------------------------------------------------------

    def _build_cv_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 5))
        if n_splits < 2:
            raise ValueError("finetune.gene_essent.cv_folds must be at least 2.")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        return list(splitter.split(np.arange(n_samples)))

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _build_loaders(
        self,
        X_expression_train,
        Y_train: np.ndarray,
        X_expression_test,
        Y_test: np.ndarray,
    ) -> None:
        if self.fold_gene_indices is None:
            raise RuntimeError("Training-fold HVGs have not been selected.")
        batch_size = int(getattr(self.task_cfg, "batch_size", 8))
        num_workers = int(getattr(self.task_cfg, "num_workers", 0))
        if num_workers < 0:
            raise ValueError("finetune.gene_essent.num_workers must be non-negative.")
        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            prefetch_factor = int(getattr(self.task_cfg, "prefetch_factor", 2))
            if prefetch_factor <= 0:
                raise ValueError(
                    "finetune.gene_essent.prefetch_factor must be positive."
                )
            loader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": False,
                }
            )

        dataset_kwargs = {
            "fixed_gene_indices": self.fold_gene_indices,
            "bin_num": int(self.model_cfg.bin_num),
            "cls_gene_id": self.cls_gene_id,
            "gene_token_offset": self.gene_token_offset,
            "cls_value": self.cls_value,
            "seed": int(getattr(self.task_cfg, "random_seed", 42)),
            "do_binning": bool(getattr(self.task_cfg, "preprocess", True)),
        }
        train_dataset = GeneEssentDataset(
            X_expression_train,
            Y_train,
            **dataset_kwargs,
        )
        test_dataset = GeneEssentDataset(
            X_expression_test,
            Y_test,
            **dataset_kwargs,
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

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

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

        head = GeneEssentPredHead(
            embedding_dim=int(self.model_cfg.embsize),
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 512)),
            bottleneck_dim=int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
            context_pooling=str(getattr(self.task_cfg, "head_pooling", "none")),
        )
        if self.is_master:
            input_dim = int(self.model_cfg.embsize) * (
                3 if head.context_pooling == "mean_cls"
                else 2 if head.context_pooling in {"cls", "mean"}
                else 1
            )
            log.info(
                "Gene essentiality head pooling: %s | dims: %d -> %d -> %d -> 1",
                head.context_pooling,
                input_dim,
                int(getattr(self.task_cfg, "head_hidden_dim", 512)),
                int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
            )

        if finetune_mode == "adapters":
            backbone.add_adapters(
                bottleneck_dim=int(getattr(self.task_cfg, "adapter_bottleneck_dim", 32)),
                dropout=float(getattr(self.task_cfg, "adapter_dropout", 0.0)),
                after_attention=bool(getattr(self.task_cfg, "adapter_after_attention", True)),
                after_ff=bool(getattr(self.task_cfg, "adapter_after_ff", True)),
            )

        model = CancerFoundationGeneEssentModel(backbone, head)

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

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _build_optimization(self) -> None:
        if not hasattr(self.task_cfg, "head_learning_rate"):
            raise ValueError("finetune.gene_essent.head_learning_rate must be set.")
        if not hasattr(self.task_cfg, "backbone_learning_rate"):
            raise ValueError("finetune.gene_essent.backbone_learning_rate must be set.")
        head_learning_rate = float(self.task_cfg.head_learning_rate)
        backbone_learning_rate = float(self.task_cfg.backbone_learning_rate)
        adapter_learning_rate = float(
            getattr(self.task_cfg, "adapter_learning_rate", head_learning_rate)
        )

        model = self.model.module if isinstance(self.model, DDP) else self.model
        head_params = [p for p in model.to_out.parameters() if p.requires_grad]
        head_param_ids = {id(p) for p in head_params}
        adapter_params = [p for p in model.adapter_parameters() if p.requires_grad]
        adapter_param_ids = {id(p) for p in adapter_params}
        backbone_params = [
            p
            for p in model.parameters()
            if p.requires_grad and id(p) not in head_param_ids and id(p) not in adapter_param_ids
        ]

        param_groups = []
        if backbone_params and self.backbone_optimizer_enabled:
            param_groups.append({"params": backbone_params, "lr": backbone_learning_rate, "name": "backbone"})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": adapter_learning_rate, "name": "adapters"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_learning_rate, "name": "head"})
        if not param_groups:
            raise ValueError("No trainable parameters found for gene essentiality.")

        max_lrs = [float(g["lr"]) for g in param_groups]
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
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 20)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 1)),
            max_lrs=max_lrs,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 5)),
            gamma=float(getattr(self.task_cfg, "gamma", 1.0)),
        )

        if self.is_master:
            group_summaries = [
                f"{g.get('name', i)}: params={sum(p.numel() for p in g['params'])}, "
                f"max_lr={lr:.2e}, min_lr={lr * min_lr_ratio:.2e}"
                for i, (g, lr) in enumerate(zip(param_groups, max_lrs))
            ]
            log.info("Optimizer parameter groups: %s", "; ".join(group_summaries))

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        burn_in_epochs = int(getattr(self.task_cfg, "burn_in_epochs", 0))
        if (
            self._finetune_mode() != "full_ft"
            or burn_in_epochs <= 0
            or self.backbone_optimizer_enabled
            or epoch <= burn_in_epochs
        ):
            return
        self.backbone_optimizer_enabled = True
        self._build_optimization()
        if self.is_master:
            log.info(
                "Finished %d burn-in epochs; enabled backbone optimization.", burn_in_epochs
            )

    def _optimizer_parameters(self) -> list[torch.nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _move_batch_to_device(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE over finite target values only — handles NaN CRISPR scores for unscreened entries."""
        finite = torch.isfinite(targets)
        if not finite.any():
            return torch.zeros(1, device=preds.device, requires_grad=True).squeeze()
        return F.mse_loss(preds[finite], targets[finite])

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_enable_backbone_optimizer(epoch)
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 1)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss = 0.0
        if self.fold_valid_gene_mask is None:
            raise RuntimeError("Fold-level valid-gene mask is unavailable.")
        valid_mask = torch.as_tensor(
            self.fold_valid_gene_mask,
            device=self.device,
        )

        for step_idx, (batch, targets) in enumerate(self.train_loader, start=1):
            batch = self._move_batch_to_device(batch)
            targets = targets.to(self.device, non_blocking=True)

            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and step_idx % grad_acc_steps != 0
            )
            sync_context = self.model.no_sync() if use_no_sync else nullcontext()

            with sync_context:
                preds = self.model(batch)
                loss = self._masked_mse(preds[:, valid_mask], targets[:, valid_mask])
                (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)
        self.scheduler.step()
        return {"loss": epoch_loss}

    def _evaluate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        if self.fold_valid_gene_mask is None:
            raise RuntimeError("Fold-level valid-gene mask is unavailable.")
        valid_mask = torch.as_tensor(
            self.fold_valid_gene_mask,
            device=self.device,
        )

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for batch, targets in self.test_loader:
                batch = self._move_batch_to_device(batch)
                targets = targets.to(self.device, non_blocking=True)
                preds = self.model(batch)
                loss = self._masked_mse(preds[:, valid_mask], targets[:, valid_mask])
                running_loss += loss.item()
                all_preds.append(preds[:, valid_mask])
                all_targets.append(targets[:, valid_mask])

        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)

        if self.is_distributed:
            all_preds_t = distributed_concat(all_preds_t, self.test_dataset_size, self.world_size)
            all_targets_t = distributed_concat(all_targets_t, self.test_dataset_size, self.world_size)

        preds_np = all_preds_t.cpu().float().numpy()    # (n_test, n_valid_genes)
        targets_np = all_targets_t.cpu().float().numpy()

        # PCC and SCC per cell line, then mean
        pccs, sccs = [], []
        for i in range(preds_np.shape[0]):
            p, t = preds_np[i], targets_np[i]
            finite_mask = np.isfinite(p) & np.isfinite(t)
            if finite_mask.sum() < 2:
                continue
            pcc, _ = pearsonr(p[finite_mask], t[finite_mask])
            scc, _ = spearmanr(p[finite_mask], t[finite_mask])
            pccs.append(float(pcc) if np.isfinite(pcc) else float("nan"))
            sccs.append(float(scc) if np.isfinite(scc) else float("nan"))

        test_loss = running_loss / max(1, len(self.test_loader))
        if self.is_distributed:
            test_loss = get_reduced(test_loss, self.local_rank, 0, self.world_size)

        return {
            "loss": float(test_loss),
            "pcc": float(np.nanmean(pccs)) if pccs else float("nan"),
            "scc": float(np.nanmean(sccs)) if sccs else float("nan"),
            "pcc_per_cell_line": pccs,
            "scc_per_cell_line": sccs,
            "n_test_samples": preds_np.shape[0],
        }

    # ------------------------------------------------------------------
    # Results writing
    # ------------------------------------------------------------------

    def _flatten_fold_metrics(
        self,
        model_key: str,
        fold: int,
        n_folds: int,
        checkpoint_path: str,
        train_metrics: dict[str, float],
        test_metrics: dict[str, object],
    ) -> dict[str, object]:
        return {
            "model": model_key,
            "fold": fold,
            "n_folds": n_folds,
            "finetune_mode": self._finetune_mode(),
            "checkpoint_path": checkpoint_path,
            "train_loss": float(train_metrics["loss"]),
            "test_loss": float(test_metrics["loss"]),
            "test_pcc": float(test_metrics["pcc"]),
            "test_scc": float(test_metrics["scc"]),
            "n_test_samples": int(test_metrics["n_test_samples"]),
            "n_valid_genes": self.fold_n_valid_genes,
        }

    def _per_cell_line_rows(
        self,
        model_key: str,
        fold: int,
        checkpoint_path: str,
        test_cell_ids: list[str],
        test_metrics: dict[str, object],
    ) -> list[dict[str, object]]:
        pccs = test_metrics["pcc_per_cell_line"]
        sccs = test_metrics["scc_per_cell_line"]
        rows = []
        for cell_id, pcc, scc in zip(test_cell_ids, pccs, sccs):
            rows.append(
                {
                    "model": model_key,
                    "fold": fold,
                    "finetune_mode": self._finetune_mode(),
                    "checkpoint_path": checkpoint_path,
                    "cell_line_id": cell_id,
                    "pcc": pcc,
                    "scc": scc,
                }
            )
        return rows

    def _write_model_results(
        self,
        checkpoint_path: str,
        fold_rows: list[dict[str, object]],
        cell_line_rows: list[dict[str, object]],
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
        self._write_csv(out_dir / f"{prefix}_{model_key}_cell_line_metrics.csv", cell_line_rows)
        return aggregate

    def _cleanup_fold_state(self) -> None:
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.fold_gene_indices = None
        self.fold_valid_gene_mask = None
        self.fold_n_valid_genes = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        try:
            self._setup_runtime()

            X_expression, Y_targets, cell_ids = self._load_depmap_data()
            n_samples = len(cell_ids)
            splits = self._build_cv_splits(n_samples)
            fold_hvg_indices = [
                self._select_training_hvg_indices(X_expression, train_idx)
                for train_idx, _ in splits
            ]
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            if self.is_master:
                log.info(
                    "Gene essentiality CV: cell_lines=%d, valid_genes=%d, folds=%d",
                    n_samples,
                    self.n_valid_genes,
                    len(splits),
                )

            epochs = int(getattr(self.task_cfg, "epochs", 20))
            aggregate_rows: list[dict[str, object]] = []

            for model_key, checkpoint_path in checkpoint_paths.items():
                fold_rows: list[dict[str, object]] = []
                cell_line_rows: list[dict[str, object]] = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + fold_idx
                    )

                    self.fold_gene_indices = fold_hvg_indices[fold_idx - 1]
                    self.fold_valid_gene_mask = self.valid_gene_mask[
                        self.fold_gene_indices
                    ]
                    self.fold_n_valid_genes = int(self.fold_valid_gene_mask.sum())
                    if self.fold_n_valid_genes < 2:
                        raise ValueError(
                            "Fewer than two selected HVGs have DepMap CRISPR targets."
                        )

                    X_train = X_expression[train_idx]
                    Y_train = Y_targets[train_idx]
                    X_test = X_expression[test_idx]
                    Y_test = Y_targets[test_idx]
                    test_cell_ids = [cell_ids[i] for i in test_idx]

                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | train=%d, test=%d",
                            model_key, fold_idx, len(splits), len(train_idx), len(test_idx),
                        )

                    self._build_loaders(X_train, Y_train, X_test, Y_test)
                    self._build_model(checkpoint_path)
                    self._build_optimization()

                    last_train_metrics = {"loss": float("nan")}
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        if self.is_master:
                            log.info(
                                "Model %s | Fold %d/%d | Epoch %d | Train Loss: %.6f",
                                model_key, fold_idx, len(splits), epoch, last_train_metrics["loss"],
                            )

                    test_metrics = self._evaluate()
                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | Test Loss: %.6f | PCC: %.4f | SCC: %.4f",
                            model_key, fold_idx, len(splits),
                            test_metrics["loss"], test_metrics["pcc"], test_metrics["scc"],
                        )
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
                        cell_line_rows.extend(
                            self._per_cell_line_rows(
                                model_key=model_key,
                                fold=fold_idx,
                                checkpoint_path=checkpoint_path,
                                test_cell_ids=test_cell_ids,
                                test_metrics=test_metrics,
                            )
                        )
                    self._cleanup_fold_state()

                if self.is_master:
                    aggregate_rows.append(
                        self._write_model_results(checkpoint_path, fold_rows, cell_line_rows)
                    )
                if self.is_distributed:
                    dist.barrier()

            if self.is_master:
                out_dir = self._task_output_dir()
                output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
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
