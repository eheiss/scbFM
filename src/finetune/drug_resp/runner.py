from __future__ import annotations

import csv
import fcntl
import hashlib
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
import pandas as pd
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
from finetune.canc_type_class.runner import (
    GroupedCosineWarmupUpdateScheduler,
    _quantile_bin_expression,
    _set_finetune_training_mode,
)
from finetune.training_correctness import (
    add_optimizer_parameter_group,
    is_accumulation_boundary,
    normalize_accumulated_gradients,
    validate_backbone_checkpoint,
)
from preprocess import (
    reindex_adata_genes,
    validate_token_matrix,
)
from run_provenance import complete_run_metadata, start_run_metadata
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
TASK_NAME = "drug_resp"
CHECKPOINT_MODEL_KEYS = ("pretrain_sc", "pretrain_bulk", "preadapt_sc", "preadapt_bulk")
RANDOM_INIT_MODEL_KEY = "random_init"
MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class DrugRespPredHead(nn.Module):
    """[cell_emb ‖ drug_emb] → MLP → IC50 scalar."""

    def __init__(
        self,
        cell_emb_dim: int,
        drug_emb_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = cell_emb_dim + drug_emb_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SELU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, cell_emb: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        # cell_emb: (B, cell_emb_dim)   drug_emb: (B, drug_emb_dim)
        x = torch.cat([cell_emb, drug_emb], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B,)


class DrugRespModel(nn.Module):
    """CancerFoundation backbone + sample/drug fusion head."""

    def __init__(
        self,
        backbone: CancerFoundationBackbone,
        head: DrugRespPredHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def pool_cell_embeddings(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0, :]

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None,
        drug_emb: torch.Tensor,
        cell_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cell_emb is None:
            if batch is None:
                raise ValueError("A backbone input batch is required when cell_emb is absent.")
            hidden = self.backbone(
                batch["gene_ids"],
                batch["expr"],
                src_key_padding_mask=batch.get("attention_key_padding_mask"),
            )
            cell_emb = self.pool_cell_embeddings(hidden)
        return self.head(cell_emb, drug_emb)  # (B,)

    def add_adapters(self, **kwargs) -> nn.ModuleList:
        return self.backbone.add_adapters(**kwargs)

    def adapter_parameters(self):
        return self.backbone.adapter_parameters()

    def enable_grad_checkpoint(self):
        self.backbone.enable_grad_checkpoint()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DrugRespDataset(Dataset):
    """Each sample is one (cell_line, drug) pair → IC50."""

    def __init__(
        self,
        X_cell: np.ndarray,           # (n_cells, gene_num) expression matrix
        drug_emb_matrix: np.ndarray,  # (n_drugs, drug_emb_dim) float32
        cell_idxs: np.ndarray,        # (n_pairs,) int — index into X_cell
        drug_idxs: np.ndarray,        # (n_pairs,) int — index into drug_emb_matrix
        ic50_values: np.ndarray,      # (n_pairs,) float32
        fixed_gene_indices: np.ndarray,
        bin_num: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
        seed: int,
        do_binning: bool,
    ) -> None:
        self.X_cell = X_cell
        self.drug_emb_matrix = drug_emb_matrix
        self.cell_idxs = cell_idxs.astype(np.int64)
        self.drug_idxs = drug_idxs.astype(np.int64)
        self.ic50_values = ic50_values.astype(np.float32)
        self.fixed_gene_indices = np.asarray(fixed_gene_indices, dtype=np.int64)
        self.bin_num = int(bin_num)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.seed = int(seed)
        self.do_binning = bool(do_binning)

    def __len__(self) -> int:
        return len(self.ic50_values)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        cell_idx = int(self.cell_idxs[idx])
        row = self.X_cell[cell_idx]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()
        selected_values = values[self.fixed_gene_indices].astype(np.float32, copy=False)
        if self.do_binning:
            rng = np.random.default_rng(self.seed + cell_idx)
            selected_values = _quantile_bin_expression(
                selected_values,
                self.bin_num,
                rng,
            ).astype(np.float32)

        gene_ids = torch.from_numpy(
            self.fixed_gene_indices.astype(np.int64, copy=False)
            + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        expression = torch.from_numpy(selected_values)
        expression = torch.cat((torch.tensor([self.cls_value]), expression))

        drug_emb = torch.as_tensor(
            self.drug_emb_matrix[self.drug_idxs[idx]], dtype=torch.float32
        )
        target = torch.as_tensor(self.ic50_values[idx], dtype=torch.float32)
        return (
            torch.tensor(cell_idx, dtype=torch.long),
            {"gene_ids": gene_ids, "expr": expression},
            drug_emb,
            target,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class DrugRespRunner:
    task_name = TASK_NAME
    config_node = TASK_NAME

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
                "Drug response expects max_seq_len to equal selected_gene_count + 1 "
                "for the <cls> token."
            )

        self.train_loader: DataLoader | None = None
        self.train_dataset_size: int = 0
        self.test_loader: DataLoader | None = None
        self.test_dataset_size: int = 0
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.backbone_optimizer_enabled = False
        self.cell_emb_cache: torch.Tensor | None = None
        self.fold_gene_indices: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "drug_resp" in cfg.finetune:
            return cfg.finetune.drug_resp
        raise ValueError("Could not find drug response config. Expected cfg.finetune.drug_resp.")

    @staticmethod
    def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
        if "pretrain" in cfg:
            return cfg.pretrain
        raise ValueError("Could not find model architecture config. Expected cfg.pretrain.")

    def _finetune_mode(self) -> str:
        return str(getattr(self.task_cfg, "finetune_mode", "head_only"))

    def _output_suffix(self) -> str:
        configured = str(getattr(self.task_cfg, "output_suffix", "") or "").strip()
        if configured and not re.fullmatch(r"[A-Za-z0-9_-]+", configured):
            raise ValueError(
                "finetune.drug_resp.output_suffix may contain only letters, numbers, "
                "underscores, and hyphens."
            )
        return configured

    def _task_output_dir(self) -> Path:
        mode = self._finetune_mode()
        suffix = self._output_suffix()
        directory = f"{mode}_{suffix}" if suffix else mode
        return ROOT / "output" / TASK_NAME / directory

    def _output_prefix(self) -> str:
        mode = self._finetune_mode()
        suffix = self._output_suffix()
        return f"{TASK_NAME}_{mode}_{suffix}" if suffix else f"{TASK_NAME}_{mode}"

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
    def _write_csv(path: Path, rows: list[dict], comment: str = "") -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        preferred = ["model", "fold", "n_folds", "finetune_mode", "checkpoint_path", "cell_line_id"]
        fieldnames = [f for f in preferred if any(f in row for row in rows)]
        extra_fields = sorted({f for row in rows for f in row if f not in fieldnames})
        with path.open("w", newline="") as handle:
            if comment:
                handle.write(f"# {comment}\n")
            writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _get_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT / "scbFM",
                check=True, capture_output=True, text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def _save_run_metadata(self, checkpoint_paths: dict) -> None:
        if not self.is_master:
            return
        out_dir = self._task_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_prefix()
        (out_dir / f"{prefix}_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8"
        )
        self._run_metadata_path = out_dir / f"{prefix}_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": self.task_name,
                "finetune_mode": self._finetune_mode(),
                "head_only_backbone_eval": self._finetune_mode() == "head_only",
                "output_suffix": self._output_suffix(),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "cv_fold_manifest_path": str(
                    getattr(self, "_cv_fold_manifest_path", "")
                ),
                "cv_fold_fingerprint": str(
                    getattr(self, "_cv_fold_fingerprint", "")
                ),
                "epochs": int(getattr(self.task_cfg, "epochs", 20)),
                "warmup_epochs": int(getattr(self.task_cfg, "warmup_epochs", 2)),
                "batch_size_per_gpu": int(getattr(self.task_cfg, "batch_size", 4)),
                "gradient_accumulation_steps": int(
                    getattr(self.task_cfg, "grad_accumulation_steps", 4)
                ),
                "world_size": int(self.world_size),
                "global_effective_batch_size": int(
                    getattr(self.task_cfg, "batch_size", 4)
                )
                * int(getattr(self.task_cfg, "grad_accumulation_steps", 4))
                * int(self.world_size),
                "resume_completed_models": bool(
                    getattr(self.task_cfg, "resume_completed_models", False)
                ),
                "reused_completed_models": list(
                    getattr(self, "_reused_completed_models", [])
                ),
            },
            checkpoint_paths=checkpoint_paths,
            repo_dir=ROOT / "scbFM",
        )

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(
                csv.DictReader(line for line in handle if not line.startswith("#"))
            )

    def _load_completed_model_results(
        self,
        checkpoint_paths: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        if not bool(getattr(self.task_cfg, "resume_completed_models", False)):
            return {}

        out_dir = self._task_output_dir()
        prefix = self._output_prefix()
        completed_candidates = {
            model_key: out_dir / f"{prefix}_{model_key}_evaluation_metrics.csv"
            for model_key in checkpoint_paths
        }
        completed_candidates = {
            model_key: path
            for model_key, path in completed_candidates.items()
            if path.is_file()
        }
        if not completed_candidates:
            return {}

        metadata_path = out_dir / f"{prefix}_run_metadata.json"
        if not metadata_path.is_file():
            raise ValueError(
                "Cannot resume drug response: completed model outputs exist without "
                f"run metadata at {metadata_path}."
            )
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)

        config_path = out_dir / f"{prefix}_config.yaml"
        if not config_path.is_file():
            raise ValueError(
                "Cannot resume drug response: completed model outputs exist without "
                f"the resolved run config at {config_path}."
            )
        previous_cfg = OmegaConf.load(config_path)
        previous_task_cfg = previous_cfg.finetune.drug_resp
        previous_model_cfg = previous_cfg.pretrain

        expected_metadata = {
            "task": self.task_name,
            "finetune_mode": self._finetune_mode(),
            "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
            "cv_fold_fingerprint": str(self._cv_fold_fingerprint),
            "epochs": int(getattr(self.task_cfg, "epochs", 20)),
            "warmup_epochs": int(getattr(self.task_cfg, "warmup_epochs", 2)),
            "batch_size_per_gpu": int(getattr(self.task_cfg, "batch_size", 4)),
            "gradient_accumulation_steps": int(
                getattr(self.task_cfg, "grad_accumulation_steps", 4)
            ),
            "world_size": int(self.world_size),
        }
        mismatches = {
            field: {"expected": expected, "observed": metadata.get(field)}
            for field, expected in expected_metadata.items()
            if metadata.get(field) != expected
        }
        previous_checkpoints = metadata.get("checkpoint_paths", {})
        for model_key in completed_candidates:
            expected_path = str(checkpoint_paths[model_key])
            observed_path = str(previous_checkpoints.get(model_key, ""))
            if observed_path != expected_path:
                mismatches[f"checkpoint_paths.{model_key}"] = {
                    "expected": expected_path,
                    "observed": observed_path,
                }
        task_config_fields = (
            "expression_data_path",
            "ic50_data_path",
            "drug_features_path",
            "gene_list_path",
            "preprocess",
            "hvg_selection_method",
            "random_seed",
            "burn_in_epochs",
            "adapter_bottleneck_dim",
            "adapter_dropout",
            "adapter_learning_rate",
            "adapter_after_attention",
            "adapter_after_ff",
            "head_hidden_dim",
            "head_bottleneck_dim",
            "head_learning_rate",
            "backbone_learning_rate",
            "max_grad_norm",
            "min_lr",
            "min_lr_ratio",
        )
        model_config_fields = (
            "gene_num",
            "selected_gene_count",
            "max_seq_len",
            "bin_num",
            "embsize",
            "nlayers",
            "nheads",
            "d_hid",
            "dropout",
            "value_encoder_max_value",
        )
        for field in task_config_fields:
            expected = getattr(self.task_cfg, field, None)
            observed = getattr(previous_task_cfg, field, None)
            if observed != expected:
                mismatches[f"finetune.drug_resp.{field}"] = {
                    "expected": expected,
                    "observed": observed,
                }
        for field in model_config_fields:
            expected = getattr(self.model_cfg, field, None)
            observed = getattr(previous_model_cfg, field, None)
            if observed != expected:
                mismatches[f"pretrain.{field}"] = {
                    "expected": expected,
                    "observed": observed,
                }
        if mismatches:
            raise ValueError(
                "Cannot resume drug response because the interrupted run is not "
                f"benchmark-compatible with this run: {mismatches}"
            )

        expected_folds = {str(fold) for fold in range(1, expected_metadata["cv_folds"] + 1)}
        expected_epoch_pairs = {
            (str(fold), str(epoch))
            for fold in range(1, expected_metadata["cv_folds"] + 1)
            for epoch in range(1, expected_metadata["epochs"] + 1)
        }
        completed: dict[str, dict[str, str]] = {}
        for model_key, evaluation_path in completed_candidates.items():
            artifact_paths = {
                "evaluation": evaluation_path,
                "fold": out_dir / f"{prefix}_{model_key}_fold_metrics.csv",
                "cell_line": out_dir / f"{prefix}_{model_key}_cell_line_metrics.csv",
                "curve": out_dir / f"{prefix}_{model_key}_training_curves.csv",
            }
            missing = [name for name, path in artifact_paths.items() if not path.is_file()]
            if missing:
                raise ValueError(
                    f"Cannot reuse completed model {model_key!r}; missing artifacts: {missing}"
                )
            rows = {
                name: self._read_csv_rows(path)
                for name, path in artifact_paths.items()
            }
            if len(rows["evaluation"]) != 1:
                raise ValueError(
                    f"Completed model {model_key!r} must have one aggregate row."
                )
            if {row.get("fold") for row in rows["fold"]} != expected_folds:
                raise ValueError(
                    f"Completed model {model_key!r} does not contain all CV folds."
                )
            observed_epoch_pairs = {
                (row.get("fold", ""), row.get("epoch", ""))
                for row in rows["curve"]
            }
            if observed_epoch_pairs != expected_epoch_pairs:
                raise ValueError(
                    f"Completed model {model_key!r} does not contain every fold/epoch."
                )
            for artifact, artifact_rows in rows.items():
                if not artifact_rows or any(
                    row.get("model") != model_key for row in artifact_rows
                ):
                    raise ValueError(
                        f"Completed model {model_key!r} has invalid {artifact} rows."
                    )
            aggregate = rows["evaluation"][0]
            if aggregate.get("checkpoint_path", "") != str(checkpoint_paths[model_key]):
                raise ValueError(
                    f"Completed model {model_key!r} uses a different checkpoint."
                )
            completed[model_key] = aggregate
        return completed

    @staticmethod
    def _aggregate_numeric_rows(rows: list[dict]) -> dict:
        aggregate: dict = {"n_folds": len(rows)}
        skip_fields = {"model", "fold", "n_folds", "finetune_mode", "checkpoint_path"}
        numeric_fields = sorted({
            f for row in rows for f, v in row.items()
            if f not in skip_fields and isinstance(v, (int, float, np.integer, np.floating))
        })
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in rows if field in row], dtype=float)
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
                f"finetune.drug_resp.pretrained_model_paths must define {', '.join(CHECKPOINT_MODEL_KEYS)}."
            )
        configured_model_keys = getattr(self.task_cfg, "model_keys", None)
        if configured_model_keys is None:
            selected_model_keys = list(MODEL_KEYS)
        else:
            selected_model_keys = [str(key) for key in configured_model_keys]
            if not selected_model_keys:
                raise ValueError("finetune.drug_resp.model_keys may not be empty.")
            invalid = sorted(set(selected_model_keys).difference(MODEL_KEYS))
            if invalid:
                raise ValueError(
                    f"Invalid finetune.drug_resp.model_keys: {invalid}; expected a subset "
                    f"of {list(MODEL_KEYS)}."
                )
            if len(set(selected_model_keys)) != len(selected_model_keys):
                raise ValueError("finetune.drug_resp.model_keys contains duplicates.")

        checkpoint_paths: dict[str, str] = {}
        missing = []
        for key in selected_model_keys:
            if key == RANDOM_INIT_MODEL_KEY:
                checkpoint_paths[key] = ""
                continue
            value = paths_cfg.get(key)
            if value:
                checkpoint_paths[key] = str(Path(hydra.utils.to_absolute_path(str(value))))
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f"Missing checkpoint paths in finetune.drug_resp.pretrained_model_paths: {missing}"
            )
        return checkpoint_paths

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_gdsc_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and align expression + IC50 + drug embeddings.

        Returns:
            X_cell          (n_cells, gene_num) token matrix (sparse or dense)
            drug_emb_matrix (n_drugs, drug_emb_dim) float32
            cell_idxs       (n_pairs,) int — index into X_cell rows
            drug_idxs       (n_pairs,) int — index into drug_emb_matrix rows
            ic50_values     (n_pairs,) float32
            pair_cell_ids   (n_pairs,) str — ModelID for each pair (for per-cell-line metrics)
        """
        expr_path = Path(hydra.utils.to_absolute_path(
            str(getattr(self.task_cfg, "expression_data_path", ""))
        ))
        ic50_path = Path(hydra.utils.to_absolute_path(
            str(getattr(self.task_cfg, "ic50_data_path", ""))
        ))
        drug_feat_path = Path(hydra.utils.to_absolute_path(
            str(getattr(self.task_cfg, "drug_features_path", ""))
        ))
        configured_gene_info = getattr(self.task_cfg, "gene_info_path", None)
        gene_info_path = Path(
            hydra.utils.to_absolute_path(
                str(
                    configured_gene_info
                    or ROOT / "scbFM" / "data" / "bulkformer_gene_info.csv"
                )
            )
        )

        paths_to_check = [
            (expr_path, "expression_data_path"),
            (ic50_path, "ic50_data_path"),
            (drug_feat_path, "drug_features_path"),
        ]
        # gene_info only needed for CSV expression input
        if expr_path.suffix != ".h5ad":
            paths_to_check.append((gene_info_path, "gene_info_path"))
        for p, name in paths_to_check:
            if not p.exists():
                raise FileNotFoundError(f"{name} not found: {p}")

        # ── Expression: h5ad or CSV ──────────────────────────────────────
        if expr_path.suffix == ".h5ad":
            log.info("Loading expression h5ad from %s", expr_path)
            adata = ad.read_h5ad(expr_path)
            adata.obs_names = adata.obs_names.astype(str)
            log.info(
                "Expression h5ad: %d cell lines, %d genes",
                adata.n_obs, adata.n_vars,
            )
        else:
            log.info("Loading gene symbol→ENSG mapping from %s", gene_info_path)
            gene_info = pd.read_csv(gene_info_path, usecols=["gene_symbol", "ensg_id"])
            symbol_to_ensg: dict[str, str] = dict(
                zip(gene_info["gene_symbol"].astype(str), gene_info["ensg_id"].astype(str))
            )

            log.info("Loading expression CSV from %s", expr_path)
            expr_df = pd.read_csv(expr_path, index_col=0)
            expr_df.index = expr_df.index.astype(str)

            symbol_cols = [c for c in expr_df.columns if c in symbol_to_ensg]
            ensg_ids = [symbol_to_ensg[c] for c in symbol_cols]
            expr_df = expr_df[symbol_cols].copy()
            expr_df.columns = ensg_ids
            expr_df = expr_df.loc[:, ~expr_df.columns.duplicated(keep="first")]
            log.info(
                "Expression CSV: %d cell lines, %d genes after symbol→ENSG mapping",
                len(expr_df), len(expr_df.columns),
            )

            adata = ad.AnnData(X=expr_df.values.astype(np.float32))
            adata.obs_names = list(expr_df.index)
            adata.var_names = list(expr_df.columns)

        gene_list_path = self._resolve_gene_list_path()
        should_preprocess = bool(getattr(self.task_cfg, "preprocess", True))
        adata, missing_genes = reindex_adata_genes(
            adata,
            gene_list_path=gene_list_path,
        )
        if should_preprocess:
            log.info(
                "Aligned raw expression for on-the-fly sequence binning: "
                "%d target genes missing, shape %s",
                len(missing_genes), adata.shape,
            )
        else:
            log.info(
                "Expression reindexed (no preprocessing): %d genes missing, shape %s",
                len(missing_genes), adata.shape,
            )

        self._missing_genes_note = (
            f"Model genes missing from GDSC GEX and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
        ) if missing_genes else ""

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes after preprocessing, got {adata.n_vars}."
            )
        if not should_preprocess:
            validate_token_matrix(
                adata.X,
                bin_num=int(self.model_cfg.bin_num),
                name="drug response expression input",
            )

        cell_id_to_row: dict[str, int] = {
            str(cid): i for i, cid in enumerate(adata.obs_names)
        }
        X_cell = adata.X  # (n_cells, gene_num), sparse or dense

        # ── Drug features ─────────────────────────────────────────────────
        log.info("Loading drug features from %s", drug_feat_path)
        drug_data = np.load(drug_feat_path, allow_pickle=True)
        drug_ids_arr = drug_data["drug_ids"].astype(str)
        drug_emb_matrix = drug_data["features"].astype(np.float32)  # (n_drugs, drug_emb_dim)
        drug_id_to_emb_idx: dict[str, int] = {did: i for i, did in enumerate(drug_ids_arr)}
        log.info(
            "Drug embeddings: %d drugs, emb_dim=%d", len(drug_ids_arr), drug_emb_matrix.shape[1]
        )

        # ── IC50 CSV ─────────────────────────────────────────────────────
        log.info("Loading IC50 CSV from %s", ic50_path)
        model_id_col = str(getattr(self.task_cfg, "model_id_col", "ModelID"))
        drug_id_col = str(getattr(self.task_cfg, "drug_id_col", "Drug ID"))
        ic50_col = str(getattr(self.task_cfg, "ic50_col", "IC50"))
        dataset_col = str(getattr(self.task_cfg, "dataset_col", "Dataset Version"))

        ic50_df = pd.read_csv(ic50_path)
        ic50_df[model_id_col] = ic50_df[model_id_col].astype(str)
        ic50_df[drug_id_col] = ic50_df[drug_id_col].astype(str)

        # Prefer GDSC2 over GDSC1 for duplicate (cell_line, drug) pairs
        ic50_df = ic50_df.sort_values(
            dataset_col,
            key=lambda s: s.map({"GDSC1": 0, "GDSC2": 1}).fillna(0),
            ascending=True,
        )
        ic50_df = ic50_df.drop_duplicates(subset=[model_id_col, drug_id_col], keep="last")
        ic50_df[ic50_col] = pd.to_numeric(ic50_df[ic50_col], errors="coerce")
        ic50_df = ic50_df[np.isfinite(ic50_df[ic50_col].to_numpy(dtype=float))]
        ic50_df = ic50_df.reset_index(drop=True)

        # Keep pairs with both expression data and drug embeddings
        mask = (
            ic50_df[model_id_col].isin(cell_id_to_row)
            & ic50_df[drug_id_col].isin(drug_id_to_emb_idx)
        )
        ic50_df = ic50_df[mask].reset_index(drop=True)
        log.info("GDSC pairs after filtering: %d", len(ic50_df))

        if len(ic50_df) == 0:
            raise ValueError(
                "No valid (cell_line, drug) pairs found after filtering. "
                "Check that ModelIDs in IC50 CSV match expression CSV, "
                "and Drug IDs match drug_features.npz."
            )

        cell_idxs = np.array(
            [cell_id_to_row[cid] for cid in ic50_df[model_id_col]], dtype=np.int64
        )
        drug_idxs = np.array(
            [drug_id_to_emb_idx[did] for did in ic50_df[drug_id_col]], dtype=np.int64
        )
        ic50_values = ic50_df[ic50_col].values.astype(np.float32)
        pair_cell_ids = ic50_df[model_id_col].values

        self._canonical_adata = adata
        self._canonical_cell_ids = adata.obs_names.astype(str).to_numpy()
        self._pair_drug_ids = ic50_df[drug_id_col].astype(str).to_numpy()

        return X_cell, drug_emb_matrix, cell_idxs, drug_idxs, ic50_values, pair_cell_ids

    def _select_training_hvg_indices(
        self,
        X_cell,
        training_cell_idxs: np.ndarray,
    ) -> np.ndarray:
        """Select the highest-MAD genes on distinct training-fold cell lines."""
        unique_training_cells = np.unique(training_cell_idxs).astype(np.int64)
        if unique_training_cells.size == 0:
            raise ValueError("The drug-response training fold contains no cell lines.")
        if X_cell.shape[1] < self.selected_gene_count:
            raise ValueError(
                f"Cannot select {self.selected_gene_count} genes from "
                f"only {X_cell.shape[1]} genes."
            )
        selection_method = str(
            getattr(self.task_cfg, "hvg_selection_method", "mad")
        ).strip().lower()
        if selection_method != "mad":
            raise ValueError(
                "Drug response supports only hvg_selection_method=mad; "
                f"got '{selection_method}'."
            )

        matrix = X_cell[unique_training_cells]
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
        selected = np.sort(ranked[: self.selected_gene_count]).astype(
            np.int64,
            copy=False,
        )
        if selected.size != self.selected_gene_count:
            raise RuntimeError(
                f"MAD selection selected {selected.size} genes; "
                f"expected exactly {self.selected_gene_count}."
            )
        log.info(
            "Selected %d genes from %d distinct training-fold cell lines "
            "with method=mad | selected MAD min=%.6g median=%.6g max=%.6g",
            selected.size,
            unique_training_cells.size,
            float(np.min(scores[selected])),
            float(np.median(scores[selected])),
            float(np.max(scores[selected])),
        )
        return selected

    def _cell_backbone_batch(self, X_cell, cell_indices: np.ndarray) -> dict[str, torch.Tensor]:
        if self.fold_gene_indices is None:
            raise RuntimeError("Training-fold HVGs have not been selected.")
        expressions: list[torch.Tensor] = []
        for cell_idx in np.asarray(cell_indices, dtype=np.int64):
            row = X_cell[int(cell_idx)]
            values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
            selected_values = values[self.fold_gene_indices].astype(np.float32, copy=False)
            if bool(getattr(self.task_cfg, "preprocess", True)):
                rng = np.random.default_rng(
                    int(getattr(self.task_cfg, "random_seed", 42)) + int(cell_idx)
                )
                selected_values = _quantile_bin_expression(
                    selected_values,
                    int(self.model_cfg.bin_num),
                    rng,
                ).astype(np.float32)
            expression = torch.from_numpy(selected_values)
            expressions.append(
                torch.cat((torch.tensor([self.cls_value]), expression))
            )

        gene_ids = torch.from_numpy(
            self.fold_gene_indices.astype(np.int64, copy=False)
            + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        return {
            "gene_ids": gene_ids.unsqueeze(0).expand(len(expressions), -1),
            "expr": torch.stack(expressions),
        }

    # ------------------------------------------------------------------
    # CV splits
    # ------------------------------------------------------------------

    def _build_cv_splits(self, n_pairs: int) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 5))
        if n_splits < 2:
            raise ValueError("finetune.drug_resp.cv_folds must be at least 2.")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        return list(splitter.split(np.arange(n_pairs)))

    @staticmethod
    def _fold_assignments_from_splits(
        n_pairs: int,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        assignments = np.full(n_pairs, -1, dtype=np.int64)
        all_indices = np.arange(n_pairs, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)
            if np.intersect1d(train_idx, test_idx).size:
                raise ValueError(f"CV fold {fold} contains overlapping train and test pairs.")
            if not np.array_equal(
                np.sort(np.concatenate((train_idx, test_idx))),
                all_indices,
            ):
                raise ValueError(f"CV fold {fold} does not partition every pair exactly once.")
            if np.any(assignments[test_idx] != -1):
                raise ValueError("A pair appears in the test partition of multiple CV folds.")
            assignments[test_idx] = fold
        if np.any(assignments < 1):
            raise ValueError("Every pair must appear in exactly one CV test fold.")
        return assignments

    def _cv_manifest_source_rows(
        self,
        pair_cell_ids: np.ndarray,
        pair_drug_ids: np.ndarray,
        ic50_values: np.ndarray,
    ) -> list[dict[str, str]]:
        if not (
            len(pair_cell_ids) == len(pair_drug_ids) == len(ic50_values)
        ):
            raise ValueError("Drug-response pair metadata arrays must have equal lengths.")
        rows = [
            {
                "cell_line_id": str(cell_id),
                "drug_id": str(drug_id),
                "target": format(float(target), ".9g"),
            }
            for cell_id, drug_id, target in zip(
                pair_cell_ids,
                pair_drug_ids,
                ic50_values,
            )
        ]
        pair_keys = [(row["cell_line_id"], row["drug_id"]) for row in rows]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("Canonical drug-response pairs must be unique after GDSC deduplication.")
        return rows

    def _resolve_cv_fold_manifest_path(
        self,
        source_rows: list[dict[str, str]],
    ) -> Path | None:
        configured = getattr(self.task_cfg, "cv_fold_manifest_path", None)
        if configured is None or not str(configured).strip():
            return None
        configured_str = str(configured).strip()
        if configured_str.lower() != "auto":
            return Path(hydra.utils.to_absolute_path(configured_str))

        ic50_path = Path(
            hydra.utils.to_absolute_path(str(getattr(self.task_cfg, "ic50_data_path", "")))
        )
        signature_payload = {
            "task": self.task_name,
            "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
            "random_seed": int(getattr(self.task_cfg, "random_seed", 42)),
            "pairs": source_rows,
        }
        source_fingerprint = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        self._cv_source_fingerprint = source_fingerprint
        return ic50_path.with_name(
            f"{ic50_path.stem}_{self.task_name}_{source_fingerprint[:16]}_cv_folds.csv"
        )

    def _load_cv_fold_manifest(
        self,
        manifest_path: Path,
        source_rows: list[dict[str, str]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required_fields = {"cell_line_id", "drug_id", "target", "fold"}
        observed_fields = set(rows[0]) if rows else set()
        if not required_fields.issubset(observed_fields):
            raise ValueError(
                f"CV fold manifest {manifest_path} is missing columns "
                f"{sorted(required_fields - observed_fields)}."
            )
        if len(rows) != len(source_rows):
            raise ValueError(
                f"CV fold manifest {manifest_path} contains {len(rows)} pairs; "
                f"expected {len(source_rows)}."
            )

        assignments = np.empty(len(rows), dtype=np.int64)
        for index, (observed, expected) in enumerate(zip(rows, source_rows)):
            if {key: observed[key] for key in expected} != expected:
                raise ValueError(
                    f"CV fold manifest {manifest_path} differs from canonical pair row {index}."
                )
            assignments[index] = int(observed["fold"])

        n_folds = int(getattr(self.task_cfg, "cv_folds", 5))
        if set(assignments.tolist()) != set(range(1, n_folds + 1)):
            raise ValueError(
                f"CV fold manifest {manifest_path} must contain folds 1..{n_folds}."
            )
        indices = np.arange(len(rows), dtype=np.int64)
        return [
            (indices[assignments != fold], indices[assignments == fold])
            for fold in range(1, n_folds + 1)
        ]

    def _build_or_load_cv_splits(
        self,
        pair_cell_ids: np.ndarray,
        pair_drug_ids: np.ndarray,
        ic50_values: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        source_rows = self._cv_manifest_source_rows(
            pair_cell_ids,
            pair_drug_ids,
            ic50_values,
        )
        manifest_path = self._resolve_cv_fold_manifest_path(source_rows)
        if manifest_path is None:
            return self._build_cv_splits(len(source_rows))

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = manifest_path.with_name(f"{manifest_path.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if manifest_path.exists():
                splits = self._load_cv_fold_manifest(manifest_path, source_rows)
            else:
                splits = self._build_cv_splits(len(source_rows))
                assignments = self._fold_assignments_from_splits(len(source_rows), splits)
                temporary_path = manifest_path.with_name(
                    f"{manifest_path.name}.tmp.{os.getpid()}.{self.rank}"
                )
                try:
                    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=["cell_line_id", "drug_id", "target", "fold"],
                        )
                        writer.writeheader()
                        for row, fold in zip(source_rows, assignments):
                            writer.writerow({**row, "fold": int(fold)})
                    os.replace(temporary_path, manifest_path)
                finally:
                    temporary_path.unlink(missing_ok=True)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        assignments = self._fold_assignments_from_splits(len(source_rows), splits)
        fold_fingerprint = hashlib.sha256(
            "\n".join(
                f"{row['cell_line_id']}\t{row['drug_id']}\t{int(fold)}"
                for row, fold in zip(source_rows, assignments)
            ).encode("utf-8")
        ).hexdigest()
        self._cv_fold_manifest_path = manifest_path
        self._cv_fold_fingerprint = fold_fingerprint
        log.info(
            "Using shared drug-response CV fold manifest %s | fingerprint=%s",
            manifest_path,
            fold_fingerprint,
        )
        return splits

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _loader_kwargs(self) -> dict[str, object]:
        num_workers = int(getattr(self.task_cfg, "num_workers", 0))
        if num_workers < 0:
            raise ValueError("finetune.drug_resp.num_workers must be non-negative.")
        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            prefetch_factor = int(getattr(self.task_cfg, "prefetch_factor", 2))
            if prefetch_factor <= 0:
                raise ValueError("finetune.drug_resp.prefetch_factor must be positive.")
            loader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": False,
                }
            )
        return loader_kwargs

    def _build_train_loader(
        self,
        X_cell: np.ndarray,
        drug_emb_matrix: np.ndarray,
        cell_idxs_train: np.ndarray,
        drug_idxs_train: np.ndarray,
        ic50_train: np.ndarray,
    ) -> None:
        """Build random pair-level training batches."""
        if self.fold_gene_indices is None:
            raise RuntimeError("Training-fold HVGs have not been selected.")
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        train_dataset = DrugRespDataset(
            X_cell, drug_emb_matrix, cell_idxs_train, drug_idxs_train, ic50_train,
            fixed_gene_indices=self.fold_gene_indices,
            bin_num=int(self.model_cfg.bin_num),
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=self.cls_value,
            seed=int(getattr(self.task_cfg, "random_seed", 42)),
            do_binning=bool(getattr(self.task_cfg, "preprocess", True)),
        )
        self.train_dataset_size = len(train_dataset)
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=int(getattr(self.task_cfg, "random_seed", 42)),
                drop_last=False,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,
                **self._loader_kwargs(),
            )
        else:
            generator = torch.Generator()
            generator.manual_seed(int(getattr(self.task_cfg, "random_seed", 42)))
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                **self._loader_kwargs(),
            )
        if self.is_master:
            log.info(
                "Training DataLoader: random pair-level batches | pairs=%d | batch_size=%d",
                self.train_dataset_size,
                batch_size,
            )

    def _build_test_loader(
        self,
        X_cell: np.ndarray,
        drug_emb_matrix: np.ndarray,
        cell_idxs_test: np.ndarray,
        drug_idxs_test: np.ndarray,
        ic50_test: np.ndarray,
    ) -> None:
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        loader_kwargs = self._loader_kwargs()
        if self.fold_gene_indices is None:
            raise RuntimeError("Training-fold HVGs have not been selected.")
        test_dataset = DrugRespDataset(
            X_cell, drug_emb_matrix, cell_idxs_test, drug_idxs_test, ic50_test,
            fixed_gene_indices=self.fold_gene_indices,
            bin_num=int(self.model_cfg.bin_num),
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=self.cls_value,
            seed=int(getattr(self.task_cfg, "random_seed", 42)),
            do_binning=bool(getattr(self.task_cfg, "preprocess", True)),
        )
        self.test_dataset_size = len(test_dataset)
        if self.is_distributed:
            test_sampler = SequentialDistributedSampler(
                test_dataset, batch_size=batch_size, world_size=self.world_size, rank=self.rank,
            )
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        else:
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
                int(getattr(self.task_cfg, "num_workers", 0)),
                (
                    int(getattr(self.task_cfg, "prefetch_factor", 2))
                    if int(getattr(self.task_cfg, "num_workers", 0)) > 0
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

    def _build_model(self, checkpoint_path: str, drug_emb_dim: int) -> None:
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
            resolved = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved, map_location="cpu")
            validate_backbone_checkpoint(
                checkpoint,
                resolved,
                gene_num=int(self.model_cfg.gene_num),
                selected_gene_count=self.selected_gene_count,
                max_seq_len=self.max_seq_len,
                bin_num=int(self.model_cfg.bin_num),
            )
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            backbone.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved)
        else:
            log.info("Using randomly initialized backbone")

        cell_emb_dim = int(self.model_cfg.embsize)
        head = DrugRespPredHead(
            cell_emb_dim=cell_emb_dim,
            drug_emb_dim=drug_emb_dim,
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 512)),
            bottleneck_dim=int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
        )

        if finetune_mode == "adapters":
            backbone.add_adapters(
                bottleneck_dim=int(getattr(self.task_cfg, "adapter_bottleneck_dim", 32)),
                dropout=float(getattr(self.task_cfg, "adapter_dropout", 0.0)),
                after_attention=bool(getattr(self.task_cfg, "adapter_after_attention", True)),
                after_ff=bool(getattr(self.task_cfg, "adapter_after_ff", True)),
            )

        model = DrugRespModel(
            backbone,
            head,
        )
        if self.is_master:
            log.info(
                "Drug response cell representation: CLS token | fusion head dims: %d -> %d -> %d -> 1",
                cell_emb_dim + drug_emb_dim,
                int(getattr(self.task_cfg, "head_hidden_dim", 512)),
                int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
            )

        if finetune_mode == "head_only":
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
            self.backbone_optimizer_enabled = False
        elif finetune_mode == "adapters":
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.backbone.adapter_parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            self.backbone_optimizer_enabled = False
        elif finetune_mode == "full_ft":
            for param in model.parameters():
                param.requires_grad = True
            self.backbone_optimizer_enabled = (
                int(getattr(self.task_cfg, "burn_in_epochs", 3)) <= 0
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
        head_lr = float(self.task_cfg.head_learning_rate)
        backbone_lr = float(self.task_cfg.backbone_learning_rate)
        adapter_lr = float(getattr(self.task_cfg, "adapter_learning_rate", head_lr))

        raw = self.model.module if isinstance(self.model, DDP) else self.model
        head_params = [p for p in raw.head.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head_params}
        adapter_params = [p for p in raw.backbone.adapter_parameters() if p.requires_grad]
        adapter_ids = {id(p) for p in adapter_params}
        backbone_params = [
            p for p in raw.backbone.parameters()
            if p.requires_grad and id(p) not in head_ids and id(p) not in adapter_ids
        ]

        param_groups = []
        if backbone_params and self.backbone_optimizer_enabled:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": adapter_lr, "name": "adapters"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
        if not param_groups:
            raise ValueError("No trainable parameters found for drug response.")

        max_lrs = [float(g["lr"]) for g in param_groups]
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = float(configured_ratio) if configured_ratio is not None else min_lr / max(head_lr, 1e-12)

        grad_acc_steps = max(
            1,
            int(getattr(self.task_cfg, "grad_accumulation_steps", 4)),
        )
        updates_per_epoch = math.ceil(len(self.train_loader) / grad_acc_steps)
        epochs = int(getattr(self.task_cfg, "epochs", 20))
        warmup_epochs = int(getattr(self.task_cfg, "warmup_epochs", 2))

        self.optimizer = Adam(param_groups)
        self.scheduler = GroupedCosineWarmupUpdateScheduler(
            self.optimizer,
            max_lrs=max_lrs,
            min_lr_ratio=min_lr_ratio,
            updates_per_epoch=updates_per_epoch,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
        )
        if self.is_master:
            log.info(
                "Update-based LR schedule: updates_per_epoch=%d | total_updates=%d | "
                "warmup_epochs=%d | warmup_updates=%d",
                updates_per_epoch,
                self.scheduler.total_updates,
                warmup_epochs,
                self.scheduler.warmup_updates,
            )

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        burn_in = int(getattr(self.task_cfg, "burn_in_epochs", 3))
        if (
            self._finetune_mode() != "full_ft"
            or burn_in <= 0
            or self.backbone_optimizer_enabled
            or epoch <= burn_in
        ):
            return
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        add_optimizer_parameter_group(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            parameters=raw_model.backbone.parameters(),
            max_lr=float(self.task_cfg.backbone_learning_rate),
            name="backbone",
        )
        self.backbone_optimizer_enabled = True
        if self.is_master:
            log.info("Finished %d burn-in epochs; enabled backbone optimization.", burn_in)

    def _optimizer_parameters(self) -> list[nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _precompute_cell_embs(self, X_cell: np.ndarray) -> None:
        """Precompute frozen backbone embeddings for all cell lines once per fold (head_only only)."""
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        raw.backbone.eval()
        n_cells = X_cell.shape[0]
        infer_batch = int(getattr(self.task_cfg, "batch_size", 4)) * int(
            getattr(self.task_cfg, "grad_accumulation_steps", 4)
        )
        infer_batch = max(1, infer_batch)
        all_embs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n_cells, infer_batch):
                batch = self._cell_backbone_batch(
                    X_cell,
                    np.arange(start, min(start + infer_batch, n_cells)),
                )
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                hidden = raw.backbone(batch["gene_ids"], batch["expr"])
                all_embs.append(raw.pool_cell_embeddings(hidden).cpu())
        self.cell_emb_cache = torch.cat(all_embs, dim=0).to(self.device)  # (n_cells, dim)
        if self.is_master:
            log.info(
                "Precomputed cell embeddings: representation=CLS token, shape=%s",
                tuple(self.cell_emb_cache.shape),
            )

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_enable_backbone_optimizer(epoch)
        if self.is_distributed:
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        _set_finetune_training_mode(self.model, self._finetune_mode())
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 4)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss_numerator = 0.0
        running_loss_normalizer = 0.0
        accumulated_normalizer = 0.0
        total_steps = len(self.train_loader)

        for step_idx, (cell_idxs_b, batch, drug_emb, targets) in enumerate(self.train_loader, start=1):
            drug_emb = drug_emb.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.cell_emb_cache is not None:
                cell_emb = self.cell_emb_cache[cell_idxs_b.to(self.device)]
                batch_arg = None
            else:
                batch_arg = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                cell_emb = None

            should_step = is_accumulation_boundary(
                step_idx,
                total_steps,
                grad_acc_steps,
            )
            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and not should_step
            )
            sync_context = self.model.no_sync() if use_no_sync else nullcontext()
            with sync_context:
                preds = self.model(batch_arg, drug_emb, cell_emb)  # (B,)
                loss_sum = F.mse_loss(preds, targets, reduction="sum")
                loss_normalizer = float(targets.numel())
                loss_sum.backward()

            accumulated_normalizer += loss_normalizer

            if should_step:
                normalize_accumulated_gradients(
                    self._optimizer_parameters(),
                    accumulated_normalizer,
                    device=self.device,
                    is_distributed=self.is_distributed,
                    world_size=self.world_size,
                )
                torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), max_grad_norm)
                self.scheduler.step()
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_normalizer = 0.0

            running_loss_numerator += float(loss_sum.detach().item())
            running_loss_normalizer += loss_normalizer

        loss_totals = torch.tensor(
            [running_loss_numerator, running_loss_normalizer],
            dtype=torch.float64,
            device=self.device,
        )
        if self.is_distributed:
            dist.all_reduce(loss_totals, op=dist.ReduceOp.SUM)
        epoch_loss = float(loss_totals[0].item() / loss_totals[1].item())
        return {"loss": epoch_loss}

    def _evaluate(self, test_pair_cell_ids: np.ndarray) -> dict:
        self.model.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for cell_idxs_b, batch, drug_emb, targets in self.test_loader:
                drug_emb = drug_emb.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.cell_emb_cache is not None:
                    cell_emb = self.cell_emb_cache[cell_idxs_b.to(self.device)]
                    preds = self.model(None, drug_emb, cell_emb)    # (B,)
                else:
                    batch = {
                        key: value.to(self.device, non_blocking=True)
                        for key, value in batch.items()
                    }
                    preds = self.model(batch, drug_emb)    # (B,)
                all_preds.append(preds)
                all_targets.append(targets)

        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)

        if self.is_distributed:
            all_preds_t = distributed_concat(all_preds_t, self.test_dataset_size, self.world_size)
            all_targets_t = distributed_concat(all_targets_t, self.test_dataset_size, self.world_size)

        preds_np = all_preds_t.cpu().float().numpy()    # (n_test_pairs,)
        targets_np = all_targets_t.cpu().float().numpy()

        # Global PCC and SCC across all pairs
        finite = np.isfinite(preds_np) & np.isfinite(targets_np)
        if finite.sum() >= 2:
            global_pcc, _ = pearsonr(preds_np[finite], targets_np[finite])
            global_scc, _ = spearmanr(preds_np[finite], targets_np[finite])
        else:
            global_pcc = global_scc = float("nan")

        # Per-cell-line PCC and SCC (average across drugs for each cell line)
        cell_pccs: dict[str, float] = {}
        cell_sccs: dict[str, float] = {}
        for cell_id in np.unique(test_pair_cell_ids):
            mask = test_pair_cell_ids == cell_id
            p, t = preds_np[mask], targets_np[mask]
            fm = np.isfinite(p) & np.isfinite(t)
            if fm.sum() < 2:
                continue
            pcc, _ = pearsonr(p[fm], t[fm])
            scc, _ = spearmanr(p[fm], t[fm])
            cell_pccs[cell_id] = float(pcc) if np.isfinite(pcc) else float("nan")
            cell_sccs[cell_id] = float(scc) if np.isfinite(scc) else float("nan")

        mean_pcc = float(np.nanmean(list(cell_pccs.values()))) if cell_pccs else float("nan")
        mean_scc = float(np.nanmean(list(cell_sccs.values()))) if cell_sccs else float("nan")

        test_loss = F.mse_loss(all_preds_t, all_targets_t).item()

        row = {
            "loss": float(test_loss),
            "global_pcc": float(global_pcc) if np.isfinite(global_pcc) else float("nan"),
            "global_scc": float(global_scc) if np.isfinite(global_scc) else float("nan"),
            "mean_pcc_per_cell": mean_pcc,
            "mean_scc_per_cell": mean_scc,
            "cell_pccs": cell_pccs,
            "cell_sccs": cell_sccs,
            "n_test_pairs": int(preds_np.shape[0]),
            "n_test_cell_lines": len(cell_pccs),
        }
        return row

    # ------------------------------------------------------------------
    # Results writing
    # ------------------------------------------------------------------

    def _flatten_fold_metrics(
        self,
        model_key: str,
        fold: int,
        n_folds: int,
        checkpoint_path: str,
        train_metrics: dict,
        test_metrics: dict,
    ) -> dict:
        row = {
            "model": model_key,
            "fold": fold,
            "n_folds": n_folds,
            "finetune_mode": self._finetune_mode(),
            "checkpoint_path": checkpoint_path,
            "train_loss": float(train_metrics["loss"]),
            "test_loss": float(test_metrics["loss"]),
            "global_pcc": float(test_metrics["global_pcc"]),
            "global_scc": float(test_metrics["global_scc"]),
            "mean_pcc_per_cell": float(test_metrics["mean_pcc_per_cell"]),
            "mean_scc_per_cell": float(test_metrics["mean_scc_per_cell"]),
            "n_test_pairs": int(test_metrics["n_test_pairs"]),
            "n_test_cell_lines": int(test_metrics["n_test_cell_lines"]),
        }
        for field in ("pca_components", "cell_embedding_dim", "pair_feature_dim"):
            if field in test_metrics:
                row[field] = int(test_metrics[field])
        return row

    def _per_cell_line_rows(
        self,
        model_key: str,
        fold: int,
        checkpoint_path: str,
        test_metrics: dict,
    ) -> list[dict]:
        rows = []
        for cell_id in sorted(test_metrics["cell_pccs"]):
            rows.append({
                "model": model_key,
                "fold": fold,
                "finetune_mode": self._finetune_mode(),
                "checkpoint_path": checkpoint_path,
                "cell_line_id": cell_id,
                "pcc": test_metrics["cell_pccs"][cell_id],
                "scc": test_metrics["cell_sccs"].get(cell_id, float("nan")),
            })
        return rows

    def _write_model_results(
        self,
        checkpoint_path: str,
        fold_rows: list[dict],
        cell_line_rows: list[dict],
    ) -> dict:
        aggregate = self._aggregate_numeric_rows(fold_rows)
        aggregate.update({
            "model": fold_rows[0]["model"],
            "finetune_mode": fold_rows[0]["finetune_mode"],
            "checkpoint_path": checkpoint_path,
        })
        out_dir = self._task_output_dir()
        model_key = str(fold_rows[0]["model"])
        prefix = self._output_prefix()
        self._write_csv(out_dir / f"{prefix}_{model_key}_fold_metrics.csv", fold_rows)
        self._write_csv(out_dir / f"{prefix}_{model_key}_evaluation_metrics.csv", [aggregate], comment=getattr(self, "_missing_genes_note", ""))
        self._write_csv(out_dir / f"{prefix}_{model_key}_cell_line_metrics.csv", cell_line_rows)
        return aggregate

    def _cleanup_fold_state(self) -> None:
        self.train_loader = None
        self.train_dataset_size = 0
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.cell_emb_cache = None
        self.fold_gene_indices = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        try:
            self._setup_runtime()

            X_cell, drug_emb_matrix, cell_idxs, drug_idxs, ic50_values, pair_cell_ids = (
                self._load_gdsc_data()
            )
            drug_emb_dim = drug_emb_matrix.shape[1]
            n_pairs = len(ic50_values)
            splits = self._build_or_load_cv_splits(
                pair_cell_ids,
                self._pair_drug_ids,
                ic50_values,
            )
            fold_hvg_indices = [
                self._select_training_hvg_indices(X_cell, cell_idxs[train_idx])
                for train_idx, _ in splits
            ]
            checkpoint_paths = self._get_checkpoint_paths()
            if self.is_distributed:
                dist.barrier()
            completed_model_results = self._load_completed_model_results(
                checkpoint_paths
            )
            if self.is_distributed:
                dist.barrier()
            self._reused_completed_models = [
                model_key
                for model_key in checkpoint_paths
                if model_key in completed_model_results
            ]
            self._save_run_metadata(checkpoint_paths)

            if self.is_master:
                log.info(
                    "Drug response CV: pairs=%d, cell_lines=%d, drug_emb_dim=%d, folds=%d",
                    n_pairs,
                    len(np.unique(cell_idxs)),
                    drug_emb_dim,
                    len(splits),
                )

            epochs = int(getattr(self.task_cfg, "epochs", 20))
            aggregate_rows: list[dict] = [
                completed_model_results[model_key]
                for model_key in checkpoint_paths
                if model_key in completed_model_results
            ]

            for model_key, checkpoint_path in checkpoint_paths.items():
                if model_key in completed_model_results:
                    if self.is_master:
                        log.info(
                            "Reusing completed model %s after validating all resume artifacts.",
                            model_key,
                        )
                    if self.is_distributed:
                        dist.barrier()
                    continue
                fold_rows: list[dict] = []
                cell_line_rows: list[dict] = []
                curve_rows: list[dict] = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + fold_idx
                    )

                    self.fold_gene_indices = fold_hvg_indices[fold_idx - 1]

                    test_pair_cell_ids = pair_cell_ids[test_idx]

                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | train=%d pairs, test=%d pairs",
                            model_key, fold_idx, len(splits), len(train_idx), len(test_idx),
                        )

                    self._build_train_loader(
                        X_cell, drug_emb_matrix,
                        cell_idxs[train_idx], drug_idxs[train_idx], ic50_values[train_idx],
                    )
                    self._build_test_loader(
                        X_cell, drug_emb_matrix,
                        cell_idxs[test_idx], drug_idxs[test_idx], ic50_values[test_idx],
                    )
                    self._build_model(checkpoint_path, drug_emb_dim)
                    if self._finetune_mode() == "head_only":
                        self._precompute_cell_embs(X_cell)
                    self._build_optimization()

                    last_train_metrics = {"loss": float("nan")}
                    last_validation_metrics: dict | None = None
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        last_validation_metrics = self._evaluate(test_pair_cell_ids)
                        if self.is_master:
                            curve_rows.append(
                                {
                                    "model": model_key,
                                    "fold": fold_idx,
                                    "epoch": epoch,
                                    "finetune_mode": self._finetune_mode(),
                                    "train_loss": last_train_metrics["loss"],
                                    "validation_loss": last_validation_metrics["loss"],
                                    "validation_global_pcc": last_validation_metrics["global_pcc"],
                                    "validation_global_scc": last_validation_metrics["global_scc"],
                                    "learning_rates": ";".join(
                                        f"{float(group['lr']):.6g}"
                                        for group in self.optimizer.param_groups
                                    ),
                                    "optimizer_updates": self.scheduler.completed_updates,
                                }
                            )
                            self._write_csv(
                                self._task_output_dir()
                                / f"{self._output_prefix()}_{model_key}_training_curves.csv",
                                curve_rows,
                            )
                            log.info(
                                "Model %s | Fold %d/%d | Epoch %d | "
                                "Train Loss: %.6f | Validation Loss: %.6f",
                                model_key, fold_idx, len(splits), epoch,
                                last_train_metrics["loss"],
                                last_validation_metrics["loss"],
                            )

                    test_metrics = (
                        last_validation_metrics
                        if last_validation_metrics is not None
                        else self._evaluate(test_pair_cell_ids)
                    )
                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | "
                            "Test Loss: %.4f | Global PCC: %.4f | Mean PCC/cell: %.4f",
                            model_key, fold_idx, len(splits),
                            test_metrics["loss"],
                            test_metrics["global_pcc"],
                            test_metrics["mean_pcc_per_cell"],
                        )
                        fold_rows.append(self._flatten_fold_metrics(
                            model_key=model_key, fold=fold_idx, n_folds=len(splits),
                            checkpoint_path=checkpoint_path,
                            train_metrics=last_train_metrics, test_metrics=test_metrics,
                        ))
                        cell_line_rows.extend(self._per_cell_line_rows(
                            model_key=model_key, fold=fold_idx,
                            checkpoint_path=checkpoint_path, test_metrics=test_metrics,
                        ))
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
                complete_run_metadata(self._run_metadata_path, output_path)
                return {"results_path": str(output_path), "results": aggregate_rows}
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
