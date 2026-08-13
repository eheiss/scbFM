from __future__ import annotations

import csv
import fcntl
import gc
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
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
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
from preprocess import reindex_adata_genes, validate_token_matrix
from run_provenance import complete_run_metadata, start_run_metadata
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
TASK_NAME = "deconv"
CHECKPOINT_MODEL_KEYS = ("pretrain_sc", "pretrain_bulk", "preadapt_sc", "preadapt_bulk")
RANDOM_INIT_MODEL_KEY = "random_init"
MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)


class DeconvPredHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pooling = "cls"
        input_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SELU(),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x[:, 0, :])


class CancerFoundationDeconvModel(nn.Module):
    def __init__(self, backbone: CancerFoundationBackbone, head: DeconvPredHead) -> None:
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


class DeconvDataset(Dataset):
    def __init__(
        self,
        data,
        targets: np.ndarray,
        bin_num: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
        selected_gene_count: int,
        seed: int,
        do_binning: bool,
        fixed_gene_indices: np.ndarray,
    ) -> None:
        self.data = data
        targets = np.asarray(targets, dtype=np.float32)
        target_sums = targets.sum(axis=1, keepdims=True)
        if np.any(target_sums <= 0):
            raise ValueError("Every deconvolution target row must have a positive sum.")
        self.targets = targets / target_sums
        self.bin_num = int(bin_num)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.selected_gene_count = int(selected_gene_count)
        self.seed = int(seed)
        self.do_binning = bool(do_binning)
        self.fixed_gene_indices = np.asarray(fixed_gene_indices, dtype=np.int64)

        if self.fixed_gene_indices.shape != (self.selected_gene_count,):
            raise ValueError(
                "fixed_gene_indices must contain exactly "
                f"{self.selected_gene_count} genes."
            )
        if self.data.shape[1] != self.selected_gene_count:
            raise ValueError(
                "DeconvDataset data must already contain exactly the selected "
                f"{self.selected_gene_count} expression columns."
            )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.data[index]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()

        selected_values = values.astype(np.float32, copy=False)
        if self.do_binning:
            rng = np.random.default_rng(self.seed + index)
            selected_values = _quantile_bin_expression(
                selected_values,
                self.bin_num,
                rng,
            ).astype(np.float32)

        gene_ids = torch.from_numpy(
            self.fixed_gene_indices.astype(np.int64, copy=False) + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        expression = torch.from_numpy(selected_values)
        expression = torch.cat((torch.tensor([self.cls_value]), expression))
        target = torch.from_numpy(self.targets[index]).float()
        return {"gene_ids": gene_ids, "expr": expression}, target


class DeconvRunner:
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
                "Deconvolution expects max_seq_len to equal selected_gene_count + 1 "
                "for the <cls> token."
            )

        self.cell_types: list[str] = []
        self.target_columns: list[str] = []
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size = 0
        self.fold_train_target_mean: np.ndarray | None = None
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.backbone_optimizer_enabled = False

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "deconv" in cfg.finetune:
            return cfg.finetune.deconv
        raise ValueError("Could not find config at cfg.finetune.deconv.")

    @staticmethod
    def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
        if "pretrain" in cfg:
            return cfg.pretrain
        raise ValueError("Could not find model architecture config. Expected cfg.pretrain.")

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
            "dataset_id",
            "donor_id",
            "tissue_general",
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
        self._run_metadata_path = out_dir / f"{prefix}_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": self.task_name,
                "finetune_mode": self._finetune_mode(),
                "head_only_backbone_eval": self._finetune_mode() == "head_only",
                "output_suffix": self._output_suffix(),
                "representation": "cls",
                "head_hidden_dim": int(getattr(self.task_cfg, "head_hidden_dim", 512)),
                "head_bottleneck_dim": int(
                    getattr(self.task_cfg, "head_bottleneck_dim", 256)
                ),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "cv_fold_manifest_path": str(
                    getattr(self, "_cv_fold_manifest_path", "")
                ),
                "cv_fold_fingerprint": str(
                    getattr(self, "_cv_fold_fingerprint", "")
                ),
                "gene_selection": str(
                    getattr(self.task_cfg, "hvg_selection_method", "mad")
                ),
                "expression_transform": (
                    "log1p" if self._should_preprocess_input() else "pretokenized"
                ),
            },
            checkpoint_paths=checkpoint_paths,
            repo_dir=ROOT / "scbFM",
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
                "finetune.deconv.pretrained_model_paths must define "
                f"{', '.join(CHECKPOINT_MODEL_KEYS)}."
            )

        configured_model_keys = getattr(self.task_cfg, "model_keys", None)
        if configured_model_keys is None:
            selected_model_keys = list(MODEL_KEYS)
        else:
            selected_model_keys = [str(key) for key in configured_model_keys]
            if not selected_model_keys:
                raise ValueError("finetune.deconv.model_keys may not be empty.")
            duplicates = sorted(
                key for key in set(selected_model_keys)
                if selected_model_keys.count(key) > 1
            )
            invalid = sorted(set(selected_model_keys).difference(MODEL_KEYS))
            if duplicates or invalid:
                raise ValueError(
                    "Invalid finetune.deconv.model_keys: "
                    f"duplicates={duplicates}, unsupported={invalid}; "
                    f"supported={list(MODEL_KEYS)}."
                )

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
                "Missing checkpoint paths in finetune.deconv.pretrained_model_paths: "
                f"{missing}"
            )
        return checkpoint_paths

    @staticmethod
    def _strip_module_prefix(state_dict: dict) -> dict:
        if not state_dict:
            return state_dict
        if not all(key.startswith("module.") for key in state_dict):
            return state_dict
        return {key.removeprefix("module."): value for key, value in state_dict.items()}

    def _load_input_adata(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "pseudo_bulk_data_path", None)
        if not configured_path:
            raise ValueError("finetune.deconv.pseudo_bulk_data_path must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"Pseudo-bulk h5ad file not found: {data_path}")
        return ad.read_h5ad(data_path)

    def _resolve_gene_list_path(self) -> Path:
        gene_list_path = getattr(self.task_cfg, "gene_list_path", None)
        if gene_list_path:
            return Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        return ROOT / "scbFM" / "data" / "gene_list.txt"

    def _should_preprocess_input(self) -> bool:
        return bool(getattr(self.task_cfg, "preprocess", False))

    @staticmethod
    def _as_string(value) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.str_):
            return str(value)
        return None

    def _as_string_list(self, values) -> list[str] | None:
        if values is None:
            return None
        array = np.asarray(values).reshape(-1)
        strings: list[str] = []
        for value in array:
            value_string = self._as_string(value)
            if value_string is None:
                return None
            strings.append(value_string)
        return strings

    def _normalize_proportion_mapping(self, raw_mapping, obs_columns) -> dict[str, str]:
        obs_columns = set(map(str, obs_columns))
        mapping_out: dict[str, str] = {}

        def visit(cell_type_parts: list[str], value) -> None:
            value_string = self._as_string(value)
            if value_string is not None:
                mapping_out["/".join(cell_type_parts)] = value_string
                return

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    visit(cell_type_parts, value.item())
                    return
                if value.size == 1:
                    visit(cell_type_parts, value.reshape(-1)[0])
                    return

            if isinstance(value, (list, tuple)) and len(value) == 1:
                visit(cell_type_parts, value[0])
                return

            if isinstance(value, dict):
                direct_value = None
                for key in ("column", "obs_column", "value"):
                    if key in value:
                        direct_value = self._as_string(value[key])
                        if direct_value is not None:
                            break
                if direct_value is not None:
                    mapping_out["/".join(cell_type_parts)] = direct_value
                    return

                # HDF5-backed AnnData stores '/' in uns dict keys as nested groups.
                for key, child in value.items():
                    key_string = self._as_string(key)
                    if key_string is None:
                        key_string = str(key)
                    child_string = self._as_string(child)
                    if child_string is None and key_string in obs_columns:
                        mapping_out["/".join(cell_type_parts)] = key_string
                    else:
                        visit([*cell_type_parts, key_string], child)
                return

            raise ValueError(
                "Unsupported cell_type_proportion_columns entry for "
                f"{'/'.join(cell_type_parts)!r}: expected a column-name string, got {value!r}."
            )

        try:
            mapping = dict(raw_mapping)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "uns['cell_type_proportion_columns'] must be a mapping from cell type "
                "to proportion-column name."
            ) from exc

        for cell_type, value in mapping.items():
            cell_type_string = self._as_string(cell_type)
            if cell_type_string is None:
                cell_type_string = str(cell_type)
            visit([cell_type_string], value)

        invalid_columns = [col for col in mapping_out.values() if col not in obs_columns]
        if invalid_columns:
            inferred = self._infer_proportion_mapping_from_obs(obs_columns)
            if inferred:
                log.warning(
                    "Ignoring unusable uns['cell_type_proportion_columns'] entries and "
                    "inferring targets from prop__* obs columns instead. Invalid columns: %s",
                    invalid_columns[:10],
                )
                return inferred

        return mapping_out

    @staticmethod
    def _infer_proportion_mapping_from_obs(obs_columns) -> dict[str, str]:
        prop_columns = sorted(str(col) for col in obs_columns if str(col).startswith("prop__"))
        return {col.removeprefix("prop__"): col for col in prop_columns}

    def _load_targets(self, adata: ad.AnnData) -> np.ndarray:
        cell_types = self._as_string_list(adata.uns.get("cell_type_proportion_cell_types"))
        obs_columns = self._as_string_list(adata.uns.get("cell_type_proportion_obs_columns"))
        if cell_types is not None and obs_columns is not None:
            if len(cell_types) != len(obs_columns):
                raise ValueError(
                    "uns['cell_type_proportion_cell_types'] and "
                    "uns['cell_type_proportion_obs_columns'] must have the same length."
                )
            mapping = dict(zip(cell_types, obs_columns))
        else:
            mapping = adata.uns.get("cell_type_proportion_columns")

        if mapping is None:
            mapping = self._infer_proportion_mapping_from_obs(adata.obs.columns)
            if not mapping:
                raise ValueError(
                    "Pseudo-bulk AnnData must contain uns['cell_type_proportion_columns'] "
                    "or obs columns prefixed with 'prop__'."
                )
        else:
            mapping = self._normalize_proportion_mapping(mapping, adata.obs.columns)
        self.cell_types = sorted(mapping)
        self.target_columns = [mapping[cell_type] for cell_type in self.cell_types]
        missing = [col for col in self.target_columns if col not in adata.obs]
        if missing:
            raise ValueError(f"Missing proportion columns in adata.obs: {missing}")

        targets = adata.obs[self.target_columns].to_numpy(dtype=np.float32)
        if np.any(targets < 0):
            raise ValueError("Cell type proportions must be non-negative.")
        return targets

    def _prepare_cv_data(self) -> tuple[ad.AnnData, np.ndarray, np.ndarray | None]:
        adata = self._load_input_adata()
        if self._should_preprocess_input():
            adata, missing_genes = reindex_adata_genes(
                adata, gene_list_path=self._resolve_gene_list_path()
            )
            log.info(
                "Aligned raw input for on-the-fly sequence binning: "
                "%d target genes missing, output shape %s",
                len(missing_genes),
                adata.shape,
            )
        else:
            adata, missing_genes = reindex_adata_genes(
                adata, gene_list_path=self._resolve_gene_list_path()
            )
            log.info(
                "Reindexed preprocessed input to gene list: %d target genes missing, output shape %s",
                len(missing_genes),
                adata.shape,
            )

        self._missing_genes_note = (
            f"Model genes missing from deconv GEX and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
        ) if missing_genes else ""

        if self._should_preprocess_input():
            if sparse.issparse(adata.X):
                matrix = adata.X.tocsr().astype(np.float32, copy=True)
                if not np.all(np.isfinite(matrix.data)):
                    raise ValueError("Raw deconvolution expression contains non-finite values.")
                if np.any(matrix.data < 0):
                    raise ValueError("Raw deconvolution expression must be non-negative.")
                matrix.data = np.log1p(matrix.data)
                adata.X = matrix
            else:
                matrix = np.asarray(adata.X, dtype=np.float32)
                if not np.all(np.isfinite(matrix)):
                    raise ValueError("Raw deconvolution expression contains non-finite values.")
                if np.any(matrix < 0):
                    raise ValueError("Raw deconvolution expression must be non-negative.")
                adata.X = np.log1p(matrix).astype(np.float32, copy=False)
            log.info("Applied log1p to raw deconvolution expression after gene alignment.")

        targets = self._load_targets(adata)

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes for the scbFM backbone, got {adata.n_vars}."
            )

        if not self._should_preprocess_input():
            validate_token_matrix(
                adata.X,
                bin_num=int(self.model_cfg.bin_num),
                name="deconvolution input data",
            )
        groups = None
        if bool(getattr(self.task_cfg, "split_by_context", True)):
            context_columns = list(
                getattr(self.task_cfg, "context_columns", ["dataset_id", "donor_id", "tissue_general"])
            )
            missing_context = [col for col in context_columns if col not in adata.obs]
            if missing_context:
                raise ValueError(
                    f"split_by_context=true but these context columns are missing: {missing_context}"
                )
            groups = adata.obs[context_columns].astype(str).agg("||".join, axis=1).to_numpy()

        return adata, targets, groups

    def _select_training_hvg_indices(self, adata: ad.AnnData) -> np.ndarray:
        """Select the fixed sequence vocabulary from the training fold only."""
        if adata.n_vars < self.selected_gene_count:
            raise ValueError(
                f"Cannot select {self.selected_gene_count} HVGs from only {adata.n_vars} genes."
            )

        selection_method = str(
            getattr(self.task_cfg, "hvg_selection_method", "mad")
        ).lower()
        if selection_method != "mad":
            raise ValueError(
                "finetune.deconv.hvg_selection_method must be 'mad' to match the thesis."
            )

        matrix = adata.X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
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

        log.info(
            "Selected %d training-fold genes with MAD on log1p expression | "
            "selected MAD min=%.6g median=%.6g max=%.6g",
            selected.size,
            float(np.min(scores[selected])),
            float(np.median(scores[selected])),
            float(np.max(scores[selected])),
        )
        return selected

    def _select_distributed_training_hvg_indices(
        self,
        adata: ad.AnnData,
    ) -> np.ndarray:
        """Compute fold-specific MAD genes once and share them across DDP ranks."""
        if not self.is_distributed:
            return self._select_training_hvg_indices(adata)

        selected_tensor = torch.empty(
            self.selected_gene_count,
            dtype=torch.long,
            device=self.device,
        )
        if self.is_master:
            selected = self._select_training_hvg_indices(adata)
            selected_tensor.copy_(torch.from_numpy(selected).to(self.device))
        dist.broadcast(selected_tensor, src=0)
        return selected_tensor.cpu().numpy()

    def _build_cv_splits(
        self,
        adata: ad.AnnData,
        groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 5))
        if n_splits < 2:
            raise ValueError("finetune.deconv.cv_folds must be at least 2.")

        if groups is not None:
            unique_groups = np.unique(groups)
            if n_splits > unique_groups.size:
                raise ValueError(
                    f"cv_folds={n_splits} is larger than the number of context groups ({unique_groups.size})."
                )
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(np.zeros(adata.n_obs), groups=groups))

        if n_splits > adata.n_obs:
            raise ValueError(f"cv_folds={n_splits} is larger than sample count ({adata.n_obs}).")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        return list(splitter.split(np.arange(adata.n_obs)))

    def _cv_manifest_source_rows(
        self,
        adata: ad.AnnData,
        groups: np.ndarray | None,
    ) -> list[dict[str, str]]:
        sample_ids = adata.obs_names.astype(str).to_numpy()
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("Deconvolution CV fold manifests require unique sample IDs.")
        group_ids = (
            np.asarray(groups).astype(str)
            if groups is not None
            else np.asarray([""] * adata.n_obs)
        )
        if group_ids.shape[0] != adata.n_obs:
            raise ValueError(
                f"CV groups contain {group_ids.shape[0]} rows, but AnnData contains "
                f"{adata.n_obs}."
            )
        return [
            {"sample_id": str(sample_id), "group_id": str(group_id)}
            for sample_id, group_id in zip(sample_ids, group_ids)
        ]

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

        data_path = Path(
            hydra.utils.to_absolute_path(str(self.task_cfg.pseudo_bulk_data_path))
        )
        signature_payload = {
            "task": self.task_name,
            "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
            "random_seed": int(getattr(self.task_cfg, "random_seed", 42)),
            "split_by_context": bool(getattr(self.task_cfg, "split_by_context", True)),
            "context_columns": [
                str(value) for value in getattr(self.task_cfg, "context_columns", [])
            ],
            "samples": source_rows,
        }
        source_fingerprint = hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._cv_source_fingerprint = source_fingerprint
        return data_path.with_name(
            f"{data_path.stem}_{self.task_name}_{source_fingerprint[:16]}_cv_folds.csv"
        )

    @staticmethod
    def _fold_assignments_from_splits(
        n_samples: int,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        assignments = np.full(n_samples, -1, dtype=np.int64)
        all_indices = np.arange(n_samples, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)
            if np.intersect1d(train_idx, test_idx).size:
                raise ValueError(f"CV fold {fold} overlaps between train and test.")
            if not np.array_equal(
                np.sort(np.concatenate((train_idx, test_idx))),
                all_indices,
            ):
                raise ValueError(f"CV fold {fold} does not partition all samples.")
            if np.any(assignments[test_idx] != -1):
                raise ValueError("A sample appears in multiple CV test folds.")
            assignments[test_idx] = fold
        if np.any(assignments < 1):
            raise ValueError("Every sample must appear in exactly one CV test fold.")
        return assignments

    def _load_cv_fold_manifest(
        self,
        manifest_path: Path,
        source_rows: list[dict[str, str]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(source_rows):
            raise ValueError(
                f"CV fold manifest {manifest_path} contains {len(rows)} samples; "
                f"expected {len(source_rows)}."
            )
        assignments = np.empty(len(rows), dtype=np.int64)
        for index, (observed, expected) in enumerate(zip(rows, source_rows)):
            if {key: observed.get(key) for key in expected} != expected:
                raise ValueError(
                    f"CV fold manifest {manifest_path} differs from canonical sample "
                    f"row {index}."
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
        adata: ad.AnnData,
        groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        source_rows = self._cv_manifest_source_rows(adata, groups)
        manifest_path = self._resolve_cv_fold_manifest_path(source_rows)
        if manifest_path is None:
            return self._build_cv_splits(adata, groups)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = manifest_path.with_name(f"{manifest_path.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if manifest_path.exists():
                splits = self._load_cv_fold_manifest(manifest_path, source_rows)
            else:
                splits = self._build_cv_splits(adata, groups)
                assignments = self._fold_assignments_from_splits(adata.n_obs, splits)
                temporary_path = manifest_path.with_name(
                    f"{manifest_path.name}.tmp.{os.getpid()}.{self.rank}"
                )
                try:
                    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=["sample_id", "group_id", "fold"],
                        )
                        writer.writeheader()
                        for row, fold in zip(source_rows, assignments):
                            writer.writerow({**row, "fold": int(fold)})
                    os.replace(temporary_path, manifest_path)
                finally:
                    temporary_path.unlink(missing_ok=True)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        assignments = self._fold_assignments_from_splits(adata.n_obs, splits)
        if groups is not None:
            groups_array = np.asarray(groups).astype(str)
            split_groups = [
                group
                for group in np.unique(groups_array)
                if np.unique(assignments[groups_array == group]).size != 1
            ]
            if split_groups:
                raise ValueError(
                    "CV fold manifest splits deconvolution context groups. "
                    f"Affected groups include: {split_groups[:10]}"
                )
        fold_fingerprint = hashlib.sha256(
            "\n".join(
                f"{row['sample_id']}\t{row['group_id']}\t{int(fold)}"
                for row, fold in zip(source_rows, assignments)
            ).encode("utf-8")
        ).hexdigest()
        self._cv_fold_manifest_path = manifest_path
        self._cv_fold_fingerprint = fold_fingerprint
        log.info(
            "Using shared deconvolution CV fold manifest %s | fingerprint=%s",
            manifest_path,
            fold_fingerprint,
        )
        return splits

    def _build_loaders(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
    ) -> None:
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        num_workers = int(getattr(self.task_cfg, "num_workers", 0))
        if num_workers < 0:
            raise ValueError("finetune.deconv.num_workers must be non-negative.")

        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            prefetch_factor = int(getattr(self.task_cfg, "prefetch_factor", 2))
            if prefetch_factor <= 0:
                raise ValueError("finetune.deconv.prefetch_factor must be positive.")
            loader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": False,
                }
            )

        random_seed = int(getattr(self.task_cfg, "random_seed", 42))
        do_binning = self._should_preprocess_input()
        fold_hvg_indices = self._select_distributed_training_hvg_indices(train_adata)
        train_expression = train_adata.X[:, fold_hvg_indices].copy()
        test_expression = test_adata.X[:, fold_hvg_indices].copy()
        train_target_sums = train_targets.sum(axis=1, keepdims=True)
        if np.any(train_target_sums <= 0):
            raise ValueError("Every deconvolution target row must have a positive sum.")
        self.fold_train_target_mean = np.mean(
            train_targets / train_target_sums,
            axis=0,
        )
        train_dataset = DeconvDataset(
            train_expression,
            train_targets,
            bin_num=int(self.model_cfg.bin_num),
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=self.cls_value,
            selected_gene_count=self.selected_gene_count,
            seed=random_seed,
            do_binning=do_binning,
            fixed_gene_indices=fold_hvg_indices,
        )
        test_dataset = DeconvDataset(
            test_expression,
            test_targets,
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
            validate_backbone_checkpoint(
                checkpoint,
                resolved_path,
                gene_num=int(self.model_cfg.gene_num),
                selected_gene_count=self.selected_gene_count,
                max_seq_len=self.max_seq_len,
                bin_num=int(self.model_cfg.bin_num),
            )
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            backbone.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved_path)
        else:
            log.info("Using randomly initialized backbone")

        head = DeconvPredHead(
            embedding_dim=int(self.model_cfg.embsize),
            output_dim=len(self.cell_types),
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 512)),
            bottleneck_dim=int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
        )
        if self.is_master:
            log.info(
                "Deconvolution representation: CLS token | head dims: %d -> %d -> %d -> %d",
                int(self.model_cfg.embsize),
                int(getattr(self.task_cfg, "head_hidden_dim", 512)),
                int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
                len(self.cell_types),
            )
        model = CancerFoundationDeconvModel(backbone=backbone, head=head)

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

    def _build_optimization(self) -> None:
        if not hasattr(self.task_cfg, "head_learning_rate"):
            raise ValueError("finetune.deconv.head_learning_rate must be set.")
        if not hasattr(self.task_cfg, "backbone_learning_rate"):
            raise ValueError("finetune.deconv.backbone_learning_rate must be set.")
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
            param_groups.append({"params": backbone_params, "lr": backbone_learning_rate, "name": "backbone"})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": adapter_learning_rate, "name": "adapters"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_learning_rate, "name": "head"})
        if not param_groups:
            raise ValueError("No trainable parameters found for deconvolution.")

        max_lrs = [float(group["lr"]) for group in param_groups]
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_min_lr_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = (
            float(configured_min_lr_ratio)
            if configured_min_lr_ratio is not None
            else min_lr / max(head_learning_rate, 1e-12)
        )

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
            group_summaries = [
                f"{group.get('name', idx)}: params={sum(p.numel() for p in group['params'])}, "
                f"max_lr={max_lr:.2e}, min_lr={max_lr * min_lr_ratio:.2e}"
                for idx, (group, max_lr) in enumerate(zip(param_groups, max_lrs))
            ]
            log.info("Optimizer parameter groups: %s", "; ".join(group_summaries))

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        finetune_mode = self._finetune_mode()
        burn_in_epochs = int(getattr(self.task_cfg, "burn_in_epochs", 3))
        if (
            finetune_mode != "full_ft"
            or burn_in_epochs <= 0
            or self.backbone_optimizer_enabled
            or epoch <= burn_in_epochs
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

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_sum, normalizer = self._loss_sum_and_normalizer(logits, targets)
        return loss_sum / normalizer

    def _loss_sum_and_normalizer(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        loss_name = str(getattr(self.task_cfg, "loss", "kl")).lower()
        if loss_name == "kl":
            return (
                F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="sum"),
                float(targets.shape[0]),
            )
        pred_props = F.softmax(logits, dim=-1)
        if loss_name == "mse":
            return F.mse_loss(pred_props, targets, reduction="sum"), float(targets.numel())
        if loss_name in {"mae", "l1"}:
            return F.l1_loss(pred_props, targets, reduction="sum"), float(targets.numel())
        raise ValueError("Unsupported deconvolution loss. Expected one of: kl, mse, mae.")

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_enable_backbone_optimizer(epoch)

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        _set_finetune_training_mode(self.model, self._finetune_mode())
        self.model.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 4)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss_numerator = 0.0
        running_loss_normalizer = 0.0
        accumulated_normalizer = 0.0
        total_steps = len(self.train_loader)

        for step_idx, (data, targets) in enumerate(self.train_loader, start=1):
            data = {
                key: value.to(self.device, non_blocking=True)
                for key, value in data.items()
            }
            targets = targets.to(self.device, non_blocking=True)

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
                logits = self.model(data)
                loss_sum, loss_normalizer = self._loss_sum_and_normalizer(logits, targets)
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
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad(set_to_none=True)
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

    @staticmethod
    def _safe_pearson(pred: np.ndarray, truth: np.ndarray) -> float:
        if np.std(pred) == 0 or np.std(truth) == 0:
            return float("nan")
        return float(np.corrcoef(pred, truth)[0, 1])

    @staticmethod
    def _safe_spearman(pred: np.ndarray, truth: np.ndarray) -> float:
        if np.std(pred) == 0 or np.std(truth) == 0:
            return float("nan")
        return float(spearmanr(pred, truth).correlation)

    def _cell_type_correlation_across_samples_metrics(
        self,
        predictions: np.ndarray,
        truths: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float], float, float]:
        pearson_by_type: dict[str, float] = {}
        spearman_by_type: dict[str, float] = {}
        for idx, cell_type in enumerate(self.cell_types):
            pearson_by_type[cell_type] = self._safe_pearson(predictions[:, idx], truths[:, idx])
            spearman_by_type[cell_type] = self._safe_spearman(predictions[:, idx], truths[:, idx])

        pearson_values = np.asarray(list(pearson_by_type.values()), dtype=float)
        spearman_values = np.asarray(list(spearman_by_type.values()), dtype=float)
        mean_pearson = (
            float(np.nanmean(pearson_values))
            if np.any(~np.isnan(pearson_values))
            else float("nan")
        )
        mean_spearman = (
            float(np.nanmean(spearman_values))
            if np.any(~np.isnan(spearman_values))
            else float("nan")
        )
        return pearson_by_type, spearman_by_type, mean_pearson, mean_spearman

    def _sample_correlation_across_cell_types_metrics(
        self,
        predictions: np.ndarray,
        truths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        sample_pearson = np.asarray(
            [
                self._safe_pearson(predictions[idx, :], truths[idx, :])
                for idx in range(predictions.shape[0])
            ],
            dtype=float,
        )
        sample_spearman = np.asarray(
            [
                self._safe_spearman(predictions[idx, :], truths[idx, :])
                for idx in range(predictions.shape[0])
            ],
            dtype=float,
        )
        mean_sample_pearson = (
            float(np.nanmean(sample_pearson))
            if np.any(~np.isnan(sample_pearson))
            else float("nan")
        )
        mean_sample_spearman = (
            float(np.nanmean(sample_spearman))
            if np.any(~np.isnan(sample_spearman))
            else float("nan")
        )
        return sample_pearson, sample_spearman, mean_sample_pearson, mean_sample_spearman

    @staticmethod
    def _distribution_metrics(predictions: np.ndarray, truths: np.ndarray) -> tuple[float, float, float]:
        eps = 1e-8
        pred = np.clip(predictions, eps, 1.0)
        truth = np.clip(truths, eps, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        truth = truth / truth.sum(axis=1, keepdims=True)
        midpoint = 0.5 * (pred + truth)
        kl_truth_pred = np.sum(truth * np.log(truth / pred), axis=1)
        js = 0.5 * (
            np.sum(truth * np.log(truth / midpoint), axis=1)
            + np.sum(pred * np.log(pred / midpoint), axis=1)
        )
        return (
            float(np.mean(kl_truth_pred)),
            float(np.mean(np.sqrt(js))),
            float(np.mean(js)),
        )

    @staticmethod
    def _normalize_composition_predictions(
        predictions: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        predictions = np.clip(predictions, 0.0, None)
        row_sums = predictions.sum(axis=1, keepdims=True)
        invalid = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= 0)
        if np.any(invalid):
            predictions[invalid] = np.asarray(fallback, dtype=np.float64)
            row_sums = predictions.sum(axis=1, keepdims=True)
        return (predictions / row_sums).astype(np.float32, copy=False)

    def _evaluation_metrics_from_arrays(
        self,
        predictions_np: np.ndarray,
        truths_np: np.ndarray,
        *,
        test_loss: float,
    ) -> dict[str, object]:
        predictions_np = np.asarray(predictions_np, dtype=np.float32)
        truths_np = np.asarray(truths_np, dtype=np.float32)
        per_type_mae = np.mean(np.abs(predictions_np - truths_np), axis=0)
        per_type_rmse = np.sqrt(np.mean((predictions_np - truths_np) ** 2, axis=0))
        per_type_prediction_mean = np.mean(predictions_np, axis=0)
        per_type_prediction_std = np.std(predictions_np, axis=0)
        per_type_truth_mean = np.mean(truths_np, axis=0)
        per_type_truth_std = np.std(truths_np, axis=0)
        (
            cell_type_pearson_across_samples,
            cell_type_spearman_across_samples,
            mean_cell_type_pearson_across_samples,
            mean_cell_type_spearman_across_samples,
        ) = self._cell_type_correlation_across_samples_metrics(
            predictions_np,
            truths_np,
        )
        (
            sample_pearson_across_cell_types,
            sample_spearman_across_cell_types,
            mean_sample_pearson_across_cell_types,
            mean_sample_spearman_across_cell_types,
        ) = self._sample_correlation_across_cell_types_metrics(
            predictions_np,
            truths_np,
        )
        kl_divergence, js_distance, js_divergence = self._distribution_metrics(
            predictions_np,
            truths_np,
        )
        if self.fold_train_target_mean is None:
            raise RuntimeError("Training-fold target mean is unavailable during evaluation.")
        mean_baseline = np.broadcast_to(
            self.fold_train_target_mean,
            predictions_np.shape,
        )
        (
            mean_baseline_kl_divergence,
            mean_baseline_js_distance,
            mean_baseline_js_divergence,
        ) = self._distribution_metrics(mean_baseline, truths_np)
        return {
            "loss": float(test_loss),
            "mae": float(mean_absolute_error(truths_np, predictions_np)),
            "rmse": float(np.sqrt(mean_squared_error(truths_np, predictions_np))),
            "mean_cell_type_pearson_across_samples": mean_cell_type_pearson_across_samples,
            "mean_cell_type_spearman_across_samples": mean_cell_type_spearman_across_samples,
            "mean_sample_pearson_across_cell_types": mean_sample_pearson_across_cell_types,
            "mean_sample_spearman_across_cell_types": mean_sample_spearman_across_cell_types,
            "kl_divergence": kl_divergence,
            "js_distance": js_distance,
            "js_divergence": js_divergence,
            "mean_baseline_mae": float(mean_absolute_error(truths_np, mean_baseline)),
            "mean_baseline_rmse": float(
                np.sqrt(mean_squared_error(truths_np, mean_baseline))
            ),
            "mean_baseline_kl_divergence": mean_baseline_kl_divergence,
            "mean_baseline_js_distance": mean_baseline_js_distance,
            "mean_baseline_js_divergence": mean_baseline_js_divergence,
            "prediction_mae_from_train_mean": float(
                np.mean(np.abs(predictions_np - mean_baseline))
            ),
            "per_cell_type_mae": dict(zip(self.cell_types, per_type_mae.astype(float))),
            "per_cell_type_rmse": dict(zip(self.cell_types, per_type_rmse.astype(float))),
            "per_cell_type_prediction_mean": dict(
                zip(self.cell_types, per_type_prediction_mean.astype(float))
            ),
            "per_cell_type_prediction_std": dict(
                zip(self.cell_types, per_type_prediction_std.astype(float))
            ),
            "per_cell_type_truth_mean": dict(
                zip(self.cell_types, per_type_truth_mean.astype(float))
            ),
            "per_cell_type_truth_std": dict(
                zip(self.cell_types, per_type_truth_std.astype(float))
            ),
            "per_cell_type_train_mean": dict(
                zip(self.cell_types, self.fold_train_target_mean.astype(float))
            ),
            "per_cell_type_pearson_across_samples": cell_type_pearson_across_samples,
            "per_cell_type_spearman_across_samples": cell_type_spearman_across_samples,
            "cell_types": self.cell_types,
            "target_columns": self.target_columns,
            "n_test_samples": int(len(truths_np)),
            "predictions": predictions_np,
            "truths": truths_np,
            "sample_pearson_across_cell_types": sample_pearson_across_cell_types,
            "sample_spearman_across_cell_types": sample_spearman_across_cell_types,
        }

    def _evaluate(self) -> dict:
        self.model.eval()
        predictions = []
        truths = []

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for data, targets in self.test_loader:
                data = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in data.items()
                }
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(data)
                predictions.append(F.softmax(logits, dim=-1))
                truths.append(targets)

        predictions = torch.cat(predictions, dim=0)
        truths = torch.cat(truths, dim=0)

        if self.is_distributed:
            predictions = distributed_concat(predictions, self.test_dataset_size, self.world_size)
            truths = distributed_concat(truths, self.test_dataset_size, self.world_size)

        predictions_np = predictions.cpu().numpy()
        truths_np = truths.cpu().numpy()
        loss_name = str(getattr(self.task_cfg, "loss", "kl")).lower()
        if loss_name == "kl":
            test_loss = F.kl_div(
                predictions.clamp_min(1e-12).log(),
                truths,
                reduction="batchmean",
            ).item()
        elif loss_name == "mse":
            test_loss = F.mse_loss(predictions, truths).item()
        else:
            test_loss = F.l1_loss(predictions, truths).item()

        return self._evaluation_metrics_from_arrays(
            predictions_np,
            truths_np,
            test_loss=float(test_loss),
        )

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
            }
        for key, value in test_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[key] = float(value)
        for metric_key, prefix in (
            ("per_cell_type_mae", "mae"),
            ("per_cell_type_rmse", "rmse"),
            ("per_cell_type_prediction_mean", "prediction_mean"),
            ("per_cell_type_prediction_std", "prediction_std"),
            ("per_cell_type_truth_mean", "truth_mean"),
            ("per_cell_type_truth_std", "truth_std"),
            ("per_cell_type_train_mean", "train_mean"),
            ("per_cell_type_pearson_across_samples", "pearson_across_samples"),
            ("per_cell_type_spearman_across_samples", "spearman_across_samples"),
        ):
            values = test_metrics.get(metric_key, {})
            if isinstance(values, dict):
                for cell_type, value in values.items():
                    row[f"{prefix}_{cell_type}"] = float(value)
        return row

    def _prediction_rows(
        self,
        model_key: str,
        fold: int,
        checkpoint_path: str,
        test_adata: ad.AnnData,
        test_metrics: dict[str, object],
    ) -> list[dict[str, object]]:
        predictions = np.asarray(test_metrics["predictions"], dtype=float)
        truths = np.asarray(test_metrics["truths"], dtype=float)
        sample_pearson = np.asarray(
            test_metrics["sample_pearson_across_cell_types"],
            dtype=float,
        )
        sample_spearman = np.asarray(
            test_metrics["sample_spearman_across_cell_types"],
            dtype=float,
        )
        rows: list[dict[str, object]] = []
        context_columns = [
            col
            for col in ["dataset_id", "donor_id", "tissue_general", "total_cells", "n_cell_types"]
            if col in test_adata.obs
        ]
        context_values = {
            col: test_adata.obs[col].astype(str).to_numpy()
            for col in context_columns
        }
        if self.fold_train_target_mean is None:
            raise RuntimeError("Training-fold target mean is unavailable for prediction export.")

        for sample_idx, sample_id in enumerate(test_adata.obs_names.astype(str)):
            row: dict[str, object] = {
                "model": model_key,
                "fold": fold,
                "finetune_mode": self._finetune_mode(),
                "checkpoint_path": checkpoint_path,
                "sample_id": sample_id,
                "sample_pearson_across_cell_types": float(sample_pearson[sample_idx]),
                "sample_spearman_across_cell_types": float(sample_spearman[sample_idx]),
            }
            for col in context_columns:
                row[col] = context_values[col][sample_idx]
            for cell_idx, cell_type in enumerate(self.cell_types):
                pred_value = float(predictions[sample_idx, cell_idx])
                true_value = float(truths[sample_idx, cell_idx])
                row[f"pred_{cell_type}"] = pred_value
                row[f"true_{cell_type}"] = true_value
                row[f"train_mean_{cell_type}"] = float(
                    self.fold_train_target_mean[cell_idx]
                )
                row[f"abs_error_{cell_type}"] = abs(pred_value - true_value)
            rows.append(row)
        return rows

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
        return aggregate

    def _cleanup_fold_state(self) -> None:
        self.train_loader = None
        self.test_loader = None
        self.fold_train_target_mean = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> dict:
        try:
            self._setup_runtime()
            adata, targets, groups = self._prepare_cv_data()
            splits = self._build_or_load_cv_splits(adata, groups)
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)
            if self.is_master:
                log.info(
                    "Prepared deconvolution CV data: samples=%d, genes=%d, cell_types=%d, folds=%d",
                    adata.n_obs,
                    adata.n_vars,
                    len(self.cell_types),
                    len(splits),
                )

            epochs = int(getattr(self.task_cfg, "epochs", 20))
            aggregate_rows: list[dict[str, object]] = []

            for model_key, checkpoint_path in checkpoint_paths.items():
                fold_rows: list[dict[str, object]] = []
                prediction_rows: list[dict[str, object]] = []
                curve_rows: list[dict[str, object]] = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + fold_idx
                    )
                    train_adata = adata[train_idx]
                    test_adata = adata[test_idx]
                    train_targets = targets[train_idx]
                    test_targets = targets[test_idx]
                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | train=%d, test=%d",
                            model_key,
                            fold_idx,
                            len(splits),
                            train_adata.n_obs,
                            test_adata.n_obs,
                        )

                    self._build_loaders(train_adata, test_adata, train_targets, test_targets)
                    self._build_model(checkpoint_path)
                    self._build_optimization()

                    last_train_metrics = {"loss": float("nan")}
                    last_validation_metrics: dict[str, object] | None = None
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        last_validation_metrics = self._evaluate()
                        if self.is_master:
                            curve_rows.append(
                                {
                                    "model": model_key,
                                    "fold": fold_idx,
                                    "epoch": epoch,
                                    "finetune_mode": self._finetune_mode(),
                                    "train_loss": last_train_metrics["loss"],
                                    "validation_loss": last_validation_metrics["loss"],
                                    "validation_mae": last_validation_metrics["mae"],
                                    "validation_rmse": last_validation_metrics["rmse"],
                                    "learning_rates": ";".join(
                                        f"{float(group['lr']):.6g}"
                                        for group in self.optimizer.param_groups
                                    ),
                                }
                            )
                            self._write_csv(
                                self._task_output_dir()
                                / f"{self._output_prefix()}_{model_key}_training_curves.csv",
                                curve_rows,
                            )
                            log.info(
                                "Model %s | Fold %d/%d | Epoch %d | "
                                "Training Loss: %.6f | Validation Loss: %.6f",
                                model_key,
                                fold_idx,
                                len(splits),
                                epoch,
                                last_train_metrics["loss"],
                                last_validation_metrics["loss"],
                            )

                    test_metrics = (
                        last_validation_metrics
                        if last_validation_metrics is not None
                        else self._evaluate()
                    )
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
                    self._cleanup_fold_state()
                    del train_adata, test_adata, train_targets, test_targets
                    gc.collect()

                if self.is_master:
                    aggregate_rows.append(
                        self._write_model_results(
                            checkpoint_path,
                            fold_rows,
                            prediction_rows,
                        )
                    )
                if self.is_distributed:
                    dist.barrier()

            if self.is_master:
                combined_dir = self._task_output_dir()
                output_path = combined_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
                self._write_csv(output_path, aggregate_rows, comment=getattr(self, "_missing_genes_note", ""))
                complete_run_metadata(self._run_metadata_path, output_path)
                return {
                    "results_path": str(output_path),
                    "results": aggregate_rows,
                }
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
