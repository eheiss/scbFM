from __future__ import annotations

import csv
import json
import logging
import os
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

from finetune.canc_type_class.runner import GroupedCosineAnnealingWarmupRestarts
from performer_pytorch import PerformerLM
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
        seq_len: int,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, embedding_dim))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(seq_len, 512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, hidden_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        return self.fc3(x)


class DeconvDataset(Dataset):
    def __init__(
        self,
        data,
        targets: np.ndarray,
        bin_num: int,
        special_token_id: int,
    ) -> None:
        self.data = data
        targets = np.asarray(targets, dtype=np.float32)
        target_sums = targets.sum(axis=1, keepdims=True)
        if np.any(target_sums <= 0):
            raise ValueError("Every deconvolution target row must have a positive sum.")
        self.targets = targets / target_sums
        self.bin_num = bin_num
        self.special_token_id = special_token_id

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data[index]
        if sparse.issparse(row):
            full_seq = row.toarray().ravel()
        else:
            full_seq = np.asarray(row).ravel()
        full_seq = np.clip(full_seq, 0, self.bin_num)
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat(
            (full_seq, torch.tensor([self.special_token_id], dtype=torch.long))
        )
        target = torch.from_numpy(self.targets[index]).float()
        return full_seq, target


class DeconvRunner:
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

        self.class_count = int(self.model_cfg.bin_num) + 2
        self.vocab_size = self.class_count + 1
        self.special_token_id = self.class_count

        self.cell_types: list[str] = []
        self.target_columns: list[str] = []
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size = 0
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
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
        (out_dir / "config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        self._write_json(
            out_dir / "run_metadata.json",
            {
                "task": TASK_NAME,
                "finetune_mode": str(getattr(self.task_cfg, "finetune_mode", "head_only")),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 10)),
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
                "finetune.deconv.pretrained_model_paths must define "
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
                "Missing checkpoint paths in finetune.deconv.pretrained_model_paths: "
                f"{missing}"
            )
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
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
        normalized: dict[str, str] = {}

        def visit(cell_type_parts: list[str], value) -> None:
            value_string = self._as_string(value)
            if value_string is not None:
                normalized["/".join(cell_type_parts)] = value_string
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
                    normalized["/".join(cell_type_parts)] = direct_value
                    return

                # HDF5-backed AnnData stores '/' in uns dict keys as nested groups.
                for key, child in value.items():
                    key_string = self._as_string(key)
                    if key_string is None:
                        key_string = str(key)
                    child_string = self._as_string(child)
                    if child_string is None and key_string in obs_columns:
                        normalized["/".join(cell_type_parts)] = key_string
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

        invalid_columns = [col for col in normalized.values() if col not in obs_columns]
        if invalid_columns:
            inferred = self._infer_proportion_mapping_from_obs(obs_columns)
            if inferred:
                log.warning(
                    "Ignoring unusable uns['cell_type_proportion_columns'] entries and "
                    "inferring targets from prop__* obs columns instead. Invalid columns: %s",
                    invalid_columns[:10],
                )
                return inferred

        return normalized

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
        adata.var_names_make_unique()
        targets = self._load_targets(adata)

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes for the scbFM backbone, got {adata.n_vars}."
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

    def _build_cv_splits(
        self,
        adata: ad.AnnData,
        groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 10))
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

    def _build_loaders(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
    ) -> None:
        batch_size = int(getattr(self.task_cfg, "batch_size", 2))
        train_dataset = DeconvDataset(
            train_adata.X,
            train_targets,
            bin_num=int(self.model_cfg.bin_num),
            special_token_id=self.special_token_id,
        )
        test_dataset = DeconvDataset(
            test_adata.X,
            test_targets,
            bin_num=int(self.model_cfg.bin_num),
            special_token_id=self.special_token_id,
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
            )
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                shuffle=False,
            )
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def _build_model(self, checkpoint_path: str) -> None:
        finetune_mode = str(getattr(self.task_cfg, "finetune_mode", "head_only"))
        valid_modes = {"head_only", "full_ft", "adapters"}
        if finetune_mode not in valid_modes:
            raise ValueError(
                f"Unsupported finetune_mode '{finetune_mode}'. Expected one of {sorted(valid_modes)}."
            )

        model = PerformerLM(
            num_tokens=self.vocab_size,
            max_seq_len=int(self.model_cfg.gene_num) + 1,
            dim=int(self.model_cfg.dim),
            depth=int(self.model_cfg.depth),
            heads=int(self.model_cfg.heads),
            dim_head=int(self.model_cfg.dim_head),
            ff_mult=int(self.model_cfg.ff_mult),
            nb_features=self.model_cfg.nb_features,
            feature_redraw_interval=int(self.model_cfg.feature_redraw_interval),
            ff_chunks=int(self.model_cfg.ff_chunks),
            ff_glu=bool(self.model_cfg.ff_glu),
            emb_dropout=float(self.model_cfg.emb_dropout),
            ff_dropout=float(self.model_cfg.ff_dropout),
            attn_dropout=float(self.model_cfg.attn_dropout),
            use_scalenorm=bool(self.model_cfg.use_scalenorm),
            use_rezero=bool(self.model_cfg.use_rezero),
            no_projection=bool(self.model_cfg.no_projection),
            tie_embed=bool(self.model_cfg.tie_embed),
            g2v_position_emb=bool(self.model_cfg.g2v_position_emb),
            auto_check_redraw=bool(self.model_cfg.auto_check_redraw),
            qkv_bias=bool(self.model_cfg.qkv_bias),
        )

        if checkpoint_path:
            resolved_path = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved_path, map_location="cpu")
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            model.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved_path)
        else:
            log.info("Using randomly initialized backbone")

        model.to_out = DeconvPredHead(
            seq_len=int(self.model_cfg.gene_num) + 1,
            embedding_dim=int(self.model_cfg.dim),
            output_dim=len(self.cell_types),
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 128)),
            dropout=float(getattr(self.task_cfg, "head_dropout", 0.0)),
        )

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

        if self.is_master:
            group_summaries = [
                f"{group.get('name', idx)}: params={sum(p.numel() for p in group['params'])}, "
                f"max_lr={max_lr:.2e}, min_lr={max_lr * min_lr_ratio:.2e}"
                for idx, (group, max_lr) in enumerate(zip(param_groups, max_lrs))
            ]
            log.info("Optimizer parameter groups: %s", "; ".join(group_summaries))

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        finetune_mode = str(getattr(self.task_cfg, "finetune_mode", "head_only"))
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

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_name = str(getattr(self.task_cfg, "loss", "kl")).lower()
        if loss_name == "kl":
            return F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="batchmean")
        pred_props = F.softmax(logits, dim=-1)
        if loss_name == "mse":
            return F.mse_loss(pred_props, targets)
        if loss_name in {"mae", "l1"}:
            return F.l1_loss(pred_props, targets)
        raise ValueError("Unsupported deconvolution loss. Expected one of: kl, mse, mae.")

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

        for step_idx, (data, targets) in enumerate(self.train_loader, start=1):
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and step_idx % grad_acc_steps != 0
            )
            sync_context = self.model.no_sync() if use_no_sync else nullcontext()

            with sync_context:
                logits = self.model(data)
                loss = self._compute_loss(logits, targets)
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

    def _evaluate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        predictions = []
        truths = []

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(data)
                loss = self._compute_loss(logits, targets)
                running_loss += loss.item()
                predictions.append(F.softmax(logits, dim=-1))
                truths.append(targets)

        predictions = torch.cat(predictions, dim=0)
        truths = torch.cat(truths, dim=0)

        if self.is_distributed:
            predictions = distributed_concat(predictions, self.test_dataset_size, self.world_size)
            truths = distributed_concat(truths, self.test_dataset_size, self.world_size)

        predictions_np = predictions.cpu().numpy()
        truths_np = truths.cpu().numpy()
        test_loss = running_loss / len(self.test_loader)
        if self.is_distributed:
            test_loss = get_reduced(test_loss, self.local_rank, 0, self.world_size)

        per_type_mae = np.mean(np.abs(predictions_np - truths_np), axis=0)
        per_type_rmse = np.sqrt(np.mean((predictions_np - truths_np) ** 2, axis=0))
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
            "per_cell_type_mae": {
                cell_type: float(value)
                for cell_type, value in zip(self.cell_types, per_type_mae.tolist())
            },
            "per_cell_type_rmse": {
                cell_type: float(value)
                for cell_type, value in zip(self.cell_types, per_type_rmse.tolist())
            },
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
            "finetune_mode": str(getattr(self.task_cfg, "finetune_mode", "head_only")),
            "checkpoint_path": checkpoint_path,
            "train_loss": float(train_metrics["loss"]),
        }
        for key, value in test_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[key] = float(value)
        for metric_key, prefix in (
            ("per_cell_type_mae", "mae"),
            ("per_cell_type_rmse", "rmse"),
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

        for sample_idx, sample_id in enumerate(test_adata.obs_names.astype(str)):
            row: dict[str, object] = {
                "model": model_key,
                "fold": fold,
                "finetune_mode": str(getattr(self.task_cfg, "finetune_mode", "head_only")),
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
                row[f"abs_error_{cell_type}"] = abs(pred_value - true_value)
            rows.append(row)
        return rows

    def _task_output_dir(self) -> Path:
        finetune_mode = str(getattr(self.task_cfg, "finetune_mode", "head_only"))
        return ROOT / "output" / TASK_NAME / finetune_mode

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
        self._write_csv(out_dir / f"{model_key}_fold_metrics.csv", fold_rows)
        self._write_csv(out_dir / f"{model_key}_evaluation_metrics.csv", [aggregate])
        self._write_csv(out_dir / f"{model_key}_predictions.csv", prediction_rows)
        return aggregate

    def _cleanup_fold_state(self) -> None:
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> dict:
        try:
            self._setup_runtime()
            adata, targets, groups = self._prepare_cv_data()
            splits = self._build_cv_splits(adata, groups)
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

            epochs = int(getattr(self.task_cfg, "epochs", 10))
            aggregate_rows: list[dict[str, object]] = []

            for model_idx, (model_key, checkpoint_path) in enumerate(checkpoint_paths.items()):
                fold_rows: list[dict[str, object]] = []
                prediction_rows: list[dict[str, object]] = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + model_idx * 10000
                        + fold_idx
                    )
                    train_adata = adata[train_idx].copy()
                    test_adata = adata[test_idx].copy()
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
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        if self.is_master:
                            log.info(
                                "Model %s | Fold %d/%d | Epoch %d | Training Loss: %.6f",
                                model_key,
                                fold_idx,
                                len(splits),
                                epoch,
                                last_train_metrics["loss"],
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
                    self._cleanup_fold_state()

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
                self._write_csv(combined_dir / "evaluation_metrics.csv", aggregate_rows)
                return {
                    "results_path": str(combined_dir / "evaluation_metrics.csv"),
                    "results": aggregate_rows,
                }
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
