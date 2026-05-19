from __future__ import annotations

import csv
import json
import logging
import math
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
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
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from performer_pytorch import PerformerLM
from preprocess import (
    preprocess_adata_for_tokens,
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
TASK_NAME = "disease_class"
CHECKPOINT_MODEL_KEYS = ("pretrain_sc", "pretrain_bulk", "preadapt_sc", "preadapt_bulk")
RANDOM_INIT_MODEL_KEY = "random_init"
MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)


class DiseasePredHead(nn.Module):
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


class GroupedCosineAnnealingWarmupRestarts:
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


class DiseaseClassDataset(Dataset):
    def __init__(
        self,
        data,
        labels: np.ndarray,
        bin_num: int,
        special_token_id: int,
    ) -> None:
        self.data = data
        self.labels = np.asarray(labels, dtype=np.int64)
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
        full_seq = torch.as_tensor(full_seq, dtype=torch.long)
        full_seq = torch.cat(
            (full_seq, torch.tensor([self.special_token_id], dtype=torch.long))
        )
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return full_seq, label


class DiseaseClassRunner:
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

        self.class_count = int(self.model_cfg.bin_num) + 2
        self.vocab_size = self.class_count + 1
        self.special_token_id = self.class_count

        self.label_dict: np.ndarray | None = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size = 0
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.loss_fn: nn.Module | None = None
        self.backbone_optimizer_enabled = False

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "disease_class" in cfg.finetune:
            return cfg.finetune.disease_class
        raise ValueError(
            "Could not find a disease classification config. "
            "Expected cfg.finetune.disease_class."
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
                "finetune.disease_class.pretrained_model_paths must define "
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
                "Missing checkpoint paths in finetune.disease_class.pretrained_model_paths: "
                f"{missing}"
            )
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
        return checkpoint_paths

    def _load_disignatlas(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "disignatlas_data_path", None)
        if not configured_path:
            raise ValueError("finetune.disease_class.disignatlas_data_path must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"DiSignAtlas h5ad file not found: {data_path}")

        adata = ad.read_h5ad(data_path)

        disease_label_col = str(getattr(self.task_cfg, "disease_label_col", "disease"))
        if disease_label_col not in adata.obs:
            raise ValueError(
                f"obs column '{disease_label_col}' not found in DiSignAtlas h5ad. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        adata.obs["disease_label"] = adata.obs[disease_label_col].astype(str)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        return adata

    def _resolve_gene_list_path(self) -> Path:
        gene_list_path = getattr(self.task_cfg, "gene_list_path", None)
        if gene_list_path:
            return Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        return ROOT / "scbFM" / "data" / "gene_list.txt"

    def _should_preprocess_input(self) -> bool:
        return bool(getattr(self.task_cfg, "preprocess", True))

    def _preprocess_adata(self, adata: ad.AnnData) -> ad.AnnData:
        gene_list_path = self._resolve_gene_list_path()
        if self._should_preprocess_input():
            min_genes = int(getattr(self.task_cfg, "min_genes", 0))
            adata, missing_genes = preprocess_adata_for_tokens(
                adata,
                gene_list_path=gene_list_path,
                min_genes=min_genes,
                target_sum=float(getattr(self.task_cfg, "target_sum", 1e4)),
                bin_num=int(self.model_cfg.bin_num),
                reindex_genes=True,
            )
            log.info(
                "Applied shared raw preprocessing: %d target genes missing, output shape %s",
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

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes for the scbFM backbone, got {adata.n_vars}. "
                "Provide a matching gene list or aligned input matrix."
            )

        validate_token_matrix(adata.X, bin_num=int(self.model_cfg.bin_num), name="disease class input data")
        return adata

    def _prepare_cv_data(self) -> tuple[ad.AnnData, np.ndarray]:
        adata = self._load_disignatlas()
        adata = self._preprocess_adata(adata)
        self.label_dict = np.unique(np.asarray(adata.obs["disease_label"]).astype(str))
        labels = np.asarray(adata.obs["disease_label"]).astype(str)
        log.info(
            "DiSignAtlas dataset: %d samples, %d diseases",
            adata.n_obs,
            len(self.label_dict),
        )
        return adata, labels

    def _build_cv_splits(self, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 5))
        if n_splits < 2:
            raise ValueError("finetune.disease_class.cv_folds must be at least 2.")

        _, class_counts = np.unique(labels, return_counts=True)
        min_class_count = int(class_counts.min())
        if n_splits > min_class_count:
            raise ValueError(
                f"cv_folds={n_splits} is larger than the smallest class size ({min_class_count})."
            )

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        return list(splitter.split(np.zeros(labels.shape[0]), labels))

    def _build_loaders(self, train_adata: ad.AnnData, test_adata: ad.AnnData) -> None:
        if self.label_dict is None:
            self.label_dict = np.unique(np.asarray(train_adata.obs["disease_label"]).astype(str))
        label_to_idx = {label: idx for idx, label in enumerate(self.label_dict.tolist())}
        train_labels = np.array(
            [label_to_idx[label] for label in np.asarray(train_adata.obs["disease_label"]).astype(str)],
            dtype=np.int64,
        )
        test_labels = np.array(
            [label_to_idx[label] for label in np.asarray(test_adata.obs["disease_label"]).astype(str)],
            dtype=np.int64,
        )

        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        train_dataset = DiseaseClassDataset(
            train_adata.X,
            train_labels,
            bin_num=int(self.model_cfg.bin_num),
            special_token_id=self.special_token_id,
        )
        test_dataset = DiseaseClassDataset(
            test_adata.X,
            test_labels,
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
            embx_bin_num=int(self.model_cfg.bin_num) if str(getattr(self.model_cfg, "loss_type", "ce")).lower() == "mse" else None,
        )

        if checkpoint_path:
            resolved_path = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved_path, map_location="cpu")
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            model.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved_path)
        else:
            log.info("Using randomly initialized backbone")

        model.to_out = DiseasePredHead(
            seq_len=int(self.model_cfg.gene_num) + 1,
            embedding_dim=int(self.model_cfg.dim),
            output_dim=len(self.label_dict),
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
            raise ValueError("finetune.disease_class.head_learning_rate must be set.")
        if not hasattr(self.task_cfg, "backbone_learning_rate"):
            raise ValueError("finetune.disease_class.backbone_learning_rate must be set.")
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
            raise ValueError("No trainable parameters found for disease classification.")

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
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

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
        running_acc = 0.0

        for step_idx, (data, labels) in enumerate(self.train_loader, start=1):
            data = data.to(self.device, non_blocking=True)
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
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)
            epoch_acc = get_reduced(epoch_acc, self.local_rank, 0, self.world_size)

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
                data = data.to(self.device, non_blocking=True)
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
            test_loss = get_reduced(test_loss, self.local_rank, 0, self.world_size)

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
        dataset_ids = (
            test_adata.obs["dataset"].astype(str).to_numpy()
            if "dataset" in test_adata.obs
            else np.asarray([""] * test_adata.n_obs)
        )
        for idx, (truth_idx, pred_idx) in enumerate(zip(truth_indices, prediction_indices)):
            rows.append(
                {
                    "model": model_key,
                    "fold": fold,
                    "finetune_mode": self._finetune_mode(),
                    "checkpoint_path": checkpoint_path,
                    "sample_id": str(test_adata.obs_names[idx]),
                    "dataset_id": dataset_ids[idx],
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
        return ROOT / "output" / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _finetune_mode(self) -> str:
        return str(getattr(self.task_cfg, "finetune_mode", "head_only"))

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
        self._write_csv(out_dir / f"{prefix}_{model_key}_evaluation_metrics.csv", [aggregate])
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> dict:
        try:
            self._setup_runtime()
            adata, labels = self._prepare_cv_data()
            splits = self._build_cv_splits(labels)
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)
            if self.is_master:
                log.info(
                    "Prepared disease classification CV data: samples=%d, genes=%d, diseases=%d, folds=%d",
                    adata.n_obs,
                    adata.n_vars,
                    len(self.label_dict),
                    len(splits),
                )

            epochs = int(getattr(self.task_cfg, "epochs", 20))
            aggregate_rows: list[dict[str, object]] = []

            for model_idx, (model_key, checkpoint_path) in enumerate(checkpoint_paths.items()):
                fold_rows: list[dict[str, object]] = []
                prediction_rows: list[dict[str, object]] = []
                confusion_matrices: list[np.ndarray] = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + model_idx * 10000
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
                        log.info(
                            "Model %s | Fold %d/%d | Test Loss: %.6f | Accuracy: %.4f | "
                            "F1-Weighted: %.4f | F1-Macro: %.4f",
                            model_key,
                            fold_idx,
                            len(splits),
                            test_metrics["loss"],
                            test_metrics["accuracy"],
                            test_metrics["f1_weighted"],
                            test_metrics["f1_macro"],
                        )
                        print(test_metrics["classification_report"])
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
                self._write_csv(output_path, aggregate_rows)
                return {
                    "results_path": str(output_path),
                    "results": aggregate_rows,
                }
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
