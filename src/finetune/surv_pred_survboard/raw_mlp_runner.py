from __future__ import annotations

import math

import numpy as np
import torch
from omegaconf import DictConfig
from scipy import sparse
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from finetune.canc_type_class.raw_mlp_runner import solve_hidden_dim
from finetune.canc_type_class.runner import GroupedCosineWarmupUpdateScheduler
from finetune.surv_pred_survboard.runner import (
    SurvPredSurvBoardRunner,
)
from utils import SequentialDistributedSampler


class RawSurvivalDataset(Dataset):
    def __init__(
        self,
        X,
        times: np.ndarray,
        events: np.ndarray,
        feature_indices: np.ndarray,
        mean: np.ndarray | None,
        std: np.ndarray | None,
    ) -> None:
        self.X = X
        self.times = np.asarray(times, dtype=np.float32)
        self.events = np.asarray(events, dtype=np.float32)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int):
        row = self.X[index, self.feature_indices]
        values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        values = values.astype(np.float32, copy=False)
        if self.mean is not None and self.std is not None:
            values = (values - self.mean) / self.std
        return (
            {"raw_expr": torch.from_numpy(values)},
            torch.tensor(self.times[index], dtype=torch.float32),
            torch.tensor(self.events[index], dtype=torch.float32),
        )


class RawSurvivalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SELU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["raw_expr"]).squeeze(-1)


class SurvPredSurvBoardRawMLPRunner(SurvPredSurvBoardRunner):
    """Raw-expression MLP Cox baseline for the SurvBoard survival task."""

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "surv_pred_survboard" in cfg.finetune:
            return cfg.finetune.surv_pred_survboard
        raise ValueError(
            "Could not find SurvBoard survival config. "
            "Expected cfg.finetune.surv_pred_survboard."
        )

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "raw_mlp_variant", "") or "").strip()
        if not variant:
            feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
            variant = f"raw_mlp_{feature_mode}"
        return variant

    def _output_suffix(self) -> str:
        return ""

    def _get_checkpoint_paths(self) -> dict[str, str]:
        return {"raw_mlp": ""}

    def _select_training_hvg_indices(self, train_adata):
        feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
        if feature_mode == "all_genes":
            return np.arange(train_adata.n_vars, dtype=np.int64)
        if feature_mode == "hvg1199":
            return super()._select_training_hvg_indices(train_adata)
        raise ValueError("raw_mlp_feature_mode must be one of: all_genes, hvg1199.")

    @staticmethod
    def _feature_mean_std(data, feature_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = data[:, feature_indices]
        if sparse.issparse(selected):
            mean = np.asarray(selected.mean(axis=0)).ravel().astype(np.float32)
            mean_sq = np.asarray(selected.power(2).mean(axis=0)).ravel().astype(np.float32)
            var = np.maximum(mean_sq - mean * mean, 0.0)
        else:
            arr = np.asarray(selected, dtype=np.float32)
            mean = arr.mean(axis=0, dtype=np.float64).astype(np.float32)
            var = arr.var(axis=0, dtype=np.float64).astype(np.float32)
        std = np.sqrt(var).astype(np.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    def _build_loaders(self, train_X, train_times, train_events, test_X, test_times, test_events) -> None:
        if self.fold_gene_indices is None:
            raise RuntimeError("Raw MLP feature indices have not been selected.")
        batch_size = int(getattr(self.task_cfg, "batch_size", 32))
        num_workers = int(getattr(self.task_cfg, "num_workers", 0))
        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs.update(
                {
                    "prefetch_factor": int(getattr(self.task_cfg, "prefetch_factor", 2)),
                    "persistent_workers": False,
                }
            )
        if bool(getattr(self.task_cfg, "raw_mlp_standardize", True)):
            mean, std = self._feature_mean_std(train_X, self.fold_gene_indices)
        else:
            mean, std = None, None
        self.train_dataset = RawSurvivalDataset(train_X, train_times, train_events, self.fold_gene_indices, mean, std)
        self.test_dataset = RawSurvivalDataset(test_X, test_times, test_events, self.fold_gene_indices, mean, std)
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            test_sampler = SequentialDistributedSampler(self.test_dataset, batch_size=batch_size, world_size=self.world_size, rank=self.rank)
            train_infer_sampler = SequentialDistributedSampler(self.train_dataset, batch_size=batch_size, world_size=self.world_size, rank=self.rank)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, **loader_kwargs)
            self.train_infer_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=train_infer_sampler, shuffle=False, **loader_kwargs)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, **loader_kwargs)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
            self.train_infer_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    def _build_model(self, checkpoint_path: str) -> None:
        input_dim = int(len(self.fold_gene_indices))
        hidden_layers = int(getattr(self.task_cfg, "raw_mlp_hidden_layers", 2))
        hidden_dim_cfg = getattr(self.task_cfg, "raw_mlp_hidden_dim", None)
        hidden_dim = (
            solve_hidden_dim(
                input_dim=input_dim,
                output_dim=1,
                target_params=int(getattr(self.task_cfg, "raw_mlp_target_params", 6_560_000)),
                hidden_layers=hidden_layers,
            )
            if hidden_dim_cfg is None
            else int(hidden_dim_cfg)
        )
        self.model = RawSurvivalMLP(input_dim, hidden_dim, hidden_layers).to(self.device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank) if self.device.type == "cuda" else DDP(self.model)

    def _build_optimization(self) -> None:
        lr = float(getattr(self.task_cfg, "raw_mlp_learning_rate", 1e-4))
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": lr, "name": "raw_mlp"}])
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        grad_acc_steps = max(
            1,
            int(getattr(self.task_cfg, "grad_accumulation_steps", 4)),
        )
        updates_per_epoch = math.ceil(len(self.train_loader) / grad_acc_steps)
        self.scheduler = GroupedCosineWarmupUpdateScheduler(
            self.optimizer,
            max_lrs=[lr],
            min_lr_ratio=min_lr / max(lr, 1e-12),
            updates_per_epoch=updates_per_epoch,
            epochs=int(getattr(self.task_cfg, "epochs", 20)),
            warmup_epochs=int(getattr(self.task_cfg, "warmup_epochs", 2)),
        )

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        return
