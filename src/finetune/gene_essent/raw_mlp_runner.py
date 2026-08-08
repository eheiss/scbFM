from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from finetune.canc_type_class.raw_mlp_runner import solve_hidden_dim
from finetune.gene_essent.runner import (
    ROOT,
    GeneEssentRunner,
    GroupedCosineAnnealingWarmupRestarts,
)
from utils import SequentialDistributedSampler

log = logging.getLogger(__name__)


class RawGeneEssentDataset(Dataset):
    """Raw expression input; selected-gene CRISPR targets as multi-output regression."""

    def __init__(
        self,
        X_expression,
        Y_targets: np.ndarray,
        feature_indices: np.ndarray,
        target_gene_indices: np.ndarray,
        mean: np.ndarray | None,
        std: np.ndarray | None,
    ) -> None:
        self.X = X_expression
        self.Y = np.asarray(Y_targets, dtype=np.float32)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.target_gene_indices = np.asarray(target_gene_indices, dtype=np.int64)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.X[index, self.feature_indices]
        values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        values = values.astype(np.float32, copy=False)
        if self.mean is not None and self.std is not None:
            values = (values - self.mean) / self.std
        target = self.Y[index, self.target_gene_indices].astype(np.float32, copy=False)
        return {"raw_expr": torch.from_numpy(values)}, torch.from_numpy(target)


class RawGeneEssentMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SELU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["raw_expr"])


class GeneEssentRawMLPRunner(GeneEssentRunner):
    """Parameter-matched raw-expression MLP baseline for gene essentiality.

    The output dimension stays aligned to the same fold-selected genes used by
    the transformer variants. For ``raw_mlp_feature_mode=all_genes`` the MLP
    sees all expression features but still predicts only those selected target
    genes, keeping the metric comparable and the parameter count controlled.
    """

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "gene_essent" in cfg.finetune:
            return cfg.finetune.gene_essent
        raise ValueError("Could not find gene essentiality config. Expected cfg.finetune.gene_essent.")

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "raw_mlp_variant", "") or "").strip()
        if not variant:
            feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
            variant = f"raw_mlp_{feature_mode}"
        return variant

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _get_checkpoint_paths(self) -> dict[str, str]:
        return {"raw_mlp": ""}

    def _save_run_metadata(self, checkpoint_paths: dict | None = None) -> None:
        super()._save_run_metadata(checkpoint_paths or {"raw_mlp": ""})

    def _select_raw_feature_indices(self, X_expression_train) -> np.ndarray:
        feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
        if feature_mode == "all_genes":
            return np.arange(X_expression_train.shape[1], dtype=np.int64)
        if feature_mode == "hvg1199":
            if self.fold_gene_indices is None:
                raise RuntimeError("Fold target genes have not been selected.")
            return self.fold_gene_indices.astype(np.int64, copy=False)
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

    def _build_loaders(
        self,
        X_expression_train,
        Y_train: np.ndarray,
        X_expression_test,
        Y_test: np.ndarray,
    ) -> None:
        if self.fold_gene_indices is None:
            raise RuntimeError("Fold target genes have not been selected.")
        self.raw_feature_indices = self._select_raw_feature_indices(X_expression_train)
        if bool(getattr(self.task_cfg, "raw_mlp_standardize", True)):
            mean, std = self._feature_mean_std(X_expression_train, self.raw_feature_indices)
        else:
            mean, std = None, None

        batch_size = int(getattr(self.task_cfg, "batch_size", 8))
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

        train_dataset = RawGeneEssentDataset(
            X_expression_train,
            Y_train,
            self.raw_feature_indices,
            self.fold_gene_indices,
            mean,
            std,
        )
        test_dataset = RawGeneEssentDataset(
            X_expression_test,
            Y_test,
            self.raw_feature_indices,
            self.fold_gene_indices,
            mean,
            std,
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

    def _build_model(self, checkpoint_path: str) -> None:
        input_dim = int(len(self.raw_feature_indices))
        output_dim = int(len(self.fold_gene_indices))
        hidden_layers = int(getattr(self.task_cfg, "raw_mlp_hidden_layers", 2))
        hidden_dim_cfg = getattr(self.task_cfg, "raw_mlp_hidden_dim", None)
        hidden_dim = (
            solve_hidden_dim(
                input_dim=input_dim,
                output_dim=output_dim,
                target_params=int(getattr(self.task_cfg, "raw_mlp_target_params", 6_560_000)),
                hidden_layers=hidden_layers,
            )
            if hidden_dim_cfg is None
            else int(hidden_dim_cfg)
        )
        self.model = RawGeneEssentMLP(input_dim, output_dim, hidden_dim, hidden_layers).to(self.device)
        if self.is_distributed:
            self.model = (
                DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
                if self.device.type == "cuda"
                else DDP(self.model)
            )
        if self.is_master:
            param_count = sum(p.numel() for p in self.model.parameters())
            log.info(
                "Raw gene-essentiality MLP: input=%d | output=%d selected genes | hidden=%d x %d | params=%d",
                input_dim,
                output_dim,
                hidden_dim,
                hidden_layers,
                param_count,
            )

    def _build_optimization(self) -> None:
        lr = float(getattr(self.task_cfg, "raw_mlp_learning_rate", 1e-4))
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": lr, "name": "raw_mlp"}])
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        self.scheduler = GroupedCosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 20)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 1)),
            max_lrs=[lr],
            min_lr_ratio=min_lr / max(lr, 1e-12),
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 5)),
            gamma=float(getattr(self.task_cfg, "gamma", 1.0)),
        )

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        return

    def _optimizer_parameters(self) -> list[torch.nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _cleanup_fold_state(self) -> None:
        self.raw_feature_indices = None
        super()._cleanup_fold_state()
