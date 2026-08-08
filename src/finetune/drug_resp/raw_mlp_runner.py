from __future__ import annotations

import logging
import math
from pathlib import Path

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
from finetune.drug_resp.runner import (
    ROOT,
    DrugRespRunner,
    GroupedCosineWarmupUpdateScheduler,
)

log = logging.getLogger(__name__)


class RawDrugRespDataset(Dataset):
    """Each item is one optional raw-expression vector and drug-feature pair."""

    def __init__(
        self,
        X_cell,
        drug_emb_matrix: np.ndarray,
        cell_idxs: np.ndarray,
        drug_idxs: np.ndarray,
        ic50_values: np.ndarray,
        feature_indices: np.ndarray,
        mean: np.ndarray | None,
        std: np.ndarray | None,
    ) -> None:
        self.X_cell = X_cell
        self.drug_emb_matrix = np.asarray(drug_emb_matrix, dtype=np.float32)
        self.cell_idxs = np.asarray(cell_idxs, dtype=np.int64)
        self.drug_idxs = np.asarray(drug_idxs, dtype=np.int64)
        self.ic50_values = np.asarray(ic50_values, dtype=np.float32)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.ic50_values.shape[0])

    def __getitem__(self, idx: int):
        cell_idx = int(self.cell_idxs[idx])
        if self.feature_indices.size:
            row = self.X_cell[cell_idx, self.feature_indices]
            values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
            values = values.astype(np.float32, copy=False)
            if self.mean is not None and self.std is not None:
                values = (values - self.mean) / self.std
        else:
            values = np.empty(0, dtype=np.float32)
        return (
            torch.tensor(cell_idx, dtype=torch.long),
            {"raw_expr": torch.from_numpy(values)},
            torch.as_tensor(self.drug_emb_matrix[self.drug_idxs[idx]], dtype=torch.float32),
            torch.as_tensor(self.ic50_values[idx], dtype=torch.float32),
        )


class RawDrugRespMLP(nn.Module):
    """Optional raw expression plus drug features -> MLP -> IC50 scalar."""

    def __init__(self, expression_dim: int, drug_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = int(expression_dim) + int(drug_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None,
        drug_emb: torch.Tensor,
        cell_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cell_emb is not None:
            raw_expr = cell_emb
        else:
            if batch is None:
                raise ValueError("Raw expression batch is required when cell_emb is absent.")
            raw_expr = batch["raw_expr"]
        if raw_expr.shape[0] == 1 and drug_emb.shape[0] > 1:
            raw_expr = raw_expr.expand(drug_emb.shape[0], -1)
        return self.net(torch.cat((raw_expr, drug_emb), dim=-1)).squeeze(-1)


class DrugRespRawMLPRunner(DrugRespRunner):
    """Parameter-matched drug-response MLP baselines."""

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "drug_resp" in cfg.finetune:
            return cfg.finetune.drug_resp
        raise ValueError("Could not find drug response config. Expected cfg.finetune.drug_resp.")

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

    def _select_training_hvg_indices(self, X_cell, training_cell_idxs: np.ndarray) -> np.ndarray:
        feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
        if feature_mode == "all_genes":
            return np.arange(X_cell.shape[1], dtype=np.int64)
        if feature_mode == "hvg1199":
            return super()._select_training_hvg_indices(X_cell, training_cell_idxs)
        if feature_mode == "drug_only":
            return np.empty(0, dtype=np.int64)
        raise ValueError(
            "raw_mlp_feature_mode must be one of: all_genes, hvg1199, drug_only."
        )

    @staticmethod
    def _feature_mean_std(X_cell, cell_idxs: np.ndarray, feature_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = X_cell[np.unique(cell_idxs).astype(np.int64)][:, feature_indices]
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

    def _build_train_loader(
        self,
        X_cell,
        drug_emb_matrix: np.ndarray,
        cell_idxs_train: np.ndarray,
        drug_idxs_train: np.ndarray,
        ic50_train: np.ndarray,
    ) -> None:
        if self.fold_gene_indices is None:
            raise RuntimeError("Raw MLP feature indices have not been selected.")
        if bool(getattr(self.task_cfg, "raw_mlp_standardize", True)):
            if self.fold_gene_indices.size == 0:
                self.raw_feature_mean, self.raw_feature_std = None, None
            else:
                self.raw_feature_mean, self.raw_feature_std = self._feature_mean_std(
                    X_cell,
                    cell_idxs_train,
                    self.fold_gene_indices,
                )
        else:
            self.raw_feature_mean, self.raw_feature_std = None, None

        train_dataset = RawDrugRespDataset(
            X_cell,
            drug_emb_matrix,
            cell_idxs_train,
            drug_idxs_train,
            ic50_train,
            self.fold_gene_indices,
            self.raw_feature_mean,
            self.raw_feature_std,
        )
        self.train_dataset_size = len(train_dataset)
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
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
                "Raw MLP training loader: pairs=%d | expression_features=%d | batch_size=%d",
                self.train_dataset_size,
                self.fold_gene_indices.size,
                batch_size,
            )

    def _cell_backbone_batch(self, X_cell, cell_indices: np.ndarray) -> dict[str, torch.Tensor]:
        if self.fold_gene_indices is None:
            raise RuntimeError("Raw MLP feature indices have not been selected.")
        rows: list[torch.Tensor] = []
        for cell_idx in np.asarray(cell_indices, dtype=np.int64):
            row = X_cell[int(cell_idx), self.fold_gene_indices]
            values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
            values = values.astype(np.float32, copy=False)
            if self.raw_feature_mean is not None and self.raw_feature_std is not None:
                values = (values - self.raw_feature_mean) / self.raw_feature_std
            rows.append(torch.from_numpy(values))
        return {"raw_expr": torch.stack(rows)}

    def _build_test_loader(
        self,
        X_cell,
        drug_emb_matrix: np.ndarray,
        cell_idxs_test: np.ndarray,
        drug_idxs_test: np.ndarray,
        ic50_test: np.ndarray,
    ) -> None:
        if self.fold_gene_indices is None:
            raise RuntimeError("Raw MLP feature indices have not been selected.")
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        loader_kwargs = self._loader_kwargs()
        test_dataset = RawDrugRespDataset(
            X_cell,
            drug_emb_matrix,
            cell_idxs_test,
            drug_idxs_test,
            ic50_test,
            self.fold_gene_indices,
            getattr(self, "raw_feature_mean", None),
            getattr(self, "raw_feature_std", None),
        )
        self.test_dataset_size = len(test_dataset)
        if self.is_distributed:
            from utils import SequentialDistributedSampler

            test_sampler = SequentialDistributedSampler(
                test_dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
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

    def _build_model(self, checkpoint_path: str, drug_emb_dim: int) -> None:
        expression_dim = int(len(self.fold_gene_indices))
        hidden_dim_cfg = getattr(self.task_cfg, "raw_mlp_hidden_dim", None)
        hidden_dim = (
            solve_hidden_dim(
                input_dim=expression_dim + int(drug_emb_dim),
                output_dim=1,
                target_params=int(getattr(self.task_cfg, "raw_mlp_target_params", 6_560_000)),
                hidden_layers=2,
            )
            if hidden_dim_cfg is None
            else int(hidden_dim_cfg)
        )
        self.model = RawDrugRespMLP(
            expression_dim=expression_dim,
            drug_emb_dim=int(drug_emb_dim),
            hidden_dim=hidden_dim,
        ).to(self.device)
        if self.is_distributed:
            self.model = (
                DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
                if self.device.type == "cuda"
                else DDP(self.model)
            )
        if self.is_master:
            param_count = sum(p.numel() for p in self.model.parameters())
            log.info(
                "Raw drug-response MLP: input=%d expression features + %d drug features | "
                "hidden=%d x 2 | params=%d",
                expression_dim,
                int(drug_emb_dim),
                hidden_dim,
                param_count,
            )

    def _build_optimization(self) -> None:
        lr = float(getattr(self.task_cfg, "raw_mlp_learning_rate", 1e-4))
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": lr, "name": "raw_mlp"}])
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        grad_acc_steps = max(
            1,
            int(getattr(self.task_cfg, "grad_accumulation_steps", 4)),
        )
        self.scheduler = GroupedCosineWarmupUpdateScheduler(
            self.optimizer,
            max_lrs=[lr],
            min_lr_ratio=min_lr / max(lr, 1e-12),
            updates_per_epoch=math.ceil(len(self.train_loader) / grad_acc_steps),
            epochs=int(getattr(self.task_cfg, "epochs", 20)),
            warmup_epochs=int(getattr(self.task_cfg, "warmup_epochs", 2)),
        )

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        return

    def _cleanup_fold_state(self) -> None:
        self.raw_feature_mean = None
        self.raw_feature_std = None
        super()._cleanup_fold_state()
