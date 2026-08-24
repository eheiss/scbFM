from __future__ import annotations

import math
from contextlib import nullcontext

import anndata as ad
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy import sparse
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from finetune.canc_type_class.raw_mlp_runner import solve_hidden_dim
from finetune.canc_type_class.runner import GroupedCosineWarmupUpdateScheduler
from finetune.deconv.runner import DeconvRunner
from finetune.training_correctness import (
    is_accumulation_boundary,
    normalize_accumulated_gradients,
)
from utils import SequentialDistributedSampler, distributed_concat, get_reduced, seed_all


class RawDeconvDataset(Dataset):
    def __init__(
        self,
        data,
        targets: np.ndarray,
        feature_indices: np.ndarray,
        mean: np.ndarray | None,
        std: np.ndarray | None,
    ) -> None:
        self.data = data
        targets = np.asarray(targets, dtype=np.float32)
        target_sums = targets.sum(axis=1, keepdims=True)
        if np.any(target_sums <= 0):
            raise ValueError("Every deconvolution target row must have a positive sum.")
        self.targets = targets / target_sums
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data[index, self.feature_indices]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()
        values = values.astype(np.float32, copy=False)
        if self.mean is not None and self.std is not None:
            values = (values - self.mean) / self.std
        return torch.from_numpy(values), torch.from_numpy(self.targets[index]).float()


class RawMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SELU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvRawMLPRunner(DeconvRunner):
    """Raw-expression MLP baseline for deconvolution."""

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "deconv" in cfg.finetune:
            return cfg.finetune.deconv
        raise ValueError("Could not find config at cfg.finetune.deconv.")

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

    def _select_raw_feature_indices(self, train_adata: ad.AnnData) -> np.ndarray:
        feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
        if feature_mode == "all_genes":
            return np.arange(train_adata.n_vars, dtype=np.int64)
        if feature_mode == "hvg1199":
            return self._select_distributed_training_hvg_indices(train_adata)
        raise ValueError("raw_mlp_feature_mode must be one of: all_genes, hvg1199.")

    def _build_loaders(self, train_adata, test_adata, train_targets, test_targets) -> None:
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

        self.raw_feature_indices = self._select_raw_feature_indices(train_adata)
        local_feature_indices = np.arange(len(self.raw_feature_indices), dtype=np.int64)
        train_expression = train_adata.X[:, self.raw_feature_indices].copy()
        test_expression = test_adata.X[:, self.raw_feature_indices].copy()
        if bool(getattr(self.task_cfg, "raw_mlp_standardize", True)):
            mean, std = self._feature_mean_std(train_expression, local_feature_indices)
        else:
            mean, std = None, None

        train_target_sums = train_targets.sum(axis=1, keepdims=True)
        self.fold_train_target_mean = np.mean(train_targets / train_target_sums, axis=0)
        train_dataset = RawDeconvDataset(train_expression, train_targets, local_feature_indices, mean, std)
        test_dataset = RawDeconvDataset(test_expression, test_targets, local_feature_indices, mean, std)
        self.test_dataset_size = len(test_dataset)

        if self.is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            test_sampler = SequentialDistributedSampler(test_dataset, batch_size=batch_size, world_size=self.world_size, rank=self.rank)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, **loader_kwargs)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, **loader_kwargs)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    def _build_model(self, checkpoint_path: str) -> None:
        input_dim = int(len(self.raw_feature_indices))
        output_dim = int(len(self.cell_types))
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
        self.model = RawMLP(input_dim, output_dim, hidden_dim, hidden_layers).to(self.device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank) if self.device.type == "cuda" else DDP(self.model)

    def _build_optimization(self) -> None:
        lr = float(getattr(self.task_cfg, "raw_mlp_learning_rate", 1e-4))
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam([{"params": params, "lr": lr, "name": "raw_mlp"}])
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        min_lr_ratio = min_lr / max(lr, 1e-12)
        grad_acc_steps = max(
            1,
            int(getattr(self.task_cfg, "grad_accumulation_steps", 4)),
        )
        updates_per_epoch = math.ceil(len(self.train_loader) / grad_acc_steps)
        self.scheduler = GroupedCosineWarmupUpdateScheduler(
            self.optimizer,
            max_lrs=[lr],
            min_lr_ratio=min_lr_ratio,
            updates_per_epoch=updates_per_epoch,
            epochs=int(getattr(self.task_cfg, "epochs", 20)),
            warmup_epochs=int(getattr(self.task_cfg, "warmup_epochs", 2)),
        )

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        return

    def _optimizer_parameters(self) -> list[torch.nn.Parameter]:
        return [param for group in self.optimizer.param_groups for param in group["params"]]

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 1)))
        running_loss_numerator = 0.0
        running_loss_normalizer = 0.0
        accumulated_normalizer = 0.0
        total_steps = len(self.train_loader)
        for step_idx, (data, targets) in enumerate(self.train_loader, start=1):
            data = data.to(self.device, non_blocking=True)
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
                torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), float(getattr(self.task_cfg, "max_grad_norm", 1e6)))
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

    def _evaluate(self) -> dict:
        self.model.eval()
        predictions = []
        truths = []
        if self.is_distributed:
            dist.barrier()
        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.device, non_blocking=True)
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
        loss_name = str(getattr(self.task_cfg, "loss", "mse")).lower()
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
