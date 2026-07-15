from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from finetune.canc_type_class.runner import (
    ROOT,
    CancTypeClassRunner,
    GroupedCosineAnnealingWarmupRestarts,
)
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)


def solve_hidden_dim(
    *,
    input_dim: int,
    output_dim: int,
    target_params: int,
    hidden_layers: int = 2,
) -> int:
    """Choose hidden width for Linear/SELU MLP close to target parameter count."""
    if hidden_layers < 1:
        raise ValueError("raw_mlp_hidden_layers must be at least 1.")
    if hidden_layers == 1:
        # Params: input*h + h + h*out + out
        denom = input_dim + output_dim + 1
        return max(1, int(round((target_params - output_dim) / denom)))

    # Params: input*h + h + (hidden_layers - 1) * (h*h + h) + h*out + out
    #       = (hidden_layers - 1) h^2 + h * (input + output + hidden_layers) + output
    a = hidden_layers - 1
    b = input_dim + output_dim + hidden_layers
    c = output_dim - target_params
    discriminant = b * b - 4 * a * c
    if discriminant <= 0:
        raise ValueError(
            f"Cannot solve hidden_dim for input_dim={input_dim}, output_dim={output_dim}, "
            f"target_params={target_params}, hidden_layers={hidden_layers}."
        )
    hidden_dim = int(round((-b + math.sqrt(discriminant)) / (2 * a)))
    return max(1, hidden_dim)


class RawExpressionDataset(Dataset):
    def __init__(
        self,
        data,
        labels: np.ndarray,
        feature_indices: np.ndarray,
        mean: np.ndarray | None,
        std: np.ndarray | None,
    ) -> None:
        self.data = data
        self.labels = np.asarray(labels, dtype=np.int64)
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
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return torch.from_numpy(values), label


class RawMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
        ]
        for _ in range(hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CancTypeClassRawMLPRunner(CancTypeClassRunner):
    """Param-matched raw-expression MLP baseline for TCGA 5-type classification."""

    task_name = "canc_type_class"
    config_node = "canc_type_class"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "canc_type_class" in cfg.finetune:
            return cfg.finetune.canc_type_class
        raise ValueError(
            "Could not find cancer type classification config. "
            "Expected cfg.finetune.canc_type_class."
        )

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

    def _save_run_metadata(self, checkpoint_paths: dict[str, str] | None = None) -> None:
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
                "task": "finetune.canc_type_class_raw_mlp",
                "baseline": "raw_expression_mlp",
                "variant": self._finetune_mode(),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "git_commit": self._get_git_commit(),
            },
        )

    def _select_feature_indices(self, train_adata: ad.AnnData) -> np.ndarray:
        feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
        if feature_mode == "all_genes":
            return np.arange(train_adata.n_vars, dtype=np.int64)
        if feature_mode == "hvg1199":
            return self._select_training_hvg_indices(train_adata)
        raise ValueError(
            "finetune.canc_type_class.raw_mlp_feature_mode must be one of: "
            "all_genes, hvg1199."
        )

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

    def _build_raw_loaders(self, train_adata: ad.AnnData, test_adata: ad.AnnData) -> None:
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

        class_weight_power = float(getattr(self.task_cfg, "class_weight_power", 0.0))
        if not np.isfinite(class_weight_power) or class_weight_power < 0:
            raise ValueError("class_weight_power must be a finite non-negative number.")
        if class_weight_power == 0.0:
            self.train_class_weights = None
        else:
            class_counts = np.bincount(train_labels, minlength=len(self.label_dict)).astype(np.float64)
            if np.any(class_counts == 0):
                missing_labels = self.label_dict[class_counts == 0].tolist()
                raise ValueError(
                    "Cannot compute class weights because the training fold contains "
                    f"no samples for classes {missing_labels}."
                )
            balanced = len(train_labels) / (len(self.label_dict) * class_counts)
            self.train_class_weights = torch.as_tensor(
                np.power(balanced, class_weight_power),
                dtype=torch.float32,
            )

        self.raw_feature_indices = self._select_feature_indices(train_adata)
        if bool(getattr(self.task_cfg, "raw_mlp_standardize", True)):
            mean, std = self._feature_mean_std(train_adata.X, self.raw_feature_indices)
        else:
            mean, std = None, None

        train_dataset = RawExpressionDataset(
            train_adata.X,
            train_labels,
            self.raw_feature_indices,
            mean,
            std,
        )
        test_dataset = RawExpressionDataset(
            test_adata.X,
            test_labels,
            self.raw_feature_indices,
            mean,
            std,
        )
        self.test_dataset_size = len(test_dataset)

        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
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
                raise ValueError("finetune.canc_type_class.prefetch_factor must be positive.")
            loader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": False,
                }
            )

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
                "Raw MLP data: features=%d | standardize=%s | workers=%d",
                len(self.raw_feature_indices),
                bool(getattr(self.task_cfg, "raw_mlp_standardize", True)),
                num_workers,
            )

    def _build_raw_model(self) -> None:
        input_dim = int(len(self.raw_feature_indices))
        output_dim = int(len(self.label_dict))
        hidden_layers = int(getattr(self.task_cfg, "raw_mlp_hidden_layers", 2))
        if hidden_layers <= 0:
            raise ValueError("raw_mlp_hidden_layers must be positive.")

        configured_hidden_dim = getattr(self.task_cfg, "raw_mlp_hidden_dim", None)
        if configured_hidden_dim is None:
            target_params = int(getattr(self.task_cfg, "raw_mlp_target_params", 6_560_000))
            hidden_dim = solve_hidden_dim(
                input_dim=input_dim,
                output_dim=output_dim,
                target_params=target_params,
                hidden_layers=hidden_layers,
            )
        else:
            hidden_dim = int(configured_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("raw_mlp_hidden_dim must be positive.")

        model = RawMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        total_params = sum(param.numel() for param in model.parameters())
        if self.is_master:
            log.info(
                "Raw MLP architecture: input=%d | hidden=%d x %d | output=%d | params=%d",
                input_dim,
                hidden_dim,
                hidden_layers,
                output_dim,
                total_params,
            )

        model = model.to(self.device)
        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                model = DDP(model)
        self.model = model

    def _build_raw_optimization(self) -> None:
        learning_rate = float(getattr(self.task_cfg, "raw_mlp_learning_rate", 1e-4))
        params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = Adam([{"params": params, "lr": learning_rate, "name": "raw_mlp"}])

        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_min_lr_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = (
            float(configured_min_lr_ratio)
            if configured_min_lr_ratio is not None
            else min_lr / max(learning_rate, 1e-12)
        )
        self.scheduler = GroupedCosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 20)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 1)),
            max_lrs=[learning_rate],
            min_lr_ratio=min_lr_ratio,
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 2)),
            gamma=float(getattr(self.task_cfg, "gamma", 1.0)),
        )
        loss_weights = (
            None
            if self.train_class_weights is None
            else self.train_class_weights.to(self.device)
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights).to(self.device)

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        return

    def _move_batch_to_device(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.to(self.device, non_blocking=True)

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
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

    def run(self) -> dict:
        try:
            self._setup_runtime()
            adata, labels, groups = self._prepare_cv_data()
            splits = self._build_cv_splits(labels, groups)
            self._save_run_metadata({})

            if self.is_master:
                log.info(
                    "Prepared raw MLP TCGA 5-type classification data: "
                    "samples=%d, genes=%d, folds=%d",
                    adata.n_obs,
                    adata.n_vars,
                    len(splits),
                )

            epochs = int(getattr(self.task_cfg, "epochs", 20))
            model_key = "raw_mlp"
            fold_rows: list[dict[str, object]] = []
            prediction_rows: list[dict[str, object]] = []
            confusion_matrices: list[np.ndarray] = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + self.rank + fold_idx)
                train_adata = adata[train_idx].copy()
                test_adata = adata[test_idx].copy()
                if self.is_master:
                    log.info(
                        "Raw MLP | Fold %d/%d | train=%d, test=%d",
                        fold_idx,
                        len(splits),
                        train_adata.n_obs,
                        test_adata.n_obs,
                    )

                self._build_raw_loaders(train_adata, test_adata)
                self._build_raw_model()
                self._build_raw_optimization()

                last_train_metrics = {"loss": float("nan"), "accuracy": float("nan")}
                for epoch in range(1, epochs + 1):
                    last_train_metrics = self._train_one_epoch(epoch)
                    if self.is_master:
                        log.info(
                            "Raw MLP | Fold %d/%d | Epoch %d | Loss: %.6f | Accuracy: %.4f%%",
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
                            checkpoint_path="",
                            train_metrics=last_train_metrics,
                            test_metrics=test_metrics,
                        )
                    )
                    prediction_rows.extend(
                        self._prediction_rows(
                            model_key=model_key,
                            fold=fold_idx,
                            checkpoint_path="",
                            test_adata=test_adata,
                            test_metrics=test_metrics,
                        )
                    )
                    confusion_matrices.append(np.asarray(test_metrics["confusion_matrix"]))

                self._cleanup_fold_state()
                if self.is_distributed:
                    dist.barrier()

            if self.is_master:
                aggregate = self._write_model_results(
                    "",
                    fold_rows,
                    prediction_rows,
                    confusion_matrices,
                )
                out_dir = self._task_output_dir()
                output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
                self._write_csv(
                    output_path,
                    [aggregate],
                )
                return {"results_path": str(output_path), "results": [aggregate]}
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
