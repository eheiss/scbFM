from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from cancerfoundation_backbone import (
    CancerFoundationBackbone,
    ExpressionBinDecoder,
    ExpressionClsDecoder,
)
from preprocess import read_gene_list
from utils import (
    CosineAnnealingWarmupRestarts,
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _digitize(x: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    left_digits = np.digitize(x, bins)
    right_digits = np.digitize(x, bins, right=True)
    random_offsets = rng.random(len(x))
    digits = random_offsets * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def quantile_bin_expression(values: np.ndarray, bin_num: int, rng: np.random.Generator) -> np.ndarray:
    if bin_num < 2:
        raise ValueError("bin_num must be at least 2 because bin 0 is reserved for zero expression.")

    values = np.asarray(values, dtype=np.float32)
    binned = np.zeros(values.shape, dtype=np.int64)
    nonzero = values > 0
    if not nonzero.any():
        return binned

    nonzero_values = values[nonzero]
    bins = np.quantile(nonzero_values, np.linspace(0, 1, bin_num - 1))
    digits = _digitize(nonzero_values, bins, rng)
    binned[nonzero] = np.clip(digits, 1, bin_num - 1)
    return binned


def hybrid_sample_gene_indices(
    values: np.ndarray,
    *,
    selected_gene_count: int,
    expressed_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0.0 <= expressed_fraction <= 1.0:
        raise ValueError("expressed_fraction must be between 0 and 1.")

    expressed_cols = np.flatnonzero(values > 0)
    zero_cols = np.flatnonzero(values == 0)

    requested_expressed = int(round(selected_gene_count * expressed_fraction))
    requested_zero = selected_gene_count - requested_expressed
    sampled_expressed = min(requested_expressed, expressed_cols.shape[0])
    sampled_zero = min(requested_zero, zero_cols.shape[0])

    expressed_sample = (
        rng.choice(expressed_cols, size=sampled_expressed, replace=False)
        if sampled_expressed > 0
        else np.empty(0, dtype=np.int64)
    )
    zero_sample = (
        rng.choice(zero_cols, size=sampled_zero, replace=False)
        if sampled_zero > 0
        else np.empty(0, dtype=np.int64)
    )

    selected = np.concatenate((expressed_sample, zero_sample)).astype(np.int64, copy=False)
    if selected.shape[0] < selected_gene_count:
        selected_set = set(selected.tolist())
        remaining = np.asarray(
            [idx for idx in range(values.shape[0]) if idx not in selected_set],
            dtype=np.int64,
        )
        fill = rng.choice(
            remaining,
            size=selected_gene_count - selected.shape[0],
            replace=False,
        )
        selected = np.concatenate((selected, fill)).astype(np.int64, copy=False)

    rng.shuffle(selected)
    return selected


class RawExpressionDataset(Dataset):
    def __init__(
        self,
        *,
        data_path: str,
        indices: np.ndarray,
        gene_num: int,
        selected_gene_count: int,
        bin_num: int,
        hybrid_gene_sampling: bool,
        expressed_gene_fraction: float,
        seed: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.gene_num = int(gene_num)
        self.selected_gene_count = int(selected_gene_count)
        self.bin_num = int(bin_num)
        self.hybrid_gene_sampling = bool(hybrid_gene_sampling)
        self.expressed_gene_fraction = float(expressed_gene_fraction)
        self.seed = int(seed)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.epoch = 0
        self._adata = None

        if self.selected_gene_count <= 0:
            raise ValueError("selected_gene_count must be positive.")
        if self.selected_gene_count > self.gene_num:
            raise ValueError(
                f"Cannot sample {self.selected_gene_count} genes from only {self.gene_num} genes."
            )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_adata"] = None
        return state

    def __del__(self):
        adata = getattr(self, "_adata", None)
        if adata is not None:
            file_obj = getattr(adata, "file", None)
            if file_obj is not None:
                file_obj.close()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ensure_open(self):
        if self._adata is None:
            self._adata = ad.read_h5ad(self.data_path, backed="r")
        return self._adata

    def _read_row(self, source_index: int) -> np.ndarray:
        adata = self._ensure_open()
        row = adata.X[source_index]
        if sparse.issparse(row):
            return np.asarray(row.toarray()).ravel()
        return np.asarray(row).ravel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = int(self.indices[index])
        row = self._read_row(source_index)

        # Deterministic per epoch/sample while still changing the selected gene
        # subset across epochs.
        rng_seed = self.seed + self.epoch * 1_000_003 + source_index
        rng = np.random.default_rng(rng_seed)
        if self.hybrid_gene_sampling:
            selected_cols = hybrid_sample_gene_indices(
                row,
                selected_gene_count=self.selected_gene_count,
                expressed_fraction=self.expressed_gene_fraction,
                rng=rng,
            )
        else:
            selected_cols = rng.choice(
                self.gene_num,
                size=self.selected_gene_count,
                replace=False,
            )

        raw_values = row[selected_cols].astype(np.float32, copy=False)
        binned_values = quantile_bin_expression(raw_values, self.bin_num, rng).astype(np.float32)
        gene_ids = selected_cols.astype(np.int64, copy=False) + self.gene_token_offset

        gene_ids = np.concatenate((
            np.asarray([self.cls_gene_id], dtype=np.int64),
            gene_ids,
        ))
        values = np.concatenate((
            np.asarray([self.cls_value], dtype=np.float32),
            binned_values,
        ))

        return {
            "gene_ids": torch.from_numpy(gene_ids).long(),
            "expr": torch.from_numpy(values).float(),
        }


class PreTrainRunner:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.pretrain_cfg = cfg.pretrain

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        self.bin_num = int(self.pretrain_cfg.bin_num)
        self.value_bin_count = self.bin_num
        self.cls_loss_weight = float(getattr(self.pretrain_cfg, "cls_loss_weight", 1.0))
        self.gene_num = int(self.pretrain_cfg.gene_num)
        self.selected_gene_count = int(getattr(self.pretrain_cfg, "selected_gene_count", 1199))
        self.max_seq_len = int(getattr(self.pretrain_cfg, "max_seq_len", self.selected_gene_count + 1))
        if self.max_seq_len != self.selected_gene_count + 1:
            raise ValueError("max_seq_len must equal selected_gene_count + 1 for the <cls> token.")

        self.cls_gene_id = 0
        self.pad_gene_id = 1
        self.gene_token_offset = 2
        self.num_gene_tokens = self.gene_num + self.gene_token_offset
        self.pad_value = float(getattr(self.pretrain_cfg, "pad_value", -2.0))
        self.mask_value = float(getattr(self.pretrain_cfg, "mask_value", -1.0))
        self.label_ignore_id = -100
        self.mask_ignore_values = {
            int(value) for value in getattr(self.pretrain_cfg, "mask_ignore_values", [])
        }

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.val_dataset_size = 0
        self.model = None
        self.expr_decoder = None
        self.cls_decoder = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.resume_checkpoint_data = None
        self.loaded_optimizer_state = False

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

        seed_all(int(self.pretrain_cfg.seed) + self.rank)

    def _resolve_gene_list_path(self) -> Path:
        gene_list_path = getattr(self.pretrain_cfg, "gene_list_path", None)
        if gene_list_path:
            return Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        return ROOT / "data" / "gene_list.txt"

    def _load_data(self) -> tuple[np.ndarray, np.ndarray | None, str]:
        if bool(getattr(self.pretrain_cfg, "preprocess", False)):
            raise ValueError(
                "CancerFoundation pretraining expects the generated RAW h5ad files. "
                "Set pretrain.preprocess=false."
            )

        data_path = hydra.utils.to_absolute_path(str(self.pretrain_cfg.data_path))
        log.info("Loading pretraining metadata from %s", data_path)
        backed = ad.read_h5ad(data_path, backed="r")
        try:
            n_obs, n_vars = map(int, backed.shape)
            var_names = backed.var_names.astype(str).tolist()
        finally:
            backed.file.close()

        if n_vars != self.gene_num:
            raise ValueError(
                f"Expected {self.gene_num} genes for the scbFM CancerFoundation backbone, got {n_vars}."
            )

        gene_list = read_gene_list(self._resolve_gene_list_path())
        if len(gene_list) != self.gene_num:
            raise ValueError(
                f"pretrain.gene_num={self.gene_num} but gene_list has {len(gene_list)} genes."
            )
        if var_names != gene_list:
            first_mismatch = next(
                (
                    i
                    for i, (observed, expected) in enumerate(zip(var_names, gene_list))
                    if observed != expected
                ),
                None,
            )
            detail = (
                f"first mismatch at position {first_mismatch}: "
                f"observed={var_names[first_mismatch]!r}, expected={gene_list[first_mismatch]!r}"
                if first_mismatch is not None
                else "gene order mismatch"
            )
            raise ValueError(f"{data_path} is not aligned to gene_list.txt ({detail}).")

        indices = np.arange(n_obs, dtype=np.int64)
        val_fraction = float(self.pretrain_cfg.validation_split)
        if n_obs < 2 or val_fraction <= 0:
            return indices, None, data_path

        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=int(self.pretrain_cfg.seed),
        )
        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            data_path,
        )

    def _new_dataset(self, data_path: str, indices: np.ndarray) -> RawExpressionDataset:
        return RawExpressionDataset(
            data_path=data_path,
            indices=indices,
            gene_num=self.gene_num,
            selected_gene_count=self.selected_gene_count,
            bin_num=self.bin_num,
            hybrid_gene_sampling=bool(getattr(self.pretrain_cfg, "hybrid_gene_sampling", True)),
            expressed_gene_fraction=float(getattr(self.pretrain_cfg, "expressed_gene_fraction", 0.5)),
            seed=int(self.pretrain_cfg.seed) + self.rank,
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=self.pad_value,
        )

    def _build_loaders(self, train_indices: np.ndarray, val_indices: np.ndarray | None, data_path: str) -> None:
        batch_size = int(self.pretrain_cfg.batch_size)
        self.train_dataset = self._new_dataset(data_path, train_indices)

        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=self.device.type == "cuda",
            )

        if val_indices is None:
            self.val_loader = None
            self.val_dataset_size = 0
            return

        self.val_dataset = self._new_dataset(data_path, val_indices)
        self.val_dataset_size = len(self.val_dataset)

        if self.is_distributed:
            val_sampler = SequentialDistributedSampler(
                self.val_dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
            )
        else:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
            )

    def _build_model(self) -> None:
        model = CancerFoundationBackbone(
            num_gene_tokens=self.num_gene_tokens,
            d_model=int(self.pretrain_cfg.embsize),
            nhead=int(self.pretrain_cfg.nheads),
            d_hid=int(self.pretrain_cfg.d_hid),
            nlayers=int(self.pretrain_cfg.nlayers),
            dropout=float(self.pretrain_cfg.dropout),
            pad_gene_id=self.pad_gene_id,
            max_value=int(getattr(self.pretrain_cfg, "value_encoder_max_value", 512)),
        ).to(self.device)
        self.expr_decoder = ExpressionBinDecoder(
            d_model=int(self.pretrain_cfg.embsize),
        ).to(self.device)
        self.cls_decoder = ExpressionClsDecoder(
            d_model=int(self.pretrain_cfg.embsize),
        ).to(self.device)

        if self.pretrain_cfg.resume_checkpoint:
            checkpoint_path = hydra.utils.to_absolute_path(str(self.pretrain_cfg.resume_checkpoint))
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.resume_checkpoint_data = checkpoint
            model.load_state_dict(checkpoint["model_state_dict"])
            self.expr_decoder.load_state_dict(checkpoint["expr_decoder_state_dict"])
            if "cls_decoder_state_dict" in checkpoint:
                self.cls_decoder.load_state_dict(checkpoint["cls_decoder_state_dict"])
            elif self.cls_loss_weight > 0:
                log.warning(
                    "Checkpoint %s has no cls_decoder_state_dict; initializing CLS decoder from scratch.",
                    checkpoint_path,
                )
            log.info("Loaded checkpoint from %s", checkpoint_path)

        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
                self.expr_decoder = DDP(
                    self.expr_decoder,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
                self.cls_decoder = DDP(
                    self.cls_decoder,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
            else:
                model = DDP(model)
                self.expr_decoder = DDP(self.expr_decoder)
                self.cls_decoder = DDP(self.cls_decoder)

        self.model = model

    def _build_optimization(self) -> None:
        learning_rate = float(self.pretrain_cfg.learning_rate)
        self.loss_fn = None
        all_params = list(self.model.parameters()) + list(self.expr_decoder.parameters())
        if self.cls_loss_weight > 0:
            all_params += list(self.cls_decoder.parameters())
        self.optimizer = Adam(all_params, lr=learning_rate)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(self.pretrain_cfg.first_cycle_steps),
            cycle_mult=float(self.pretrain_cfg.cycle_mult),
            max_lr=learning_rate,
            min_lr=float(self.pretrain_cfg.min_lr),
            warmup_steps=int(self.pretrain_cfg.warmup_steps),
            gamma=float(self.pretrain_cfg.gamma),
        )
        if (
            self.resume_checkpoint_data is not None
            and bool(getattr(self.pretrain_cfg, "resume_optimizer_state", True))
        ):
            checkpoint = self.resume_checkpoint_data
            missing_keys = [
                key
                for key in ("optimizer_state_dict", "scheduler_state_dict")
                if key not in checkpoint
            ]
            if missing_keys:
                log.warning(
                    "Checkpoint has no %s; starting optimizer/scheduler from config.",
                    ", ".join(missing_keys),
                )
                return

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.loaded_optimizer_state = True
            log.info(
                "Loaded optimizer and scheduler state from checkpoint; current learning rates: %s",
                ", ".join(f"{lr:.6g}" for lr in self._current_learning_rates()),
            )

    def _current_learning_rates(self) -> list[float]:
        if self.optimizer is None:
            return []
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _mask_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gene_ids = batch["gene_ids"].to(self.device, non_blocking=True)
        expr = batch["expr"].to(self.device, non_blocking=True)

        probability = torch.full_like(expr, float(self.pretrain_cfg.mask_prob))
        probability[:, 0] = 0.0  # never mask <cls>
        probability[gene_ids.eq(self.pad_gene_id)] = 0.0
        for value in self.mask_ignore_values:
            probability[expr.eq(float(value))] = 0.0

        mask = torch.bernoulli(probability).bool()
        labels = torch.full(
            expr.shape,
            self.label_ignore_id,
            dtype=torch.float,
            device=self.device,
        )
        labels[mask] = expr[mask]

        masked_expr = expr.clone()
        masked_expr[mask] = self.mask_value
        padding_mask = gene_ids.eq(self.pad_gene_id)
        attention_key_padding_mask = padding_mask.clone()
        if bool(getattr(self.pretrain_cfg, "exclude_masked_from_attention", True)):
            # Masked genes are not usable context: visible tokens cannot attend
            # to them, while masked query positions can still attend to visible keys.
            attention_key_padding_mask |= mask
        return gene_ids, masked_expr, labels, attention_key_padding_mask

    def _forward_batch(
        self,
        gene_ids: torch.Tensor,
        masked_expr: torch.Tensor,
        attention_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        model_output = self.model(
            gene_ids,
            masked_expr,
            src_key_padding_mask=attention_key_padding_mask,
            return_gene_embeddings=self.cls_loss_weight > 0,
        )
        if self.cls_loss_weight > 0:
            hidden, gene_embeddings = model_output
        else:
            hidden = model_output
            gene_embeddings = None

        predicted_values = self.expr_decoder(hidden)
        cls_predicted_values = None
        if self.cls_loss_weight > 0:
            cls_predicted_values = self.cls_decoder(hidden[:, 0, :], gene_embeddings)

        predictions = predicted_values.round().clamp(0, self.value_bin_count - 1).long()
        return predicted_values, cls_predicted_values, predictions

    def _compute_losses(
        self,
        predicted_values: torch.Tensor,
        cls_predicted_values: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        masked_positions = labels != float(self.label_ignore_id)
        if not masked_positions.any():
            zero_loss = predicted_values.sum() * 0.0
            if cls_predicted_values is not None:
                zero_loss = zero_loss + cls_predicted_values.sum() * 0.0
            return {
                "total": zero_loss,
                "gene": zero_loss,
                "cls": zero_loss,
            }
        mask = masked_positions.float()
        gene_loss = nn.functional.mse_loss(
            predicted_values * mask,
            labels * mask,
            reduction="sum",
        )
        gene_loss = gene_loss / mask.sum()

        if self.cls_loss_weight > 0:
            if cls_predicted_values is None:
                raise RuntimeError("CLS loss is enabled but CLS predictions were not computed.")
            cls_loss = nn.functional.mse_loss(
                cls_predicted_values * mask,
                labels * mask,
                reduction="sum",
            )
            cls_loss = cls_loss / mask.sum()
        else:
            cls_loss = gene_loss.detach() * 0.0

        total_loss = gene_loss + self.cls_loss_weight * cls_loss
        return {
            "total": total_loss,
            "gene": gene_loss,
            "cls": cls_loss,
        }

    def _output_dir(self) -> Path:
        return ROOT / "output" / self._model_name()

    def _model_name(self) -> str:
        return str(self.pretrain_cfg.model_name)

    def _output_prefix(self) -> str:
        return self._model_name()

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        preferred = [
            "epoch",
            "split",
            "loss",
            "accuracy",
            "token_id",
            "target_count",
            "target_fraction",
            "correct_count",
            "predicted_count",
            "precision",
            "recall",
            "f1",
        ]
        fieldnames = [field for field in preferred if any(field in row for row in rows)]
        extra_fields = sorted(
            {
                field
                for row in rows
                for field in row
                if field not in fieldnames
            }
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
            writer.writeheader()
            writer.writerows(rows)

    def _new_mask_stats(self) -> dict[str, torch.Tensor]:
        return {
            "target_counts": torch.zeros(
                self.value_bin_count,
                dtype=torch.float64,
                device=self.device,
            ),
            "predicted_counts": torch.zeros(
                self.value_bin_count,
                dtype=torch.float64,
                device=self.device,
            ),
            "correct_counts": torch.zeros(
                self.value_bin_count,
                dtype=torch.float64,
                device=self.device,
            ),
            "sample_count": torch.zeros((), dtype=torch.float64, device=self.device),
        }

    def _update_mask_stats(
        self,
        stats: dict[str, torch.Tensor],
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        valid_mask = labels != self.label_ignore_id
        valid_labels = labels[valid_mask].long()
        valid_predictions = predictions[valid_mask]

        stats["sample_count"] += labels.shape[0]
        if valid_labels.numel() == 0:
            return

        stats["target_counts"] += torch.bincount(
            valid_labels,
            minlength=self.value_bin_count,
        )[: self.value_bin_count].to(torch.float64)
        stats["predicted_counts"] += torch.bincount(
            valid_predictions,
            minlength=self.value_bin_count,
        )[: self.value_bin_count].to(torch.float64)
        correct_labels = valid_labels[valid_predictions == valid_labels]
        if correct_labels.numel() > 0:
            stats["correct_counts"] += torch.bincount(
                correct_labels,
                minlength=self.value_bin_count,
            )[: self.value_bin_count].to(torch.float64)

    def _reduce_mask_stats(self, stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.is_distributed:
            return stats
        reduced = {}
        for key, value in stats.items():
            reduced_value = value.clone()
            dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)
            reduced[key] = reduced_value
        return reduced

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else float("nan")

    def _format_mask_stats(
        self,
        *,
        epoch: int,
        split: str,
        loss: float,
        stats: dict[str, torch.Tensor],
        reduce_stats: bool = True,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        if reduce_stats:
            stats = self._reduce_mask_stats(stats)
        target_counts = stats["target_counts"].detach().cpu().numpy()
        predicted_counts = stats["predicted_counts"].detach().cpu().numpy()
        correct_counts = stats["correct_counts"].detach().cpu().numpy()
        sample_count = float(stats["sample_count"].detach().cpu().item())

        total_targets = float(target_counts.sum())
        total_correct = float(correct_counts.sum())
        zero_targets = float(target_counts[0]) if target_counts.size > 0 else 0.0
        zero_correct = float(correct_counts[0]) if correct_counts.size > 0 else 0.0
        nonzero_targets = total_targets - zero_targets
        nonzero_correct = total_correct - zero_correct

        precision = np.divide(
            correct_counts,
            predicted_counts,
            out=np.full_like(correct_counts, np.nan, dtype=float),
            where=predicted_counts > 0,
        )
        recall = np.divide(
            correct_counts,
            target_counts,
            out=np.full_like(correct_counts, np.nan, dtype=float),
            where=target_counts > 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.full_like(correct_counts, np.nan, dtype=float),
            where=(precision + recall) > 0,
        )
        supported_bins = target_counts > 0

        if total_targets > 0:
            majority_token_id = int(np.argmax(target_counts))
            majority_count = float(target_counts[majority_token_id])
            majority_fraction = majority_count / total_targets
            target_distribution = target_counts / total_targets
            nonzero_distribution = target_distribution[target_distribution > 0]
            target_entropy = float(-np.sum(nonzero_distribution * np.log(nonzero_distribution)))
        else:
            majority_token_id = -1
            majority_fraction = float("nan")
            target_entropy = float("nan")

        summary_row: dict[str, object] = {
            "epoch": epoch,
            "split": split,
            "loss": float(loss),
            "accuracy": 100.0 * self._safe_divide(total_correct, total_targets),
            "macro_f1": float(np.nanmean(f1[supported_bins])) if np.any(supported_bins) else float("nan"),
            "weighted_f1": self._safe_divide(
                float(np.nansum(np.nan_to_num(f1) * target_counts)),
                total_targets,
            ),
            "masked_token_count": int(total_targets),
            "avg_masked_tokens_per_sample": self._safe_divide(total_targets, sample_count),
            "zero_target_count": int(zero_targets),
            "zero_target_fraction": self._safe_divide(zero_targets, total_targets),
            "zero_accuracy": 100.0 * self._safe_divide(zero_correct, zero_targets),
            "nonzero_target_count": int(nonzero_targets),
            "nonzero_target_fraction": self._safe_divide(nonzero_targets, total_targets),
            "nonzero_accuracy": 100.0 * self._safe_divide(nonzero_correct, nonzero_targets),
            "majority_token_id": majority_token_id,
            "majority_token_fraction": majority_fraction,
            "majority_baseline_accuracy": 100.0 * majority_fraction,
            "target_entropy": target_entropy,
            "mask_prob": float(self.pretrain_cfg.mask_prob),
            "mask_ignore_values": ";".join(map(str, sorted(self.mask_ignore_values))),
            "exclude_masked_from_attention": bool(
                getattr(self.pretrain_cfg, "exclude_masked_from_attention", True)
            ),
            "selected_gene_count": self.selected_gene_count,
            "hybrid_gene_sampling": bool(getattr(self.pretrain_cfg, "hybrid_gene_sampling", True)),
            "expressed_gene_fraction": float(getattr(self.pretrain_cfg, "expressed_gene_fraction", 0.5)),
            "bin_num": self.bin_num,
            "loss_type": "mse",
            "cls_loss_weight": self.cls_loss_weight,
            "loaded_optimizer_state": self.loaded_optimizer_state,
            "learning_rates": ";".join(f"{lr:.6g}" for lr in self._current_learning_rates()),
        }

        bin_rows = []
        for token_id in range(self.value_bin_count):
            target_count = float(target_counts[token_id])
            bin_rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "token_id": token_id,
                    "target_count": int(target_count),
                    "target_fraction": self._safe_divide(target_count, total_targets),
                    "correct_count": int(correct_counts[token_id]),
                    "predicted_count": int(predicted_counts[token_id]),
                    "precision": float(precision[token_id]),
                    "recall": float(recall[token_id]),
                    "f1": float(f1[token_id]),
                    "ignored_for_masking": int(token_id in self.mask_ignore_values),
                }
            )
        return summary_row, bin_rows

    def _write_diagnostics(
        self,
        epoch_rows: list[dict[str, object]],
        bin_rows: list[dict[str, object]],
    ) -> None:
        if not self.is_master:
            return
        out_dir = self._output_dir()
        prefix = self._output_prefix()
        self._write_csv(out_dir / f"{prefix}_pretrain_epoch_metrics.csv", epoch_rows)
        self._write_csv(out_dir / f"{prefix}_pretrain_bin_metrics.csv", bin_rows)

    def _save_checkpoint(self, epoch: int, train_loss: float) -> Path | None:
        if not self.is_master:
            return None

        model_name = self._model_name()
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"{model_name}.pth"

        model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_decoder = self.expr_decoder.module if isinstance(self.expr_decoder, DDP) else self.expr_decoder
        raw_cls_decoder = self.cls_decoder.module if isinstance(self.cls_decoder, DDP) else self.cls_decoder
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "expr_decoder_state_dict": raw_decoder.state_dict(),
            "cls_decoder_state_dict": raw_cls_decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "losses": train_loss,
            "backbone": "cancerfoundation",
            "gene_num": self.gene_num,
            "selected_gene_count": self.selected_gene_count,
            "max_seq_len": self.max_seq_len,
            "bin_num": self.bin_num,
            "cls_loss_weight": self.cls_loss_weight,
            "loaded_optimizer_state": self.loaded_optimizer_state,
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _train_one_epoch(self, epoch: int) -> tuple[dict, dict[str, object], list[dict[str, object]]]:
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        self.model.train()
        self.expr_decoder.train()
        self.cls_decoder.train()
        grad_acc_steps = int(self.pretrain_cfg.grad_acc)
        max_grad_norm = float(self.pretrain_cfg.max_grad_norm)

        running_loss = 0.0
        running_gene_loss = 0.0
        running_cls_loss = 0.0
        num_batches = 0
        mask_stats = self._new_mask_stats()
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(self.train_loader, start=1):
            gene_ids, masked_expr, labels, attention_key_padding_mask = self._mask_batch(batch)

            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and step_idx % grad_acc_steps != 0
            )

            if use_no_sync:
                from contextlib import ExitStack
                with ExitStack() as stack:
                    stack.enter_context(self.model.no_sync())
                    if isinstance(self.expr_decoder, DDP):
                        stack.enter_context(self.expr_decoder.no_sync())
                    if isinstance(self.cls_decoder, DDP):
                        stack.enter_context(self.cls_decoder.no_sync())
                    predicted_values, cls_predicted_values, predictions = self._forward_batch(
                        gene_ids,
                        masked_expr,
                        attention_key_padding_mask,
                    )
                    loss_dict = self._compute_losses(predicted_values, cls_predicted_values, labels)
                    loss = loss_dict["total"]
                    (loss / grad_acc_steps).backward()
            else:
                predicted_values, cls_predicted_values, predictions = self._forward_batch(
                    gene_ids,
                    masked_expr,
                    attention_key_padding_mask,
                )
                loss_dict = self._compute_losses(predicted_values, cls_predicted_values, labels)
                loss = loss_dict["total"]
                (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                all_params = list(self.model.parameters()) + list(self.expr_decoder.parameters())
                if self.cls_loss_weight > 0:
                    all_params += list(self.cls_decoder.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                self._update_mask_stats(mask_stats, labels, predictions)

            running_loss += loss.item()
            running_gene_loss += loss_dict["gene"].item()
            running_cls_loss += loss_dict["cls"].item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_gene_loss = running_gene_loss / max(num_batches, 1)
        epoch_cls_loss = running_cls_loss / max(num_batches, 1)

        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.device, 0, self.world_size)
            epoch_gene_loss = get_reduced(epoch_gene_loss, self.device, 0, self.world_size)
            epoch_cls_loss = get_reduced(epoch_cls_loss, self.device, 0, self.world_size)

        self.scheduler.step()
        summary_row, bin_rows = self._format_mask_stats(
            epoch=epoch,
            split="train",
            loss=epoch_loss,
            stats=mask_stats,
        )
        summary_row["gene_loss"] = epoch_gene_loss
        summary_row["cls_loss"] = epoch_cls_loss
        return (
            {
                "train_loss": epoch_loss,
                "train_gene_loss": epoch_gene_loss,
                "train_cls_loss": epoch_cls_loss,
                "train_accuracy": summary_row["accuracy"],
                "train_masked_token_count": summary_row["masked_token_count"],
                "train_nonzero_accuracy": summary_row["nonzero_accuracy"],
                "train_zero_target_fraction": summary_row["zero_target_fraction"],
                "train_majority_baseline_accuracy": summary_row["majority_baseline_accuracy"],
            },
            summary_row,
            bin_rows,
        )

    def _validate(self, epoch: int) -> tuple[dict, dict[str, object], list[dict[str, object]]] | None:
        if self.val_loader is None:
            return None
        if self.val_dataset is not None:
            self.val_dataset.set_epoch(0)

        self.model.eval()
        self.expr_decoder.eval()
        self.cls_decoder.eval()
        if self.is_distributed:
            dist.barrier()

        running_loss = 0.0
        running_gene_loss = 0.0
        running_cls_loss = 0.0
        predictions = []
        truths = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                gene_ids, masked_expr, labels, attention_key_padding_mask = self._mask_batch(batch)
                predicted_values, cls_predicted_values, prediction = self._forward_batch(
                    gene_ids,
                    masked_expr,
                    attention_key_padding_mask,
                )
                loss_dict = self._compute_losses(predicted_values, cls_predicted_values, labels)
                loss = loss_dict["total"]

                running_loss += loss.item()
                running_gene_loss += loss_dict["gene"].item()
                running_cls_loss += loss_dict["cls"].item()
                predictions.append(prediction)
                truths.append(labels)
                num_batches += 1

        prediction_tensor = torch.cat(predictions, dim=0)
        truth_tensor = torch.cat(truths, dim=0)
        val_loss = running_loss / max(num_batches, 1)
        val_gene_loss = running_gene_loss / max(num_batches, 1)
        val_cls_loss = running_cls_loss / max(num_batches, 1)

        if self.is_distributed:
            prediction_tensor = distributed_concat(
                prediction_tensor,
                self.val_dataset_size,
                self.world_size,
            )
            truth_tensor = distributed_concat(
                truth_tensor,
                self.val_dataset_size,
                self.world_size,
            )
            val_loss = get_reduced(val_loss, self.device, 0, self.world_size)
            val_gene_loss = get_reduced(val_gene_loss, self.device, 0, self.world_size)
            val_cls_loss = get_reduced(val_cls_loss, self.device, 0, self.world_size)

        mask_stats = self._new_mask_stats()
        self._update_mask_stats(mask_stats, truth_tensor, prediction_tensor)
        summary_row, bin_rows = self._format_mask_stats(
            epoch=epoch,
            split="val",
            loss=val_loss,
            stats=mask_stats,
            reduce_stats=False,
        )
        summary_row["gene_loss"] = val_gene_loss
        summary_row["cls_loss"] = val_cls_loss
        return (
            {
                "val_loss": val_loss,
                "val_gene_loss": val_gene_loss,
                "val_cls_loss": val_cls_loss,
                "val_accuracy": summary_row["accuracy"],
                "val_masked_token_count": summary_row["masked_token_count"],
                "val_nonzero_accuracy": summary_row["nonzero_accuracy"],
                "val_zero_target_fraction": summary_row["zero_target_fraction"],
                "val_majority_baseline_accuracy": summary_row["majority_baseline_accuracy"],
            },
            summary_row,
            bin_rows,
        )

    def run(self) -> dict:
        self._setup_runtime()
        train_indices, val_indices, data_path = self._load_data()
        self._build_loaders(train_indices, val_indices, data_path)
        self._build_model()
        self._build_optimization()

        epochs = int(self.pretrain_cfg.epochs)
        validate_every = int(self.pretrain_cfg.valid_every)
        history = []
        epoch_metric_rows = []
        bin_metric_rows = []
        final_checkpoint = None
        last_train_loss = float("nan")

        if self.is_master:
            if self.pretrain_cfg.resume_checkpoint:
                log.info(
                    "Starting CancerFoundation-style pre-adaptation of %s from checkpoint %s "
                    "for up to %d epochs on device %s",
                    self._model_name(),
                    self.pretrain_cfg.resume_checkpoint,
                    epochs,
                    self.device,
                )
            else:
                log.info(
                    "Starting CancerFoundation-style pretraining of %s for %d epochs on device %s",
                    self._model_name(),
                    epochs,
                    self.device,
                )
            log.info(
                "Input: %s | gene_num=%d | selected genes/sample=%d | max_seq_len=%d | bins=%d",
                data_path,
                self.gene_num,
                self.selected_gene_count,
                self.max_seq_len,
                self.bin_num,
            )
            log.info(
                "Gene sampling: hybrid=%s | expressed_gene_fraction=%.3f | "
                "exclude_masked_from_attention=%s | cls_loss_weight=%.3f",
                bool(getattr(self.pretrain_cfg, "hybrid_gene_sampling", True)),
                float(getattr(self.pretrain_cfg, "expressed_gene_fraction", 0.5)),
                bool(getattr(self.pretrain_cfg, "exclude_masked_from_attention", True)),
                self.cls_loss_weight,
            )
            log.info(
                "Optimizer: resume_optimizer_state=%s | loaded_optimizer_state=%s | learning_rates=%s",
                bool(getattr(self.pretrain_cfg, "resume_optimizer_state", True)),
                self.loaded_optimizer_state,
                ", ".join(f"{lr:.6g}" for lr in self._current_learning_rates()),
            )

        try:
            for epoch in range(1, epochs + 1):
                train_metrics, train_epoch_row, train_bin_rows = self._train_one_epoch(epoch)
                last_train_loss = train_metrics["train_loss"]

                if self.is_master:
                    log.info(
                        (
                            "Epoch %d | Training Loss: %.6f | Gene Loss: %.6f | CLS Loss: %.6f | "
                            "Accuracy: %.4f%% | "
                            "Nonzero Accuracy: %.4f%% | Majority Baseline: %.4f%% | "
                            "Zero Target Fraction: %.6f"
                        ),
                        epoch,
                        train_metrics["train_loss"],
                        train_metrics["train_gene_loss"],
                        train_metrics["train_cls_loss"],
                        train_metrics["train_accuracy"],
                        train_metrics["train_nonzero_accuracy"],
                        train_metrics["train_majority_baseline_accuracy"],
                        train_metrics["train_zero_target_fraction"],
                    )
                    epoch_metric_rows.append(train_epoch_row)
                    bin_metric_rows.extend(train_bin_rows)

                val_metrics = None
                if validate_every > 0 and epoch % validate_every == 0:
                    val_result = self._validate(epoch)
                    if val_result is not None:
                        val_metrics, val_epoch_row, val_bin_rows = val_result
                    if self.is_master and val_result is not None:
                        log.info(
                            (
                                "Epoch %d | Validation Loss: %.6f | Gene Loss: %.6f | CLS Loss: %.6f | "
                                "Accuracy: %.4f%% | "
                                "Nonzero Accuracy: %.4f%% | Majority Baseline: %.4f%% | "
                                "Zero Target Fraction: %.6f"
                            ),
                            epoch,
                            val_metrics["val_loss"],
                            val_metrics["val_gene_loss"],
                            val_metrics["val_cls_loss"],
                            val_metrics["val_accuracy"],
                            val_metrics["val_nonzero_accuracy"],
                            val_metrics["val_majority_baseline_accuracy"],
                            val_metrics["val_zero_target_fraction"],
                        )
                        epoch_metric_rows.append(val_epoch_row)
                        bin_metric_rows.extend(val_bin_rows)

                self._write_diagnostics(epoch_metric_rows, bin_metric_rows)

                history.append(
                    {
                        "epoch": epoch,
                        **train_metrics,
                        **(val_metrics or {}),
                    }
                )

            checkpoint_path = self._save_checkpoint(epochs, last_train_loss)
            if checkpoint_path is not None:
                final_checkpoint = str(checkpoint_path)
                log.info("Saved final checkpoint to %s", checkpoint_path)

            return {
                "history": history,
                "final_checkpoint": final_checkpoint,
            }
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
