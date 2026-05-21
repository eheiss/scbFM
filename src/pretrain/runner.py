from __future__ import annotations

import csv
import logging
import math
import os
from functools import reduce
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

from performer_pytorch import PerformerLM
from preprocess import preprocess_adata_for_tokens, validate_token_matrix
from utils import (
    CosineAnnealingWarmupRestarts,
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def prob_mask_like(tensor, prob):
    return torch.zeros_like(tensor).float().uniform_(0, 1) < prob


def mask_with_tokens(tensor, token_ids):
    init_no_mask = torch.full_like(tensor, False, dtype=torch.bool)
    return reduce(lambda acc, token_id: acc | (tensor == token_id), token_ids, init_no_mask)


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = torch.cat(
        (torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))
    ).reshape(mask.size(0), mask.size(-1)).to(device)
    mask_excess = mask_excess >= (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


def data_mask(
    data,
    mask_prob,
    replace_prob,
    num_tokens,
    random_token_prob,
    mask_token_id,
    pad_token_id,
    mask_ignore_token_ids,
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)
    masked_input = data.clone().detach()

    if random_token_prob > 0:
        random_token_mask = prob_mask_like(data, random_token_prob)
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)
        random_token_mask &= ~random_no_mask
        random_indices = torch.nonzero(random_token_mask, as_tuple=True)
        masked_input[random_indices] = random_tokens[random_indices]

    replace_mask = prob_mask_like(data, replace_prob)
    masked_input = masked_input.masked_fill(mask & replace_mask, mask_token_id)
    labels = data.masked_fill(~mask, pad_token_id)
    return masked_input, labels


class SCDataset(Dataset):
    def __init__(self, data, bin_num, special_token_id, device):
        super().__init__()
        self.data = data
        self.max_token_id = bin_num
        self.special_token_id = special_token_id
        self.device = device

    def __getitem__(self, index):
        row = self.data[index]
        if sparse.issparse(row):
            full_seq = row.toarray().ravel()
        else:
            full_seq = np.asarray(row).ravel()

        full_seq = torch.as_tensor(full_seq, dtype=torch.long)
        full_seq = torch.cat((
            full_seq,
            torch.tensor([self.special_token_id], dtype=torch.long),
        ))
        return full_seq.to(self.device)

    def __len__(self):
        return self.data.shape[0]


class ExprDecoder(nn.Module):
    """Scalar regression head: hidden_dim → 1, matching scGPT's ExprDecoder MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.1),
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.1),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (B, seq_len)


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

        self.class_count = self.pretrain_cfg.bin_num + 2
        self.vocab_size = self.class_count + 1
        self.pad_token_id = self.class_count - 1
        self.mask_token_id = self.class_count - 1
        self.special_token_id = self.class_count
        self.mask_ignore_token_ids = list(dict.fromkeys([
            *self.pretrain_cfg.mask_ignore_token_ids,
            self.special_token_id,
        ]))

        self.train_loader = None
        self.val_loader = None
        self.val_dataset_size = 0
        self.model = None
        self.expr_decoder = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

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

        seed_all(self.pretrain_cfg.seed + self.rank)

    def _resolve_gene_list_path(self) -> Path:
        gene_list_path = getattr(self.pretrain_cfg, "gene_list_path", None)
        if gene_list_path:
            return Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        return ROOT / "data" / "gene_list.txt"

    def _should_preprocess_input(self) -> bool:
        return bool(getattr(self.pretrain_cfg, "preprocess", False))

    def _load_data(self):
        data_path = hydra.utils.to_absolute_path(self.pretrain_cfg.data_path)
        log.info("Loading pretraining data from %s", data_path)
        adata = ad.read_h5ad(data_path)
        if self._should_preprocess_input():
            adata, missing_genes = preprocess_adata_for_tokens(
                adata,
                gene_list_path=self._resolve_gene_list_path(),
                min_genes=int(getattr(self.pretrain_cfg, "min_genes", 200)),
                bin_num=int(self.pretrain_cfg.bin_num),
                reindex_genes=bool(getattr(self.pretrain_cfg, "reindex_genes", True)),
            )
            log.info(
                "Applied shared raw preprocessing: %d target genes missing, output shape %s",
                len(missing_genes),
                adata.shape,
            )

        expected_gene_num = int(self.pretrain_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes for the scbFM backbone, got {adata.n_vars}."
            )
        matrix = adata.X
        validate_token_matrix(matrix, bin_num=int(self.pretrain_cfg.bin_num), name="pretraining data")

        val_fraction = self.pretrain_cfg.validation_split
        if matrix.shape[0] < 2 or val_fraction <= 0:
            return matrix, None

        return train_test_split(
            matrix,
            test_size=val_fraction,
            random_state=self.pretrain_cfg.seed,
        )

    def _build_loaders(self, train_data, val_data) -> None:
        batch_size = self.pretrain_cfg.batch_size
        train_dataset = SCDataset(
            train_data,
            self.pretrain_cfg.bin_num,
            self.special_token_id,
            self.device,
        )

        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,
            )
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is None:
            self.val_loader = None
            self.val_dataset_size = 0
            return

        val_dataset = SCDataset(
            val_data,
            self.pretrain_cfg.bin_num,
            self.special_token_id,
            self.device,
        )
        self.val_dataset_size = len(val_dataset)

        if self.is_distributed:
            val_sampler = SequentialDistributedSampler(
                val_dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
            )
        else:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def _build_model(self) -> None:
        loss_type = str(getattr(self.pretrain_cfg, "loss_type", "mse")).lower()
        model = PerformerLM(
            num_tokens=self.vocab_size,
            max_seq_len=self.pretrain_cfg.gene_num + 1,
            dim=self.pretrain_cfg.dim,
            depth=self.pretrain_cfg.depth,
            heads=self.pretrain_cfg.heads,
            dim_head=self.pretrain_cfg.dim_head,
            ff_mult=self.pretrain_cfg.ff_mult,
            nb_features=self.pretrain_cfg.nb_features,
            feature_redraw_interval=self.pretrain_cfg.feature_redraw_interval,
            ff_chunks=self.pretrain_cfg.ff_chunks,
            ff_glu=self.pretrain_cfg.ff_glu,
            emb_dropout=self.pretrain_cfg.emb_dropout,
            ff_dropout=self.pretrain_cfg.ff_dropout,
            attn_dropout=self.pretrain_cfg.attn_dropout,
            use_scalenorm=self.pretrain_cfg.use_scalenorm,
            use_rezero=self.pretrain_cfg.use_rezero,
            no_projection=self.pretrain_cfg.no_projection,
            tie_embed=self.pretrain_cfg.tie_embed,
            g2v_position_emb=self.pretrain_cfg.g2v_position_emb,
            auto_check_redraw=self.pretrain_cfg.auto_check_redraw,
            qkv_bias=self.pretrain_cfg.qkv_bias,
            embx_bin_num=int(self.pretrain_cfg.bin_num) if loss_type == "mse" else None,
        ).to(self.device)

        if loss_type == "mse":
            self.expr_decoder = ExprDecoder(dim=self.pretrain_cfg.dim).to(self.device)

        if self.pretrain_cfg.resume_checkpoint:
            checkpoint_path = hydra.utils.to_absolute_path(self.pretrain_cfg.resume_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "expr_decoder_state_dict" in checkpoint and self.expr_decoder is not None:
                self.expr_decoder.load_state_dict(checkpoint["expr_decoder_state_dict"])
            log.info("Loaded checkpoint from %s", checkpoint_path)

        if self.is_distributed:
            # MSE mode never calls to_out inside the DDP forward (used only for
            # no-grad accuracy stats), so those parameters appear unused to DDP.
            find_unused = loss_type == "mse"
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=find_unused)
            else:
                model = DDP(model, find_unused_parameters=find_unused)
            if self.expr_decoder is not None:
                if self.device.type == "cuda":
                    self.expr_decoder = DDP(self.expr_decoder, device_ids=[self.local_rank], output_device=self.local_rank)
                else:
                    self.expr_decoder = DDP(self.expr_decoder)

        self.model = model

    def _compute_loss(self, hidden_or_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute masked MLM loss. CE receives logits; MSE receives hidden states."""
        loss_type = str(getattr(self.pretrain_cfg, "loss_type", "mse")).lower()
        if loss_type == "ce":
            return self.loss_fn(hidden_or_logits.transpose(1, 2), labels)

        # MSE: scalar decoder head on hidden states, masked positions only (scGPT style).
        pred = self.expr_decoder(hidden_or_logits)  # (B, seq_len)
        valid = labels != self.pad_token_id
        if not valid.any():
            return pred.sum() * 0.0
        mask = valid.float()
        loss = nn.functional.mse_loss(pred * mask, labels.to(pred.dtype) * mask, reduction="sum")
        return loss / mask.sum()

    def _build_optimization(self) -> None:
        learning_rate = self.pretrain_cfg.learning_rate
        loss_type = str(getattr(self.pretrain_cfg, "loss_type", "mse")).lower()
        if loss_type == "ce":
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=self.pad_token_id,
                reduction="mean",
            ).to(self.device)
            all_params = self.model.parameters()
        else:
            self.loss_fn = None
            all_params = list(self.model.parameters()) + list(self.expr_decoder.parameters())
        self.optimizer = Adam(all_params, lr=learning_rate)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=self.pretrain_cfg.first_cycle_steps,
            cycle_mult=self.pretrain_cfg.cycle_mult,
            max_lr=learning_rate,
            min_lr=self.pretrain_cfg.min_lr,
            warmup_steps=self.pretrain_cfg.warmup_steps,
            gamma=self.pretrain_cfg.gamma,
        )

    def _use_mse(self) -> bool:
        return str(getattr(self.pretrain_cfg, "loss_type", "mse")).lower() == "mse"

    def _forward_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run model forward, returning (loss_input, predictions).

        CE mode: loss_input == logits (vocab logits, B×L×V); predictions = logits (argmax externally).
        MSE mode: loss_input = hidden (B×L×dim); predictions = integer bin IDs from ExprDecoder.
        """
        if not self._use_mse():
            logits = self.model(batch)
            return logits, logits

        hidden = self.model(batch, return_encodings=True)
        raw_decoder = self.expr_decoder.module if isinstance(self.expr_decoder, DDP) else self.expr_decoder
        with torch.no_grad():
            pred_int = (
                raw_decoder(hidden)
                .squeeze(-1)
                .round()
                .clamp(0, self.pretrain_cfg.bin_num)
                .long()
            )
        return hidden, pred_int

    def _mask_batch(self, batch):
        return data_mask(
            batch,
            mask_prob=self.pretrain_cfg.mask_prob,
            replace_prob=self.pretrain_cfg.replace_prob,
            num_tokens=self.vocab_size,
            random_token_prob=self.pretrain_cfg.random_token_prob,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
            mask_ignore_token_ids=self.mask_ignore_token_ids,
        )

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
        expression_token_count = self.pad_token_id
        return {
            "target_counts": torch.zeros(
                expression_token_count,
                dtype=torch.float64,
                device=self.device,
            ),
            "predicted_counts": torch.zeros(
                expression_token_count,
                dtype=torch.float64,
                device=self.device,
            ),
            "correct_counts": torch.zeros(
                expression_token_count,
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
        valid_mask = labels != self.pad_token_id
        valid_labels = labels[valid_mask]
        valid_predictions = predictions[valid_mask]

        stats["sample_count"] += labels.shape[0]
        if valid_labels.numel() == 0:
            return

        stats["target_counts"] += torch.bincount(
            valid_labels,
            minlength=self.pad_token_id,
        )[: self.pad_token_id].to(torch.float64)
        stats["predicted_counts"] += torch.bincount(
            valid_predictions,
            minlength=self.pad_token_id,
        )[: self.pad_token_id].to(torch.float64)
        correct_labels = valid_labels[valid_predictions == valid_labels]
        if correct_labels.numel() > 0:
            stats["correct_counts"] += torch.bincount(
                correct_labels,
                minlength=self.pad_token_id,
            )[: self.pad_token_id].to(torch.float64)

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
            majority_count = 0.0
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
            "replace_prob": float(self.pretrain_cfg.replace_prob),
            "random_token_prob": float(self.pretrain_cfg.random_token_prob),
            "mask_ignore_token_ids": ";".join(map(str, self.mask_ignore_token_ids)),
        }

        bin_rows = []
        ignored_token_ids = set(self.mask_ignore_token_ids)
        for token_id in range(self.pad_token_id):
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
                    "ignored_for_masking": int(token_id in ignored_token_ids),
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
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "losses": train_loss,
        }
        if self.expr_decoder is not None:
            raw_decoder = self.expr_decoder.module if isinstance(self.expr_decoder, DDP) else self.expr_decoder
            checkpoint["expr_decoder_state_dict"] = raw_decoder.state_dict()
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _train_one_epoch(self, epoch: int) -> tuple[dict, dict[str, object], list[dict[str, object]]]:
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        self.model.train()
        if self.expr_decoder is not None:
            self.expr_decoder.train()
        grad_acc_steps = self.pretrain_cfg.grad_acc
        max_grad_norm = self.pretrain_cfg.max_grad_norm

        running_loss = 0.0
        num_batches = 0
        mask_stats = self._new_mask_stats()
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(self.train_loader, start=1):
            batch = batch.to(self.device, non_blocking=True)
            masked_batch, labels = self._mask_batch(batch)

            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and step_idx % grad_acc_steps != 0
            )

            no_sync_contexts = [self.model.no_sync()]
            if self._use_mse() and isinstance(self.expr_decoder, DDP):
                no_sync_contexts.append(self.expr_decoder.no_sync())

            if use_no_sync:
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for ctx in no_sync_contexts:
                        stack.enter_context(ctx)
                    loss_input, logits = self._forward_batch(masked_batch)
                    loss = self._compute_loss(loss_input, labels)
                    (loss / grad_acc_steps).backward()
            else:
                loss_input, logits = self._forward_batch(masked_batch)
                loss = self._compute_loss(loss_input, labels)
                (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                all_params = (
                    list(self.model.parameters()) + list(self.expr_decoder.parameters())
                    if self._use_mse()
                    else self.model.parameters()
                )
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if self._use_mse():
                    predictions = logits  # already integer bin IDs from ExprDecoder
                else:
                    predictions = logits[..., : self.pad_token_id].argmax(dim=-1)
                self._update_mask_stats(mask_stats, labels, predictions)

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)

        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.device, 0, self.world_size)

        self.scheduler.step()
        summary_row, bin_rows = self._format_mask_stats(
            epoch=epoch,
            split="train",
            loss=epoch_loss,
            stats=mask_stats,
        )
        return (
            {
                "train_loss": epoch_loss,
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

        self.model.eval()
        if self.expr_decoder is not None:
            self.expr_decoder.eval()
        if self.is_distributed:
            dist.barrier()

        running_loss = 0.0
        predictions = []
        truths = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device, non_blocking=True)
                masked_batch, labels = self._mask_batch(batch)
                loss_input, logits = self._forward_batch(masked_batch)
                loss = self._compute_loss(loss_input, labels)

                running_loss += loss.item()
                if self._use_mse():
                    predictions.append(logits)  # already integer bin IDs from ExprDecoder
                else:
                    predictions.append(logits[..., : self.pad_token_id].argmax(dim=-1))
                truths.append(labels)
                num_batches += 1

        prediction_tensor = torch.cat(predictions, dim=0)
        truth_tensor = torch.cat(truths, dim=0)
        val_loss = running_loss / max(num_batches, 1)

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

        mask_stats = self._new_mask_stats()
        self._update_mask_stats(mask_stats, truth_tensor, prediction_tensor)
        summary_row, bin_rows = self._format_mask_stats(
            epoch=epoch,
            split="val",
            loss=val_loss,
            stats=mask_stats,
            reduce_stats=False,
        )
        return (
            {
                "val_loss": val_loss,
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
        train_data, val_data = self._load_data()
        self._build_loaders(train_data, val_data)
        self._build_model()
        self._build_optimization()

        epochs = self.pretrain_cfg.epochs
        validate_every = self.pretrain_cfg.valid_every
        history = []
        epoch_metric_rows = []
        bin_metric_rows = []
        final_checkpoint = None
        last_train_loss = float("nan")

        if self.is_master:

            if self.pretrain_cfg.resume_checkpoint:
                log.info(
                    "Starting pre-adaptation of %s from checkpoint %s for up to %d epochs on device %s",
                    self._model_name(),
                    self.pretrain_cfg.resume_checkpoint,
                    epochs,
                    self.device,
                )
            else:   
                log.info(
                    "Starting pretraining of %s for %d epochs on device %s",
                    self._model_name(),
                    epochs,
                    self.device,
                )

        try:
            for epoch in range(1, epochs + 1):
                train_metrics, train_epoch_row, train_bin_rows = self._train_one_epoch(epoch)
                last_train_loss = train_metrics["train_loss"]

                if self.is_master:
                    log.info(
                        (
                            "Epoch %d | Training Loss: %.6f | Accuracy: %.4f%% | "
                            "Nonzero Accuracy: %.4f%% | Majority Baseline: %.4f%% | "
                            "Zero Target Fraction: %.6f"
                        ),
                        epoch,
                        train_metrics["train_loss"],
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
                                "Epoch %d | Validation Loss: %.6f | Accuracy: %.4f%% | "
                                "Nonzero Accuracy: %.4f%% | Majority Baseline: %.4f%% | "
                                "Zero Target Fraction: %.6f"
                            ),
                            epoch,
                            val_metrics["val_loss"],
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
