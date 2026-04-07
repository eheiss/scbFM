from __future__ import annotations

import logging
import math
import os
from functools import reduce
from pathlib import Path

import hydra
import numpy as np
import scanpy as sc
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
from utils import (
    CosineAnnealingWarmupRestarts,
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)


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

        full_seq = np.clip(full_seq, 0, self.max_token_id)
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((
            full_seq,
            torch.tensor([self.special_token_id], dtype=torch.long),
        ))
        return full_seq.to(self.device)

    def __len__(self):
        return self.data.shape[0]


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
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.softmax = nn.Softmax(dim=-1)

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

    def _load_data(self):
        data_path = hydra.utils.to_absolute_path(self.pretrain_cfg.data_path)
        log.info("Loading pretraining data from %s", data_path)
        adata = sc.read_h5ad(data_path)
        matrix = adata.X

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
        ).to(self.device)

        if self.pretrain_cfg.resume_checkpoint:
            checkpoint_path = hydra.utils.to_absolute_path(self.pretrain_cfg.resume_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info("Loaded checkpoint from %s", checkpoint_path)

        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                model = DDP(model)

        self.model = model

    def _build_optimization(self) -> None:
        learning_rate = self.pretrain_cfg.learning_rate
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=self.pretrain_cfg.first_cycle_steps,
            cycle_mult=self.pretrain_cfg.cycle_mult,
            max_lr=learning_rate,
            min_lr=self.pretrain_cfg.min_lr,
            warmup_steps=self.pretrain_cfg.warmup_steps,
            gamma=self.pretrain_cfg.gamma,
        )
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id,
            reduction="mean",
        ).to(self.device)

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

    def _save_checkpoint(self, epoch: int, train_loss: float) -> Path | None:
        if not self.is_master:
            return None

        checkpoint_dir = Path(hydra.utils.to_absolute_path(self.pretrain_cfg.ckpt_dir))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_name = self.pretrain_cfg.model_name
        checkpoint_path = checkpoint_dir / f"{model_name}_{epoch}.pth"

        model = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "losses": train_loss,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _train_one_epoch(self, epoch: int) -> dict:
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        self.model.train()
        grad_acc_steps = self.pretrain_cfg.grad_acc
        max_grad_norm = self.pretrain_cfg.max_grad_norm

        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(self.train_loader, start=1):
            batch = batch.to(self.device, non_blocking=True)
            masked_batch, labels = self._mask_batch(batch)

            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and step_idx % grad_acc_steps != 0
            )

            if use_no_sync:
                with self.model.no_sync():
                    logits = self.model(masked_batch)
                    loss = self.loss_fn(logits.transpose(1, 2), labels)
                    (loss / grad_acc_steps).backward()
            else:
                logits = self.model(masked_batch)
                loss = self.loss_fn(logits.transpose(1, 2), labels)
                (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                predictions = self.softmax(logits)[..., 1:self.pad_token_id].argmax(dim=-1) + 1
                valid_token_counts = (labels != self.pad_token_id).sum(dim=-1)
                correct_token_counts = (
                    (labels != self.pad_token_id) & (predictions == labels)
                ).sum(dim=-1)
                batch_acc = torch.where(
                    valid_token_counts > 0,
                    correct_token_counts.float() / valid_token_counts.float(),
                    torch.zeros_like(valid_token_counts, dtype=torch.float),
                ).mean()

            running_loss += loss.item()
            running_acc += batch_acc.item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_acc = 100.0 * running_acc / max(num_batches, 1)

        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.device, 0, self.world_size)
            epoch_acc = get_reduced(epoch_acc, self.device, 0, self.world_size)

        self.scheduler.step()
        return {"train_loss": epoch_loss, "train_accuracy": epoch_acc}

    def _validate(self) -> dict | None:
        if self.val_loader is None:
            return None

        self.model.eval()
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
                logits = self.model(masked_batch)
                loss = self.loss_fn(logits.transpose(1, 2), labels)

                running_loss += loss.item()
                predictions.append(
                    self.softmax(logits)[..., 1:self.pad_token_id].argmax(dim=-1) + 1
                )
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

        valid_token_count = (truth_tensor != self.pad_token_id).sum().item()
        correct_token_count = (
            (truth_tensor != self.pad_token_id) & (prediction_tensor == truth_tensor)
        ).sum().item()
        val_acc = 100.0 * correct_token_count / max(valid_token_count, 1)

        return {"val_loss": val_loss, "val_accuracy": val_acc}

    def run(self) -> dict:
        self._setup_runtime()
        train_data, val_data = self._load_data()
        self._build_loaders(train_data, val_data)
        self._build_model()
        self._build_optimization()

        epochs = self.pretrain_cfg.epochs
        validate_every = self.pretrain_cfg.valid_every
        history = []
        last_checkpoint = None

        if self.is_master:

            if self.pretrain_cfg.resume_checkpoint:
                log.info(
                    "Starting pre-adaptation of %s from checkpoint %s for up to %d epochs on device %s",
                    self.pretrain_cfg.model_name,
                    self.pretrain_cfg.resume_checkpoint,
                    epochs,
                    self.device,
                )
            else:   
                log.info(
                    "Starting pretraining of %s for %d epochs on device %s",
                    self.pretrain_cfg.model_name,
                    epochs,
                    self.device,
                )

        try:
            for epoch in range(1, epochs + 1):
                train_metrics = self._train_one_epoch(epoch)

                if self.is_master:
                    log.info(
                        "Epoch %d | Training Loss: %.6f | Accuracy: %.4f%%",
                        epoch,
                        train_metrics["train_loss"],
                        train_metrics["train_accuracy"],
                    )

                val_metrics = None
                if validate_every > 0 and epoch % validate_every == 0:
                    val_metrics = self._validate()
                    if self.is_master and val_metrics is not None:
                        log.info(
                            "Epoch %d | Validation Loss: %.6f | Accuracy: %.4f%%",
                            epoch,
                            val_metrics["val_loss"],
                            val_metrics["val_accuracy"],
                        )

                checkpoint_path = self._save_checkpoint(epoch, train_metrics["train_loss"])
                if checkpoint_path is not None:
                    last_checkpoint = str(checkpoint_path)
                    log.info("Saved checkpoint to %s", checkpoint_path)

                history.append(
                    {
                        "epoch": epoch,
                        **train_metrics,
                        **(val_metrics or {}),
                    }
                )

            return {
                "history": history,
                "last_checkpoint": last_checkpoint,
            }
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
