from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from scipy.stats import chi2
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.canc_type_class.runner import (
    _quantile_bin_expression,
    _set_finetune_training_mode,
)
from finetune.training_correctness import (
    add_optimizer_parameter_group,
    is_accumulation_boundary,
    validate_backbone_checkpoint,
)
from preprocess import reindex_adata_genes, validate_token_matrix
from run_provenance import complete_run_metadata, start_run_metadata
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
TASK_NAME = "surv_pred_survboard"
CHECKPOINT_MODEL_KEYS = ("pretrain_sc", "pretrain_bulk", "preadapt_sc", "preadapt_bulk")
RANDOM_INIT_MODEL_KEY = "random_init"
MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)


# ---------------------------------------------------------------------------
# Scheduler (mirrors canc_type_class runner exactly)
# ---------------------------------------------------------------------------

class GroupedCosineAnnealingWarmupRestarts:
    """Cosine warmup scheduler that preserves per-parameter-group max LRs."""

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
                    self.cur_cycle_steps = (
                        self.first_cycle_steps * self.cycle_mult**self.cycle
                    )
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


# ---------------------------------------------------------------------------
# Survival prediction head
# ---------------------------------------------------------------------------

class SurvivalPredHead(nn.Module):
    """Sample embedding → MLP → log-hazard scalar for neural Cox PH."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pooling = "cls"
        input_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SELU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x[:, 0, :]).squeeze(-1)  # (B,)


class CancerFoundationSurvivalModel(nn.Module):
    def __init__(self, backbone: CancerFoundationBackbone, head: SurvivalPredHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.to_out = head

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.backbone(
            batch["gene_ids"],
            batch["expr"],
            src_key_padding_mask=batch.get("attention_key_padding_mask"),
        )
        return self.to_out(hidden)

    def add_adapters(self, **kwargs) -> nn.ModuleList:
        return self.backbone.add_adapters(**kwargs)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return self.backbone.adapter_parameters()

    def enable_grad_checkpoint(self) -> None:
        self.backbone.enable_grad_checkpoint()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurvivalDataset(Dataset):
    def __init__(
        self,
        X,
        times: np.ndarray,
        events: np.ndarray,
        bin_num: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
        selected_gene_count: int,
        seed: int,
        do_binning: bool,
        fixed_gene_indices: np.ndarray,
    ) -> None:
        self.X = X
        self.times = np.asarray(times, dtype=np.float32)
        self.events = np.asarray(events, dtype=np.float32)
        self.bin_num = int(bin_num)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.selected_gene_count = int(selected_gene_count)
        self.seed = int(seed)
        self.do_binning = bool(do_binning)
        self.fixed_gene_indices = np.asarray(fixed_gene_indices, dtype=np.int64)
        if self.fixed_gene_indices.shape != (self.selected_gene_count,):
            raise ValueError(
                "fixed_gene_indices must contain exactly "
                f"{self.selected_gene_count} genes."
            )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        row = self.X[index]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()

        selected_values = values[self.fixed_gene_indices].astype(np.float32, copy=False)
        if self.do_binning:
            rng = np.random.default_rng(self.seed + index)
            selected_values = _quantile_bin_expression(
                selected_values,
                self.bin_num,
                rng,
            ).astype(np.float32)

        gene_ids = torch.from_numpy(
            self.fixed_gene_indices.astype(np.int64, copy=False) + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        expression = torch.from_numpy(selected_values)
        expression = torch.cat((torch.tensor([self.cls_value]), expression))
        return (
            {"gene_ids": gene_ids, "expr": expression},
            torch.tensor(self.times[index], dtype=torch.float32),
            torch.tensor(self.events[index], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Loss, metrics, Breslow estimator
# ---------------------------------------------------------------------------

def cox_partial_log_likelihood(
    log_hazard: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Breslow-approximation negative partial log-likelihood (batch-level Cox loss)."""
    order = torch.argsort(time, descending=True)
    lh = log_hazard[order]
    e = event[order]
    n_events = e.sum().clamp(min=1.0)
    log_cumsum_exp = torch.logcumsumexp(lh, dim=0)
    return -((lh - log_cumsum_exp) * e).sum() / n_events


def gather_cox_update_batch(
    log_hazard: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    *,
    is_distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather one effective Cox batch while preserving hazard gradients."""
    if not is_distributed:
        return log_hazard, time, event

    from torch.distributed.nn.functional import all_gather as differentiable_all_gather

    gathered_hazards = torch.cat(differentiable_all_gather(log_hazard.contiguous()))
    gathered_times = [torch.empty_like(time) for _ in range(dist.get_world_size())]
    gathered_events = [torch.empty_like(event) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_times, time.contiguous())
    dist.all_gather(gathered_events, event.contiguous())
    return (
        gathered_hazards,
        torch.cat(gathered_times),
        torch.cat(gathered_events),
    )


def antolini_concordance(
    survival_probs: np.ndarray,
    time_points: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Antolini's time-dependent concordance (exact match with SurvBoard leaderboard)."""
    from pycox.evaluation import EvalSurv
    surv_df = pd.DataFrame(survival_probs.T, index=time_points.astype(float))
    ev = EvalSurv(surv_df, time, event, censor_surv="km", steps="post")
    return float(ev.concordance_td())


def harrell_c_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Harrell's C-index for risk scores; higher risk should fail earlier."""
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        if not event[i]:
            continue
        mask = time[i] < time
        if not np.any(mask):
            continue
        comparable += float(mask.sum())
        concordant += float(np.sum(risk[i] > risk[mask]))
        concordant += 0.5 * float(np.sum(risk[i] == risk[mask]))
    return float(concordant / comparable) if comparable > 0 else float("nan")


def _km_survival(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier step function values after each unique event time."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    event_times = np.unique(times[events])
    surv = []
    current = 1.0
    for t in event_times:
        at_risk = np.sum(times >= t)
        observed = np.sum((times == t) & events)
        if at_risk > 0:
            current *= 1.0 - observed / at_risk
        surv.append(current)
    return event_times, np.asarray(surv, dtype=float)


def _step_survival_at(
    query_times: np.ndarray,
    step_times: np.ndarray,
    step_survival: np.ndarray,
) -> np.ndarray:
    query_times = np.asarray(query_times, dtype=float)
    if len(step_times) == 0:
        return np.ones_like(query_times, dtype=float)
    idx = np.searchsorted(step_times, query_times, side="right") - 1
    out = np.ones_like(query_times, dtype=float)
    valid = idx >= 0
    out[valid] = step_survival[np.clip(idx[valid], 0, len(step_survival) - 1)]
    return out


def ipcw_weighted_c_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Uno/IPCW-style weighted C-index using train-set censoring KM weights."""
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    # Censoring distribution G(t) = P(C >= t); censoring event is 1 - event.
    censor_times, censor_surv = _km_survival(train_time, ~np.asarray(train_event, dtype=bool))
    g_at_time = np.clip(_step_survival_at(time, censor_times, censor_surv), eps, None)

    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        if not event[i]:
            continue
        mask = time[i] < time
        if not np.any(mask):
            continue
        weight = 1.0 / (g_at_time[i] ** 2)
        comparable += weight * float(mask.sum())
        concordant += weight * float(np.sum(risk[i] > risk[mask]))
        concordant += 0.5 * weight * float(np.sum(risk[i] == risk[mask]))
    return float(concordant / comparable) if comparable > 0 else float("nan")


def _survival_at_observed_times(
    survival_probs: np.ndarray,
    time_points: np.ndarray,
    observed_times: np.ndarray,
) -> np.ndarray:
    """Stepwise survival probability at each observed time."""
    observed_times = np.asarray(observed_times, dtype=float)
    idx = np.searchsorted(time_points.astype(float), observed_times, side="right") - 1
    out = np.ones(observed_times.shape[0], dtype=float)
    valid = idx >= 0
    out[valid] = survival_probs[
        np.arange(observed_times.shape[0])[valid],
        np.clip(idx[valid], 0, survival_probs.shape[1] - 1),
    ]
    return np.clip(out, 0.0, 1.0)


def d_calibration(
    survival_probs: np.ndarray,
    time_points: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """D-calibration chi-square statistic and p-value.

    Uncensored samples contribute their predicted S(T) to one bin. Censored
    samples contribute uniformly over [0, S(C)] plus mass above S(C), following
    the standard D-calibration censored-sample treatment.
    """
    s_obs = _survival_at_observed_times(survival_probs, time_points, time)
    event = np.asarray(event, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)

    for s, is_event in zip(s_obs, event):
        if is_event:
            bin_idx = min(np.searchsorted(edges, s, side="right") - 1, n_bins - 1)
            counts[max(bin_idx, 0)] += 1.0
            continue

        if s <= 0:
            continue
        # Conditional on surviving past censoring, S(T) is uniform on [0, S(C)].
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            overlap = max(0.0, min(hi, s) - lo)
            if overlap > 0:
                counts[b] += overlap / s

    expected = len(time) / n_bins
    if expected <= 0:
        return float("nan"), float("nan")
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(1.0 - chi2.cdf(statistic, df=n_bins - 1))
    return statistic, p_value


def survival_metrics(
    survival_probs: np.ndarray,
    time_points: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_log_hazard: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
) -> dict[str, float]:
    """Compute SurvBoard and BulkRNABert-style survival metrics."""
    from pycox.evaluation import EvalSurv

    surv_df = pd.DataFrame(survival_probs.T, index=time_points.astype(float))
    ev = EvalSurv(surv_df, test_time, test_event, censor_surv="km", steps="post")
    antolini = float(ev.concordance_td())

    brier_grid = time_points[
        (time_points > np.min(test_time)) & (time_points < np.max(test_time))
    ].astype(float)
    if brier_grid.size >= 2:
        try:
            ibs = float(ev.integrated_brier_score(brier_grid))
        except Exception as exc:  # pragma: no cover - depends on pycox internals/data.
            log.warning("Could not compute integrated Brier score: %s", exc)
            ibs = float("nan")
    else:
        ibs = float("nan")

    dcal_stat, dcal_p = d_calibration(survival_probs, time_points, test_time, test_event)
    risk = np.asarray(test_log_hazard, dtype=float)
    return {
        "test_c_index": harrell_c_index(risk, test_time, test_event),
        "test_weighted_c_index": ipcw_weighted_c_index(
            risk,
            test_time,
            test_event,
            train_time,
            train_event,
        ),
        "test_antolini_cindex": antolini,
        "test_integrated_brier_score": ibs,
        "test_d_calibration_chi2": dcal_stat,
        "test_d_calibration_p_value": dcal_p,
    }


class BreslowEstimator:
    """Non-parametric baseline cumulative hazard estimator for Cox PH."""

    def __init__(self) -> None:
        self.event_times: np.ndarray | None = None
        self.baseline_cumhazard: np.ndarray | None = None

    def fit(
        self,
        log_hazard: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> None:
        event = event.astype(bool)
        order = np.argsort(time)
        time_s = time[order]
        event_s = event[order]
        exp_lh = np.exp(log_hazard[order])

        unique_event_times = np.unique(time_s[event_s])
        baseline_hazard = np.zeros(len(unique_event_times))
        for k, t in enumerate(unique_event_times):
            risk_set_exp = exp_lh[time_s >= t]
            n_events_at_t = event_s[time_s == t].sum()
            denom = risk_set_exp.sum()
            baseline_hazard[k] = n_events_at_t / denom if denom > 0 else 0.0

        self.event_times = unique_event_times
        self.baseline_cumhazard = np.cumsum(baseline_hazard)

    def predict_survival(
        self,
        log_hazard: np.ndarray,
        time_points: np.ndarray,
    ) -> np.ndarray:
        """Return S(t|x) = exp(-H0(t) * exp(h(x))), shape (n_samples, n_time_points)."""
        assert self.event_times is not None, "Call fit() before predict_survival()."
        exp_lh = np.exp(log_hazard)
        idx = np.searchsorted(self.event_times, time_points, side="right") - 1
        idx = np.clip(idx, 0, len(self.baseline_cumhazard) - 1)
        H0 = self.baseline_cumhazard[idx].copy()
        H0[time_points < self.event_times[0]] = 0.0
        return np.exp(-np.outer(exp_lh, H0))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SurvPredSurvBoardRunner:
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

        self.cls_gene_id = 0
        self.pad_gene_id = 1
        self.gene_token_offset = 2
        self.num_gene_tokens = int(self.model_cfg.gene_num) + self.gene_token_offset
        self.cls_value = float(getattr(self.model_cfg, "pad_value", -2.0))
        self.selected_gene_count = int(getattr(self.model_cfg, "selected_gene_count", 1199))
        self.max_seq_len = int(
            getattr(self.model_cfg, "max_seq_len", self.selected_gene_count + 1)
        )
        if self.max_seq_len != self.selected_gene_count + 1:
            raise ValueError(
                "Survival prediction expects max_seq_len to equal "
                "selected_gene_count + 1 for the <cls> token."
            )

        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.backbone_optimizer_enabled = False

        self.train_loader: DataLoader | None = None
        self.train_infer_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.train_dataset: SurvivalDataset | None = None
        self.test_dataset: SurvivalDataset | None = None
        self.fold_gene_indices: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "surv_pred_survboard" in cfg.finetune:
            return cfg.finetune.surv_pred_survboard
        raise ValueError(
            "Could not find survival prediction config. Expected cfg.finetune.surv_pred_survboard."
        )

    @staticmethod
    def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
        if "pretrain" in cfg:
            return cfg.pretrain
        raise ValueError("Could not find model architecture config. Expected cfg.pretrain.")

    def _finetune_mode(self) -> str:
        return str(getattr(self.task_cfg, "finetune_mode", "head_only"))

    def _output_suffix(self) -> str:
        configured = getattr(self.task_cfg, "output_suffix", "")
        suffix = "" if configured is None else str(configured).strip()
        if suffix and re.fullmatch(r"[A-Za-z0-9_-]+", suffix) is None:
            raise ValueError(
                "output_suffix may contain only letters, numbers, underscores, and hyphens."
            )
        return suffix

    def _output_variant(self) -> str:
        suffix = self._output_suffix()
        return f"{self._finetune_mode()}_{suffix}" if suffix else self._finetune_mode()

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / TASK_NAME / self._output_variant()

    def _output_prefix(self) -> str:
        return f"{TASK_NAME}_{self._output_variant()}"

    def _resolve_gene_list_path(self) -> Path:
        gene_list_path = getattr(self.task_cfg, "gene_list_path", None)
        if gene_list_path:
            return Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        return ROOT / "scbFM" / "data" / "gene_list.txt"

    # ------------------------------------------------------------------
    # Runtime setup
    # ------------------------------------------------------------------

    def _setup_runtime(self) -> None:
        if self.is_distributed and not dist.is_initialized():
            backend = (
                "nccl"
                if torch.cuda.is_available() and dist.is_nccl_available()
                else "gloo"
            )
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

    # ------------------------------------------------------------------
    # I/O helpers (mirrors canc_type_class runner)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]], comment: str = "") -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        preferred = [
            "model",
            "split",
            "n_splits",
            "finetune_mode",
            "checkpoint_path",
            "cancer",
            "project",
        ]
        fieldnames = [f for f in preferred if any(f in row for row in rows)]
        extra_fields = sorted(
            {f for row in rows for f in row if f not in fieldnames}
        )
        with path.open("w", newline="") as handle:
            if comment:
                handle.write(f"# {comment}\n")
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
            OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8"
        )
        self._run_metadata_path = out_dir / f"{prefix}_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": self.task_name,
                "finetune_mode": self._finetune_mode(),
                "head_only_backbone_eval": self._finetune_mode() == "head_only",
                "cancer": str(getattr(self.task_cfg, "cancer", "")),
                "project": str(getattr(self.task_cfg, "project", "TCGA")),
            },
            checkpoint_paths=checkpoint_paths,
            repo_dir=ROOT / "scbFM",
        )

    @staticmethod
    def _aggregate_numeric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
        aggregate: dict[str, object] = {"n_splits": len(rows)}
        skip_fields = {"model", "split", "n_splits", "finetune_mode", "checkpoint_path", "cancer", "project"}
        numeric_fields = sorted(
            {
                field
                for row in rows
                for field, value in row.items()
                if field not in skip_fields
                and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for field in numeric_fields:
            values = np.asarray(
                [float(row[field]) for row in rows if field in row], dtype=float
            )
            values = values[~np.isnan(values)]
            aggregate[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            aggregate[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        return aggregate

    # ------------------------------------------------------------------
    # Checkpoint paths
    # ------------------------------------------------------------------

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                "finetune.surv_pred_survboard.pretrained_model_paths must define "
                f"{', '.join(CHECKPOINT_MODEL_KEYS)}."
            )
        checkpoint_paths: dict[str, str] = {}
        missing = []
        for key in CHECKPOINT_MODEL_KEYS:
            value = paths_cfg.get(key)
            if value:
                checkpoint_paths[key] = str(
                    Path(hydra.utils.to_absolute_path(str(value)))
                )
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f"Missing checkpoint paths in finetune.surv_pred_survboard.pretrained_model_paths: {missing}"
            )
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
        return checkpoint_paths

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_survboard_data(self) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
        """Load SurvBoard preprocessed CSV, build AnnData from GEX columns, tokenize."""
        survboard_data_dir = getattr(self.task_cfg, "survboard_data_dir", None)
        if not survboard_data_dir:
            raise ValueError("finetune.surv_pred_survboard.survboard_data_dir must be set.")
        cancer = str(getattr(self.task_cfg, "cancer", ""))
        project = str(getattr(self.task_cfg, "project", "TCGA"))
        if not cancer:
            raise ValueError("finetune.surv_pred_survboard.cancer must be set (e.g. 'BRCA').")

        base = Path(hydra.utils.to_absolute_path(str(survboard_data_dir)))
        data_path = base / project / f"{cancer}_data_complete_modalities_preprocessed.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"SurvBoard data not found: {data_path}")

        os_col = str(getattr(self.task_cfg, "os_col", "OS"))
        os_days_col = str(getattr(self.task_cfg, "os_days_col", "OS_days"))

        df = pd.read_csv(data_path, low_memory=False)
        if "patient_id" in df.columns:
            df = df.drop(columns=["patient_id"])

        for col in (os_col, os_days_col):
            if col not in df.columns:
                raise ValueError(f"Survival column '{col}' not found in {data_path}.")

        times = df[os_days_col].values.astype(np.float32)
        events = df[os_col].values.astype(np.float32)

        gex_cols = [c for c in df.columns if c.startswith("gex_")]
        if not gex_cols:
            raise ValueError(f"No 'gex_*' columns found in {data_path}.")

        # SurvBoard TCGA columns are "gex_HUGO|ENTREZ_ID" (Entrez, not ENSG).
        # Map HUGO symbols → ENSG IDs via bulkformer_gene_info.csv, then
        # reindex_to_gene_list will align to gene_list.txt (ENSG).
        gene_info_path_cfg = getattr(self.task_cfg, "gene_info_path", None)
        if gene_info_path_cfg:
            gene_info_path_abs = Path(hydra.utils.to_absolute_path(str(gene_info_path_cfg)))
        else:
            gene_info_path_abs = Path(__file__).resolve().parents[3] / "data" / "bulkformer_gene_info.csv"
        gene_info = pd.read_csv(gene_info_path_abs)
        sym2ensg = dict(zip(gene_info["gene_symbol"].astype(str), gene_info["ensg_id"].astype(str)))

        # Extract HUGO symbols (part before "|") and map to ENSG
        hugo_symbols = [c[len("gex_"):].split("|")[0] for c in gex_cols]
        mapped_cols, mapped_ensg = [], []
        seen_ensg: set[str] = set()
        for col, sym in zip(gex_cols, hugo_symbols):
            ensg = sym2ensg.get(sym)
            if ensg and ensg not in seen_ensg:
                mapped_cols.append(col)
                mapped_ensg.append(ensg)
                seen_ensg.add(ensg)

        if not mapped_cols:
            raise ValueError(
                f"No SurvBoard GEX columns could be mapped to ENSG IDs via {gene_info_path_abs}. "
                f"Sample columns: {gex_cols[:5]}"
            )
        if self.is_master:
            log.info(
                "SurvBoard GEX: %d gex columns → %d mapped to ENSG IDs (%d unmapped/duplicate)",
                len(gex_cols), len(mapped_cols), len(gex_cols) - len(mapped_cols),
            )

        X = df[mapped_cols].values.astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = mapped_ensg

        gene_list_path = self._resolve_gene_list_path()
        if bool(getattr(self.task_cfg, "preprocess", True)):
            adata, missing_genes = reindex_adata_genes(
                adata,
                gene_list_path=gene_list_path,
            )
            if self.is_master:
                log.info(
                    "SurvBoard GEX aligned for on-the-fly sequence binning: "
                    "shape=%s, missing_genes=%d",
                    adata.shape,
                    len(missing_genes),
                )
        else:
            adata, missing_genes = reindex_adata_genes(
                adata, gene_list_path=gene_list_path
            )
            if self.is_master:
                log.info(
                    "SurvBoard GEX reindexed: shape=%s, missing_genes=%d",
                    adata.shape,
                    len(missing_genes),
                )

        self._missing_genes_note = (
            f"Model genes missing from SurvBoard GEX and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
        ) if missing_genes else ""

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes, got {adata.n_vars}. "
                "Ensure gene_list_path matches the SurvBoard GEX gene IDs."
            )

        if not bool(getattr(self.task_cfg, "preprocess", True)):
            validate_token_matrix(
                adata.X,
                bin_num=int(self.model_cfg.bin_num),
                name="surv_pred input",
            )

        # Re-align times/events if preprocess dropped samples
        if adata.n_obs != len(times):
            obs_names = adata.obs_names.astype(int).to_numpy() if adata.obs_names[0].isdigit() else None
            if obs_names is not None:
                times = times[obs_names]
                events = events[obs_names]
            else:
                raise ValueError(
                    f"Sample count mismatch after preprocessing ({adata.n_obs} vs {len(times)}). "
                    "Check SurvBoard sample identifiers and expression/clinical row alignment."
                )

        return adata, times, events

    def _load_splits(self, n_samples: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Load SurvBoard pre-computed train/test split CSVs."""
        survboard_data_dir = getattr(self.task_cfg, "survboard_data_dir", None)
        cancer = str(getattr(self.task_cfg, "cancer", ""))
        project = str(getattr(self.task_cfg, "project", "TCGA"))
        base = Path(hydra.utils.to_absolute_path(str(survboard_data_dir)))
        splits_dir = base / "splits" / project

        train_path = splits_dir / f"{cancer}_train_splits.csv"
        test_path = splits_dir / f"{cancer}_test_splits.csv"
        for p in (train_path, test_path):
            if not p.exists():
                raise FileNotFoundError(f"SurvBoard split file not found: {p}")

        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)

        outer_splits = list(getattr(self.task_cfg, "outer_splits", list(range(len(train_df)))))
        train_splits, test_splits = [], []
        for s in outer_splits:
            if s >= len(train_df):
                raise ValueError(
                    f"outer_split {s} out of range (split CSV has {len(train_df)} rows)."
                )
            train_ix = train_df.iloc[s].dropna().values.astype(int)
            test_ix = test_df.iloc[s].dropna().values.astype(int)
            if train_ix.max() >= n_samples or test_ix.max() >= n_samples:
                raise ValueError(
                    f"Split {s} contains index out of bounds for dataset of size {n_samples}."
                )
            train_splits.append(train_ix)
            test_splits.append(test_ix)

        return train_splits, test_splits

    def _select_training_hvg_indices(self, train_adata: ad.AnnData) -> np.ndarray:
        """Fit the 1,199-gene input vocabulary on the training split only."""
        if train_adata.n_vars < self.selected_gene_count:
            raise ValueError(
                f"Cannot select {self.selected_gene_count} HVGs from "
                f"only {train_adata.n_vars} genes."
            )
        batch_key = getattr(self.task_cfg, "hvg_batch_key", None)
        if batch_key is not None:
            batch_key = str(batch_key).strip() or None
        if batch_key is not None and batch_key not in train_adata.obs:
            raise ValueError(f"HVG batch key '{batch_key}' is not present in adata.obs.")

        hvg_stats = sc.pp.highly_variable_genes(
            train_adata,
            n_top_genes=self.selected_gene_count,
            flavor=str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
            batch_key=batch_key,
            inplace=False,
        )
        selected = np.flatnonzero(hvg_stats["highly_variable"].to_numpy())
        if selected.size > self.selected_gene_count:
            ranking_column = (
                "highly_variable_rank"
                if "highly_variable_rank" in hvg_stats
                else "dispersions_norm"
            )
            scores = hvg_stats[ranking_column].to_numpy()[selected]
            if ranking_column == "highly_variable_rank":
                order = np.argsort(np.nan_to_num(scores, nan=np.inf), kind="stable")
            else:
                order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
            selected = selected[order[: self.selected_gene_count]]
        if selected.size != self.selected_gene_count:
            raise RuntimeError(
                f"Scanpy selected {selected.size} HVGs; "
                f"expected exactly {self.selected_gene_count}."
            )
        log.info(
            "Selected %d training-split HVGs with flavor=%s, batch_key=%s",
            selected.size,
            str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
            batch_key,
        )
        return selected.astype(np.int64, copy=False)

    def _build_loaders(
        self,
        train_X,
        train_times: np.ndarray,
        train_events: np.ndarray,
        test_X,
        test_times: np.ndarray,
        test_events: np.ndarray,
    ) -> None:
        if self.fold_gene_indices is None:
            raise RuntimeError("Training-split HVGs have not been selected.")
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        num_workers = int(getattr(self.task_cfg, "num_workers", 0))
        if num_workers < 0:
            raise ValueError("finetune.surv_pred_survboard.num_workers must be non-negative.")
        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            prefetch_factor = int(getattr(self.task_cfg, "prefetch_factor", 2))
            if prefetch_factor <= 0:
                raise ValueError("finetune.surv_pred_survboard.prefetch_factor must be positive.")
            loader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": False,
                }
            )
        dataset_kwargs = {
            "bin_num": int(self.model_cfg.bin_num),
            "cls_gene_id": self.cls_gene_id,
            "gene_token_offset": self.gene_token_offset,
            "cls_value": self.cls_value,
            "selected_gene_count": self.selected_gene_count,
            "seed": int(getattr(self.task_cfg, "random_seed", 42)),
            "do_binning": bool(getattr(self.task_cfg, "preprocess", True)),
            "fixed_gene_indices": self.fold_gene_indices,
        }

        self.train_dataset = SurvivalDataset(
            train_X, train_times, train_events, **dataset_kwargs
        )
        self.test_dataset = SurvivalDataset(
            test_X, test_times, test_events, **dataset_kwargs
        )

        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            test_sampler = SequentialDistributedSampler(
                self.test_dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            train_infer_sampler = SequentialDistributedSampler(
                self.train_dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,
                **loader_kwargs,
            )
            self.train_infer_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=train_infer_sampler,
                shuffle=False,
                **loader_kwargs,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                **loader_kwargs,
            )
            self.train_infer_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )

        if self.is_master:
            log.info(
                "DataLoader: num_workers=%d | prefetch_factor=%s | pin_memory=%s | "
                "persistent_workers=false",
                num_workers,
                (
                    int(getattr(self.task_cfg, "prefetch_factor", 2))
                    if num_workers > 0
                    else "disabled"
                ),
                self.device.type == "cuda",
            )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

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

        backbone = CancerFoundationBackbone(
            num_gene_tokens=self.num_gene_tokens,
            d_model=int(self.model_cfg.embsize),
            nhead=int(self.model_cfg.nheads),
            d_hid=int(self.model_cfg.d_hid),
            nlayers=int(self.model_cfg.nlayers),
            dropout=float(self.model_cfg.dropout),
            pad_gene_id=self.pad_gene_id,
            max_value=int(getattr(self.model_cfg, "value_encoder_max_value", 512)),
        )

        if checkpoint_path:
            resolved_path = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved_path, map_location="cpu")
            validate_backbone_checkpoint(
                checkpoint,
                resolved_path,
                gene_num=int(self.model_cfg.gene_num),
                selected_gene_count=self.selected_gene_count,
                max_seq_len=self.max_seq_len,
                bin_num=int(self.model_cfg.bin_num),
            )
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            backbone.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved_path)
        else:
            log.info("Using randomly initialised backbone")

        head = SurvivalPredHead(
            embedding_dim=int(self.model_cfg.embsize),
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 512)),
            bottleneck_dim=int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
        )
        if self.is_master:
            log.info(
                "Survival representation: CLS token | head dims: %d -> %d -> %d -> 1",
                int(self.model_cfg.embsize),
                int(getattr(self.task_cfg, "head_hidden_dim", 512)),
                int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
            )
        model = CancerFoundationSurvivalModel(backbone=backbone, head=head)

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
                int(getattr(self.task_cfg, "burn_in_epochs", 3)) <= 0
            )

        if finetune_mode in ("adapters", "full_ft"):
            model.enable_grad_checkpoint()

        model = model.to(self.device)
        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                model = DDP(model)

        self.model = model

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _build_optimization(self) -> None:
        head_learning_rate = float(
            getattr(self.task_cfg, "head_learning_rate", 1e-4)
        )
        backbone_learning_rate = float(
            getattr(self.task_cfg, "backbone_learning_rate", 1e-4)
        )
        adapter_learning_rate = float(
            getattr(self.task_cfg, "adapter_learning_rate", head_learning_rate)
        )

        model = self.model.module if isinstance(self.model, DDP) else self.model
        head_params = [p for p in model.to_out.parameters() if p.requires_grad]
        head_param_ids = {id(p) for p in head_params}
        adapter_params = [p for p in model.adapter_parameters() if p.requires_grad]
        adapter_param_ids = {id(p) for p in adapter_params}
        backbone_params = [
            p
            for p in model.parameters()
            if p.requires_grad
            and id(p) not in head_param_ids
            and id(p) not in adapter_param_ids
        ]

        param_groups = []
        if backbone_params and self.backbone_optimizer_enabled:
            param_groups.append({"params": backbone_params, "lr": backbone_learning_rate, "name": "backbone"})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": adapter_learning_rate, "name": "adapters"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_learning_rate, "name": "head"})
        if not param_groups:
            raise ValueError("No trainable parameters found for survival prediction.")

        max_lrs = [float(g["lr"]) for g in param_groups]
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
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 30)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 1)),
            max_lrs=max_lrs,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 2)),
            gamma=float(getattr(self.task_cfg, "gamma", 1.0)),
        )

        if self.is_master:
            summaries = [
                f"{g.get('name', i)}: params={sum(p.numel() for p in g['params'])}, "
                f"max_lr={lr:.2e}"
                for i, (g, lr) in enumerate(zip(param_groups, max_lrs))
            ]
            log.info("Optimizer parameter groups: %s", "; ".join(summaries))

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        burn_in = int(getattr(self.task_cfg, "burn_in_epochs", 3))
        if (
            self._finetune_mode() != "full_ft"
            or burn_in <= 0
            or self.backbone_optimizer_enabled
            or epoch <= burn_in
        ):
            return
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        add_optimizer_parameter_group(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            parameters=raw_model.backbone.parameters(),
            max_lr=float(self.task_cfg.backbone_learning_rate),
            name="backbone",
        )
        self.backbone_optimizer_enabled = True
        if self.is_master:
            log.info("Finished %d burn-in epochs; enabled backbone optimisation.", burn_in)

    def _optimizer_parameters(self) -> list[nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_enable_backbone_optimizer(epoch)

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        _set_finetune_training_mode(self.model, self._finetune_mode())
        self.model.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 4)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss_numerator = 0.0
        running_event_count = 0.0
        window_batches: dict[str, list[torch.Tensor]] = {}
        window_times: list[torch.Tensor] = []
        window_events: list[torch.Tensor] = []
        total_steps = len(self.train_loader)

        for step_idx, (batch, time, event) in enumerate(self.train_loader, start=1):
            batch = {
                key: value.to(self.device, non_blocking=True)
                for key, value in batch.items()
            }
            time = time.to(self.device, non_blocking=True)
            event = event.to(self.device, non_blocking=True)

            is_update_step = is_accumulation_boundary(
                step_idx,
                total_steps,
                grad_acc_steps,
            )
            for key, value in batch.items():
                window_batches.setdefault(key, []).append(value)
            window_times.append(time)
            window_events.append(event)

            if is_update_step:
                update_batch = {
                    key: torch.cat(values)
                    for key, values in window_batches.items()
                }
                log_hazard, update_time, update_event = gather_cox_update_batch(
                    self.model(update_batch),
                    torch.cat(window_times),
                    torch.cat(window_events),
                    is_distributed=self.is_distributed,
                )
                loss = cox_partial_log_likelihood(
                    log_hazard,
                    update_time,
                    update_event,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)
                event_count = float(update_event.sum().item())
                running_loss_numerator += float(loss.detach().item()) * event_count
                running_event_count += event_count
                window_batches.clear()
                window_times.clear()
                window_events.clear()

        if running_event_count <= 0:
            raise ValueError("A survival training epoch contained no observed events.")
        epoch_loss = running_loss_numerator / running_event_count

        self.scheduler.step()
        return {"loss": epoch_loss}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_log_hazard(self, loader: DataLoader, dataset_size: int) -> np.ndarray:
        """Run inference and return gathered log-hazard predictions as numpy array."""
        self.model.eval()
        if self.is_distributed:
            dist.barrier()

        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch, _time, _event in loader:
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                preds.append(self.model(batch).cpu())

        pred_tensor = torch.cat(preds, dim=0)

        if self.is_distributed:
            pred_tensor = pred_tensor.to(self.device)
            pred_tensor = distributed_concat(pred_tensor, dataset_size, self.world_size)

        return pred_tensor.cpu().numpy()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _flatten_fold_metrics(
        self,
        model_key: str,
        split_idx: int,
        n_splits: int,
        checkpoint_path: str,
        cancer: str,
        project: str,
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "model": model_key,
            "split": split_idx,
            "n_splits": n_splits,
            "finetune_mode": self._finetune_mode(),
            "checkpoint_path": checkpoint_path,
            "cancer": cancer,
            "project": project,
            "train_loss": float(train_metrics["loss"]),
        }
        row.update({key: float(value) for key, value in test_metrics.items()})
        return row

    def _write_model_results(
        self,
        checkpoint_path: str,
        fold_rows: list[dict[str, object]],
        curves_rows: list[dict[str, object]],
    ) -> dict[str, object]:
        aggregate = self._aggregate_numeric_rows(fold_rows)
        aggregate.update(
            {
                "model": fold_rows[0]["model"],
                "finetune_mode": fold_rows[0]["finetune_mode"],
                "checkpoint_path": checkpoint_path,
                "cancer": fold_rows[0].get("cancer", ""),
                "project": fold_rows[0].get("project", ""),
            }
        )
        out_dir = self._task_output_dir()
        model_key = str(fold_rows[0]["model"])
        prefix = self._output_prefix()
        self._write_csv(out_dir / f"{prefix}_{model_key}_fold_metrics.csv", fold_rows)
        self._write_csv(out_dir / f"{prefix}_{model_key}_evaluation_metrics.csv", [aggregate], comment=getattr(self, "_missing_genes_note", ""))
        self._write_csv(out_dir / f"{prefix}_{model_key}_curves.csv", curves_rows)
        return aggregate

    def _save_survboard_predictions(
        self,
        model_key: str,
        split_idx: int,
        survival_probs: np.ndarray,
        time_points: np.ndarray,
    ) -> None:
        out_dir = self._task_output_dir() / "survboard" / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(survival_probs, columns=time_points.astype(str))
        df.to_csv(out_dir / f"split_{split_idx}.csv", index=False)

    def _cleanup_fold_state(self) -> None:
        self.train_loader = None
        self.train_infer_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.fold_gene_indices = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.backbone_optimizer_enabled = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        try:
            self._setup_runtime()

            cancer = str(getattr(self.task_cfg, "cancer", ""))
            project = str(getattr(self.task_cfg, "project", "TCGA"))
            epochs = int(getattr(self.task_cfg, "epochs", 20))

            if self.is_master:
                log.info("Loading and preprocessing SurvBoard data (%s / %s)...", project, cancer)

            adata, times, events = self._load_survboard_data()
            train_splits, test_splits = self._load_splits(adata.n_obs)
            n_splits = len(train_splits)
            outer_splits = list(getattr(self.task_cfg, "outer_splits", list(range(n_splits))))

            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            if self.is_master:
                log.info(
                    "Data ready: samples=%d, genes=%d, splits=%d, epochs=%d",
                    adata.n_obs, adata.n_vars, n_splits, epochs,
                )

            aggregate_rows: list[dict[str, object]] = []

            for model_idx, (model_key, checkpoint_path) in enumerate(checkpoint_paths.items()):
                fold_rows: list[dict[str, object]] = []
                curves_rows: list[dict[str, object]] = []

                for split_pos, split_idx in enumerate(outer_splits):
                    if split_idx < 0 or split_idx >= n_splits:
                        raise ValueError(
                            f"outer_splits contains invalid split {split_idx}; "
                            f"available splits are 0..{n_splits - 1}."
                        )
                    train_ix = train_splits[split_idx]
                    test_ix = test_splits[split_idx]
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + model_idx * 10000
                        + split_pos
                    )

                    train_adata = adata[train_ix].copy()
                    test_adata = adata[test_ix].copy()
                    self.fold_gene_indices = self._select_training_hvg_indices(train_adata)
                    train_X = train_adata.X
                    test_X = test_adata.X
                    train_times, train_events = times[train_ix], events[train_ix]
                    test_times, test_events = times[test_ix], events[test_ix]

                    if self.is_master:
                        log.info(
                            "Model %s | Split %d (%d/%d) | train=%d (events=%d), test=%d (events=%d)",
                            model_key, split_idx, split_pos + 1, n_splits,
                            len(train_ix), int(train_events.sum()),
                            len(test_ix), int(test_events.sum()),
                        )

                    self._build_loaders(
                        train_X, train_times, train_events,
                        test_X, test_times, test_events,
                    )
                    self._build_model(checkpoint_path)
                    self._build_optimization()

                    last_train_metrics = {"loss": float("nan")}
                    last_test_lh: np.ndarray | None = None
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        last_test_lh = self._predict_log_hazard(
                            self.test_loader,
                            len(test_ix),
                        )
                        if self.is_master:
                            validation_loss = float(
                                cox_partial_log_likelihood(
                                    torch.as_tensor(last_test_lh),
                                    torch.as_tensor(test_times),
                                    torch.as_tensor(test_events),
                                ).item()
                            )
                            validation_c_index = harrell_c_index(
                                last_test_lh,
                                test_times,
                                test_events,
                            )
                            log.info(
                                "Model %s | Split %d | Epoch %d/%d | "
                                "Loss: %.6f | Validation Loss: %.6f | C-index: %.4f",
                                model_key, split_idx, epoch, epochs,
                                last_train_metrics["loss"],
                                validation_loss,
                                validation_c_index,
                            )
                            curves_rows.append(
                                {
                                    "model": model_key,
                                    "split": split_idx,
                                    "epoch": epoch,
                                    "train_loss": last_train_metrics["loss"],
                                    "validation_loss": validation_loss,
                                    "validation_c_index": validation_c_index,
                                    "learning_rates": ";".join(
                                        f"{float(group['lr']):.6g}"
                                        for group in self.optimizer.param_groups
                                    ),
                                }
                            )
                            self._write_csv(
                                self._task_output_dir()
                                / f"{self._output_prefix()}_{model_key}_curves.csv",
                                curves_rows,
                            )

                    # Predict on test and train sets — all ranks must participate
                    # because _predict_log_hazard uses distributed_concat (all_gather).
                    test_lh = (
                        last_test_lh
                        if last_test_lh is not None
                        else self._predict_log_hazard(self.test_loader, len(test_ix))
                    )
                    train_lh = self._predict_log_hazard(self.train_infer_loader, len(train_ix))

                    if self.is_master:
                        # Fit Breslow on train set (shared by Antolini C-index + SurvBoard output)
                        breslow = BreslowEstimator()
                        breslow.fit(train_lh, train_times, train_events)
                        event_times = np.unique(train_times[train_events.astype(bool)])
                        survival_probs = breslow.predict_survival(test_lh, event_times)

                        test_metrics = survival_metrics(
                            survival_probs=survival_probs,
                            time_points=event_times,
                            test_time=test_times,
                            test_event=test_events,
                            test_log_hazard=test_lh,
                            train_time=train_times,
                            train_event=train_events,
                        )
                        log.info(
                            "Model %s | Split %d | C-index: %.4f | Weighted C-index: %.4f | "
                            "Antolini: %.4f | IBS: %.4f | D-cal p: %.4f",
                            model_key,
                            split_idx,
                            test_metrics["test_c_index"],
                            test_metrics["test_weighted_c_index"],
                            test_metrics["test_antolini_cindex"],
                            test_metrics["test_integrated_brier_score"],
                            test_metrics["test_d_calibration_p_value"],
                        )
                        fold_rows.append(
                            self._flatten_fold_metrics(
                                model_key=model_key,
                                split_idx=split_idx,
                                n_splits=n_splits,
                                checkpoint_path=checkpoint_path,
                                cancer=cancer,
                                project=project,
                                train_metrics=last_train_metrics,
                                test_metrics=test_metrics,
                            )
                        )

                        # SurvBoard-style output
                        self._save_survboard_predictions(
                            model_key, split_idx, survival_probs, event_times
                        )

                    self._cleanup_fold_state()

                if self.is_master:
                    aggregate_rows.append(
                        self._write_model_results(checkpoint_path, fold_rows, curves_rows)
                    )

                if self.is_distributed:
                    dist.barrier()

            if self.is_master:
                out_dir = self._task_output_dir()
                output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
                self._write_csv(output_path, aggregate_rows, comment=getattr(self, "_missing_genes_note", ""))
                complete_run_metadata(self._run_metadata_path, output_path)
                return {"results_path": str(output_path), "results": aggregate_rows}
            return {}

        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
