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
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from performer_pytorch import PerformerLM
from preprocess import preprocess_adata_for_tokens, reindex_adata_genes, validate_token_matrix
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
TASK_NAME = "surv_pred"
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
    """Mean-pool over gene positions → LayerNorm → MLP → log-hazard scalar.

    Mirrors BulkRNABert's survival head design (Gélard et al., 2024).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)          # (B, seq_len, dim) → (B, dim)
        x = self.norm(x)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurvivalDataset(Dataset):
    def __init__(
        self,
        X,
        times: np.ndarray,
        events: np.ndarray,
        special_token_id: int,
    ) -> None:
        self.X = X
        self.times = np.asarray(times, dtype=np.float32)
        self.events = np.asarray(events, dtype=np.float32)
        self.special_token_id = special_token_id

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.X[index]
        if sparse.issparse(row):
            seq = row.toarray().ravel()
        else:
            seq = np.asarray(row).ravel()
        seq = torch.as_tensor(seq, dtype=torch.long)
        seq = torch.cat([seq, torch.tensor([self.special_token_id], dtype=torch.long)])
        return (
            seq,
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

class SurvPredRunner:
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

        self.vocab_size = int(self.model_cfg.bin_num) + 3  # bins + pad + mask + EOS
        self.special_token_id = int(self.model_cfg.bin_num) + 1

        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.backbone_optimizer_enabled = False

        self.train_loader: DataLoader | None = None
        self.train_infer_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.train_dataset: SurvivalDataset | None = None
        self.test_dataset: SurvivalDataset | None = None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "surv_pred" in cfg.finetune:
            return cfg.finetune.surv_pred
        raise ValueError(
            "Could not find survival prediction config. Expected cfg.finetune.surv_pred."
        )

    @staticmethod
    def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
        if "pretrain" in cfg:
            return cfg.pretrain
        raise ValueError("Could not find model architecture config. Expected cfg.pretrain.")

    def _finetune_mode(self) -> str:
        return str(getattr(self.task_cfg, "finetune_mode", "head_only"))

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / TASK_NAME / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{TASK_NAME}_{self._finetune_mode()}"

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
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
        self._write_json(
            out_dir / f"{prefix}_run_metadata.json",
            {
                "task": self.task_name,
                "finetune_mode": self._finetune_mode(),
                "cancer": str(getattr(self.task_cfg, "cancer", "")),
                "project": str(getattr(self.task_cfg, "project", "TCGA")),
                "git_commit": self._get_git_commit(),
                "checkpoint_paths": checkpoint_paths,
            },
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
                "finetune.surv_pred.pretrained_model_paths must define "
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
                f"Missing checkpoint paths in finetune.surv_pred.pretrained_model_paths: {missing}"
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
            raise ValueError("finetune.surv_pred.survboard_data_dir must be set.")
        cancer = str(getattr(self.task_cfg, "cancer", ""))
        project = str(getattr(self.task_cfg, "project", "TCGA"))
        if not cancer:
            raise ValueError("finetune.surv_pred.cancer must be set (e.g. 'BRCA').")

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

        gene_ids = [c[len("gex_"):] for c in gex_cols]
        X = df[gex_cols].values.astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = gene_ids

        gene_list_path = self._resolve_gene_list_path()
        if bool(getattr(self.task_cfg, "preprocess", True)):
            adata, missing_genes = preprocess_adata_for_tokens(
                adata,
                gene_list_path=gene_list_path,
                min_genes=int(getattr(self.task_cfg, "min_genes", 0)),
                target_sum=float(getattr(self.task_cfg, "target_sum", 1e4)),
                bin_num=int(self.model_cfg.bin_num),
                reindex_genes=True,
            )
            if self.is_master:
                log.info(
                    "SurvBoard GEX preprocessed: shape=%s, missing_genes=%d",
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

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes, got {adata.n_vars}. "
                "Ensure gene_list_path matches the SurvBoard GEX gene IDs."
            )

        validate_token_matrix(adata.X, bin_num=int(self.model_cfg.bin_num), name="surv_pred input")

        # Re-align times/events if preprocess dropped samples
        if adata.n_obs != len(times):
            obs_names = adata.obs_names.astype(int).to_numpy() if adata.obs_names[0].isdigit() else None
            if obs_names is not None:
                times = times[obs_names]
                events = events[obs_names]
            else:
                raise ValueError(
                    f"Sample count mismatch after preprocessing ({adata.n_obs} vs {len(times)}). "
                    "Set min_genes=0 to avoid dropping samples."
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

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

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

    def _build_loaders(
        self,
        train_X,
        train_times: np.ndarray,
        train_events: np.ndarray,
        test_X,
        test_times: np.ndarray,
        test_events: np.ndarray,
    ) -> None:
        batch_size = int(getattr(self.task_cfg, "batch_size", 16))

        self.train_dataset = SurvivalDataset(
            train_X, train_times, train_events, self.special_token_id
        )
        self.test_dataset = SurvivalDataset(
            test_X, test_times, test_events, self.special_token_id
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
                self.train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False
            )
            self.train_infer_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, sampler=train_infer_sampler, shuffle=False
            )
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True
            )
            self.train_infer_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=False
            )
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=False
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
            log.info("Using randomly initialised backbone")

        model.to_out = SurvivalPredHead(
            embedding_dim=int(self.model_cfg.dim),
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 256)),
            dropout=float(getattr(self.task_cfg, "head_dropout", 0.1)),
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

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _build_optimization(self) -> None:
        head_learning_rate = float(
            getattr(self.task_cfg, "head_learning_rate", 1e-4)
        )
        backbone_learning_rate = float(
            getattr(self.task_cfg, "backbone_learning_rate", 1e-6)
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
        burn_in = int(getattr(self.task_cfg, "burn_in_epochs", 0))
        if (
            self._finetune_mode() != "full_ft"
            or burn_in <= 0
            or self.backbone_optimizer_enabled
            or epoch <= burn_in
        ):
            return
        self.backbone_optimizer_enabled = True
        self._build_optimization()
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

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 1)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss = 0.0
        n_batches = 0

        for step_idx, (seq, time, event) in enumerate(self.train_loader, start=1):
            seq = seq.to(self.device, non_blocking=True)
            time = time.to(self.device, non_blocking=True)
            event = event.to(self.device, non_blocking=True)

            use_no_sync = (
                self.is_distributed
                and isinstance(self.model, DDP)
                and step_idx % grad_acc_steps != 0
            )
            sync_ctx = self.model.no_sync() if use_no_sync else nullcontext()

            with sync_ctx:
                log_hazard = self.model(seq)
                loss = cox_partial_log_likelihood(log_hazard, time, event)
                (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(n_batches, 1)
        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)

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
            for seq, _time, _event in loader:
                seq = seq.to(self.device, non_blocking=True)
                preds.append(self.model(seq).cpu())

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
        test_antolini_cindex: float,
    ) -> dict[str, object]:
        return {
            "model": model_key,
            "split": split_idx,
            "n_splits": n_splits,
            "finetune_mode": self._finetune_mode(),
            "checkpoint_path": checkpoint_path,
            "cancer": cancer,
            "project": project,
            "train_loss": float(train_metrics["loss"]),
            "test_antolini_cindex": float(test_antolini_cindex),
        }

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
        self._write_csv(out_dir / f"{prefix}_{model_key}_evaluation_metrics.csv", [aggregate])
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
            epochs = int(getattr(self.task_cfg, "epochs", 30))

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

                for split_pos, (split_idx, train_ix, test_ix) in enumerate(
                    zip(outer_splits, train_splits, test_splits)
                ):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + model_idx * 10000
                        + split_pos
                    )

                    train_X = adata.X[train_ix]
                    test_X = adata.X[test_ix]
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
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        if self.is_master:
                            log.info(
                                "Model %s | Split %d | Epoch %d/%d | Loss: %.6f",
                                model_key, split_idx, epoch, epochs,
                                last_train_metrics["loss"],
                            )
                            curves_rows.append(
                                {
                                    "model": model_key,
                                    "split": split_idx,
                                    "epoch": epoch,
                                    "train_loss": last_train_metrics["loss"],
                                }
                            )

                    # Predict on test set
                    test_lh = self._predict_log_hazard(self.test_loader, len(test_ix))

                    if self.is_master:
                        # Fit Breslow on train set (shared by Antolini C-index + SurvBoard output)
                        train_lh = self._predict_log_hazard(
                            self.train_infer_loader, len(train_ix)
                        )
                        breslow = BreslowEstimator()
                        breslow.fit(train_lh, train_times, train_events)
                        event_times = np.unique(train_times[train_events.astype(bool)])
                        survival_probs = breslow.predict_survival(test_lh, event_times)

                        test_antolini_cindex = antolini_concordance(
                            survival_probs, event_times, test_times, test_events
                        )
                        log.info(
                            "Model %s | Split %d | Test Antolini C-index: %.4f",
                            model_key, split_idx, test_antolini_cindex,
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
                                test_antolini_cindex=test_antolini_cindex,
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
                self._write_csv(output_path, aggregate_rows)
                return {"results_path": str(output_path), "results": aggregate_rows}
            return {}

        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
