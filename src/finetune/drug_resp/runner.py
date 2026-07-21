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
import scanpy as sc
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.canc_type_class.runner import (
    GroupedCosineAnnealingWarmupRestarts,
    _quantile_bin_expression,
)
from preprocess import (
    reindex_adata_genes,
    validate_token_matrix,
)
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
TASK_NAME = "drug_resp"
CHECKPOINT_MODEL_KEYS = ("pretrain_sc", "pretrain_bulk", "preadapt_sc", "preadapt_bulk")
RANDOM_INIT_MODEL_KEY = "random_init"
MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class DrugRespPredHead(nn.Module):
    """[cell_emb ‖ drug_emb] → MLP → IC50 scalar."""

    def __init__(
        self,
        cell_emb_dim: int,
        drug_emb_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = cell_emb_dim + drug_emb_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SELU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, cell_emb: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        # cell_emb: (B, cell_emb_dim)   drug_emb: (B, drug_emb_dim)
        x = torch.cat([cell_emb, drug_emb], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B,)


class DrugRespModel(nn.Module):
    """CancerFoundation backbone + sample/drug fusion head."""

    def __init__(
        self,
        backbone: CancerFoundationBackbone,
        head: DrugRespPredHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def pool_cell_embeddings(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0, :]

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None,
        drug_emb: torch.Tensor,
        cell_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cell_emb is None:
            if batch is None:
                raise ValueError("A backbone input batch is required when cell_emb is absent.")
            hidden = self.backbone(
                batch["gene_ids"],
                batch["expr"],
                src_key_padding_mask=batch.get("attention_key_padding_mask"),
            )
            cell_emb = self.pool_cell_embeddings(hidden)
        # cell-centric: single cell embedding → expand to match all its drugs
        if cell_emb.shape[0] == 1 and drug_emb.shape[0] > 1:
            cell_emb = cell_emb.expand(drug_emb.shape[0], -1)
        return self.head(cell_emb, drug_emb)  # (B,)

    def add_adapters(self, **kwargs) -> nn.ModuleList:
        return self.backbone.add_adapters(**kwargs)

    def adapter_parameters(self):
        return self.backbone.adapter_parameters()

    def enable_grad_checkpoint(self):
        self.backbone.enable_grad_checkpoint()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DrugRespDataset(Dataset):
    """Each sample is one (cell_line, drug) pair → IC50."""

    def __init__(
        self,
        X_cell: np.ndarray,           # (n_cells, gene_num) expression matrix
        drug_emb_matrix: np.ndarray,  # (n_drugs, drug_emb_dim) float32
        cell_idxs: np.ndarray,        # (n_pairs,) int — index into X_cell
        drug_idxs: np.ndarray,        # (n_pairs,) int — index into drug_emb_matrix
        ic50_values: np.ndarray,      # (n_pairs,) float32
        fixed_gene_indices: np.ndarray,
        bin_num: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
        seed: int,
        do_binning: bool,
    ) -> None:
        self.X_cell = X_cell
        self.drug_emb_matrix = drug_emb_matrix
        self.cell_idxs = cell_idxs.astype(np.int64)
        self.drug_idxs = drug_idxs.astype(np.int64)
        self.ic50_values = ic50_values.astype(np.float32)
        self.fixed_gene_indices = np.asarray(fixed_gene_indices, dtype=np.int64)
        self.bin_num = int(bin_num)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.seed = int(seed)
        self.do_binning = bool(do_binning)

    def __len__(self) -> int:
        return len(self.ic50_values)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        cell_idx = int(self.cell_idxs[idx])
        row = self.X_cell[cell_idx]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()
        selected_values = values[self.fixed_gene_indices].astype(np.float32, copy=False)
        if self.do_binning:
            rng = np.random.default_rng(self.seed + cell_idx)
            selected_values = _quantile_bin_expression(
                selected_values,
                self.bin_num,
                rng,
            ).astype(np.float32)

        gene_ids = torch.from_numpy(
            self.fixed_gene_indices.astype(np.int64, copy=False)
            + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        expression = torch.from_numpy(selected_values)
        expression = torch.cat((torch.tensor([self.cls_value]), expression))

        drug_emb = torch.as_tensor(
            self.drug_emb_matrix[self.drug_idxs[idx]], dtype=torch.float32
        )
        target = torch.as_tensor(self.ic50_values[idx], dtype=torch.float32)
        return (
            torch.tensor(cell_idx, dtype=torch.long),
            {"gene_ids": gene_ids, "expr": expression},
            drug_emb,
            target,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class DrugRespRunner:
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
                "Drug response expects max_seq_len to equal selected_gene_count + 1 "
                "for the <cls> token."
            )

        self.test_loader: DataLoader | None = None
        self.test_dataset_size: int = 0
        self.cell_to_train_pairs: dict | None = None
        self.train_drug_idxs: np.ndarray | None = None
        self.train_ic50: np.ndarray | None = None
        self.X_cell_ref: np.ndarray | None = None
        self.drug_emb_matrix_ref: np.ndarray | None = None
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.backbone_optimizer_enabled = False
        self.cell_emb_cache: torch.Tensor | None = None
        self.fold_gene_indices: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "drug_resp" in cfg.finetune:
            return cfg.finetune.drug_resp
        raise ValueError("Could not find drug response config. Expected cfg.finetune.drug_resp.")

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
        seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + self.rank)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: list[dict], comment: str = "") -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        preferred = ["model", "fold", "n_folds", "finetune_mode", "checkpoint_path", "cell_line_id"]
        fieldnames = [f for f in preferred if any(f in row for row in rows)]
        extra_fields = sorted({f for row in rows for f in row if f not in fieldnames})
        with path.open("w", newline="") as handle:
            if comment:
                handle.write(f"# {comment}\n")
            writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _get_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT / "scbFM",
                check=True, capture_output=True, text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def _save_run_metadata(self, checkpoint_paths: dict) -> None:
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
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "git_commit": self._get_git_commit(),
                "checkpoint_paths": checkpoint_paths,
            },
        )

    @staticmethod
    def _aggregate_numeric_rows(rows: list[dict]) -> dict:
        aggregate: dict = {"n_folds": len(rows)}
        skip_fields = {"model", "fold", "n_folds", "finetune_mode", "checkpoint_path"}
        numeric_fields = sorted({
            f for row in rows for f, v in row.items()
            if f not in skip_fields and isinstance(v, (int, float, np.integer, np.floating))
        })
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in rows if field in row], dtype=float)
            values = values[~np.isnan(values)]
            aggregate[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            aggregate[f"{field}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return aggregate

    # ------------------------------------------------------------------
    # Checkpoint paths
    # ------------------------------------------------------------------

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                f"finetune.drug_resp.pretrained_model_paths must define {', '.join(CHECKPOINT_MODEL_KEYS)}."
            )
        checkpoint_paths: dict[str, str] = {}
        missing = []
        for key in CHECKPOINT_MODEL_KEYS:
            value = paths_cfg.get(key)
            if value:
                checkpoint_paths[key] = str(Path(hydra.utils.to_absolute_path(str(value))))
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f"Missing checkpoint paths in finetune.drug_resp.pretrained_model_paths: {missing}"
            )
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
        return checkpoint_paths

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_gdsc_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and align expression + IC50 + drug embeddings.

        Returns:
            X_cell          (n_cells, gene_num) token matrix (sparse or dense)
            drug_emb_matrix (n_drugs, drug_emb_dim) float32
            cell_idxs       (n_pairs,) int — index into X_cell rows
            drug_idxs       (n_pairs,) int — index into drug_emb_matrix rows
            ic50_values     (n_pairs,) float32
            pair_cell_ids   (n_pairs,) str — ModelID for each pair (for per-cell-line metrics)
        """
        expr_path = Path(hydra.utils.to_absolute_path(
            str(getattr(self.task_cfg, "expression_data_path", ""))
        ))
        ic50_path = Path(hydra.utils.to_absolute_path(
            str(getattr(self.task_cfg, "ic50_data_path", ""))
        ))
        drug_feat_path = Path(hydra.utils.to_absolute_path(
            str(getattr(self.task_cfg, "drug_features_path", ""))
        ))
        gene_info_path = Path(hydra.utils.to_absolute_path(
            str(getattr(self.task_cfg, "gene_info_path",
                        str(ROOT / "scbFM" / "data" / "bulkformer_gene_info.csv")))
        ))

        paths_to_check = [
            (expr_path, "expression_data_path"),
            (ic50_path, "ic50_data_path"),
            (drug_feat_path, "drug_features_path"),
        ]
        # gene_info only needed for CSV expression input
        if expr_path.suffix != ".h5ad":
            paths_to_check.append((gene_info_path, "gene_info_path"))
        for p, name in paths_to_check:
            if not p.exists():
                raise FileNotFoundError(f"{name} not found: {p}")

        # ── Expression: h5ad or CSV ──────────────────────────────────────
        if expr_path.suffix == ".h5ad":
            log.info("Loading expression h5ad from %s", expr_path)
            adata = ad.read_h5ad(expr_path)
            adata.obs_names = adata.obs_names.astype(str)
            log.info(
                "Expression h5ad: %d cell lines, %d genes",
                adata.n_obs, adata.n_vars,
            )
        else:
            log.info("Loading gene symbol→ENSG mapping from %s", gene_info_path)
            gene_info = pd.read_csv(gene_info_path, usecols=["gene_symbol", "ensg_id"])
            symbol_to_ensg: dict[str, str] = dict(
                zip(gene_info["gene_symbol"].astype(str), gene_info["ensg_id"].astype(str))
            )

            log.info("Loading expression CSV from %s", expr_path)
            expr_df = pd.read_csv(expr_path, index_col=0)
            expr_df.index = expr_df.index.astype(str)

            symbol_cols = [c for c in expr_df.columns if c in symbol_to_ensg]
            ensg_ids = [symbol_to_ensg[c] for c in symbol_cols]
            expr_df = expr_df[symbol_cols].copy()
            expr_df.columns = ensg_ids
            expr_df = expr_df.loc[:, ~expr_df.columns.duplicated(keep="first")]
            log.info(
                "Expression CSV: %d cell lines, %d genes after symbol→ENSG mapping",
                len(expr_df), len(expr_df.columns),
            )

            adata = ad.AnnData(X=expr_df.values.astype(np.float32))
            adata.obs_names = list(expr_df.index)
            adata.var_names = list(expr_df.columns)

        gene_list_path = self._resolve_gene_list_path()
        should_preprocess = bool(getattr(self.task_cfg, "preprocess", True))
        adata, missing_genes = reindex_adata_genes(
            adata,
            gene_list_path=gene_list_path,
        )
        if should_preprocess:
            log.info(
                "Aligned raw expression for on-the-fly sequence binning: "
                "%d target genes missing, shape %s",
                len(missing_genes), adata.shape,
            )
        else:
            log.info(
                "Expression reindexed (no preprocessing): %d genes missing, shape %s",
                len(missing_genes), adata.shape,
            )

        self._missing_genes_note = (
            f"Model genes missing from GDSC GEX and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
        ) if missing_genes else ""

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes after preprocessing, got {adata.n_vars}."
            )
        if not should_preprocess:
            validate_token_matrix(
                adata.X,
                bin_num=int(self.model_cfg.bin_num),
                name="drug response expression input",
            )

        cell_id_to_row: dict[str, int] = {
            str(cid): i for i, cid in enumerate(adata.obs_names)
        }
        X_cell = adata.X  # (n_cells, gene_num), sparse or dense

        # ── Drug features ─────────────────────────────────────────────────
        log.info("Loading drug features from %s", drug_feat_path)
        drug_data = np.load(drug_feat_path, allow_pickle=True)
        drug_ids_arr = drug_data["drug_ids"].astype(str)
        drug_emb_matrix = drug_data["features"].astype(np.float32)  # (n_drugs, drug_emb_dim)
        drug_id_to_emb_idx: dict[str, int] = {did: i for i, did in enumerate(drug_ids_arr)}
        log.info(
            "Drug embeddings: %d drugs, emb_dim=%d", len(drug_ids_arr), drug_emb_matrix.shape[1]
        )

        # ── IC50 CSV ─────────────────────────────────────────────────────
        log.info("Loading IC50 CSV from %s", ic50_path)
        model_id_col = str(getattr(self.task_cfg, "model_id_col", "ModelID"))
        drug_id_col = str(getattr(self.task_cfg, "drug_id_col", "Drug ID"))
        ic50_col = str(getattr(self.task_cfg, "ic50_col", "IC50"))
        dataset_col = str(getattr(self.task_cfg, "dataset_col", "Dataset Version"))

        ic50_df = pd.read_csv(ic50_path)
        ic50_df[model_id_col] = ic50_df[model_id_col].astype(str)
        ic50_df[drug_id_col] = ic50_df[drug_id_col].astype(str)

        # Prefer GDSC2 over GDSC1 for duplicate (cell_line, drug) pairs
        ic50_df = ic50_df.sort_values(
            dataset_col,
            key=lambda s: s.map({"GDSC1": 0, "GDSC2": 1}).fillna(0),
            ascending=True,
        )
        ic50_df = ic50_df.drop_duplicates(subset=[model_id_col, drug_id_col], keep="last")
        ic50_df = ic50_df.dropna(subset=[ic50_col]).reset_index(drop=True)

        # Keep pairs with both expression data and drug embeddings
        mask = (
            ic50_df[model_id_col].isin(cell_id_to_row)
            & ic50_df[drug_id_col].isin(drug_id_to_emb_idx)
        )
        ic50_df = ic50_df[mask].reset_index(drop=True)
        log.info("GDSC pairs after filtering: %d", len(ic50_df))

        if len(ic50_df) == 0:
            raise ValueError(
                "No valid (cell_line, drug) pairs found after filtering. "
                "Check that ModelIDs in IC50 CSV match expression CSV, "
                "and Drug IDs match drug_features.npz."
            )

        cell_idxs = np.array(
            [cell_id_to_row[cid] for cid in ic50_df[model_id_col]], dtype=np.int64
        )
        drug_idxs = np.array(
            [drug_id_to_emb_idx[did] for did in ic50_df[drug_id_col]], dtype=np.int64
        )
        ic50_values = ic50_df[ic50_col].values.astype(np.float32)
        pair_cell_ids = ic50_df[model_id_col].values

        return X_cell, drug_emb_matrix, cell_idxs, drug_idxs, ic50_values, pair_cell_ids

    def _select_training_hvg_indices(
        self,
        X_cell,
        training_cell_idxs: np.ndarray,
    ) -> np.ndarray:
        """Fit the fixed sequence vocabulary on training-fold cell lines only."""
        unique_training_cells = np.unique(training_cell_idxs).astype(np.int64)
        if unique_training_cells.size == 0:
            raise ValueError("The drug-response training fold contains no cell lines.")

        hvg_adata = ad.AnnData(X=X_cell[unique_training_cells])
        hvg_stats = sc.pp.highly_variable_genes(
            hvg_adata,
            n_top_genes=self.selected_gene_count,
            flavor=str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
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
            "Selected %d HVGs from %d training-fold cell lines with flavor=%s",
            selected.size,
            unique_training_cells.size,
            str(getattr(self.task_cfg, "hvg_flavor", "cell_ranger")),
        )
        return selected.astype(np.int64, copy=False)

    def _cell_backbone_batch(self, X_cell, cell_indices: np.ndarray) -> dict[str, torch.Tensor]:
        if self.fold_gene_indices is None:
            raise RuntimeError("Training-fold HVGs have not been selected.")
        expressions: list[torch.Tensor] = []
        for cell_idx in np.asarray(cell_indices, dtype=np.int64):
            row = X_cell[int(cell_idx)]
            values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
            selected_values = values[self.fold_gene_indices].astype(np.float32, copy=False)
            if bool(getattr(self.task_cfg, "preprocess", True)):
                rng = np.random.default_rng(
                    int(getattr(self.task_cfg, "random_seed", 42)) + int(cell_idx)
                )
                selected_values = _quantile_bin_expression(
                    selected_values,
                    int(self.model_cfg.bin_num),
                    rng,
                ).astype(np.float32)
            expression = torch.from_numpy(selected_values)
            expressions.append(
                torch.cat((torch.tensor([self.cls_value]), expression))
            )

        gene_ids = torch.from_numpy(
            self.fold_gene_indices.astype(np.int64, copy=False)
            + self.gene_token_offset
        )
        gene_ids = torch.cat((torch.tensor([self.cls_gene_id]), gene_ids))
        return {
            "gene_ids": gene_ids.unsqueeze(0).expand(len(expressions), -1),
            "expr": torch.stack(expressions),
        }

    # ------------------------------------------------------------------
    # CV splits
    # ------------------------------------------------------------------

    def _build_cv_splits(self, n_pairs: int) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 5))
        if n_splits < 2:
            raise ValueError("finetune.drug_resp.cv_folds must be at least 2.")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        return list(splitter.split(np.arange(n_pairs)))

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _build_cell_centric_train(
        self,
        X_cell: np.ndarray,
        drug_emb_matrix: np.ndarray,
        cell_idxs_train: np.ndarray,
        drug_idxs_train: np.ndarray,
        ic50_train: np.ndarray,
    ) -> None:
        """Index training pairs by cell line for cell-centric training."""
        from collections import defaultdict
        mapping: dict[int, list[int]] = defaultdict(list)
        for i, ci in enumerate(cell_idxs_train):
            mapping[int(ci)].append(i)
        self.cell_to_train_pairs = dict(mapping)
        self.train_drug_idxs = drug_idxs_train
        self.train_ic50 = ic50_train
        self.X_cell_ref = X_cell
        self.drug_emb_matrix_ref = drug_emb_matrix

    def _build_test_loader(
        self,
        X_cell: np.ndarray,
        drug_emb_matrix: np.ndarray,
        cell_idxs_test: np.ndarray,
        drug_idxs_test: np.ndarray,
        ic50_test: np.ndarray,
    ) -> None:
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        num_workers = int(getattr(self.task_cfg, "num_workers", 0))
        if num_workers < 0:
            raise ValueError("finetune.drug_resp.num_workers must be non-negative.")
        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            prefetch_factor = int(getattr(self.task_cfg, "prefetch_factor", 2))
            if prefetch_factor <= 0:
                raise ValueError("finetune.drug_resp.prefetch_factor must be positive.")
            loader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": False,
                }
            )
        if self.fold_gene_indices is None:
            raise RuntimeError("Training-fold HVGs have not been selected.")
        test_dataset = DrugRespDataset(
            X_cell, drug_emb_matrix, cell_idxs_test, drug_idxs_test, ic50_test,
            fixed_gene_indices=self.fold_gene_indices,
            bin_num=int(self.model_cfg.bin_num),
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=self.cls_value,
            seed=int(getattr(self.task_cfg, "random_seed", 42)),
            do_binning=bool(getattr(self.task_cfg, "preprocess", True)),
        )
        self.test_dataset_size = len(test_dataset)
        if self.is_distributed:
            test_sampler = SequentialDistributedSampler(
                test_dataset, batch_size=batch_size, world_size=self.world_size, rank=self.rank,
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

    def _build_model(self, checkpoint_path: str, drug_emb_dim: int) -> None:
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
            resolved = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved, map_location="cpu")
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            backbone.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved)
        else:
            log.info("Using randomly initialized backbone")

        cell_emb_dim = int(self.model_cfg.embsize)
        head = DrugRespPredHead(
            cell_emb_dim=cell_emb_dim,
            drug_emb_dim=drug_emb_dim,
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 512)),
            bottleneck_dim=int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
        )

        if finetune_mode == "adapters":
            backbone.add_adapters(
                bottleneck_dim=int(getattr(self.task_cfg, "adapter_bottleneck_dim", 32)),
                dropout=float(getattr(self.task_cfg, "adapter_dropout", 0.0)),
                after_attention=bool(getattr(self.task_cfg, "adapter_after_attention", True)),
                after_ff=bool(getattr(self.task_cfg, "adapter_after_ff", True)),
            )

        model = DrugRespModel(
            backbone,
            head,
        )
        if self.is_master:
            log.info(
                "Drug response cell representation: CLS token | fusion head dims: %d -> %d -> %d -> 1",
                cell_emb_dim + drug_emb_dim,
                int(getattr(self.task_cfg, "head_hidden_dim", 512)),
                int(getattr(self.task_cfg, "head_bottleneck_dim", 256)),
            )

        if finetune_mode == "head_only":
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
            self.backbone_optimizer_enabled = False
        elif finetune_mode == "adapters":
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.backbone.adapter_parameters():
                param.requires_grad = True
            for param in model.head.parameters():
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
    # Optimization
    # ------------------------------------------------------------------

    def _build_optimization(self) -> None:
        head_lr = float(self.task_cfg.head_learning_rate)
        backbone_lr = float(self.task_cfg.backbone_learning_rate)
        adapter_lr = float(getattr(self.task_cfg, "adapter_learning_rate", head_lr))

        raw = self.model.module if isinstance(self.model, DDP) else self.model
        head_params = [p for p in raw.head.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head_params}
        adapter_params = [p for p in raw.backbone.adapter_parameters() if p.requires_grad]
        adapter_ids = {id(p) for p in adapter_params}
        backbone_params = [
            p for p in raw.backbone.parameters()
            if p.requires_grad and id(p) not in head_ids and id(p) not in adapter_ids
        ]

        param_groups = []
        if backbone_params and self.backbone_optimizer_enabled:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": adapter_lr, "name": "adapters"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
        if not param_groups:
            raise ValueError("No trainable parameters found for drug response.")

        max_lrs = [float(g["lr"]) for g in param_groups]
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = float(configured_ratio) if configured_ratio is not None else min_lr / max(head_lr, 1e-12)

        self.optimizer = Adam(param_groups)
        self.scheduler = GroupedCosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 20)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 1)),
            max_lrs=max_lrs,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 2)),
            gamma=float(getattr(self.task_cfg, "gamma", 1.0)),
        )

    def _maybe_enable_backbone_optimizer(self, epoch: int) -> None:
        burn_in = int(getattr(self.task_cfg, "burn_in_epochs", 3))
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
            log.info("Finished %d burn-in epochs; enabled backbone optimization.", burn_in)

    def _optimizer_parameters(self) -> list[nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _precompute_cell_embs(self, X_cell: np.ndarray) -> None:
        """Precompute frozen backbone embeddings for all cell lines once per fold (head_only only)."""
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        raw.backbone.eval()
        n_cells = X_cell.shape[0]
        infer_batch = int(getattr(self.task_cfg, "batch_size", 4)) * int(
            getattr(self.task_cfg, "grad_accumulation_steps", 4)
        )
        infer_batch = max(1, infer_batch)
        all_embs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n_cells, infer_batch):
                batch = self._cell_backbone_batch(
                    X_cell,
                    np.arange(start, min(start + infer_batch, n_cells)),
                )
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                hidden = raw.backbone(batch["gene_ids"], batch["expr"])
                all_embs.append(raw.pool_cell_embeddings(hidden).cpu())
        self.cell_emb_cache = torch.cat(all_embs, dim=0).to(self.device)  # (n_cells, dim)
        if self.is_master:
            log.info(
                "Precomputed cell embeddings: representation=CLS token, shape=%s",
                tuple(self.cell_emb_cache.shape),
            )

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_enable_backbone_optimizer(epoch)
        if self.is_distributed:
            dist.barrier()

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))

        # Deterministic per-epoch shuffle identical across all ranks
        all_cell_idxs = list(self.cell_to_train_pairs.keys())
        rng = np.random.default_rng(
            int(getattr(self.task_cfg, "random_seed", 42)) + epoch * 7919
        )
        rng.shuffle(all_cell_idxs)

        # Pad to a multiple of world_size so every rank gets the same step count
        n_cells = len(all_cell_idxs)
        n_per_rank = math.ceil(n_cells / self.world_size)
        if n_cells < self.world_size * n_per_rank:
            all_cell_idxs = all_cell_idxs + all_cell_idxs[: self.world_size * n_per_rank - n_cells]
        rank_cells = all_cell_idxs[self.rank :: self.world_size]

        running_loss = 0.0

        for cell_idx in rank_cells:
            pair_indices = np.array(self.cell_to_train_pairs[cell_idx])

            drug_embs_t = torch.as_tensor(
                self.drug_emb_matrix_ref[self.train_drug_idxs[pair_indices]], dtype=torch.float32
            ).to(self.device, non_blocking=True)
            targets_t = torch.as_tensor(
                self.train_ic50[pair_indices], dtype=torch.float32
            ).to(self.device, non_blocking=True)

            if self.cell_emb_cache is not None:
                # head_only: frozen embedding cached once per fold, shape (1, dim)
                cell_emb = self.cell_emb_cache[cell_idx].unsqueeze(0)
                batch_arg = None
            else:
                # adapters / full_ft: run backbone once for this cell
                batch_arg = self._cell_backbone_batch(
                    self.X_cell_ref,
                    np.asarray([cell_idx]),
                )
                batch_arg = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch_arg.items()
                }
                cell_emb = None

            preds = self.model(batch_arg, drug_embs_t, cell_emb)  # (n_pairs,)
            loss = F.mse_loss(preds, targets_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), max_grad_norm)
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)

            running_loss += loss.item()

        epoch_loss = running_loss / len(rank_cells)
        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)
        self.scheduler.step()
        return {"loss": epoch_loss}

    def _evaluate(self, test_pair_cell_ids: np.ndarray) -> dict:
        self.model.eval()
        running_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for cell_idxs_b, batch, drug_emb, targets in self.test_loader:
                drug_emb = drug_emb.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.cell_emb_cache is not None:
                    cell_emb = self.cell_emb_cache[cell_idxs_b.to(self.device)]
                    preds = self.model(None, drug_emb, cell_emb)    # (B,)
                else:
                    batch = {
                        key: value.to(self.device, non_blocking=True)
                        for key, value in batch.items()
                    }
                    preds = self.model(batch, drug_emb)    # (B,)
                loss = F.mse_loss(preds, targets)
                running_loss += loss.item()
                all_preds.append(preds)
                all_targets.append(targets)

        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)

        if self.is_distributed:
            all_preds_t = distributed_concat(all_preds_t, self.test_dataset_size, self.world_size)
            all_targets_t = distributed_concat(all_targets_t, self.test_dataset_size, self.world_size)

        preds_np = all_preds_t.cpu().float().numpy()    # (n_test_pairs,)
        targets_np = all_targets_t.cpu().float().numpy()

        # Global PCC and SCC across all pairs
        finite = np.isfinite(preds_np) & np.isfinite(targets_np)
        if finite.sum() >= 2:
            global_pcc, _ = pearsonr(preds_np[finite], targets_np[finite])
            global_scc, _ = spearmanr(preds_np[finite], targets_np[finite])
        else:
            global_pcc = global_scc = float("nan")

        # Per-cell-line PCC and SCC (average across drugs for each cell line)
        cell_pccs: dict[str, float] = {}
        cell_sccs: dict[str, float] = {}
        for cell_id in np.unique(test_pair_cell_ids):
            mask = test_pair_cell_ids == cell_id
            p, t = preds_np[mask], targets_np[mask]
            fm = np.isfinite(p) & np.isfinite(t)
            if fm.sum() < 2:
                continue
            pcc, _ = pearsonr(p[fm], t[fm])
            scc, _ = spearmanr(p[fm], t[fm])
            cell_pccs[cell_id] = float(pcc) if np.isfinite(pcc) else float("nan")
            cell_sccs[cell_id] = float(scc) if np.isfinite(scc) else float("nan")

        mean_pcc = float(np.nanmean(list(cell_pccs.values()))) if cell_pccs else float("nan")
        mean_scc = float(np.nanmean(list(cell_sccs.values()))) if cell_sccs else float("nan")

        test_loss = running_loss / max(1, len(self.test_loader))
        if self.is_distributed:
            test_loss = get_reduced(test_loss, self.local_rank, 0, self.world_size)

        return {
            "loss": float(test_loss),
            "global_pcc": float(global_pcc) if np.isfinite(global_pcc) else float("nan"),
            "global_scc": float(global_scc) if np.isfinite(global_scc) else float("nan"),
            "mean_pcc_per_cell": mean_pcc,
            "mean_scc_per_cell": mean_scc,
            "cell_pccs": cell_pccs,
            "cell_sccs": cell_sccs,
            "n_test_pairs": int(preds_np.shape[0]),
            "n_test_cell_lines": len(cell_pccs),
        }

    # ------------------------------------------------------------------
    # Results writing
    # ------------------------------------------------------------------

    def _flatten_fold_metrics(
        self,
        model_key: str,
        fold: int,
        n_folds: int,
        checkpoint_path: str,
        train_metrics: dict,
        test_metrics: dict,
    ) -> dict:
        return {
            "model": model_key,
            "fold": fold,
            "n_folds": n_folds,
            "finetune_mode": self._finetune_mode(),
            "checkpoint_path": checkpoint_path,
            "train_loss": float(train_metrics["loss"]),
            "test_loss": float(test_metrics["loss"]),
            "global_pcc": float(test_metrics["global_pcc"]),
            "global_scc": float(test_metrics["global_scc"]),
            "mean_pcc_per_cell": float(test_metrics["mean_pcc_per_cell"]),
            "mean_scc_per_cell": float(test_metrics["mean_scc_per_cell"]),
            "n_test_pairs": int(test_metrics["n_test_pairs"]),
            "n_test_cell_lines": int(test_metrics["n_test_cell_lines"]),
        }

    def _per_cell_line_rows(
        self,
        model_key: str,
        fold: int,
        checkpoint_path: str,
        test_metrics: dict,
    ) -> list[dict]:
        rows = []
        for cell_id in sorted(test_metrics["cell_pccs"]):
            rows.append({
                "model": model_key,
                "fold": fold,
                "finetune_mode": self._finetune_mode(),
                "checkpoint_path": checkpoint_path,
                "cell_line_id": cell_id,
                "pcc": test_metrics["cell_pccs"][cell_id],
                "scc": test_metrics["cell_sccs"].get(cell_id, float("nan")),
            })
        return rows

    def _write_model_results(
        self,
        checkpoint_path: str,
        fold_rows: list[dict],
        cell_line_rows: list[dict],
    ) -> dict:
        aggregate = self._aggregate_numeric_rows(fold_rows)
        aggregate.update({
            "model": fold_rows[0]["model"],
            "finetune_mode": fold_rows[0]["finetune_mode"],
            "checkpoint_path": checkpoint_path,
        })
        out_dir = self._task_output_dir()
        model_key = str(fold_rows[0]["model"])
        prefix = self._output_prefix()
        self._write_csv(out_dir / f"{prefix}_{model_key}_fold_metrics.csv", fold_rows)
        self._write_csv(out_dir / f"{prefix}_{model_key}_evaluation_metrics.csv", [aggregate], comment=getattr(self, "_missing_genes_note", ""))
        self._write_csv(out_dir / f"{prefix}_{model_key}_cell_line_metrics.csv", cell_line_rows)
        return aggregate

    def _cleanup_fold_state(self) -> None:
        self.test_loader = None
        self.cell_to_train_pairs = None
        self.train_drug_idxs = None
        self.train_ic50 = None
        self.X_cell_ref = None
        self.drug_emb_matrix_ref = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.cell_emb_cache = None
        self.fold_gene_indices = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        try:
            self._setup_runtime()

            X_cell, drug_emb_matrix, cell_idxs, drug_idxs, ic50_values, pair_cell_ids = (
                self._load_gdsc_data()
            )
            drug_emb_dim = drug_emb_matrix.shape[1]
            n_pairs = len(ic50_values)
            splits = self._build_cv_splits(n_pairs)
            fold_hvg_indices = [
                self._select_training_hvg_indices(X_cell, cell_idxs[train_idx])
                for train_idx, _ in splits
            ]
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            if self.is_master:
                log.info(
                    "Drug response CV: pairs=%d, cell_lines=%d, drug_emb_dim=%d, folds=%d",
                    n_pairs,
                    len(np.unique(cell_idxs)),
                    drug_emb_dim,
                    len(splits),
                )

            epochs = int(getattr(self.task_cfg, "epochs", 20))
            aggregate_rows: list[dict] = []

            for model_key, checkpoint_path in checkpoint_paths.items():
                fold_rows: list[dict] = []
                cell_line_rows: list[dict] = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + self.rank
                        + fold_idx
                    )

                    self.fold_gene_indices = fold_hvg_indices[fold_idx - 1]

                    test_pair_cell_ids = pair_cell_ids[test_idx]

                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | train=%d pairs, test=%d pairs",
                            model_key, fold_idx, len(splits), len(train_idx), len(test_idx),
                        )

                    self._build_cell_centric_train(
                        X_cell, drug_emb_matrix,
                        cell_idxs[train_idx], drug_idxs[train_idx], ic50_values[train_idx],
                    )
                    self._build_test_loader(
                        X_cell, drug_emb_matrix,
                        cell_idxs[test_idx], drug_idxs[test_idx], ic50_values[test_idx],
                    )
                    self._build_model(checkpoint_path, drug_emb_dim)
                    if self._finetune_mode() == "head_only":
                        self._precompute_cell_embs(X_cell)
                    self._build_optimization()

                    last_train_metrics = {"loss": float("nan")}
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        if self.is_master:
                            log.info(
                                "Model %s | Fold %d/%d | Epoch %d | Train Loss: %.6f",
                                model_key, fold_idx, len(splits), epoch, last_train_metrics["loss"],
                            )

                    test_metrics = self._evaluate(test_pair_cell_ids)
                    if self.is_master:
                        log.info(
                            "Model %s | Fold %d/%d | "
                            "Test Loss: %.4f | Global PCC: %.4f | Mean PCC/cell: %.4f",
                            model_key, fold_idx, len(splits),
                            test_metrics["loss"],
                            test_metrics["global_pcc"],
                            test_metrics["mean_pcc_per_cell"],
                        )
                        fold_rows.append(self._flatten_fold_metrics(
                            model_key=model_key, fold=fold_idx, n_folds=len(splits),
                            checkpoint_path=checkpoint_path,
                            train_metrics=last_train_metrics, test_metrics=test_metrics,
                        ))
                        cell_line_rows.extend(self._per_cell_line_rows(
                            model_key=model_key, fold=fold_idx,
                            checkpoint_path=checkpoint_path, test_metrics=test_metrics,
                        ))
                    self._cleanup_fold_state()

                if self.is_master:
                    aggregate_rows.append(
                        self._write_model_results(checkpoint_path, fold_rows, cell_line_rows)
                    )
                if self.is_distributed:
                    dist.barrier()

            if self.is_master:
                out_dir = self._task_output_dir()
                output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
                self._write_csv(output_path, aggregate_rows, comment=getattr(self, "_missing_genes_note", ""))
                return {"results_path": str(output_path), "results": aggregate_rows}
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
