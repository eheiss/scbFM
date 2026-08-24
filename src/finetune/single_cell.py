from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.canc_type_class.runner import _quantile_bin_expression
from finetune.training_correctness import validate_backbone_checkpoint
from preprocess import read_gene_list
from utils import SequentialDistributedSampler, distributed_concat, seed_all

log = logging.getLogger(__name__)

SCBFM_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = SCBFM_ROOT.parent

CHECKPOINT_MODEL_KEYS = (
    "pretrain_sc",
    "pretrain_bulk",
    "preadapt_sc",
    "preadapt_bulk",
)
RANDOM_INIT_MODEL_KEY = "random_init"
SCBFM_MODEL_KEYS = (*CHECKPOINT_MODEL_KEYS, RANDOM_INIT_MODEL_KEY)
EXTERNAL_SCGPT_MODEL_KEYS = ("scgpt", "scgpt_preadapt")
RAW_PCA_MODEL_KEY = "raw_pca"
DISPLAY_NAMES = {
    "random_init": "Random init",
    "pretrain_sc": "Pretrain sc",
    "preadapt_sc": "Preadapt sc",
    "pretrain_bulk": "Pretrain bulk",
    "preadapt_bulk": "Preadapt bulk",
    "scgpt": "scGPT",
    "scgpt_preadapt": "Preadapted scGPT",
    "raw_pca": "Raw PCA",
}

SINGLE_CELL_BENCHMARK_MODEL_ORDER = (
    "random_init",
    "pretrain_sc",
    "preadapt_sc",
    "pretrain_bulk",
    "preadapt_bulk",
    "raw_pca",
    "scgpt",
    "scgpt_preadapt",
)


def plot_single_cell_benchmark_summary(
    path: Path,
    rows: list[dict[str, object]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot one cross-dataset primary score in the standard benchmark layout."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    frame = pd.DataFrame(rows)
    summary = frame[frame["dataset"] == "macro_average"].set_index("model")
    models = [
        model for model in SINGLE_CELL_BENCHMARK_MODEL_ORDER if model in summary.index
    ]
    if not models:
        raise ValueError("No macro-average single-cell results are available to plot.")
    if metric not in summary.columns:
        raise ValueError(f"Single-cell summary does not contain metric '{metric}'.")

    positions = {
        "random_init": 0.0,
        "pretrain_sc": 1.1,
        "preadapt_sc": 1.9,
        "pretrain_bulk": 3.0,
        "preadapt_bulk": 3.8,
        "raw_pca": 5.0,
        "scgpt": 6.1,
        "scgpt_preadapt": 6.9,
    }
    colors = {
        model: "#4C72B0"
        for model in (
            "random_init",
            "pretrain_sc",
            "preadapt_sc",
            "pretrain_bulk",
            "preadapt_bulk",
        )
    }
    colors.update(
        {
            "raw_pca": "#8C8C8C",
            "scgpt": "#8172B3",
            "scgpt_preadapt": "#8172B3",
        }
    )

    figure, axis = plt.subplots(figsize=(12.5, 5.3))
    x = np.asarray([positions[model] for model in models])
    values = summary.loc[models, metric].astype(float).to_numpy()
    axis.bar(
        x,
        values,
        width=0.58,
        color=[colors[model] for model in models],
        alpha=0.88,
    )

    group_specs = (
        (("pretrain_sc", "preadapt_sc"), "sc pre-trained", "steelblue", 0.07),
        (("pretrain_bulk", "preadapt_bulk"), "bulk pre-trained", "salmon", 0.07),
        (("raw_pca",), "raw baseline", "darkgray", 0.06),
        (("scgpt", "scgpt_preadapt"), "external scGPT", "#8172B3", 0.07),
    )
    y_max = 1.16
    for group_models, label, color, alpha in group_specs:
        group_x = [positions[model] for model in group_models if model in models]
        if not group_x:
            continue
        lower, upper = min(group_x) - 0.43, max(group_x) + 0.43
        axis.axvspan(lower, upper, color=color, alpha=alpha, zorder=0)
        axis.text(
            (lower + upper) / 2,
            y_max - 0.015,
            label,
            ha="center",
            va="top",
            fontsize=9,
            color={
                "bulk pre-trained": "firebrick",
                "raw baseline": "dimgray",
            }.get(label, color),
            style="italic",
        )

    axis.set_xticks(
        x,
        [DISPLAY_NAMES.get(model, model).replace(" ", "\n", 1) for model in models],
    )
    axis.set_ylim(0, y_max)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.set_title(title, fontsize=12)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor="#4C72B0", alpha=0.88, label="scbFM"),
            Patch(facecolor="#8C8C8C", alpha=0.88, label="Raw PCA"),
            Patch(facecolor="#8172B3", alpha=0.88, label="scGPT"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.9,
    )
    figure.tight_layout(rect=(0, 0, 0.86, 1))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    plt.close(figure)


@dataclass
class SingleCellDatasetBundle:
    key: str
    display_name: str
    reference: ad.AnnData
    query: ad.AnnData
    combined: ad.AnnData
    model_gene_indices: np.ndarray
    manifest: pd.DataFrame
    label_key: str
    batch_key: str
    mapping_stats: dict[str, object]


class _SingleCellExpressionDataset(Dataset):
    def __init__(
        self,
        matrix,
        *,
        model_gene_indices: np.ndarray,
        bin_num: int,
        cls_gene_id: int,
        gene_token_offset: int,
        cls_value: float,
        seed: int,
    ) -> None:
        self.matrix = matrix
        self.model_gene_indices = np.asarray(model_gene_indices, dtype=np.int64)
        self.bin_num = int(bin_num)
        self.cls_gene_id = int(cls_gene_id)
        self.gene_token_offset = int(gene_token_offset)
        self.cls_value = float(cls_value)
        self.seed = int(seed)
        if self.matrix.shape[1] != self.model_gene_indices.size:
            raise ValueError(
                "Expression columns and model gene indices must have equal length."
            )

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.matrix[index]
        values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        rng = np.random.default_rng(self.seed + index)
        binned = _quantile_bin_expression(values, self.bin_num, rng).astype(np.float32)
        gene_ids = self.model_gene_indices + self.gene_token_offset
        return {
            "gene_ids": torch.as_tensor(
                np.concatenate(([self.cls_gene_id], gene_ids)), dtype=torch.long
            ),
            "expr": torch.as_tensor(
                np.concatenate(([self.cls_value], binned)), dtype=torch.float32
            ),
        }


class FrozenSingleCellBenchmark:
    """Shared frozen-embedding protocol for the scGPT single-cell benchmarks."""

    def __init__(self, cfg: DictConfig, task_cfg: DictConfig) -> None:
        self.cfg = cfg
        self.task_cfg = task_cfg
        self.model_cfg = cfg.pretrain
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        self.cls_gene_id = 0
        self.pad_gene_id = 1
        self.gene_token_offset = 2
        self.selected_gene_count = int(
            getattr(self.model_cfg, "selected_gene_count", 1199)
        )
        self.max_seq_len = int(
            getattr(self.model_cfg, "max_seq_len", self.selected_gene_count + 1)
        )
        if self.max_seq_len != self.selected_gene_count + 1:
            raise ValueError(
                "Single-cell evaluation expects max_seq_len to equal "
                "selected_gene_count + 1."
            )
        self.mapping_stats: dict[str, dict[str, object]] = {}
        self.external_scgpt_load_reports: dict[str, dict[str, object]] = {}

    def setup_runtime(self) -> None:
        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
            timeout_minutes = int(
                getattr(self.task_cfg, "distributed_timeout_minutes", 360)
            )
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(minutes=timeout_minutes),
            )
        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + self.rank)

    def close_runtime(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    @staticmethod
    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            raise ValueError(f"Refusing to write an empty result table to {path}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _required_path(value: object, description: str) -> Path:
        if value is None or not str(value).strip():
            raise ValueError(f"Missing required path for {description}.")
        path = Path(hydra.utils.to_absolute_path(str(value)))
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
        return path

    def dataset_keys(self) -> list[str]:
        datasets_cfg = getattr(self.task_cfg, "datasets", None)
        if datasets_cfg is None:
            raise ValueError("Single-cell task config must define datasets.")
        configured = getattr(self.task_cfg, "dataset_keys", None)
        keys = list(datasets_cfg.keys()) if configured is None else [str(key) for key in configured]
        if not keys:
            raise ValueError("dataset_keys may not be empty.")
        invalid = sorted(set(keys).difference(datasets_cfg.keys()))
        if invalid:
            raise ValueError(f"Unknown single-cell dataset keys: {invalid}")
        return keys

    def checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError("pretrained_model_paths must be configured.")
        configured = getattr(self.task_cfg, "model_keys", None)
        keys = list(SCBFM_MODEL_KEYS) if configured is None else [str(key) for key in configured]
        if not keys:
            raise ValueError("model_keys may not be empty.")
        invalid = sorted(set(keys).difference(SCBFM_MODEL_KEYS))
        if invalid:
            raise ValueError(
                f"Unsupported single-cell scbFM model keys {invalid}; "
                f"supported={list(SCBFM_MODEL_KEYS)}."
            )
        paths: dict[str, str] = {}
        for key in keys:
            if key == RANDOM_INIT_MODEL_KEY:
                paths[key] = ""
                continue
            value = paths_cfg.get(key)
            path = self._required_path(value, f"{key} checkpoint")
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint is not a file: {path}")
            paths[key] = str(path)
        return paths

    @staticmethod
    def _strip_ensembl_version(values: pd.Series) -> pd.Series:
        return values.astype("string").str.replace(r"\.\d+$", "", regex=True)

    @staticmethod
    def _validate_obs(
        adata: ad.AnnData,
        *,
        path: Path,
        label_key: str,
        batch_key: str,
    ) -> None:
        missing = sorted({label_key, batch_key}.difference(adata.obs.columns))
        if missing:
            raise ValueError(f"{path} is missing required obs columns: {missing}")
        if adata.obs[[label_key, batch_key]].isna().any().any():
            raise ValueError(f"{path} contains missing cell-type or batch labels.")
        adata.obs[label_key] = adata.obs[label_key].astype(str)
        adata.obs[batch_key] = adata.obs[batch_key].astype(str)

    @staticmethod
    def _preprocess_expression(adata: ad.AnnData, mode: str) -> ad.AnnData:
        if mode == "none":
            return adata
        if mode != "normalize_log1p":
            raise ValueError(
                f"Unsupported single-cell preprocessing mode '{mode}'; "
                "expected one of: none, normalize_log1p."
            )
        adata = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    def _gene_mapping_table(
        self,
    ) -> tuple[list[str], dict[str, str], dict[str, str], int]:
        gene_list_path = self._required_path(
            getattr(self.task_cfg, "gene_list_path", None), "scbFM gene list"
        )
        gene_info_path = self._required_path(
            getattr(self.task_cfg, "gene_info_path", None), "gene symbol mapping"
        )
        target_genes = read_gene_list(gene_list_path)
        expected_gene_num = int(self.model_cfg.gene_num)
        if len(target_genes) != expected_gene_num:
            raise ValueError(
                f"Model gene list has {len(target_genes)} genes; expected {expected_gene_num}."
            )
        target_set = set(target_genes)
        gene_info = pd.read_csv(gene_info_path)
        missing = sorted({"gene_symbol", "ensg_id"}.difference(gene_info.columns))
        if missing:
            raise ValueError(f"Gene mapping is missing columns: {missing}")
        gene_info = gene_info.dropna(subset=["gene_symbol", "ensg_id"]).copy()
        gene_info["gene_symbol"] = gene_info["gene_symbol"].astype(str)
        gene_info["ensg_id"] = self._strip_ensembl_version(gene_info["ensg_id"])
        gene_info = gene_info[gene_info["ensg_id"].isin(target_set)]
        counts = gene_info.groupby("gene_symbol")["ensg_id"].nunique()
        ambiguous = set(counts[counts > 1].index.astype(str))
        unambiguous = gene_info[~gene_info["gene_symbol"].isin(ambiguous)]
        symbol_to_ensg = (
            unambiguous.drop_duplicates("gene_symbol", keep="first")
            .set_index("gene_symbol")["ensg_id"]
            .astype(str)
            .to_dict()
        )
        ensg_to_symbol = (
            gene_info.drop_duplicates("ensg_id", keep="first")
            .set_index("ensg_id")["gene_symbol"]
            .astype(str)
            .to_dict()
        )
        return target_genes, symbol_to_ensg, ensg_to_symbol, len(ambiguous)

    def _map_to_model_vocabulary(
        self,
        adata: ad.AnnData,
        *,
        gene_key: str,
        target_genes: list[str],
        symbol_to_ensg: dict[str, str],
        ambiguous_symbol_count: int,
        split: str,
    ) -> tuple[ad.AnnData, dict[str, object]]:
        if gene_key == "index":
            symbols = adata.var_names.astype(str).to_numpy()
        else:
            if gene_key not in adata.var:
                raise ValueError(f"AnnData is missing configured var column '{gene_key}'.")
            symbols = adata.var[gene_key].astype(str).to_numpy()
        target_position = {gene: index for index, gene in enumerate(target_genes)}
        source_positions: list[int] = []
        mapped_genes: list[str] = []
        mapped_symbols: list[str] = []
        seen: set[str] = set()
        for source_index, symbol in enumerate(symbols):
            ensembl = symbol_to_ensg.get(str(symbol))
            if ensembl is None or ensembl in seen:
                continue
            seen.add(ensembl)
            source_positions.append(source_index)
            mapped_genes.append(ensembl)
            mapped_symbols.append(str(symbol))
        order = np.argsort(
            np.asarray([target_position[gene] for gene in mapped_genes]), kind="stable"
        )
        source_positions = np.asarray(source_positions, dtype=np.int64)[order].tolist()
        mapped_genes = np.asarray(mapped_genes, dtype=object)[order].astype(str).tolist()
        mapped_symbols = np.asarray(mapped_symbols, dtype=object)[order].astype(str).tolist()
        mapped = adata[:, source_positions].copy()
        mapped.var_names = pd.Index(mapped_genes, name="ensembl_id")
        mapped.var["gene_symbol"] = mapped_symbols
        mapped.var["model_gene_index"] = [target_position[gene] for gene in mapped_genes]
        stats = {
            "split": split,
            "cell_count": int(adata.n_obs),
            "source_gene_count": int(adata.n_vars),
            "matched_unique_gene_count": int(mapped.n_vars),
            "model_gene_count": len(target_genes),
            "missing_model_gene_count": int(len(target_genes) - mapped.n_vars),
            "ambiguous_gene_symbols_in_mapping": int(ambiguous_symbol_count),
        }
        return mapped, stats

    @staticmethod
    def _check_expected_count(
        observed: int,
        expected: object,
        description: str,
    ) -> None:
        if expected is None:
            return
        expected_int = int(expected)
        if observed != expected_int:
            raise ValueError(f"Expected {expected_int} {description}, found {observed}.")

    def _select_reference_genes(
        self,
        reference: ad.AnnData,
        *,
        dataset_key: str,
        target_genes: list[str],
        ensg_to_symbol: dict[str, str],
    ) -> tuple[list[str], pd.DataFrame]:
        method = str(getattr(self.task_cfg, "gene_selection_method", "mad"))
        if method != "mad":
            raise ValueError("Single-cell gene_selection_method currently must be 'mad'.")
        if reference.n_vars == 0:
            raise ValueError(f"Dataset {dataset_key} has no mapped reference genes.")
        matrix = reference.X.toarray() if sparse.issparse(reference.X) else np.asarray(reference.X)
        matrix = np.asarray(matrix, dtype=np.float32)
        medians = np.nanmedian(matrix, axis=0)
        scores = np.nanmedian(np.abs(matrix - medians), axis=0)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        ranked = np.argsort(-scores, kind="stable")
        selected_source_count = min(self.selected_gene_count, reference.n_vars)
        selected_positions = ranked[:selected_source_count]
        if not np.all(np.isfinite(scores[selected_positions])):
            raise ValueError(f"Dataset {dataset_key} has insufficient finite MAD scores.")

        selected_source_genes = set(
            reference.var_names[selected_positions].astype(str).tolist()
        )
        zero_fill_count = self.selected_gene_count - selected_source_count
        filler_genes = [
            gene for gene in target_genes if gene not in reference.var_names
        ][:zero_fill_count]
        if len(filler_genes) != zero_fill_count:
            raise ValueError(
                f"Could not construct the fixed {self.selected_gene_count}-gene input "
                f"for {dataset_key}."
            )
        target_position = {gene: index for index, gene in enumerate(target_genes)}
        selected_genes = sorted(
            [*selected_source_genes, *filler_genes], key=target_position.__getitem__
        )
        score_by_gene = {
            str(reference.var_names[index]): float(scores[index])
            for index in selected_positions
        }
        manifest = pd.DataFrame(
            {
                "ensembl_id": selected_genes,
                "gene_symbol": [ensg_to_symbol.get(gene, "") for gene in selected_genes],
                "model_gene_index": [target_position[gene] for gene in selected_genes],
                "source_present": [gene in selected_source_genes for gene in selected_genes],
                "reference_mad": [score_by_gene.get(gene, np.nan) for gene in selected_genes],
                "selection_reason": [
                    "reference_mad" if gene in selected_source_genes else "zero_fill_fixed_length"
                    for gene in selected_genes
                ],
            }
        )
        manifest.insert(0, "sequence_position", np.arange(self.selected_gene_count))
        manifest["selection_scope"] = "reference_cells_only"
        manifest["selection_method"] = method
        manifest["dataset"] = dataset_key
        return selected_genes, manifest

    @staticmethod
    def _materialize_selected_genes(
        adata: ad.AnnData,
        *,
        selected_genes: list[str],
        target_genes: list[str],
        ensg_to_symbol: dict[str, str],
    ) -> ad.AnnData:
        source_position = {str(gene): index for index, gene in enumerate(adata.var_names)}
        present_genes = [gene for gene in selected_genes if gene in source_position]
        present_source_positions = [source_position[gene] for gene in present_genes]
        source = adata.X[:, present_source_positions]
        destination_position = {gene: index for index, gene in enumerate(selected_genes)}
        destination_columns = np.asarray(
            [destination_position[gene] for gene in present_genes], dtype=np.int64
        )
        if sparse.issparse(source):
            source = source.tocoo()
            matrix = sparse.csr_matrix(
                (
                    source.data,
                    (source.row, destination_columns[source.col]),
                ),
                shape=(adata.n_obs, len(selected_genes)),
                dtype=source.dtype,
            )
        else:
            source = np.asarray(source)
            matrix = np.zeros(
                (adata.n_obs, len(selected_genes)), dtype=source.dtype
            )
            matrix[:, destination_columns] = source

        target_position = {gene: index for index, gene in enumerate(target_genes)}
        symbol_by_gene = {
            str(gene): str(symbol)
            for gene, symbol in zip(
                adata.var_names.astype(str), adata.var["gene_symbol"].astype(str)
            )
        }
        var = pd.DataFrame(
            {
                "gene_symbol": [
                    symbol_by_gene.get(gene, ensg_to_symbol.get(gene, ""))
                    for gene in selected_genes
                ],
                "model_gene_index": [target_position[gene] for gene in selected_genes],
                "source_present": [gene in source_position for gene in selected_genes],
            },
            index=pd.Index(selected_genes, name="ensembl_id"),
        )
        return ad.AnnData(X=matrix, obs=adata.obs.copy(), var=var)

    def load_dataset(self, dataset_key: str) -> SingleCellDatasetBundle:
        dataset_cfg = self.task_cfg.datasets[dataset_key]
        reference_path = self._required_path(
            dataset_cfg.reference_path, f"{dataset_key} reference AnnData"
        )
        query_path = self._required_path(
            dataset_cfg.query_path, f"{dataset_key} query AnnData"
        )
        reference = ad.read_h5ad(reference_path)
        query = ad.read_h5ad(query_path)
        label_key = str(dataset_cfg.label_key)
        batch_key = str(dataset_cfg.batch_key)
        gene_key = str(getattr(dataset_cfg, "gene_key", "gene_name"))
        self._validate_obs(
            reference, path=reference_path, label_key=label_key, batch_key=batch_key
        )
        self._validate_obs(query, path=query_path, label_key=label_key, batch_key=batch_key)
        self._check_expected_count(
            reference.n_obs,
            getattr(dataset_cfg, "expected_reference_cells", None),
            f"{dataset_key} reference cells",
        )
        self._check_expected_count(
            query.n_obs,
            getattr(dataset_cfg, "expected_query_cells", None),
            f"{dataset_key} query cells",
        )
        preprocessing = str(getattr(dataset_cfg, "preprocessing", "none"))
        reference = self._preprocess_expression(reference, preprocessing)
        query = self._preprocess_expression(query, preprocessing)

        target_genes, symbol_to_ensg, ensg_to_symbol, ambiguous = self._gene_mapping_table()
        reference, reference_stats = self._map_to_model_vocabulary(
            reference,
            gene_key=gene_key,
            target_genes=target_genes,
            symbol_to_ensg=symbol_to_ensg,
            ambiguous_symbol_count=ambiguous,
            split="reference",
        )
        query, query_stats = self._map_to_model_vocabulary(
            query,
            gene_key=gene_key,
            target_genes=target_genes,
            symbol_to_ensg=symbol_to_ensg,
            ambiguous_symbol_count=ambiguous,
            split="query",
        )
        common_genes = reference.var_names.intersection(query.var_names, sort=False)
        if common_genes.size == 0:
            raise ValueError(
                f"No mapped genes are shared by the {dataset_key} reference and query files."
            )
        reference = reference[:, common_genes].copy()
        query = query[:, common_genes].copy()

        selected_genes: list[str] | None = None
        manifest: pd.DataFrame | None = None
        if self.is_master:
            selected_genes, manifest = self._select_reference_genes(
                reference,
                dataset_key=dataset_key,
                target_genes=target_genes,
                ensg_to_symbol=ensg_to_symbol,
            )
        if self.is_distributed:
            payload = [selected_genes]
            dist.broadcast_object_list(payload, src=0)
            selected_genes = payload[0]
        if selected_genes is None:
            raise RuntimeError("Reference-only gene selection produced no genes.")
        reference = self._materialize_selected_genes(
            reference,
            selected_genes=selected_genes,
            target_genes=target_genes,
            ensg_to_symbol=ensg_to_symbol,
        )
        query = self._materialize_selected_genes(
            query,
            selected_genes=selected_genes,
            target_genes=target_genes,
            ensg_to_symbol=ensg_to_symbol,
        )
        if manifest is None:
            manifest = pd.DataFrame()

        reference.obs["mapping_split"] = "reference"
        query.obs["mapping_split"] = "query"
        reference.obs_names = pd.Index(
            [f"{dataset_key}:reference:{name}" for name in reference.obs_names.astype(str)]
        )
        query.obs_names = pd.Index(
            [f"{dataset_key}:query:{name}" for name in query.obs_names.astype(str)]
        )
        combined = ad.concat(
            [reference, query],
            axis=0,
            join="inner",
            merge="same",
            uns_merge="same",
        )
        required_var_columns = {"gene_symbol", "model_gene_index"}
        missing_var_columns = required_var_columns.difference(combined.var.columns)
        if missing_var_columns:
            raise ValueError(
                f"{dataset_key} lost required gene metadata during concatenation: "
                f"{sorted(missing_var_columns)}"
            )
        if combined.n_vars != self.selected_gene_count:
            raise ValueError(
                f"{dataset_key} has {combined.n_vars} selected genes after concatenation; "
                f"expected {self.selected_gene_count}."
            )
        combined.obs[label_key] = combined.obs[label_key].astype(str)
        combined.obs[batch_key] = combined.obs[batch_key].astype(str)
        self._check_expected_count(
            combined.n_obs,
            getattr(dataset_cfg, "expected_total_cells", None),
            f"{dataset_key} total cells",
        )
        self._check_expected_count(
            combined.obs[batch_key].nunique(),
            getattr(dataset_cfg, "expected_batch_count", None),
            f"{dataset_key} batches",
        )
        model_gene_indices = combined.var["model_gene_index"].to_numpy(dtype=np.int64)
        reference_labels = set(reference.obs[label_key].astype(str))
        query_labels = set(query.obs[label_key].astype(str))
        unseen_query_labels = sorted(query_labels.difference(reference_labels))
        stats = {
            "reference": reference_stats,
            "query": query_stats,
            "shared_mapped_gene_count": int(common_genes.size),
            "selected_gene_count": self.selected_gene_count,
            "selected_source_gene_count": int(combined.var["source_present"].sum()),
            "zero_filled_gene_count": int((~combined.var["source_present"]).sum()),
            "reference_cell_count": int(reference.n_obs),
            "query_cell_count": int(query.n_obs),
            "total_cell_count": int(combined.n_obs),
            "batch_count": int(combined.obs[batch_key].nunique()),
            "cell_type_count": int(combined.obs[label_key].nunique()),
            "query_cell_types_absent_from_reference": unseen_query_labels,
            "reference_path": str(reference_path),
            "query_path": str(query_path),
            "preprocessing": preprocessing,
            "gene_selection_method": "mad",
            "gene_selection_scope": "reference_cells_only",
        }
        self.mapping_stats[dataset_key] = stats
        log.info(
            "%s | cells reference=%d query=%d | shared mapped genes=%d | batches=%d",
            dataset_key,
            reference.n_obs,
            query.n_obs,
            common_genes.size,
            combined.obs[batch_key].nunique(),
        )
        return SingleCellDatasetBundle(
            key=dataset_key,
            display_name=str(getattr(dataset_cfg, "display_name", dataset_key)),
            reference=reference,
            query=query,
            combined=combined,
            model_gene_indices=model_gene_indices,
            manifest=manifest,
            label_key=label_key,
            batch_key=batch_key,
            mapping_stats=stats,
        )

    def make_loader(self, adata: ad.AnnData, model_gene_indices: np.ndarray) -> DataLoader:
        dataset = _SingleCellExpressionDataset(
            adata.X,
            model_gene_indices=model_gene_indices,
            bin_num=int(self.model_cfg.bin_num),
            cls_gene_id=self.cls_gene_id,
            gene_token_offset=self.gene_token_offset,
            cls_value=float(getattr(self.model_cfg, "pad_value", -2.0)),
            seed=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        batch_size = int(getattr(self.task_cfg, "batch_size", 4))
        sampler = (
            SequentialDistributedSampler(
                dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            if self.is_distributed
            else None
        )
        workers = int(getattr(self.task_cfg, "num_workers", 2))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
        )

    @staticmethod
    def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if state_dict and all(key.startswith("module.") for key in state_dict):
            return {key.removeprefix("module."): value for key, value in state_dict.items()}
        return state_dict

    def build_backbone(self, checkpoint_path: str) -> CancerFoundationBackbone:
        seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
        backbone = CancerFoundationBackbone(
            num_gene_tokens=int(self.model_cfg.gene_num) + self.gene_token_offset,
            d_model=int(self.model_cfg.embsize),
            nhead=int(self.model_cfg.nheads),
            d_hid=int(self.model_cfg.d_hid),
            nlayers=int(self.model_cfg.nlayers),
            dropout=float(self.model_cfg.dropout),
            pad_gene_id=self.pad_gene_id,
            max_value=int(getattr(self.model_cfg, "value_encoder_max_value", 512)),
        )
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            validate_backbone_checkpoint(
                checkpoint,
                checkpoint_path,
                gene_num=int(self.model_cfg.gene_num),
                selected_gene_count=self.selected_gene_count,
                max_seq_len=self.max_seq_len,
                bin_num=int(self.model_cfg.bin_num),
            )
            backbone.load_state_dict(
                self._strip_module_prefix(checkpoint["model_state_dict"]), strict=True
            )
        backbone.requires_grad_(False)
        backbone.eval()
        return backbone.to(self.device)

    def extract_scbfm_embeddings(
        self,
        backbone: CancerFoundationBackbone,
        loader: DataLoader,
    ) -> np.ndarray | None:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                hidden = backbone(
                    batch["gene_ids"].to(self.device, non_blocking=True),
                    batch["expr"].to(self.device, non_blocking=True),
                )
                cls = torch.nn.functional.normalize(hidden[:, 0, :], p=2, dim=1)
                chunks.append(cls.cpu().numpy().astype(np.float32, copy=False))
        if not chunks:
            raise RuntimeError("Frozen scbFM embedding extraction produced no cells.")
        embeddings = np.vstack(chunks)
        if self.is_distributed:
            tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
            tensor = distributed_concat(tensor, len(loader.dataset), self.world_size)
            embeddings = tensor.cpu().numpy().astype(np.float32, copy=False)
        return embeddings if self.is_master else None

    def extract_scbfm_model_embeddings(
        self,
        bundle: SingleCellDatasetBundle,
        checkpoint_paths: dict[str, str],
    ) -> dict[str, np.ndarray]:
        loader = self.make_loader(bundle.combined, bundle.model_gene_indices)
        embeddings: dict[str, np.ndarray] = {}
        for model_key, checkpoint_path in checkpoint_paths.items():
            log.info("%s | extracting frozen embeddings for %s", bundle.key, model_key)
            backbone = self.build_backbone(checkpoint_path)
            values = self.extract_scbfm_embeddings(backbone, loader)
            if self.is_master and values is not None:
                embeddings[model_key] = values
            del backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return embeddings

    def _external_scgpt_entries(self) -> list[tuple[str, DictConfig]]:
        if not bool(getattr(self.task_cfg, "include_external_scgpt", True)):
            return []
        config = getattr(self.task_cfg, "external_scgpt_models", None)
        if config is None:
            raise ValueError(
                "include_external_scgpt=true requires external_scgpt_models."
            )
        entries: list[tuple[str, DictConfig]] = []
        for key in EXTERNAL_SCGPT_MODEL_KEYS:
            if key not in config:
                raise ValueError(f"Missing external_scgpt_models.{key} configuration.")
            entries.append((key, config[key]))
        return entries

    def extract_external_scgpt_embeddings(
        self,
        bundle: SingleCellDatasetBundle,
    ) -> dict[str, np.ndarray]:
        entries = self._external_scgpt_entries()
        if not entries:
            return {}
        from finetune.canc_type_class.scgpt_pca_rf_runner import (
            CancTypeClassScGPTPCARFRunner,
        )

        repo_dir = self._required_path(
            getattr(self.task_cfg, "scgpt_repo_dir", None), "scGPT repository"
        )
        embeddings: dict[str, np.ndarray] = {}
        for model_key, model_cfg in entries:
            model_dir = self._required_path(model_cfg.model_dir, f"{model_key} model directory")
            paths = {
                "repo_dir": repo_dir,
                "model_dir": model_dir,
                "args": model_dir / str(model_cfg.args_filename),
                "vocab": model_dir / str(model_cfg.vocab_filename),
                "checkpoint": model_dir / str(model_cfg.checkpoint_filename),
            }
            missing = [
                str(paths[key])
                for key in ("args", "vocab", "checkpoint")
                if not paths[key].is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing {model_key} files: {missing}")

            helper = object.__new__(CancTypeClassScGPTPCARFRunner)
            helper.task_cfg = OmegaConf.create(
                {
                    "scgpt_use_fast_transformer": False,
                    "scgpt_batch_size": int(
                        getattr(self.task_cfg, "scgpt_batch_size", 4)
                    ),
                    "scgpt_max_seq_len": self.max_seq_len,
                    "num_workers": int(getattr(self.task_cfg, "num_workers", 2)),
                }
            )
            helper.device = self.device
            helper.rank = self.rank
            helper.world_size = self.world_size
            helper.is_distributed = self.is_distributed
            helper.is_master = self.is_master
            model, vocab, model_configs = helper._build_scgpt_model(paths)

            symbols = bundle.combined.var["gene_symbol"].astype(str).to_numpy()
            source_present = bundle.combined.var["source_present"].to_numpy(dtype=bool)
            in_vocab = np.asarray(
                [present and symbol in vocab for present, symbol in zip(source_present, symbols)],
                dtype=bool,
            )
            unique_mask = np.zeros(in_vocab.size, dtype=bool)
            seen_tokens: set[int] = set()
            for index in np.flatnonzero(in_vocab):
                token = int(vocab[symbols[index]])
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                unique_mask[index] = True
            selected = bundle.combined[:, unique_mask].copy()
            source_gene_count = int(source_present.sum())
            minimum_coverage = float(
                getattr(self.task_cfg, "min_external_scgpt_gene_coverage", 0.9)
            )
            if not 0 < minimum_coverage <= 1:
                raise ValueError(
                    "min_external_scgpt_gene_coverage must be in (0, 1]."
                )
            minimum_matches = int(np.ceil(minimum_coverage * source_gene_count))
            if selected.n_vars < minimum_matches:
                raise ValueError(
                    f"Only {selected.n_vars}/{source_gene_count} source-present "
                    f"{bundle.key} genes match the {model_key} vocabulary; at least "
                    f"{minimum_matches} ({minimum_coverage:.0%}) are required."
                )
            gene_ids = np.asarray(
                vocab(selected.var["gene_symbol"].astype(str).tolist()),
                dtype=np.int64,
            )
            dummy_labels = np.zeros(selected.n_obs, dtype=np.int64)
            loader = helper._make_scgpt_loader(
                selected,
                dummy_labels,
                gene_ids,
                vocab,
                model_configs,
                stage=f"{bundle.key}_{model_key}",
            )
            local_values, local_labels = helper._extract_scgpt_embeddings_safely(
                model, loader, stage=f"{bundle.key}_{model_key}"
            )
            values, _ = helper._gather_scgpt_embeddings(
                local_values, local_labels, selected.n_obs
            )
            if self.is_master:
                embeddings[model_key] = values
                self.external_scgpt_load_reports.setdefault(bundle.key, {})[
                    model_key
                ] = {
                    **helper._scgpt_load_report,
                    "matched_selected_gene_count": int(selected.n_vars),
                    "source_present_gene_count": source_gene_count,
                    "matched_source_gene_fraction": float(
                        selected.n_vars / source_gene_count
                    ),
                    "selected_gene_count": int(bundle.combined.n_vars),
                    "checkpoint": str(paths["checkpoint"]),
                }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return embeddings

    def extract_model_embeddings(
        self,
        bundle: SingleCellDatasetBundle,
        checkpoint_paths: dict[str, str],
    ) -> dict[str, np.ndarray]:
        embeddings = self.extract_scbfm_model_embeddings(bundle, checkpoint_paths)
        external = self.extract_external_scgpt_embeddings(bundle)
        if self.is_master:
            embeddings.update(external)
        return embeddings

    def raw_pca_embeddings(
        self,
        bundle: SingleCellDatasetBundle,
        *,
        fit_reference_only: bool,
    ) -> np.ndarray:
        reference_matrix = bundle.reference.X
        query_matrix = bundle.query.X
        if sparse.issparse(reference_matrix):
            reference_matrix = reference_matrix.toarray()
        if sparse.issparse(query_matrix):
            query_matrix = query_matrix.toarray()
        reference_matrix = np.asarray(reference_matrix, dtype=np.float32)
        query_matrix = np.asarray(query_matrix, dtype=np.float32)
        requested = int(getattr(self.task_cfg, "raw_pca_components", 256))
        if fit_reference_only:
            n_components = min(
                requested,
                reference_matrix.shape[0] - 1,
                reference_matrix.shape[1],
            )
            pca = PCA(
                n_components=n_components,
                random_state=int(getattr(self.task_cfg, "random_seed", 42)),
            ).fit(reference_matrix)
            values = np.vstack([pca.transform(reference_matrix), pca.transform(query_matrix)])
        else:
            combined = np.vstack([reference_matrix, query_matrix])
            n_components = min(requested, combined.shape[0] - 1, combined.shape[1])
            values = PCA(
                n_components=n_components,
                random_state=int(getattr(self.task_cfg, "random_seed", 42)),
            ).fit_transform(combined)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return (values / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)

    @staticmethod
    def forgetting_rows(
        rows: list[dict[str, object]],
        *,
        metric_fields: list[str],
    ) -> list[dict[str, object]]:
        frame = pd.DataFrame(rows)
        comparisons = (
            ("sc_preadaptation", "pretrain_sc", "preadapt_sc"),
            ("bulk_preadaptation", "pretrain_bulk", "preadapt_bulk"),
            ("scgpt_preadaptation", "scgpt", "scgpt_preadapt"),
        )
        output: list[dict[str, object]] = []
        for dataset in frame["dataset"].unique():
            dataset_frame = frame[frame["dataset"] == dataset].set_index("model")
            for comparison, before, after in comparisons:
                if before not in dataset_frame.index or after not in dataset_frame.index:
                    continue
                for metric in metric_fields:
                    before_value = float(dataset_frame.loc[before, metric])
                    after_value = float(dataset_frame.loc[after, metric])
                    output.append(
                        {
                            "dataset": dataset,
                            "comparison": comparison,
                            "before_model": before,
                            "after_model": after,
                            "metric": metric,
                            "before_value": before_value,
                            "after_value": after_value,
                            "delta_after_minus_before": after_value - before_value,
                        }
                    )
        return output

    @staticmethod
    def append_macro_average(
        rows: list[dict[str, object]],
        *,
        metric_fields: list[str],
    ) -> list[dict[str, object]]:
        frame = pd.DataFrame(rows)
        output = list(rows)
        for model, model_frame in frame.groupby("model", sort=False):
            row: dict[str, object] = {
                "dataset": "macro_average",
                "display_name": DISPLAY_NAMES.get(str(model), str(model)),
                "model": str(model),
            }
            for metric in metric_fields:
                row[metric] = float(model_frame[metric].astype(float).mean())
            output.append(row)
        return output
