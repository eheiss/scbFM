from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc
import json
import math
import os
import re
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

cellxgene_census = None


# =========================
# Paths
# =========================

GENE_LIST_PATH = Path(__file__).resolve().parent / "gene_list.txt"

OUT_DIR = Path("/cluster/work/boeva/eheiss/datasets/pseudo_bulk")
CHUNK_DIR = OUT_DIR / "pseudo_bulk_RAW_chunks"
SOURCE_CHUNK_DIR = OUT_DIR / "source_cell_chunks"
MERGE_TMP_DIR = OUT_DIR / "RAW_merge_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT = OUT_DIR / "pseudo_bulk_RAW.h5ad"
PLAN_OUT = OUT_DIR / "pseudo_bulk_sampling_plan.csv"
SUMMARY_OUT = OUT_DIR / "pseudo_bulk_summary_RAW.json"
MISSING_GENES_OUT = OUT_DIR / "cellxgene_missing_genes.json"
ELIGIBLE_CONTEXTS_OUT = OUT_DIR / "eligible_contexts.csv"
SOURCE_POOL_QUOTAS_OUT = OUT_DIR / "source_cell_pool_quotas.csv"
SAMPLED_SOURCE_CELLS_OUT = OUT_DIR / "sampled_source_cells.csv"
SOURCE_ADATA_OUT = OUT_DIR / "sampled_source_cells_aligned.h5ad"


# =========================
# Settings
# =========================

ORGANISM = "Homo sapiens"
CENSUS_VERSION = "2025-11-08"
MIN_GENES = 200
TARGET_PSEUDO_BULKS = int(os.getenv("SCBFM_TARGET_PSEUDO_BULKS", "20000"))
CELLS_PER_PSEUDO_BULK = int(os.getenv("SCBFM_CELLS_PER_PSEUDO_BULK", "1000"))
DOWNLOAD_CHUNK_SIZE = int(os.getenv("SCBFM_PSEUDO_DOWNLOAD_CHUNK_SIZE", "5000"))
WRITE_CHUNK_SIZE = int(os.getenv("SCBFM_PSEUDO_WRITE_CHUNK_SIZE", "500"))
MERGE_BATCH_SIZE = int(os.getenv("SCBFM_PSEUDO_MERGE_BATCH_SIZE", "8"))
RANDOM_SEED = 2021

MIN_CONTEXT_CELL_TYPES = int(os.getenv("SCBFM_MIN_CONTEXT_CELL_TYPES", "2"))
MIN_AVAILABLE_CELLS_PER_CELLTYPE = int(
    os.getenv("SCBFM_MIN_AVAILABLE_CELLS_PER_CELLTYPE", "20")
)
MIN_CONTEXT_TOTAL_CELLS = int(os.getenv("SCBFM_MIN_CONTEXT_TOTAL_CELLS", "100"))
MIN_SOURCE_POOL_PER_CELLTYPE = int(
    os.getenv("SCBFM_MIN_SOURCE_POOL_PER_CELLTYPE", "32")
)
MAX_SOURCE_POOL_PER_CELLTYPE = int(
    os.getenv("SCBFM_MAX_SOURCE_POOL_PER_CELLTYPE", "256")
)
SPARSE_SAMPLE_PROB = float(os.getenv("SCBFM_SPARSE_SAMPLE_PROB", "0.5"))
TISSUE_COLUMN = os.getenv("SCBFM_TISSUE_COLUMN", "tissue_general")
RESUME = os.getenv("SCBFM_PSEUDO_RESUME", "1") != "0"
OFFLINE = os.getenv("SCBFM_PSEUDO_OFFLINE", "0") == "1"

TILEDB_CONFIG = {
    "py.init_buffer_bytes": 256 * 1024**2,
    "soma.init_buffer_bytes": 256 * 1024**2,
}

OBS_CONTEXT_COLUMNS = [
    "soma_joinid",
    "dataset_id",
    "donor_id",
    TISSUE_COLUMN,
    "cell_type",
]


# =========================
# Helpers
# =========================

def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_json(data, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def require_cellxgene_census():
    global cellxgene_census
    if cellxgene_census is None:
        try:
            import cellxgene_census as census_module
        except ImportError as exc:
            raise ImportError(
                "cellxgene_census is required unless SCBFM_PSEUDO_OFFLINE=1 and all "
                "metadata/source chunks are already cached."
            ) from exc
        cellxgene_census = census_module
    return cellxgene_census


def normalize_obs_chunk(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col == "soma_joinid":
            df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype("string").fillna("unknown").replace("<NA>", "unknown")
            df[col] = df[col].astype(str)
    return df


def filter_raw_dense_block(
    x: np.ndarray,
    min_genes: int = MIN_GENES,
) -> tuple[np.ndarray, np.ndarray]:
    n_genes_by_sample = (x > 0).sum(axis=1)
    keep_mask = n_genes_by_sample >= min_genes
    x = x[keep_mask]

    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32), keep_mask

    return x.astype(np.float32, copy=False), keep_mask


def merge_h5ad_group(paths: list[Path], out_path: Path) -> Path:
    adatas = [ad.read_h5ad(p) for p in paths]
    merged = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    merged.write(out_path)

    del adatas, merged
    gc.collect()
    return out_path


def iter_obs_tables(census, column_names: list[str]):
    exp = census["census_data"]["homo_sapiens"]
    return exp.obs.read(
        column_names=column_names,
        value_filter="is_primary_data == True",
    )


def build_var_coords(census, gene_list: list[str]) -> tuple[list[int], list[str]]:
    census_api = require_cellxgene_census()
    var_df = census_api.get_var(
        census=census,
        organism=ORGANISM,
        column_names=["soma_joinid", "feature_id"],
    )
    var_df["feature_id"] = var_df["feature_id"].astype(str)
    var_df = var_df.drop_duplicates("feature_id", keep="first")
    joinid_by_feature = dict(zip(var_df["feature_id"], var_df["soma_joinid"].astype(int)))

    missing = [g for g in gene_list if g not in joinid_by_feature]
    var_coords = [int(joinid_by_feature[g]) for g in gene_list if g in joinid_by_feature]
    return var_coords, missing


def make_context_key(row) -> tuple[str, str, str]:
    return (row.dataset_id, row.donor_id, getattr(row, TISSUE_COLUMN))


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip().lower()).strip("_")
    return sanitized or "cell_type"


def build_proportion_column_map(cell_types: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for cell_type in sorted(cell_types):
        base = f"prop__{sanitize_name(cell_type)}"
        column = base
        suffix = 2
        while column in used:
            column = f"{base}_{suffix}"
            suffix += 1
        mapping[cell_type] = column
        used.add(column)

    return mapping


def store_proportion_metadata(adata: ad.AnnData, proportion_column_map: dict[str, str]) -> None:
    cell_types = sorted(proportion_column_map)
    adata.uns["cell_type_proportion_cell_types"] = np.asarray(cell_types, dtype=object)
    adata.uns["cell_type_proportion_obs_columns"] = np.asarray(
        [proportion_column_map[cell_type] for cell_type in cell_types],
        dtype=object,
    )
    adata.uns["cell_type_proportion_columns"] = proportion_column_map


def parse_json_dict_column(value: object, cast):
    if isinstance(value, dict):
        return {str(k): cast(v) for k, v in value.items()}
    if pd.isna(value):
        return {}
    parsed = json.loads(value)
    return {str(k): cast(v) for k, v in parsed.items()}


def load_eligible_contexts_from_csv(
    path: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, int]]]:
    eligible_df = pd.read_csv(path)
    if eligible_df.empty:
        return eligible_df, {}

    eligible_df = normalize_obs_chunk(
        eligible_df,
        [col for col in ["dataset_id", "donor_id", TISSUE_COLUMN] if col in eligible_df.columns],
    )

    eligible_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in eligible_df.itertuples(index=False):
        context_key = (row.dataset_id, row.donor_id, getattr(row, TISSUE_COLUMN))
        eligible_counts[context_key] = parse_json_dict_column(row.cell_type_counts, int)
    return eligible_df, eligible_counts


def load_sample_plan_from_csv(
    plan_path: Path,
    quota_path: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[tuple[tuple[str, str, str], str], int], list[str]]:
    plan_df = pd.read_csv(plan_path)
    plan_df = normalize_obs_chunk(
        plan_df,
        [col for col in ["sample_id", "dataset_id", "donor_id", TISSUE_COLUMN] if col in plan_df.columns],
    )

    plan_rows: list[dict[str, object]] = []
    all_cell_types: set[str] = set()
    for row in plan_df.itertuples(index=False):
        cell_type_counts = parse_json_dict_column(row.cell_type_counts, int)
        cell_type_proportions = parse_json_dict_column(row.cell_type_proportions, float)
        plan_rows.append(
            {
                "sample_id": row.sample_id,
                "dataset_id": row.dataset_id,
                "donor_id": row.donor_id,
                TISSUE_COLUMN: getattr(row, TISSUE_COLUMN),
                "total_cells": int(row.total_cells),
                "n_cell_types": int(row.n_cell_types),
                "cell_type_counts": cell_type_counts,
                "cell_type_proportions": cell_type_proportions,
            }
        )
        all_cell_types.update(cell_type_counts)

    quota_df = pd.read_csv(quota_path)
    quota_df = normalize_obs_chunk(
        quota_df,
        [col for col in ["dataset_id", "donor_id", TISSUE_COLUMN, "cell_type"] if col in quota_df.columns],
    )
    reservoir_quotas: dict[tuple[tuple[str, str, str], str], int] = {}
    for row in quota_df.itertuples(index=False):
        context_key = (row.dataset_id, row.donor_id, getattr(row, TISSUE_COLUMN))
        reservoir_quotas[(context_key, row.cell_type)] = int(row.reservoir_quota)
        all_cell_types.add(row.cell_type)

    return plan_rows, plan_df, reservoir_quotas, sorted(all_cell_types)


def load_sampled_source_meta(path: Path) -> pd.DataFrame:
    sampled_meta = pd.read_csv(path)
    return normalize_obs_chunk(sampled_meta, OBS_CONTEXT_COLUMNS)


def count_context_cell_types(census) -> dict[tuple[str, str, str], dict[str, int]]:
    counts_by_context: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for table in iter_obs_tables(census, OBS_CONTEXT_COLUMNS):
        df = normalize_obs_chunk(table.to_pandas(), OBS_CONTEXT_COLUMNS)
        df = df[
            (df["donor_id"] != "unknown")
            & (df[TISSUE_COLUMN] != "unknown")
            & (df["cell_type"] != "unknown")
        ]
        for row in df.itertuples(index=False):
            context_key = make_context_key(row)
            counts_by_context[context_key][row.cell_type] += 1

        del df
        gc.collect()

    return counts_by_context


def build_eligible_contexts(
    counts_by_context: dict[tuple[str, str, str], dict[str, int]]
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, int]]]:
    records: list[dict[str, object]] = []
    filtered_counts: dict[tuple[str, str, str], dict[str, int]] = {}

    for context_key, cell_type_counts in counts_by_context.items():
        filtered_cell_types = {
            cell_type: count
            for cell_type, count in cell_type_counts.items()
            if count >= MIN_AVAILABLE_CELLS_PER_CELLTYPE
        }
        total_cells = int(sum(filtered_cell_types.values()))
        if len(filtered_cell_types) < MIN_CONTEXT_CELL_TYPES:
            continue
        if total_cells < MIN_CONTEXT_TOTAL_CELLS:
            continue

        dataset_id, donor_id, tissue = context_key
        filtered_counts[context_key] = filtered_cell_types
        records.append(
            {
                "dataset_id": dataset_id,
                "donor_id": donor_id,
                TISSUE_COLUMN: tissue,
                "n_cell_types": int(len(filtered_cell_types)),
                "n_cells": total_cells,
                "cell_types": json.dumps(sorted(filtered_cell_types)),
                "cell_type_counts": json.dumps(filtered_cell_types, sort_keys=True),
            }
        )

    eligible_df = pd.DataFrame(
        records,
        columns=[
            "dataset_id",
            "donor_id",
            TISSUE_COLUMN,
            "n_cell_types",
            "n_cells",
            "cell_types",
            "cell_type_counts",
        ],
    )
    if not eligible_df.empty:
        eligible_df = eligible_df.sort_values(
            ["n_cells", "n_cell_types", "dataset_id", "donor_id", TISSUE_COLUMN],
            ascending=[False, False, True, True, True],
        )
    return eligible_df, filtered_counts


def simulate_sample_plan(
    eligible_counts: dict[tuple[str, str, str], dict[str, int]],
    target_samples: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[tuple[tuple[str, str, str], str], int], list[str]]:
    if not eligible_counts:
        raise ValueError("No eligible contexts found for pseudo-bulk generation.")

    rng = np.random.default_rng(RANDOM_SEED)
    context_keys = list(eligible_counts)
    context_weights = np.array(
        [math.sqrt(sum(eligible_counts[key].values())) for key in context_keys],
        dtype=np.float64,
    )
    context_weights /= context_weights.sum()

    plan_rows: list[dict[str, object]] = []
    reservoir_demands: dict[tuple[tuple[str, str, str], str], int] = {}
    all_cell_types: set[str] = set()

    for sample_idx in range(target_samples):
        context_key = context_keys[int(rng.choice(len(context_keys), p=context_weights))]
        cell_type_counts = eligible_counts[context_key]
        available_cell_types = sorted(cell_type_counts)
        active_cell_types = list(available_cell_types)

        if len(available_cell_types) > 1 and rng.random() < SPARSE_SAMPLE_PROB:
            n_absent = int(rng.integers(0, len(available_cell_types)))
            if n_absent >= len(available_cell_types):
                n_absent = len(available_cell_types) - 1
            if n_absent > 0:
                absent_idx = rng.choice(len(available_cell_types), size=n_absent, replace=False)
                active_mask = np.ones(len(available_cell_types), dtype=bool)
                active_mask[absent_idx] = False
                active_cell_types = [
                    cell_type for keep, cell_type in zip(active_mask.tolist(), available_cell_types) if keep
                ]

        weights = rng.random(len(active_cell_types))
        weights /= weights.sum()
        realized_counts = rng.multinomial(CELLS_PER_PSEUDO_BULK, weights)

        realized_cell_type_counts = {
            cell_type: int(count)
            for cell_type, count in zip(active_cell_types, realized_counts.tolist())
            if count > 0
        }
        realized_cell_type_props = {
            cell_type: count / CELLS_PER_PSEUDO_BULK
            for cell_type, count in realized_cell_type_counts.items()
        }

        for cell_type, count in realized_cell_type_counts.items():
            key = (context_key, cell_type)
            current = reservoir_demands.get(key, 0)
            reservoir_demands[key] = max(current, count)
            all_cell_types.add(cell_type)

        dataset_id, donor_id, tissue = context_key
        plan_rows.append(
            {
                "sample_id": f"pseudo_bulk:{sample_idx:06d}",
                "dataset_id": dataset_id,
                "donor_id": donor_id,
                TISSUE_COLUMN: tissue,
                "total_cells": CELLS_PER_PSEUDO_BULK,
                "n_cell_types": int(len(realized_cell_type_counts)),
                "cell_type_counts": realized_cell_type_counts,
                "cell_type_proportions": realized_cell_type_props,
            }
        )

    quota_records: list[dict[str, object]] = []
    for (context_key, cell_type), max_count in reservoir_demands.items():
        available = eligible_counts[context_key][cell_type]
        quota = min(
            available,
            max(
                min(available, MIN_SOURCE_POOL_PER_CELLTYPE),
                min(MAX_SOURCE_POOL_PER_CELLTYPE, int(math.ceil(math.sqrt(max_count * target_samples / max(1, len(context_keys)))))),
                max_count,
            ),
        )
        reservoir_demands[(context_key, cell_type)] = int(quota)
        dataset_id, donor_id, tissue = context_key
        quota_records.append(
            {
                "dataset_id": dataset_id,
                "donor_id": donor_id,
                TISSUE_COLUMN: tissue,
                "cell_type": cell_type,
                "available_cells": int(available),
                "reservoir_quota": int(quota),
            }
        )

    plan_df = pd.DataFrame(
        [
            {
                "sample_id": row["sample_id"],
                "dataset_id": row["dataset_id"],
                "donor_id": row["donor_id"],
                TISSUE_COLUMN: row[TISSUE_COLUMN],
                "total_cells": row["total_cells"],
                "n_cell_types": row["n_cell_types"],
                "cell_type_counts": json.dumps(row["cell_type_counts"], sort_keys=True),
                "cell_type_proportions": json.dumps(row["cell_type_proportions"], sort_keys=True),
            }
            for row in plan_rows
        ]
    )
    if quota_records:
        quota_df = pd.DataFrame(quota_records).sort_values(
            ["dataset_id", "donor_id", TISSUE_COLUMN, "cell_type"]
        )
        quota_df.to_csv(SOURCE_POOL_QUOTAS_OUT, index=False)

    return plan_rows, plan_df, reservoir_demands, sorted(all_cell_types)


def reservoir_sample_source_cells(
    census,
    reservoir_quotas: dict[tuple[tuple[str, str, str], str], int],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    seen_counts: dict[tuple[tuple[str, str, str], str], int] = defaultdict(int)
    reservoirs: dict[tuple[tuple[str, str, str], str], list[dict[str, object]]] = {
        key: [] for key, quota in reservoir_quotas.items() if quota > 0
    }

    for table in iter_obs_tables(census, OBS_CONTEXT_COLUMNS):
        df = normalize_obs_chunk(table.to_pandas(), OBS_CONTEXT_COLUMNS)
        df = df[
            (df["donor_id"] != "unknown")
            & (df[TISSUE_COLUMN] != "unknown")
            & (df["cell_type"] != "unknown")
        ]
        for row in df.itertuples(index=False):
            context_key = make_context_key(row)
            quota_key = (context_key, row.cell_type)
            quota = reservoir_quotas.get(quota_key, 0)
            if quota <= 0:
                continue

            seen_counts[quota_key] += 1
            sample_row = {
                "soma_joinid": int(row.soma_joinid),
                "dataset_id": row.dataset_id,
                "donor_id": row.donor_id,
                TISSUE_COLUMN: getattr(row, TISSUE_COLUMN),
                "cell_type": row.cell_type,
            }
            reservoir = reservoirs[quota_key]
            if len(reservoir) < quota:
                reservoir.append(sample_row)
            else:
                j = int(rng.integers(0, seen_counts[quota_key]))
                if j < quota:
                    reservoir[j] = sample_row

        del df
        gc.collect()

    sampled_rows = [row for reservoir in reservoirs.values() for row in reservoir]
    sampled_meta = pd.DataFrame(sampled_rows).drop_duplicates("soma_joinid")
    sampled_meta = sampled_meta.sort_values("soma_joinid").reset_index(drop=True)
    return sampled_meta


def download_source_cells(
    census,
    sampled_meta: pd.DataFrame,
    var_coords: list[int],
    gene_list: list[str],
) -> ad.AnnData:
    census_api = require_cellxgene_census()
    chunk_paths: list[Path] = []

    for chunk_id, start in enumerate(range(0, sampled_meta.shape[0], DOWNLOAD_CHUNK_SIZE)):
        end = min(start + DOWNLOAD_CHUNK_SIZE, sampled_meta.shape[0])
        chunk_path = SOURCE_CHUNK_DIR / f"source_cells_chunk_{chunk_id:05d}.h5ad"
        if RESUME and chunk_path.exists():
            print(
                f"Reusing downloaded source CELLxGENE chunk {chunk_id}: "
                f"cells {start}:{end}"
            )
            chunk_paths.append(chunk_path)
            continue

        meta_chunk = sampled_meta.iloc[start:end].copy()
        obs_coords = meta_chunk["soma_joinid"].astype(np.int64).tolist()

        print(f"Downloading source CELLxGENE chunk {chunk_id}: cells {start}:{end}")
        t0 = time.perf_counter()
        adata = census_api.get_anndata(
            census=census,
            organism=ORGANISM,
            obs_coords=obs_coords,
            var_coords=var_coords,
            obs_column_names=OBS_CONTEXT_COLUMNS,
            var_column_names=["feature_id"],
        )
        t1 = time.perf_counter()
        print(
            f"Source chunk {chunk_id}: download finished in {t1 - t0:.1f}s "
            f"with shape {adata.n_obs} x {adata.n_vars}"
        )

        feature_ids = adata.var["feature_id"].astype(str).tolist()
        reorder_idx = pd.Index(feature_ids).get_indexer(gene_list)
        if (reorder_idx < 0).any():
            raise ValueError("Downloaded source chunk is missing requested genes.")

        adata = adata[:, reorder_idx].copy()
        adata.var_names = pd.Index(gene_list, dtype=str)
        adata.obs = normalize_obs_chunk(adata.obs.reset_index(drop=True), OBS_CONTEXT_COLUMNS)
        adata.obs.index = pd.Index(
            [f"cellxgene:{sid}" for sid in adata.obs["soma_joinid"].astype(str)],
            name="cell_id",
        )
        adata.var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        if sparse.issparse(adata.X):
            adata.X = adata.X.tocsr()
        else:
            adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))

        adata.write(chunk_path)
        chunk_paths.append(chunk_path)

        del adata
        gc.collect()

    if not chunk_paths:
        raise ValueError("No source cell chunks are available.")

    source_chunks = [ad.read_h5ad(path) for path in chunk_paths]
    source_adata = ad.concat(
        source_chunks,
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )
    source_adata.obs_names_make_unique()
    source_adata.var_names = pd.Index(gene_list, dtype=str)

    del source_chunks
    gc.collect()
    return source_adata


def build_source_row_index(
    source_adata: ad.AnnData,
) -> dict[tuple[tuple[str, str, str], str], np.ndarray]:
    row_index_by_context_cell_type: dict[tuple[tuple[str, str, str], str], list[int]] = defaultdict(list)

    obs = source_adata.obs.reset_index(drop=True)
    for row_idx, row in enumerate(obs.itertuples(index=False)):
        context_key = make_context_key(row)
        row_index_by_context_cell_type[(context_key, row.cell_type)].append(row_idx)

    return {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in row_index_by_context_cell_type.items()
    }


def get_source_chunk_paths(sampled_meta: pd.DataFrame) -> list[Path]:
    chunk_paths = []
    for chunk_id, _start in enumerate(range(0, sampled_meta.shape[0], DOWNLOAD_CHUNK_SIZE)):
        chunk_path = SOURCE_CHUNK_DIR / f"source_cells_chunk_{chunk_id:05d}.h5ad"
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Missing cached source chunk {chunk_path}. "
                "Run the download stage with internet access first."
            )
        chunk_paths.append(chunk_path)
    if not chunk_paths:
        raise ValueError("No cached source chunks were found.")
    return chunk_paths


def build_source_chunk_index(
    source_chunk_paths: list[Path],
) -> dict[tuple[tuple[str, str, str], str], list[tuple[int, int]]]:
    index: dict[tuple[tuple[str, str, str], str], list[tuple[int, int]]] = defaultdict(list)

    for chunk_id, chunk_path in enumerate(source_chunk_paths):
        adata = ad.read_h5ad(chunk_path, backed="r")
        obs = normalize_obs_chunk(adata.obs.reset_index(drop=True), OBS_CONTEXT_COLUMNS)
        for row_idx, row in enumerate(obs.itertuples(index=False)):
            context_key = make_context_key(row)
            index[(context_key, row.cell_type)].append((chunk_id, row_idx))
        adata.file.close()
        del adata, obs
        gc.collect()

    return index


def aggregate_selected_cached_cells(
    source_chunk_paths: list[Path],
    selected_locations: list[tuple[int, int]],
) -> sparse.csr_matrix:
    rows_by_chunk: dict[int, list[int]] = defaultdict(list)
    for chunk_id, row_idx in selected_locations:
        rows_by_chunk[int(chunk_id)].append(int(row_idx))

    aggregated = None
    for chunk_id, row_indices in rows_by_chunk.items():
        adata = ad.read_h5ad(source_chunk_paths[chunk_id])
        selected = adata.X[np.asarray(row_indices, dtype=np.int64)]
        chunk_sum = selected.sum(axis=0)
        chunk_sum = sparse.csr_matrix(chunk_sum)
        aggregated = chunk_sum if aggregated is None else aggregated + chunk_sum
        del adata, selected, chunk_sum
        gc.collect()

    if aggregated is None:
        raise ValueError("No source cells were selected for aggregation.")
    return sparse.csr_matrix(aggregated)


def generate_pseudo_bulk_chunks(
    source_adata: ad.AnnData,
    plan_rows: list[dict[str, object]],
    proportion_column_map: dict[str, str],
    gene_list: list[str],
) -> list[Path]:
    source_index = build_source_row_index(source_adata)
    chunk_paths: list[Path] = []

    for chunk_id, start in enumerate(range(0, len(plan_rows), WRITE_CHUNK_SIZE)):
        end = min(start + WRITE_CHUNK_SIZE, len(plan_rows))
        chunk_plan = plan_rows[start:end]
        out_path = CHUNK_DIR / f"pseudo_bulk_RAW_chunk_{chunk_id:05d}.h5ad"
        if RESUME and out_path.exists():
            print(f"Pseudo-bulk chunk {chunk_id}: reusing existing {out_path.name}")
            chunk_paths.append(out_path)
            continue

        rng = np.random.default_rng(RANDOM_SEED + chunk_id)

        aggregated_rows: list[sparse.csr_matrix] = []
        obs_records: list[dict[str, object]] = []

        for row in chunk_plan:
            context_key = (row["dataset_id"], row["donor_id"], row[TISSUE_COLUMN])
            selected_row_indices: list[np.ndarray] = []

            for cell_type, count in row["cell_type_counts"].items():
                pool = source_index.get((context_key, cell_type))
                if pool is None or pool.size == 0:
                    raise ValueError(
                        f"No sampled source cells available for context={context_key}, cell_type={cell_type}."
                    )
                chosen = rng.choice(pool, size=int(count), replace=pool.size < int(count))
                selected_row_indices.append(np.asarray(chosen, dtype=np.int64))

            selected_indices = np.concatenate(selected_row_indices)
            aggregated = source_adata.X[selected_indices].sum(axis=0)
            aggregated_rows.append(sparse.csr_matrix(aggregated))

            obs_record = {
                "dataset": "CELLxGENE_Census_pseudobulk",
                "dataset_id": row["dataset_id"],
                "donor_id": row["donor_id"],
                TISSUE_COLUMN: row[TISSUE_COLUMN],
                "total_cells": int(row["total_cells"]),
                "n_cell_types": int(row["n_cell_types"]),
            }
            for column in proportion_column_map.values():
                obs_record[column] = np.float32(0.0)
            for cell_type, proportion in row["cell_type_proportions"].items():
                obs_record[proportion_column_map[cell_type]] = np.float32(proportion)
            obs_records.append(obs_record)

        raw_chunk = sparse.vstack(aggregated_rows, format="csr")
        x_dense = raw_chunk.toarray().astype(np.float32, copy=False)
        x_raw, keep_mask = filter_raw_dense_block(x_dense)
        if x_raw.shape[0] == 0:
            continue

        obs = pd.DataFrame(obs_records, index=pd.Index([row["sample_id"] for row in chunk_plan], name="sample_id"))
        obs = obs.iloc[np.where(keep_mask)[0]].copy()
        var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        out = ad.AnnData(X=sparse.csr_matrix(x_raw), obs=obs, var=var)
        out.var_names = pd.Index(gene_list, dtype=str)
        store_proportion_metadata(out, proportion_column_map)

        out.write(out_path)
        chunk_paths.append(out_path)

        del raw_chunk, x_dense, x_raw, obs, var, out
        gc.collect()

        print(
            f"Pseudo-bulk chunk {chunk_id}: wrote {chunk_paths[-1].name} "
            f"for samples {start}:{end}"
        )

    return chunk_paths


def generate_pseudo_bulk_chunks_from_cached_sources(
    source_chunk_paths: list[Path],
    plan_rows: list[dict[str, object]],
    proportion_column_map: dict[str, str],
    gene_list: list[str],
) -> list[Path]:
    source_index = build_source_chunk_index(source_chunk_paths)
    chunk_paths: list[Path] = []

    for chunk_id, start in enumerate(range(0, len(plan_rows), WRITE_CHUNK_SIZE)):
        end = min(start + WRITE_CHUNK_SIZE, len(plan_rows))
        chunk_plan = plan_rows[start:end]
        out_path = CHUNK_DIR / f"pseudo_bulk_RAW_chunk_{chunk_id:05d}.h5ad"
        if RESUME and out_path.exists():
            print(f"Pseudo-bulk chunk {chunk_id}: reusing existing {out_path.name}")
            chunk_paths.append(out_path)
            continue

        rng = np.random.default_rng(RANDOM_SEED + chunk_id)
        aggregated_rows: list[sparse.csr_matrix] = []
        obs_records: list[dict[str, object]] = []

        for row in chunk_plan:
            context_key = (row["dataset_id"], row["donor_id"], row[TISSUE_COLUMN])
            selected_locations: list[tuple[int, int]] = []

            for cell_type, count in row["cell_type_counts"].items():
                pool = source_index.get((context_key, cell_type))
                if not pool:
                    raise ValueError(
                        f"No cached source cells available for context={context_key}, cell_type={cell_type}."
                    )
                chosen_idx = rng.choice(len(pool), size=int(count), replace=len(pool) < int(count))
                selected_locations.extend(pool[int(idx)] for idx in np.asarray(chosen_idx).ravel())

            aggregated_rows.append(
                aggregate_selected_cached_cells(source_chunk_paths, selected_locations)
            )

            obs_record = {
                "dataset": "CELLxGENE_Census_pseudobulk",
                "dataset_id": row["dataset_id"],
                "donor_id": row["donor_id"],
                TISSUE_COLUMN: row[TISSUE_COLUMN],
                "total_cells": int(row["total_cells"]),
                "n_cell_types": int(row["n_cell_types"]),
            }
            for column in proportion_column_map.values():
                obs_record[column] = np.float32(0.0)
            for cell_type, proportion in row["cell_type_proportions"].items():
                obs_record[proportion_column_map[cell_type]] = np.float32(proportion)
            obs_records.append(obs_record)

        raw_chunk = sparse.vstack(aggregated_rows, format="csr")
        x_dense = raw_chunk.toarray().astype(np.float32, copy=False)
        x_raw, keep_mask = filter_raw_dense_block(x_dense)
        if x_raw.shape[0] == 0:
            continue

        obs = pd.DataFrame(
            obs_records,
            index=pd.Index([row["sample_id"] for row in chunk_plan], name="sample_id"),
        )
        obs = obs.iloc[np.where(keep_mask)[0]].copy()
        var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        out = ad.AnnData(X=sparse.csr_matrix(x_raw), obs=obs, var=var)
        out.var_names = pd.Index(gene_list, dtype=str)
        store_proportion_metadata(out, proportion_column_map)

        out.write(out_path)
        chunk_paths.append(out_path)

        del raw_chunk, x_dense, x_raw, obs, var, out
        gc.collect()

        print(
            f"Pseudo-bulk chunk {chunk_id}: wrote {chunk_paths[-1].name} "
            f"for samples {start}:{end}"
        )

    return chunk_paths


def merge_chunks(chunk_paths: list[Path], proportion_column_map: dict[str, str]) -> Path:
    if len(chunk_paths) == 0:
        raise ValueError("No pseudo-bulk chunk files were created.")

    if RESUME and FINAL_OUT.exists():
        print(f"Reusing existing merged pseudo-bulk file at {FINAL_OUT}")
        return FINAL_OUT

    current_paths = list(chunk_paths)
    round_id = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for batch_id, start in enumerate(range(0, len(current_paths), MERGE_BATCH_SIZE)):
            batch_paths = current_paths[start:start + MERGE_BATCH_SIZE]
            out_path = MERGE_TMP_DIR / f"RAW_merge_r{round_id:02d}_b{batch_id:04d}.h5ad"
            print(
                f"Merging pseudo-bulk batch round {round_id}, batch {batch_id}: "
                f"{len(batch_paths)} files"
            )
            next_paths.append(merge_h5ad_group(batch_paths, out_path))
        current_paths = next_paths
        round_id += 1

    final_merged = ad.read_h5ad(current_paths[0])
    final_merged.obs_names_make_unique()
    store_proportion_metadata(final_merged, proportion_column_map)
    final_merged.write(FINAL_OUT)

    del final_merged
    gc.collect()
    print(f"Saved {FINAL_OUT}")
    return FINAL_OUT


def main() -> None:
    gene_list = read_gene_list(GENE_LIST_PATH)
    print(f"Target pseudo-bulk sample count: {TARGET_PSEUDO_BULKS}")
    print(f"Target gene count: {len(gene_list)}")

    missing_genes = read_json(MISSING_GENES_OUT) if RESUME and MISSING_GENES_OUT.exists() else None

    if RESUME and ELIGIBLE_CONTEXTS_OUT.exists():
        print(f"Reusing eligible contexts from {ELIGIBLE_CONTEXTS_OUT}")
        eligible_df, eligible_counts = load_eligible_contexts_from_csv(ELIGIBLE_CONTEXTS_OUT)
    else:
        eligible_df = pd.DataFrame()
        eligible_counts = {}

    if RESUME and PLAN_OUT.exists() and SOURCE_POOL_QUOTAS_OUT.exists():
        print(f"Reusing pseudo-bulk plan from {PLAN_OUT}")
        plan_rows, plan_df, reservoir_quotas, all_cell_types = load_sample_plan_from_csv(
            PLAN_OUT,
            SOURCE_POOL_QUOTAS_OUT,
        )
    else:
        plan_rows = []
        plan_df = pd.DataFrame()
        reservoir_quotas = {}
        all_cell_types = []

    if not OFFLINE and RESUME and SOURCE_ADATA_OUT.exists():
        print(f"Reusing downloaded source cells from {SOURCE_ADATA_OUT}")
        source_adata = ad.read_h5ad(SOURCE_ADATA_OUT)
    else:
        source_adata = None

    if not plan_rows and eligible_counts:
        plan_rows, plan_df, reservoir_quotas, all_cell_types = simulate_sample_plan(
            eligible_counts,
            TARGET_PSEUDO_BULKS,
        )
        plan_df.to_csv(PLAN_OUT, index=False)

    if missing_genes is None and (source_adata is not None or OFFLINE):
        missing_genes = []

    if OFFLINE and not eligible_counts:
        raise ValueError(
            f"SCBFM_PSEUDO_OFFLINE=1 requires cached eligible contexts at {ELIGIBLE_CONTEXTS_OUT}."
        )
    if OFFLINE and not plan_rows:
        raise ValueError(
            f"SCBFM_PSEUDO_OFFLINE=1 requires cached sampling plan at {PLAN_OUT} "
            f"and quotas at {SOURCE_POOL_QUOTAS_OUT}."
        )
    if OFFLINE and not SAMPLED_SOURCE_CELLS_OUT.exists():
        raise ValueError(
            f"SCBFM_PSEUDO_OFFLINE=1 requires cached source metadata at {SAMPLED_SOURCE_CELLS_OUT}."
        )

    need_census = (not OFFLINE) and ((not eligible_counts) or (source_adata is None))

    if need_census:
        census_api = require_cellxgene_census()
        with census_api.open_soma(
            census_version=CENSUS_VERSION,
            tiledb_config=TILEDB_CONFIG,
        ) as census:
            if missing_genes is None:
                var_coords, missing_genes = build_var_coords(census, gene_list)
                write_json(missing_genes, MISSING_GENES_OUT)
            else:
                var_coords = None

            if missing_genes:
                raise ValueError(
                    f"CELLxGENE Census is missing {len(missing_genes)} genes from gene_list.txt. "
                    f"See {MISSING_GENES_OUT}."
                )

            if not eligible_counts:
                counts_by_context = count_context_cell_types(census)
                eligible_df, eligible_counts = build_eligible_contexts(counts_by_context)
                if eligible_df.empty:
                    raise ValueError("No biologically feasible contexts were found.")
                eligible_df.to_csv(ELIGIBLE_CONTEXTS_OUT, index=False)

            if not plan_rows:
                plan_rows, plan_df, reservoir_quotas, all_cell_types = simulate_sample_plan(
                    eligible_counts,
                    TARGET_PSEUDO_BULKS,
                )
                plan_df.to_csv(PLAN_OUT, index=False)

            if source_adata is None:
                if RESUME and SAMPLED_SOURCE_CELLS_OUT.exists():
                    print(f"Reusing sampled source cell metadata from {SAMPLED_SOURCE_CELLS_OUT}")
                    sampled_source_meta = load_sampled_source_meta(SAMPLED_SOURCE_CELLS_OUT)
                else:
                    sampled_source_meta = reservoir_sample_source_cells(census, reservoir_quotas)
                    sampled_source_meta.to_csv(SAMPLED_SOURCE_CELLS_OUT, index=False)
                print(
                    f"Sampled {sampled_source_meta.shape[0]} source cells for pseudo-bulk generation"
                )

                if var_coords is None:
                    var_coords, _ = build_var_coords(census, gene_list)
                source_adata = download_source_cells(
                    census,
                    sampled_source_meta,
                    var_coords,
                    gene_list,
                )
                source_adata.write(SOURCE_ADATA_OUT)
                print(f"Cached aligned source cells at {SOURCE_ADATA_OUT}")

    if eligible_df.empty:
        raise ValueError("No biologically feasible contexts were found.")
    if not plan_rows:
        raise ValueError("No pseudo-bulk plan is available.")

    proportion_column_map = build_proportion_column_map(all_cell_types)
    if source_adata is None:
        sampled_source_meta = load_sampled_source_meta(SAMPLED_SOURCE_CELLS_OUT)
        source_chunk_paths = get_source_chunk_paths(sampled_source_meta)
        chunk_paths = generate_pseudo_bulk_chunks_from_cached_sources(
            source_chunk_paths,
            plan_rows,
            proportion_column_map,
            gene_list,
        )
        sampled_source_cells = int(sampled_source_meta.shape[0])
        source_mode = "cached_source_chunks"
    else:
        chunk_paths = generate_pseudo_bulk_chunks(
            source_adata,
            plan_rows,
            proportion_column_map,
            gene_list,
        )
        sampled_source_cells = int(source_adata.n_obs)
        source_mode = "aligned_source_adata"
    merge_chunks(chunk_paths, proportion_column_map)

    summary = {
        "census_version": CENSUS_VERSION,
        "organism": ORGANISM,
        "target_pseudo_bulks": TARGET_PSEUDO_BULKS,
        "cells_per_pseudo_bulk": CELLS_PER_PSEUDO_BULK,
        "target_gene_count": len(gene_list),
        "eligible_contexts": int(len(eligible_counts)),
        "sampled_source_cells": sampled_source_cells,
        "source_mode": source_mode,
        "distinct_cell_types": int(len(all_cell_types)),
        "tissue_column": TISSUE_COLUMN,
        "min_context_cell_types": MIN_CONTEXT_CELL_TYPES,
        "min_available_cells_per_celltype": MIN_AVAILABLE_CELLS_PER_CELLTYPE,
        "min_context_total_cells": MIN_CONTEXT_TOTAL_CELLS,
        "sparse_sample_prob": SPARSE_SAMPLE_PROB,
        "missing_genes": int(len(missing_genes)),
        "outputs": {
            "final_h5ad": str(FINAL_OUT),
            "sampling_plan_csv": str(PLAN_OUT),
            "eligible_contexts_csv": str(ELIGIBLE_CONTEXTS_OUT),
            "source_pool_quotas_csv": str(SOURCE_POOL_QUOTAS_OUT),
            "sampled_source_cells_csv": str(SAMPLED_SOURCE_CELLS_OUT),
            "source_cell_chunks_dir": str(SOURCE_CHUNK_DIR),
            "aligned_source_cells_h5ad": str(SOURCE_ADATA_OUT),
            "missing_genes_json": str(MISSING_GENES_OUT),
        },
        "resume_enabled": RESUME,
        "offline_enabled": OFFLINE,
    }
    write_json(summary, SUMMARY_OUT)


if __name__ == "__main__":
    main()
