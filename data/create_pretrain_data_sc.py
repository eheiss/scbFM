from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

try:
    import cellxgene_census
except ImportError as exc:
    raise ImportError(
        "cellxgene_census is required for create_pretrain_data_sc.py. "
        "Run this script in the census environment where the notebook works."
    ) from exc


# =========================
# Paths
# =========================

GENE_LIST_PATH = Path("/cluster/work/boeva/eheiss/datasets/gene_list.txt")

OUT_DIR = Path("/cluster/work/boeva/eheiss/datasets/preprocessed_sc")
CHUNK_DIR = OUT_DIR / "census_chunks"
MERGE_TMP_DIR = OUT_DIR / "merge_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT = OUT_DIR / "pretraining_sc_binned.h5ad"
SAMPLING_PLAN_OUT = OUT_DIR / "sampling_plan.csv"
SAMPLING_SUMMARY_OUT = OUT_DIR / "sampling_summary.json"
MISSING_GENES_OUT = OUT_DIR / "cellxgene_missing_genes.json"


# =========================
# Settings
# =========================

ORGANISM = "Homo sapiens"
CENSUS_VERSION = "2025-11-08"
BIN_NUM = 5
MIN_GENES = 200
TARGET_SUM = 1e4
TARGET_TOTAL_CELLS = 100000
DOWNLOAD_CHUNK_SIZE = 2000
MERGE_BATCH_SIZE = 16
RANDOM_SEED = 2021

TILEDB_CONFIG = {
    "py.init_buffer_bytes": 256 * 1024**2,
    "soma.init_buffer_bytes": 256 * 1024**2,
}

OBS_GROUP_COLUMNS = [
    "dataset_id",
    "cell_type",
    "tissue_general",
]

OBS_SAMPLE_COLUMNS = [
    "soma_joinid",
    "dataset_id",
    "cell_type",
    "tissue_general",
    "donor_id",
]


# =========================
# Helpers
# =========================

def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def close_backed_adata(adata: ad.AnnData) -> None:
    file_obj = getattr(adata, "file", None)
    if file_obj is not None:
        file_obj.close()


def preprocess_dense_block(
    x: np.ndarray,
    min_genes: int = MIN_GENES,
    target_sum: float = TARGET_SUM,
    bin_num: int = BIN_NUM,
) -> tuple[np.ndarray, np.ndarray]:
    n_genes_by_cell = (x > 0).sum(axis=1)
    keep_mask = n_genes_by_cell >= min_genes
    x = x[keep_mask]

    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.uint8), keep_mask

    libsize = x.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    x = x / libsize * target_sum
    x = np.log1p(x) / np.log(2.0)
    x = np.clip(np.floor(x), 0, bin_num).astype(np.uint8)
    return x, keep_mask


def write_json(data, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def normalize_obs_chunk(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col == "soma_joinid":
            df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype("string").fillna("unknown").replace("<NA>", "unknown")
            df[col] = df[col].astype(str)
    return df


def allocate_quotas(available: pd.Series, target_total: int) -> pd.Series:
    if target_total >= int(available.sum()):
        return available.astype(int)

    weights = np.sqrt(available.astype(float).to_numpy())
    raw = target_total * weights / weights.sum()
    quotas = np.floor(raw).astype(int)
    quotas = np.minimum(quotas, available.to_numpy(dtype=int))

    remaining = int(target_total - quotas.sum())
    remainders = raw - np.floor(raw)
    order = np.argsort(-remainders)

    while remaining > 0:
        advanced = False
        for idx in order:
            if quotas[idx] < int(available.iloc[idx]):
                quotas[idx] += 1
                remaining -= 1
                advanced = True
                if remaining == 0:
                    break
        if not advanced:
            break

    return pd.Series(quotas, index=available.index, dtype=int)


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
    ).tables()


def build_var_coords(census, gene_list: list[str]) -> tuple[list[int], list[str]]:
    var_df = cellxgene_census.get_var(
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


def count_sampling_groups(census) -> tuple[pd.Series, dict[str, pd.Series]]:
    dataset_counts: dict[str, int] = defaultdict(int)
    group_counts_by_dataset: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))

    for table in iter_obs_tables(census, OBS_GROUP_COLUMNS):
        df = normalize_obs_chunk(table.to_pandas(), OBS_GROUP_COLUMNS)
        for row in df.itertuples(index=False):
            dataset = row.dataset_id
            group = (row.tissue_general, row.cell_type)
            dataset_counts[dataset] += 1
            group_counts_by_dataset[dataset][group] += 1

        del df
        gc.collect()

    dataset_series = pd.Series(dataset_counts, dtype=int).sort_index()
    group_series_by_dataset = {
        dataset: pd.Series(group_counts, dtype=int).sort_index()
        for dataset, group_counts in group_counts_by_dataset.items()
    }
    return dataset_series, group_series_by_dataset


def build_group_quotas(target_total: int, dataset_counts: pd.Series, group_counts_by_dataset: dict[str, pd.Series]):
    dataset_quotas = allocate_quotas(dataset_counts, target_total)
    group_quotas: dict[tuple[str, str, str], int] = {}

    for dataset, dataset_quota in dataset_quotas.items():
        if dataset_quota <= 0:
            continue
        group_counts = group_counts_by_dataset[dataset]
        group_quota_series = allocate_quotas(group_counts, int(dataset_quota))
        for (tissue, cell_type), quota in group_quota_series.items():
            if quota > 0:
                group_quotas[(dataset, tissue, cell_type)] = int(quota)

    return dataset_quotas, group_quotas


def reservoir_sample_cells(census, group_quotas: dict[tuple[str, str, str], int]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    seen_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    reservoirs: dict[tuple[str, str, str], list[dict[str, object]]] = {
        group: [] for group, quota in group_quotas.items() if quota > 0
    }

    for table in iter_obs_tables(census, OBS_SAMPLE_COLUMNS):
        df = normalize_obs_chunk(table.to_pandas(), OBS_SAMPLE_COLUMNS)

        for row in df.itertuples(index=False):
            group = (row.dataset_id, row.tissue_general, row.cell_type)
            quota = group_quotas.get(group, 0)
            if quota <= 0:
                continue

            seen_counts[group] += 1
            sample_row = {
                "soma_joinid": int(row.soma_joinid),
                "dataset_id": row.dataset_id,
                "tissue_general": row.tissue_general,
                "cell_type": row.cell_type,
                "donor_id": row.donor_id,
            }
            reservoir = reservoirs[group]

            if len(reservoir) < quota:
                reservoir.append(sample_row)
            else:
                j = int(rng.integers(0, seen_counts[group]))
                if j < quota:
                    reservoir[j] = sample_row

        del df
        gc.collect()

    sampled_rows = [row for reservoir in reservoirs.values() for row in reservoir]
    sampled_meta = pd.DataFrame(sampled_rows)
    sampled_meta = sampled_meta.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    sampled_meta.to_csv(SAMPLING_PLAN_OUT, index=False)
    return sampled_meta


def download_and_preprocess_chunks(census, sampled_meta: pd.DataFrame, var_coords: list[int], gene_list: list[str]) -> list[Path]:
    chunk_paths: list[Path] = []

    for chunk_id, start in enumerate(range(0, sampled_meta.shape[0], DOWNLOAD_CHUNK_SIZE)):
        end = min(start + DOWNLOAD_CHUNK_SIZE, sampled_meta.shape[0])
        meta_chunk = sampled_meta.iloc[start:end].copy()
        obs_coords = meta_chunk["soma_joinid"].astype(np.int64).tolist()

        print(f"Downloading CELLxGENE chunk {chunk_id}: sampled cells {start}:{end}")
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=ORGANISM,
            obs_coords=obs_coords,
            var_coords=var_coords,
            obs_column_names=OBS_SAMPLE_COLUMNS,
            var_column_names=["feature_id"],
        )

        feature_ids = adata.var["feature_id"].astype(str).tolist()
        feature_index = pd.Index(feature_ids)
        reorder_idx = feature_index.get_indexer(gene_list)
        if (reorder_idx < 0).any():
            raise ValueError("Downloaded chunk is missing requested genes after Census filtering.")

        adata = adata[:, reorder_idx].copy()
        adata.var_names = pd.Index(gene_list, dtype=str)

        if sparse.issparse(adata.X):
            x = adata.X.toarray().astype(np.float32)
        else:
            x = np.asarray(adata.X, dtype=np.float32)

        x, keep_mask = preprocess_dense_block(x)
        if x.shape[0] == 0:
            del adata, x
            gc.collect()
            continue

        obs = adata.obs.iloc[np.where(keep_mask)[0]].copy()
        obs["dataset"] = "CELLxGENE_Census"
        obs.index = pd.Index([f"cellxgene:{x}" for x in obs["soma_joinid"].astype(str)], name="cell_id")
        var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        out = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
        out.var_names = pd.Index(gene_list, dtype=str)

        out_path = CHUNK_DIR / f"cellxgene_chunk_{chunk_id:05d}.h5ad"
        out.write(out_path)
        chunk_paths.append(out_path)

        del adata, x, obs, var, out
        gc.collect()

    return chunk_paths


def merge_chunks(chunk_paths: list[Path]) -> Path:
    if len(chunk_paths) == 0:
        raise ValueError("No CELLxGENE chunk files were created.")

    current_paths = list(chunk_paths)
    round_id = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for batch_id, start in enumerate(range(0, len(current_paths), MERGE_BATCH_SIZE)):
            batch_paths = current_paths[start:start + MERGE_BATCH_SIZE]
            out_path = MERGE_TMP_DIR / f"merge_r{round_id:02d}_b{batch_id:04d}.h5ad"
            print(
                f"Merging CELLxGENE batch round {round_id}, batch {batch_id}: "
                f"{len(batch_paths)} files"
            )
            next_paths.append(merge_h5ad_group(batch_paths, out_path))
        current_paths = next_paths
        round_id += 1

    final_merged = ad.read_h5ad(current_paths[0])
    final_merged.obs_names_make_unique()
    print(f"Pretraining single-cell merged: {final_merged.n_obs} samples x {final_merged.n_vars} genes")
    final_merged.write(FINAL_OUT)

    del final_merged
    gc.collect()
    print(f"Saved {FINAL_OUT}")
    return FINAL_OUT


def main() -> None:
    gene_list = read_gene_list(GENE_LIST_PATH)
    target_total = TARGET_TOTAL_CELLS

    print(f"Target single-cell sample count: {target_total}")
    print(f"Target gene count: {len(gene_list)}")

    with cellxgene_census.open_soma(
        census_version=CENSUS_VERSION,
        tiledb_config=TILEDB_CONFIG,
    ) as census:
        var_coords, missing_genes = build_var_coords(census, gene_list)
        write_json(missing_genes, MISSING_GENES_OUT)
        if missing_genes:
            raise ValueError(
                f"CELLxGENE Census is missing {len(missing_genes)} genes from gene_list.txt. "
                f"See {MISSING_GENES_OUT}."
            )

        dataset_counts, group_counts_by_dataset = count_sampling_groups(census)
        dataset_quotas, group_quotas = build_group_quotas(
            target_total,
            dataset_counts,
            group_counts_by_dataset,
        )
        sampled_meta = reservoir_sample_cells(census, group_quotas)

        summary = {
            "census_version": CENSUS_VERSION,
            "target_total_cells": int(target_total),
            "sampled_total_cells": int(sampled_meta.shape[0]),
            "target_gene_count": int(len(gene_list)),
            "available_primary_cells": int(dataset_counts.sum()),
            "datasets_seen": int(dataset_counts.shape[0]),
            "datasets_sampled": int((dataset_quotas > 0).sum()),
            "sampling_groups_sampled": int(sum(q > 0 for q in group_quotas.values())),
            "missing_genes": int(len(missing_genes)),
        }
        write_json(summary, SAMPLING_SUMMARY_OUT)

        chunk_paths = download_and_preprocess_chunks(census, sampled_meta, var_coords, gene_list)

    merge_chunks(chunk_paths)


if __name__ == "__main__":
    main()
