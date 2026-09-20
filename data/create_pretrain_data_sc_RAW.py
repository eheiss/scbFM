from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
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

ROOT_DIR = Path(os.environ.get("SCBFM_ROOT_DIR", Path(__file__).resolve().parents[2])).expanduser().resolve()

GENE_LIST_PATH = Path(os.getenv(
    "SCBFM_GENE_LIST_PATH",
    str(Path(__file__).resolve().parent / "gene_list.txt"),
))

OUT_DIR = Path(os.getenv(
    "SCBFM_SC_OUT_DIR",
    str(ROOT_DIR / "datasets/sc"),
))
CHUNK_DIR = OUT_DIR / "census_RAW_chunks"
MERGE_TMP_DIR = OUT_DIR / "RAW_merge_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT = OUT_DIR / "pretraining_sc_RAW.h5ad"
SAMPLING_PLAN_OUT = OUT_DIR / "sampling_plan_RAW.csv"
SAMPLING_SUMMARY_OUT = OUT_DIR / "sampling_summary_RAW.json"
MISSING_GENES_OUT = OUT_DIR / "cellxgene_missing_genes_RAW.json"


# =========================
# Settings
# =========================

ORGANISM = "Homo sapiens"
CENSUS_ORGANISM_KEY = "homo_sapiens"
CENSUS_VERSION = os.getenv("SCBFM_SC_CENSUS_VERSION", "2025-11-08")
MIN_GENES = 200
TARGET_TOTAL_CELLS = int(os.getenv(
    "SCBFM_SC_TARGET_TOTAL_CELLS",
    os.getenv("SCBFM_TARGET_TOTAL_CELLS", "642406"),
))
DOWNLOAD_CHUNK_SIZE = int(os.getenv(
    "SCBFM_SC_DOWNLOAD_CHUNK_SIZE",
    os.getenv("SCBFM_DOWNLOAD_CHUNK_SIZE", "5000"),
))
PROCESS_BATCH_SIZE = int(os.getenv(
    "SCBFM_SC_PROCESS_BATCH_SIZE",
    os.getenv("SCBFM_PROCESS_BATCH_SIZE", "2048"),
))
MERGE_BATCH_SIZE = int(os.getenv(
    "SCBFM_SC_MERGE_BATCH_SIZE",
    os.getenv("SCBFM_MERGE_BATCH_SIZE", "4"),
))
RANDOM_SEED = int(os.getenv("SCBFM_SC_RANDOM_SEED", "2021"))
INITIAL_OVERDRAW_FACTOR = float(os.getenv("SCBFM_SC_INITIAL_OVERDRAW_FACTOR", "1.10"))
MAX_SAMPLING_ATTEMPTS = int(os.getenv("SCBFM_SC_MAX_SAMPLING_ATTEMPTS", "4"))
RESUME = os.getenv("SCBFM_SC_RESUME", "1") != "0"
OFFLINE = os.getenv("SCBFM_SC_OFFLINE", "0") == "1"

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

ATTEMPT_CHUNK_RE = re.compile(r"cellxgene_RAW_a(\d+)_chunk_\d+\.h5ad$")


# =========================
# Helpers
# =========================

def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def validate_cached_final(gene_list: list[str], target_total: int) -> bool:
    if not FINAL_OUT.exists():
        return False

    cached_final = ad.read_h5ad(FINAL_OUT, backed="r")
    try:
        cached_gene_list = list(cached_final.var_names)
        cached_shape = tuple(map(int, cached_final.shape))
    finally:
        close_backed_adata(cached_final)

    if cached_gene_list == gene_list and cached_shape == (target_total, len(gene_list)):
        print(
            f"Reusing existing single-cell file at {FINAL_OUT}: "
            f"{cached_shape[0]} samples x {cached_shape[1]} genes"
        )
        return True

    print(
        f"Ignoring cached single-cell file at {FINAL_OUT} because it does not "
        f"match the current target/gene order (found {cached_shape[0]} x {cached_shape[1]})."
    )
    return False


def close_backed_adata(adata: ad.AnnData) -> None:
    file_obj = getattr(adata, "file", None)
    if file_obj is not None:
        file_obj.close()


def filter_raw_dense_block(
    x: np.ndarray,
    min_genes: int = MIN_GENES,
) -> tuple[np.ndarray, np.ndarray]:
    n_genes_by_cell = (x > 0).sum(axis=1)
    keep_mask = n_genes_by_cell >= min_genes
    x = x[keep_mask]

    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32), keep_mask

    return x.astype(np.float32, copy=False), keep_mask


def write_json(data, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def require_cellxgene_census():
    global cellxgene_census
    if cellxgene_census is None:
        try:
            import cellxgene_census as census_module
        except ImportError as exc:
            raise ImportError(
                "cellxgene_census is required for create_pretrain_data_sc_RAW.py "
                "unless SCBFM_SC_OFFLINE=1 and all needed sampling plans/chunks "
                "or the final h5ad are already cached."
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


def estimate_dense_chunk_gib(n_obs: int, n_vars: int, dtype_bytes: int = 4) -> float:
    return n_obs * n_vars * dtype_bytes / 1024**3


def sampling_plan_attempt_path(attempt_id: int) -> Path:
    return OUT_DIR / f"sampling_plan_RAW_a{attempt_id:02d}.csv"


def write_sampling_plan(sampled_meta: pd.DataFrame, attempt_id: int) -> None:
    attempt_path = sampling_plan_attempt_path(attempt_id)
    sampled_meta.to_csv(attempt_path, index=False)
    sampled_meta.to_csv(SAMPLING_PLAN_OUT, index=False)


def load_sampled_meta(path: Path) -> pd.DataFrame:
    sampled_meta = pd.read_csv(path)
    sampled_meta = normalize_obs_chunk(sampled_meta, OBS_SAMPLE_COLUMNS)
    return sampled_meta.sort_values("soma_joinid").reset_index(drop=True)


def infer_legacy_sampling_plan_attempt_id() -> int | None:
    if not SAMPLING_PLAN_OUT.exists():
        return None

    attempt_ids = []
    for path in CHUNK_DIR.glob("cellxgene_RAW_a*_chunk_*.h5ad"):
        match = ATTEMPT_CHUNK_RE.match(path.name)
        if match is not None:
            attempt_ids.append(int(match.group(1)))

    if attempt_ids:
        return max(attempt_ids)
    return 0


def resolve_cached_sampling_plan_path(attempt_id: int, legacy_attempt_id: int | None) -> Path | None:
    attempt_path = sampling_plan_attempt_path(attempt_id)
    if attempt_path.exists():
        return attempt_path
    if legacy_attempt_id is not None and attempt_id == legacy_attempt_id and SAMPLING_PLAN_OUT.exists():
        return SAMPLING_PLAN_OUT
    return None


def merge_h5ad_group(paths: list[Path], out_path: Path) -> Path:
    adatas = [ad.read_h5ad(p) for p in paths]
    merged = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    merged.write(out_path)

    del adatas, merged
    gc.collect()
    return out_path


def get_census_experiment(census):
    census_data = census["census_data"]
    for organism_key in (CENSUS_ORGANISM_KEY, ORGANISM):
        try:
            return census_data[organism_key]
        except KeyError:
            continue
    try:
        available = list(census_data.keys())
    except Exception:
        available = []
    raise KeyError(
        f"Could not find CELLxGENE Census organism collection for "
        f"{CENSUS_ORGANISM_KEY!r}. Available keys: {available}"
    )


def iter_obs_tables(census, column_names: list[str]):
    exp = get_census_experiment(census)
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
    sampled_meta = sampled_meta.sort_values("soma_joinid").reset_index(drop=True)
    return sampled_meta


class MissingChunkDownloadRequired(RuntimeError):
    pass


def download_and_preprocess_chunks(
    census,
    sampled_meta: pd.DataFrame,
    var_coords: list[int] | None,
    gene_list: list[str],
    target_total: int,
    attempt_id: int,
) -> tuple[list[Path], int]:
    chunk_paths: list[Path] = []
    kept_total = 0

    for chunk_id, start in enumerate(range(0, sampled_meta.shape[0], DOWNLOAD_CHUNK_SIZE)):
        if kept_total >= target_total:
            break

        end = min(start + DOWNLOAD_CHUNK_SIZE, sampled_meta.shape[0])
        out_path = CHUNK_DIR / f"cellxgene_RAW_a{attempt_id:02d}_chunk_{chunk_id:05d}.h5ad"
        if RESUME and out_path.exists():
            cached = ad.read_h5ad(out_path, backed="r")
            cached_gene_list = list(cached.var_names)
            if cached_gene_list != gene_list:
                close_backed_adata(cached)
                raise ValueError(
                    f"Cached chunk {out_path} does not match the current gene_list.txt ordering."
                )
            reused_n_obs = int(cached.n_obs)
            remaining = target_total - kept_total
            if reused_n_obs > remaining:
                close_backed_adata(cached)
                raise ValueError(
                    f"Cached chunk {out_path} keeps {reused_n_obs} cells but only {remaining} "
                    "fit the current target. Remove cached chunks or set SCBFM_SC_RESUME=0."
                )
            close_backed_adata(cached)
            chunk_paths.append(out_path)
            kept_total += reused_n_obs
            print(
                f"Reusing CELLxGENE chunk {chunk_id}: sampled cells {start}:{end} "
                f"(kept {reused_n_obs}, cumulative {kept_total}/{target_total})"
            )
            gc.collect()
            continue

        if var_coords is None:
            raise MissingChunkDownloadRequired(
                f"Missing cached chunk {out_path}; Census download is required."
            )

        meta_chunk = sampled_meta.iloc[start:end].copy()
        obs_coords = meta_chunk["soma_joinid"].astype(np.int64).tolist()

        print(f"Downloading CELLxGENE chunk {chunk_id}: sampled cells {start}:{end}")
        t0 = time.perf_counter()
        census_api = require_cellxgene_census()
        adata = census_api.get_anndata(
            census=census,
            organism=ORGANISM,
            obs_coords=obs_coords,
            var_coords=var_coords,
            obs_column_names=OBS_SAMPLE_COLUMNS,
            var_column_names=["feature_id"],
        )
        t1 = time.perf_counter()
        print(
            f"Chunk {chunk_id}: download finished in {t1 - t0:.1f}s "
            f"with shape {adata.n_obs} x {adata.n_vars}"
        )

        feature_ids = adata.var["feature_id"].astype(str).tolist()
        feature_index = pd.Index(feature_ids)
        reorder_idx = feature_index.get_indexer(gene_list)
        if (reorder_idx < 0).any():
            raise ValueError("Downloaded chunk is missing requested genes after Census filtering.")

        adata = adata[:, reorder_idx].copy()
        adata.var_names = pd.Index(gene_list, dtype=str)
        t2 = time.perf_counter()
        print(f"Chunk {chunk_id}: reorder finished in {t2 - t1:.1f}s")

        estimated_dense_gib = estimate_dense_chunk_gib(adata.n_obs, adata.n_vars)
        print(
            f"Chunk {chunk_id}: full dense float32 materialization would require "
            f"~{estimated_dense_gib:.2f} GiB; processing in row batches of {PROCESS_BATCH_SIZE}"
        )

        x_blocks: list[sparse.csr_matrix] = []
        obs_blocks: list[pd.DataFrame] = []

        for batch_start in range(0, adata.n_obs, PROCESS_BATCH_SIZE):
            batch_end = min(batch_start + PROCESS_BATCH_SIZE, adata.n_obs)
            x_batch = adata.X[batch_start:batch_end]
            obs_batch = adata.obs.iloc[batch_start:batch_end]

            if sparse.issparse(x_batch):
                keep_mask = np.asarray((x_batch > 0).sum(axis=1)).ravel() >= MIN_GENES
                if not keep_mask.any():
                    continue
                x_dense = x_batch[keep_mask].toarray().astype(np.float32, copy=False)
            else:
                x_dense = np.asarray(x_batch, dtype=np.float32)
                x_dense, keep_mask = filter_raw_dense_block(x_dense)
                if x_dense.shape[0] == 0:
                    continue
                x_blocks.append(sparse.csr_matrix(x_dense))
                obs_blocks.append(obs_batch.iloc[np.where(keep_mask)[0]].copy())
                continue

            x_blocks.append(sparse.csr_matrix(x_dense))
            obs_blocks.append(obs_batch.iloc[np.where(keep_mask)[0]].copy())

            del x_batch, obs_batch, x_dense, keep_mask
            gc.collect()

        t3 = time.perf_counter()
        print(f"Chunk {chunk_id}: batched raw filtering finished in {t3 - t2:.1f}s")

        if len(x_blocks) == 0:
            del adata, x_blocks, obs_blocks
            gc.collect()
            continue

        x = sparse.vstack(x_blocks, format="csr")
        obs = pd.concat(obs_blocks, axis=0).copy()
        del x_blocks, obs_blocks
        gc.collect()

        t4 = time.perf_counter()
        print(f"Chunk {chunk_id}: sparse assembly finished in {t4 - t3:.1f}s")

        remaining = target_total - kept_total
        if x.shape[0] > remaining:
            x = x[:remaining]
            obs = obs.iloc[:remaining].copy()

        obs["dataset"] = "CELLxGENE_Census"
        obs.index = pd.Index([f"cellxgene:{sid}" for sid in obs["soma_joinid"].astype(str)], name="cell_id")
        var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        out = ad.AnnData(X=x, obs=obs, var=var)
        out.var_names = pd.Index(gene_list, dtype=str)

        out.write(out_path)
        chunk_paths.append(out_path)
        kept_total += out.n_obs
        t5 = time.perf_counter()
        print(
            f"Chunk {chunk_id}: write finished in {t5 - t4:.1f}s "
            f"(kept {out.n_obs}, cumulative {kept_total}/{target_total})"
        )

        del adata, x, obs, var, out
        gc.collect()

    return chunk_paths, kept_total


def merge_chunks(chunk_paths: list[Path], gene_list: list[str], target_total: int) -> Path:
    if len(chunk_paths) == 0:
        raise ValueError("No CELLxGENE chunk files were created.")

    if RESUME and validate_cached_final(gene_list, target_total):
        return FINAL_OUT

    current_paths = list(chunk_paths)
    round_id = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for batch_id, start in enumerate(range(0, len(current_paths), MERGE_BATCH_SIZE)):
            batch_paths = current_paths[start:start + MERGE_BATCH_SIZE]
            out_path = MERGE_TMP_DIR / f"RAW_merge_r{round_id:02d}_b{batch_id:04d}.h5ad"
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
    print(f"Output directory: {OUT_DIR}")
    print(f"Resume enabled: {RESUME}")
    print(f"Offline mode: {OFFLINE}")

    if DOWNLOAD_CHUNK_SIZE <= 0:
        raise ValueError("SCBFM_SC_DOWNLOAD_CHUNK_SIZE must be positive.")
    if PROCESS_BATCH_SIZE <= 0:
        raise ValueError("SCBFM_SC_PROCESS_BATCH_SIZE must be positive.")
    if MERGE_BATCH_SIZE <= 0:
        raise ValueError("SCBFM_SC_MERGE_BATCH_SIZE must be positive.")
    if target_total <= 0:
        raise ValueError("SCBFM_SC_TARGET_TOTAL_CELLS must be positive.")
    if OFFLINE and not RESUME:
        raise ValueError("SCBFM_SC_OFFLINE=1 requires SCBFM_SC_RESUME=1.")
    if RESUME and validate_cached_final(gene_list, target_total):
        return

    previous_summary = read_json(SAMPLING_SUMMARY_OUT) if RESUME and SAMPLING_SUMMARY_OUT.exists() else {}
    missing_genes = read_json(MISSING_GENES_OUT) if RESUME and MISSING_GENES_OUT.exists() else None
    legacy_plan_attempt_id = infer_legacy_sampling_plan_attempt_id()
    dataset_counts = None
    group_counts_by_dataset = None
    var_coords = None
    census = None

    with ExitStack() as stack:
        def ensure_census():
            nonlocal census
            if census is None:
                census_api = require_cellxgene_census()
                census = stack.enter_context(
                    census_api.open_soma(
                        census_version=CENSUS_VERSION,
                        tiledb_config=TILEDB_CONFIG,
                    )
                )
            return census

        attempt_summaries = []
        chunk_paths = []

        for attempt_id in range(MAX_SAMPLING_ATTEMPTS):
            overdraw_factor = INITIAL_OVERDRAW_FACTOR * (2 ** attempt_id)
            candidate_total = math.ceil(target_total * overdraw_factor)
            print(
                f"Sampling attempt {attempt_id + 1}/{MAX_SAMPLING_ATTEMPTS}: "
                f"{candidate_total} candidate cells (factor {overdraw_factor:.2f})"
            )

            cached_plan_path = resolve_cached_sampling_plan_path(attempt_id, legacy_plan_attempt_id)
            if RESUME and cached_plan_path is not None:
                print(f"Reusing sampling plan for attempt {attempt_id} from {cached_plan_path}")
                sampled_meta = load_sampled_meta(cached_plan_path)
                datasets_sampled = int(sampled_meta["dataset_id"].nunique())
                sampling_groups_sampled = int(
                    sampled_meta[["dataset_id", "tissue_general", "cell_type"]]
                    .drop_duplicates()
                    .shape[0]
                )
            else:
                if OFFLINE:
                    raise RuntimeError(
                        f"SCBFM_SC_OFFLINE=1 requires a cached sampling plan for attempt "
                        f"{attempt_id} at {sampling_plan_attempt_path(attempt_id)} "
                        f"or {SAMPLING_PLAN_OUT}."
                    )
                if dataset_counts is None or group_counts_by_dataset is None:
                    dataset_counts, group_counts_by_dataset = count_sampling_groups(ensure_census())

                dataset_quotas, group_quotas = build_group_quotas(
                    candidate_total,
                    dataset_counts,
                    group_counts_by_dataset,
                )
                sampled_meta = reservoir_sample_cells(ensure_census(), group_quotas)
                write_sampling_plan(sampled_meta, attempt_id)
                legacy_plan_attempt_id = attempt_id
                datasets_sampled = int((dataset_quotas > 0).sum())
                sampling_groups_sampled = int(sum(q > 0 for q in group_quotas.values()))

            try:
                chunk_paths, kept_total = download_and_preprocess_chunks(
                    census,
                    sampled_meta,
                    var_coords,
                    gene_list,
                    target_total=target_total,
                    attempt_id=attempt_id,
                )
            except MissingChunkDownloadRequired as exc:
                if OFFLINE:
                    raise RuntimeError(
                        "SCBFM_SC_OFFLINE=1 requires all CELLxGENE RAW chunks for the "
                        f"cached sampling plan to exist under {CHUNK_DIR}. "
                        f"Missing chunk detail: {exc}"
                    ) from exc
                if var_coords is None:
                    var_coords, missing_genes = build_var_coords(ensure_census(), gene_list)
                    write_json(missing_genes, MISSING_GENES_OUT)
                    if missing_genes:
                        raise ValueError(
                            f"CELLxGENE Census is missing {len(missing_genes)} genes from gene_list.txt. "
                            f"See {MISSING_GENES_OUT}."
                        )

                chunk_paths, kept_total = download_and_preprocess_chunks(
                    census,
                    sampled_meta,
                    var_coords,
                    gene_list,
                    target_total=target_total,
                    attempt_id=attempt_id,
                )

            attempt_summary = {
                "attempt_id": attempt_id,
                "overdraw_factor": overdraw_factor,
                "candidate_total_requested": int(candidate_total),
                "candidate_total_sampled": int(sampled_meta.shape[0]),
                "kept_total_after_filtering": int(kept_total),
                "datasets_sampled": datasets_sampled,
                "sampling_groups_sampled": sampling_groups_sampled,
            }
            attempt_summaries.append(attempt_summary)

            if kept_total >= target_total:
                break

        if missing_genes is None:
            missing_genes = []
        available_primary_cells = (
            int(dataset_counts.sum())
            if dataset_counts is not None
            else previous_summary.get("available_primary_cells")
        )
        datasets_seen = (
            int(dataset_counts.shape[0])
            if dataset_counts is not None
            else previous_summary.get("datasets_seen")
        )

        if not chunk_paths or attempt_summaries[-1]["kept_total_after_filtering"] < target_total:
            write_json(
                {
                    "census_version": CENSUS_VERSION,
                    "target_total_cells": int(target_total),
                    "target_gene_count": int(len(gene_list)),
                    "available_primary_cells": available_primary_cells,
                    "datasets_seen": datasets_seen,
                    "missing_genes": int(len(missing_genes)),
                    "attempts": attempt_summaries,
                    "resume_enabled": RESUME,
                },
                SAMPLING_SUMMARY_OUT,
            )
            raise RuntimeError(
                f"Unable to retain {target_total} cells after filtering. "
                f"Last kept count: {attempt_summaries[-1]['kept_total_after_filtering']}."
            )

        summary = {
            "census_version": CENSUS_VERSION,
            "target_total_cells": int(target_total),
            "target_gene_count": int(len(gene_list)),
            "available_primary_cells": available_primary_cells,
            "datasets_seen": datasets_seen,
            "missing_genes": int(len(missing_genes)),
            "attempts": attempt_summaries,
            "resume_enabled": RESUME,
        }
        write_json(summary, SAMPLING_SUMMARY_OUT)

    final_path = merge_chunks(chunk_paths, gene_list, target_total)
    final = ad.read_h5ad(final_path, backed="r")
    try:
        print(f"Final single-cell dataset dimensions: {final.n_obs} samples x {final.n_vars} genes")
    finally:
        close_backed_adata(final)


if __name__ == "__main__":
    main()
