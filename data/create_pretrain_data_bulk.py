from __future__ import annotations

from pathlib import Path
import gc
import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# =========================
# Paths
# =========================

GENE_LIST_PATH = Path(__file__).resolve().parent / "gene_list.txt"
GTEX_PATH = Path("/cluster/work/boeva/eheiss/datasets/GTEx/gtex.h5ad")
ARCHS4_PATH = Path("/cluster/work/boeva/eheiss/datasets/ARCHS4/human_gene_v2.latest.h5")

OUT_DIR = Path("/cluster/work/boeva/eheiss/datasets/preprocessed_bulk")
ARCHS4_CHUNK_DIR = OUT_DIR / "archs4_chunks"
ARCHS4_MERGE_TMP_DIR = OUT_DIR / "archs4_merge_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHS4_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
ARCHS4_MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)

GTEX_OUT = OUT_DIR / "gtex_binned.h5ad"
ARCHS4_OUT = OUT_DIR / "archs4_binned.h5ad"
PRETRAIN_OUT = OUT_DIR / "pretraining_binned.h5ad"


# =========================
# Settings
# =========================

BIN_NUM = 5
MIN_GENES = 200
TARGET_SUM = 1e4
ARCHS4_CHUNK_SIZE = 2000  # samples per chunk before cell filtering
MERGE_BATCH_SIZE = 16


# =========================
# Helpers
# =========================

def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def decode_bytes_array(arr) -> list[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def build_reindexer(source_gene_ids: list[str] | pd.Index, target_gene_list: list[str]) -> tuple[list[int], list[int], list[str]]:
    """
    Returns:
        src_pos: positions in source
        tgt_pos: positions in target gene list
        missing: target genes not present in source
    Preserves exact order of target_gene_list.
    """
    first_pos = {}
    for i, g in enumerate(source_gene_ids):
        if g not in first_pos:
            first_pos[g] = i

    src_pos = []
    tgt_pos = []
    missing = []

    for j, g in enumerate(target_gene_list):
        if g in first_pos:
            src_pos.append(first_pos[g])
            tgt_pos.append(j)
        else:
            missing.append(g)

    return src_pos, tgt_pos, missing


def place_into_target_order(x_present: np.ndarray, tgt_pos: list[int], total_genes: int, dtype=np.float32) -> np.ndarray:
    """
    x_present: (n_cells, n_present_genes)
    returns dense array (n_cells, total_genes), zeros for missing genes,
    with columns in exact order of target gene list.
    """
    out = np.zeros((x_present.shape[0], total_genes), dtype=dtype)
    out[:, tgt_pos] = x_present
    return out


def preprocess_dense_block(
    x: np.ndarray,
    min_genes: int = MIN_GENES,
    target_sum: float = TARGET_SUM,
    bin_num: int = BIN_NUM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    x: dense float array, shape (cells, genes)
    returns:
        x_binned_uint8: shape (kept_cells, genes)
        keep_mask: bool mask over original cells
    """
    # filter_cells(min_genes=200)
    n_genes_by_cell = (x > 0).sum(axis=1)
    keep_mask = n_genes_by_cell >= min_genes
    x = x[keep_mask]

    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.uint8), keep_mask

    # normalize_total(target_sum=1e4)
    libsize = x.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    x = x / libsize * target_sum

    # log1p(base=2)
    x = np.log1p(x) / np.log(2.0)

    # floor + clip to [0, bin_num]
    x = np.clip(np.floor(x), 0, bin_num).astype(np.uint8)

    return x, keep_mask


def write_manifest(paths: list[Path], out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump([str(p) for p in paths], f, indent=2)


def merge_h5ad_group(paths: list[Path], out_path: Path) -> Path:
    adatas = []
    for p in paths:
        adatas.append(ad.read_h5ad(p))

    merged = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    merged.write(out_path)

    del adatas, merged
    gc.collect()

    return out_path


# =========================
# GTEx
# =========================

def preprocess_gtex(gene_list: list[str]) -> Path:
    print("Loading GTEx...")
    adata = ad.read_h5ad(GTEX_PATH)

    gtex_gene_ids = adata.var_names.astype(str)
    src_pos, tgt_pos, missing = build_reindexer(gtex_gene_ids, gene_list)

    print(f"GTEx genes present: {len(src_pos)} / {len(gene_list)}")
    print(f"GTEx genes missing: {len(missing)}")

    if sparse.issparse(adata.X):
        x_present = adata.X[:, src_pos].toarray().astype(np.float32)
    else:
        x_present = np.asarray(adata.X[:, src_pos], dtype=np.float32)

    x = place_into_target_order(x_present, tgt_pos, len(gene_list), dtype=np.float32)
    del x_present
    gc.collect()

    x, keep_mask = preprocess_dense_block(x)

    obs = adata.obs.iloc[np.where(keep_mask)[0]].copy()
    obs["dataset"] = "GTEx"

    var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

    out = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
    out.var_names = pd.Index(gene_list, dtype=str)

    out.write(GTEX_OUT)

    with open(OUT_DIR / "gtex_missing_genes.json", "w") as f:
        json.dump(missing, f)

    print(f"Saved {GTEX_OUT}")
    return GTEX_OUT


# =========================
# ARCHS4 -> chunks
# =========================

def preprocess_archs4_to_chunks(gene_list: list[str], chunk_size: int = ARCHS4_CHUNK_SIZE) -> list[Path]:
    print("Preparing ARCHS4 mappings...")
    with h5py.File(ARCHS4_PATH, "r") as f:
        archs4_gene_ids = decode_bytes_array(f["meta/genes/ensembl_gene"][:])
        sc_prob = np.asarray(f["meta/samples/singlecellprobability"][:], dtype=np.float32)
        sample_ids = decode_bytes_array(f["meta/samples/sample"][:])

    src_pos, tgt_pos, missing = build_reindexer(archs4_gene_ids, gene_list)
    bulk_like_idx = np.where(sc_prob < 0.5)[0]   # keep bulk-like samples

    print(f"ARCHS4 genes present: {len(src_pos)} / {len(gene_list)}")
    print(f"ARCHS4 genes missing: {len(missing)}")
    print(f"ARCHS4 kept samples (singlecellprobability < 0.5): {len(bulk_like_idx)} / {len(sc_prob)}")

    with open(OUT_DIR / "archs4_missing_genes.json", "w") as f:
        json.dump(missing, f)

    written_chunks: list[Path] = []

    with h5py.File(ARCHS4_PATH, "r") as f:
        expr = f["data/expression"]  # shape: (genes, samples)

        for chunk_id, start in enumerate(range(0, len(bulk_like_idx), chunk_size)):
            end = min(start + chunk_size, len(bulk_like_idx))
            cols = bulk_like_idx[start:end]

            print(f"ARCHS4 chunk {chunk_id}: source samples {start}:{end}")

            # Read only the selected sample block first to avoid materializing
            # a huge intermediate array over all ARCHS4 samples.
            sample_block = np.asarray(expr[:, cols], dtype=np.float32)
            x_present = sample_block[src_pos, :].T
            del sample_block

            # Put into exact target gene order
            x = place_into_target_order(x_present, tgt_pos, len(gene_list), dtype=np.float32)
            del x_present
            gc.collect()

            # Notebook preprocessing
            x, keep_mask = preprocess_dense_block(x)

            kept_cols = cols[np.where(keep_mask)[0]]
            obs = pd.DataFrame(index=pd.Index([sample_ids[i] for i in kept_cols], name="sample_id"))
            obs["singlecellprobability"] = sc_prob[kept_cols]
            obs["dataset"] = "ARCHS4"

            var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))
            adata_chunk = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
            adata_chunk.var_names = pd.Index(gene_list, dtype=str)

            out_path = ARCHS4_CHUNK_DIR / f"archs4_binned_chunk_{chunk_id:05d}.h5ad"
            adata_chunk.write(out_path)
            written_chunks.append(out_path)

            del x, obs, var, adata_chunk
            gc.collect()

    write_manifest(written_chunks, OUT_DIR / "archs4_chunk_manifest.json")
    print(f"Wrote {len(written_chunks)} ARCHS4 chunks")
    return written_chunks


# =========================
# ARCHS4 chunks -> one h5ad
# =========================

def merge_archs4_chunks(chunk_paths: list[Path]) -> Path:
    print("Merging ARCHS4 chunks...")
    if len(chunk_paths) == 0:
        raise ValueError("No ARCHS4 chunk files found.")

    current_paths = list(chunk_paths)
    round_id = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for batch_id, start in enumerate(range(0, len(current_paths), MERGE_BATCH_SIZE)):
            batch_paths = current_paths[start:start + MERGE_BATCH_SIZE]
            out_path = ARCHS4_MERGE_TMP_DIR / f"archs4_merge_r{round_id:02d}_b{batch_id:04d}.h5ad"
            print(
                f"Merging ARCHS4 batch round {round_id}, batch {batch_id}: "
                f"{len(batch_paths)} files"
            )
            next_paths.append(merge_h5ad_group(batch_paths, out_path))
        current_paths = next_paths
        round_id += 1

    final_merged = ad.read_h5ad(current_paths[0])
    final_merged.obs_names_make_unique()
    final_merged.write(ARCHS4_OUT)

    del final_merged
    gc.collect()

    print(f"Saved {ARCHS4_OUT}")
    return ARCHS4_OUT


# =========================
# GTEx + ARCHS4 -> one h5ad
# =========================

def merge_pretraining(gtex_path: Path, archs4_path: Path) -> Path:
    print("Merging GTEx + ARCHS4...")
    gtex = ad.read_h5ad(gtex_path)
    archs4 = ad.read_h5ad(archs4_path)

    merged = ad.concat([gtex, archs4], axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    print(f"Pretraining merged: {merged.n_obs} samples x {merged.n_vars} genes")
    merged.write(PRETRAIN_OUT)

    del gtex, archs4, merged
    gc.collect()

    print(f"Saved {PRETRAIN_OUT}")
    return PRETRAIN_OUT


# =========================
# Main
# =========================

def main():
    gene_list = read_gene_list(GENE_LIST_PATH)
    print(f"Gene list length: {len(gene_list)}")

    gtex_path = preprocess_gtex(gene_list)
    archs4_chunks = preprocess_archs4_to_chunks(gene_list, chunk_size=ARCHS4_CHUNK_SIZE)
    archs4_path = merge_archs4_chunks(archs4_chunks)
    merge_pretraining(gtex_path, archs4_path)


if __name__ == "__main__":
    main()
