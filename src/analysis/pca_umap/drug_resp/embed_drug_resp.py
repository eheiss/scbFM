#!/usr/bin/env python3
"""Embed GDSC (cell-line, drug) pairs with the current backbone."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    CHECKPOINTS,
    ROOT,
    dense_matrix,
    extract_sample_embeddings,
    load_gene_aligned_adata,
    reduce_embeddings,
    select_hvg_indices,
)


EXPR_PATH = ROOT / "datasets" / "GDSC" / "gdsc.h5ad"
IC50_PATH = ROOT / "datasets" / "GDSC" / "drug_response_prediction_IC50.csv"
DRUG_FEAT_PATH = ROOT / "datasets" / "GDSC" / "drug_features.npz"
OUTPUT_DIR = ROOT / "output" / "drug_resp" / "pca_umap"
OUTPUT_PATH = OUTPUT_DIR / "drug_resp_embeddings.npz"

MODEL_ID_COL = "ModelID"
DRUG_ID_COL = "Drug ID"
IC50_COL = "IC50"
DATASET_COL = "Dataset Version"


print("Loading GDSC expression ...")
adata = ad.read_h5ad(EXPR_PATH)
adata.obs_names = adata.obs_names.astype(str)
adata = load_gene_aligned_adata(adata, min_genes=0)
cell_id_to_row = {str(cell_id): idx for idx, cell_id in enumerate(adata.obs_names)}

print("Loading drug features ...")
drug_data = np.load(DRUG_FEAT_PATH, allow_pickle=True)
drug_ids_arr = drug_data["drug_ids"].astype(str)
drug_emb_matrix = drug_data["features"].astype(np.float32)
drug_id_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids_arr)}
print(f"{len(drug_ids_arr)} drugs | drug_emb_dim={drug_emb_matrix.shape[1]}")

print("Loading IC50 pairs ...")
ic50_df = pd.read_csv(IC50_PATH)
ic50_df[MODEL_ID_COL] = ic50_df[MODEL_ID_COL].astype(str)
ic50_df[DRUG_ID_COL] = ic50_df[DRUG_ID_COL].astype(str)
ic50_df = ic50_df.sort_values(
    DATASET_COL,
    key=lambda s: s.map({"GDSC1": 0, "GDSC2": 1}).fillna(0),
)
ic50_df = ic50_df.drop_duplicates(subset=[MODEL_ID_COL, DRUG_ID_COL], keep="last")
ic50_df = ic50_df.dropna(subset=[IC50_COL]).reset_index(drop=True)
ic50_df = ic50_df[
    ic50_df[MODEL_ID_COL].isin(cell_id_to_row)
    & ic50_df[DRUG_ID_COL].isin(drug_id_to_idx)
].reset_index(drop=True)
print(f"{len(ic50_df)} valid pairs")

cell_idxs = np.asarray([cell_id_to_row[m] for m in ic50_df[MODEL_ID_COL]], dtype=np.int64)
drug_idxs = np.asarray([drug_id_to_idx[d] for d in ic50_df[DRUG_ID_COL]], dtype=np.int64)
ic50_vals = ic50_df[IC50_COL].to_numpy(dtype=np.float32)
log_ic50 = np.log1p(np.clip(ic50_vals, 0, None)).astype(np.float32)


def build_pair_embeddings(cell_embeddings: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (cell_embeddings[cell_idxs], drug_emb_matrix[drug_idxs]),
        axis=1,
    )


hvg_indices = select_hvg_indices(adata)
all_pair_embeddings: dict[str, np.ndarray] = {
    "raw_gex": build_pair_embeddings(dense_matrix(adata.X)),
}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"Extracting cell embeddings: {name} ...")
    cell_embeddings = extract_sample_embeddings(
        ckpt_path,
        adata,
        hvg_indices,
        pooling="mean_cls",
    )
    all_pair_embeddings[name] = build_pair_embeddings(cell_embeddings)

save_dict: dict[str, np.ndarray] = {
    "log_ic50": log_ic50,
    "ic50_raw": ic50_vals,
    "cell_idxs": cell_idxs,
    "drug_idxs": drug_idxs,
}
save_dict.update(
    reduce_embeddings(
        all_pair_embeddings,
        n_neighbors=15,
        min_dist=0.1,
        low_memory=False,
    )
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **save_dict)
print(f"Saved -> {OUTPUT_PATH}")
