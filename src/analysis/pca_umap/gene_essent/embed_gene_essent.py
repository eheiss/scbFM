#!/usr/bin/env python3
"""Embed DepMap (cell-line, selected-gene) pairs with the current backbone."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    CHECKPOINTS,
    ROOT,
    extract_gene_position_embeddings,
    load_gene_aligned_adata,
    reduce_embeddings,
    select_hvg_indices,
)


DATA_PATH = ROOT / "datasets" / "DepMap" / "depmap.h5ad"
OUTPUT_DIR = ROOT / "output" / "gene_essent" / "pca_umap"
OUTPUT_PATH = OUTPUT_DIR / "gene_essent_embeddings.npz"
N_CELLS_SAMPLE = 200
RANDOM_SEED = 42


print("Loading DepMap ...")
combined = ad.read_h5ad(DATA_PATH)
print(f"Loaded: {combined.n_obs} cell lines x {combined.n_vars} genes")

essen_X = combined.layers["essen_array"]
if sparse.issparse(essen_X):
    essen_X = essen_X.toarray()
essen_X = np.asarray(essen_X, dtype=np.float32)

expr_X = combined.layers["expr_array"]
if sparse.issparse(expr_X):
    expr_X = expr_X.toarray()
expr_adata = ad.AnnData(X=np.asarray(expr_X, dtype=np.float32))
expr_adata.var_names = combined.var_names.copy()
expr_adata.obs_names = combined.obs_names.copy()
expr_adata = load_gene_aligned_adata(expr_adata, min_genes=0)

depmap_gene_to_idx = {str(gene): idx for idx, gene in enumerate(combined.var_names)}
gene_list_names = list(expr_adata.var_names.astype(str))
valid_mask = np.zeros(expr_adata.n_vars, dtype=bool)
depmap_src_cols: list[int] = []
gene_list_tgt_cols: list[int] = []
for idx, gene in enumerate(gene_list_names):
    if gene in depmap_gene_to_idx:
        depmap_src_cols.append(depmap_gene_to_idx[gene])
        gene_list_tgt_cols.append(idx)
        valid_mask[idx] = True

combined_obs_to_idx = {str(obs): idx for idx, obs in enumerate(combined.obs_names)}
essen_row_order = [combined_obs_to_idx[obs] for obs in expr_adata.obs_names]
essen_X = essen_X[essen_row_order]

Y = np.zeros((expr_adata.n_obs, expr_adata.n_vars), dtype=np.float32)
Y[:, gene_list_tgt_cols] = essen_X[:, depmap_src_cols]

hvg_indices = select_hvg_indices(expr_adata)
valid_hvg_positions = valid_mask[hvg_indices]
if not valid_hvg_positions.any():
    raise ValueError("None of the selected HVGs has matching DepMap CRISPR targets.")

rng = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(
    expr_adata.n_obs,
    size=min(N_CELLS_SAMPLE, expr_adata.n_obs),
    replace=False,
)
sample_idx = np.sort(sample_idx)
expr_adata_sample = expr_adata[sample_idx].copy()
selected_valid_genes = hvg_indices[valid_hvg_positions]
flat_essent = Y[np.ix_(sample_idx, selected_valid_genes)].ravel()
print(
    f"Subsampled {len(sample_idx)} cell lines x {len(selected_valid_genes)} "
    f"selected HVGs with CRISPR targets = {flat_essent.size:,} pairs"
)

save_dict: dict[str, np.ndarray] = {
    "essentiality": flat_essent.astype(np.float32),
    "valid_mask": valid_mask,
    "hvg_indices": hvg_indices,
    "selected_valid_genes": selected_valid_genes,
}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"Extracting gene embeddings: {name} ...")
    emb = extract_gene_position_embeddings(
        ckpt_path,
        expr_adata_sample,
        hvg_indices,
        keep_selected_positions=valid_hvg_positions,
    )
    save_dict.update(
        reduce_embeddings(
            {name: emb},
            n_neighbors=15,
            min_dist=0.1,
            low_memory=True,
        )
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **save_dict)
print(f"Saved -> {OUTPUT_PATH}")
