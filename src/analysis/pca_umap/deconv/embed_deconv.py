#!/usr/bin/env python3
"""Embed pseudo-bulk deconvolution samples with the current backbone."""

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


DATA_PATH = ROOT / "datasets" / "pseudo_bulk" / "pseudo_bulk_RAW.h5ad"
OUTPUT_DIR = ROOT / "output" / "deconv" / "pca_umap"
OUTPUT_PATH = OUTPUT_DIR / "deconv_embeddings.npz"


print("Loading pseudo-bulk data ...")
adata = ad.read_h5ad(DATA_PATH)

prop_cols = sorted(c for c in adata.obs.columns if c.startswith("prop__"))
if prop_cols:
    cell_types = [c.removeprefix("prop__") for c in prop_cols]
else:
    mapping = adata.uns.get("cell_type_proportion_columns", {})
    if not mapping:
        raise ValueError("No prop__* obs columns and no cell-type mapping in uns.")
    cell_types = sorted(mapping.keys())
    prop_cols = [mapping[cell_type] for cell_type in cell_types]
proportions_before = adata.obs[prop_cols].copy()

if "tissue_general" not in adata.obs:
    raise ValueError("obs['tissue_general'] not found in pseudo-bulk data.")
tissue_labels_before = adata.obs["tissue_general"].astype(str)

adata = load_gene_aligned_adata(adata, min_genes=0)
tissue_labels = tissue_labels_before.loc[adata.obs_names].to_numpy(dtype=str)
proportions = proportions_before.loc[adata.obs_names].to_numpy(dtype=np.float32)
print(pd.Series(tissue_labels).value_counts().to_string())
print(f"Cell types ({len(cell_types)}): {cell_types}")

hvg_indices = select_hvg_indices(adata)
all_embeddings: dict[str, np.ndarray] = {"raw_gex": dense_matrix(adata.X)}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"Extracting: {name} ...")
    all_embeddings[name] = extract_sample_embeddings(
        ckpt_path,
        adata,
        hvg_indices,
        pooling="cls",
    )

save_dict: dict[str, np.ndarray] = {
    "tissue_labels": tissue_labels,
    "proportions": proportions,
    "cell_types": np.asarray(cell_types),
}
save_dict.update(reduce_embeddings(all_embeddings, n_neighbors=30, min_dist=0.3))

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **save_dict)
print(f"Saved -> {OUTPUT_PATH}")
