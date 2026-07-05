#!/usr/bin/env python3
"""Embed TCGA 33-type cancer-classification samples with the current backbone."""

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


TCGA_PATH = ROOT / "datasets" / "TCGA" / "tcga.h5ad"
OUTPUT_DIR = ROOT / "output" / "canc_type_class_33" / "pca_umap"
OUTPUT_PATH = OUTPUT_DIR / "canc_type_class_33_embeddings.npz"
COHORTS = {
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
}


print("Loading TCGA ...")
adata = ad.read_h5ad(TCGA_PATH)
adata = adata[adata.obs["project"].astype(str).isin(COHORTS)].copy()
adata = load_gene_aligned_adata(adata, min_genes=200)
adata.obs["cancer_type"] = adata.obs["project"].astype(str)
labels = adata.obs["cancer_type"].to_numpy(dtype=str)
print(pd.Series(labels).value_counts().to_string())

hvg_indices = select_hvg_indices(adata)
all_embeddings: dict[str, np.ndarray] = {"raw_gex": dense_matrix(adata.X)}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"Extracting: {name} ...")
    all_embeddings[name] = extract_sample_embeddings(
        ckpt_path,
        adata,
        hvg_indices,
        pooling="mean_cls",
    )

save_dict: dict[str, np.ndarray] = {"labels": labels}
save_dict.update(reduce_embeddings(all_embeddings, n_neighbors=30, min_dist=0.3))

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **save_dict)
print(f"Saved -> {OUTPUT_PATH}")
