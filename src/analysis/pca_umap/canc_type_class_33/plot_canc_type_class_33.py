#!/usr/bin/env python3
"""
Load pre-computed embeddings from embed_canc_type_class_33.py and produce a
2-row × 6-column PDF figure (PCA / UMAP, one column per embedding space).

Run locally or on the login node — no GPU required.

Usage:
    python plot_canc_type_class_33.py [--input PATH] [--output PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── defaults ───────────────────────────────────────────────────────────────────
ROOT           = "/Users/enricoheiss/Thesis/eheiss"
DEFAULT_DIR    = f"{ROOT}/output/canc_type_class_33/pca_umap"
DEFAULT_INPUT  = f"{DEFAULT_DIR}/canc_type_class_33_embeddings.npz"
DEFAULT_OUTPUT = f"{DEFAULT_DIR}/canc_type_class_33_umap.pdf"

COL_ORDER  = ["raw_gex", "random_init", "pretrain_sc", "pretrain_bulk",
              "preadapt_sc", "preadapt_bulk"]
COL_TITLES = ["Raw GEX", "Random init", "Pretrain sc", "Pretrain bulk",
              "Preadapt sc", "Preadapt bulk"]

# 33 cancer types — colors drawn from tab20, tab20b, tab20c to stay distinct
_CANCER_TYPES = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]

# Build palette: 20 from tab20, then 13 from tab20b (skip first 7 which overlap)
_tab20  = plt.cm.tab20.colors          # 20 colors
_tab20b = plt.cm.tab20b.colors         # 20 colors
_palette = list(_tab20) + list(_tab20b[7:20])   # 20 + 13 = 33
CANCER_COLORS = {ct: _palette[i] for i, ct in enumerate(_CANCER_TYPES)}

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default=DEFAULT_INPUT)
parser.add_argument("--output", default=DEFAULT_OUTPUT)
args = parser.parse_args()

# ── load ───────────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
data   = np.load(args.input, allow_pickle=True)
labels = data["labels"]

pca2d  = {name: data[f"pca2d_{name}"]  for name in COL_ORDER}
umap2d = {name: data[f"umap2d_{name}"] for name in COL_ORDER}

# ── figure ─────────────────────────────────────────────────────────────────────
n_cols = len(COL_ORDER)
fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 8.5))

for col, (name, title) in enumerate(zip(COL_ORDER, COL_TITLES)):
    for row, (coords, row_label) in enumerate([(pca2d, "PCA"), (umap2d, "UMAP")]):
        ax = axes[row, col]
        xy = coords[name]
        for cancer, color in CANCER_COLORS.items():
            mask = labels == cancer
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=[color], s=3, alpha=0.5, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=6)

legend_handles = [
    mpatches.Patch(color=CANCER_COLORS[ct], label=ct) for ct in _CANCER_TYPES
]
fig.legend(
    handles=legend_handles, loc="lower center", ncol=11,
    fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.07),
    handlelength=1.2, columnspacing=0.8,
)
fig.suptitle(
    "Cancer Type Classification (33 types) — PCA & UMAP of backbone embeddings",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight", dpi=150)
print(f"Saved → {args.output}")
