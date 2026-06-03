#!/usr/bin/env python3
"""
Load pre-computed embeddings from embed_disease_class.py and produce a
2-row × 6-column PDF figure (PCA / UMAP, one column per embedding space).

Run locally or on the login node — no GPU required.

Usage:
    python plot_disease_class.py [--input PATH] [--output PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── defaults ───────────────────────────────────────────────────────────────────
ROOT           = "/Users/enricoheiss/Thesis/eheiss"
DEFAULT_DIR    = f"{ROOT}/output/disease_class/pca_umap"
DEFAULT_INPUT  = f"{DEFAULT_DIR}/disease_class_embeddings.npz"
DEFAULT_OUTPUT = f"{DEFAULT_DIR}/disease_class_umap.pdf"

COL_ORDER  = ["raw_gex", "random_init", "pretrain_sc", "pretrain_bulk",
              "preadapt_sc", "preadapt_bulk"]
COL_TITLES = ["Raw GEX", "Random init", "Pretrain sc", "Pretrain bulk",
              "Preadapt sc", "Preadapt bulk"]

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

# ── build color palette (dynamic) ─────────────────────────────────────────────
unique_diseases = sorted(set(labels))
n = len(unique_diseases)
print(f"Diseases ({n}): {unique_diseases}")

# tab20 for ≤20, tab20+tab20b for ≤40, hsv beyond that
if n <= 20:
    colors = [plt.cm.tab20(i / max(n - 1, 1)) for i in range(n)]
elif n <= 40:
    _p = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    colors = _p[:n]
else:
    colors = [plt.cm.hsv(i / n) for i in range(n)]

disease_colors = dict(zip(unique_diseases, colors))

# ── figure ─────────────────────────────────────────────────────────────────────
n_cols = len(COL_ORDER)
fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 8.0))

for col, (name, title) in enumerate(zip(COL_ORDER, COL_TITLES)):
    for row, (coords, row_label) in enumerate([(pca2d, "PCA"), (umap2d, "UMAP")]):
        ax = axes[row, col]
        xy = coords[name]
        for disease, color in disease_colors.items():
            mask = labels == disease
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=[color], s=4, alpha=0.6, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=6)

legend_handles = [
    mpatches.Patch(color=disease_colors[d], label=d) for d in unique_diseases
]
n_legend_cols = min(n, 6)
fig.legend(
    handles=legend_handles, loc="lower center", ncol=n_legend_cols,
    fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.05),
    handlelength=1.2, columnspacing=0.8,
)
fig.suptitle(
    "Disease Classification — PCA & UMAP of backbone embeddings",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight", dpi=150)
print(f"Saved → {args.output}")
