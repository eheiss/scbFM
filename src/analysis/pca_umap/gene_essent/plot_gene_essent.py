#!/usr/bin/env python3
"""
Load pre-computed embeddings from embed_gene_essent.py and produce a
2-row × 5-column PDF figure (PCA / UMAP, one column per model).
Each point is one (cell_line, gene) pair, colored by CRISPR essentiality score
(negative = essential, ~0 = non-essential).

Run locally or on the login node — no GPU required.

Usage:
    python plot_gene_essent.py [--input PATH] [--output PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── defaults ───────────────────────────────────────────────────────────────────
ROOT           = "/Users/enricoheiss/Thesis/eheiss"
DEFAULT_DIR    = f"{ROOT}/output/gene_essent/pca_umap"
DEFAULT_INPUT  = f"{DEFAULT_DIR}/gene_essent_embeddings.npz"
DEFAULT_OUTPUT = f"{DEFAULT_DIR}/gene_essent_umap.pdf"

# No raw_gex column — 5 model columns only
COL_ORDER  = ["random_init", "pretrain_sc", "pretrain_bulk",
              "preadapt_sc", "preadapt_bulk"]
COL_TITLES = ["Random init", "Pretrain sc", "Pretrain bulk",
              "Preadapt sc", "Preadapt bulk"]

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default=DEFAULT_INPUT)
parser.add_argument("--output", default=DEFAULT_OUTPUT)
args = parser.parse_args()

# ── load ───────────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
data        = np.load(args.input, allow_pickle=True)
essentiality = data["essentiality"]   # (N_CELLS_SAMPLE × n_valid_genes,)

pca2d  = {name: data[f"pca2d_{name}"]  for name in COL_ORDER}
umap2d = {name: data[f"umap2d_{name}"] for name in COL_ORDER}

print(f"Pairs: {len(essentiality):,} | essentiality: "
      f"min={essentiality.min():.2f}  max={essentiality.max():.2f}  "
      f"median={np.median(essentiality):.2f}")

# ── shared colormap ────────────────────────────────────────────────────────────
# Clip at 2nd/98th percentile to avoid outliers dominating the color scale
vmin, vmax = np.percentile(essentiality, [2, 98])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.RdBu   # red = essential (negative), blue = non-essential (positive)

# ── figure ─────────────────────────────────────────────────────────────────────
n_cols = len(COL_ORDER)
fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 8.0))

for col, (name, title) in enumerate(zip(COL_ORDER, COL_TITLES)):
    for row, (coords, row_label) in enumerate([(pca2d, "PCA"), (umap2d, "UMAP")]):
        ax = axes[row, col]
        xy = coords[name]
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=essentiality, cmap=cmap, norm=norm,
            s=1, alpha=0.3, linewidths=0, rasterized=True,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=6)

# ── shared colorbar ────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("CRISPR essentiality score\n(negative = essential)", fontsize=10, labelpad=8)

fig.suptitle(
    "Gene Essentiality — PCA & UMAP of per-gene backbone embeddings\n"
    f"(200 subsampled cell lines, colored by CRISPR score)",
    fontsize=13, fontweight="bold", y=1.02,
)

plt.subplots_adjust(right=0.90)
plt.savefig(args.output, bbox_inches="tight", dpi=150)
print(f"Saved → {args.output}")
