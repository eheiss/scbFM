#!/usr/bin/env python3
"""
Load pre-computed embeddings from embed_deconv.py and produce a
4-row × 6-column PDF figure:
  rows 0-1: PCA / UMAP colored by dominant cell type
  rows 2-3: PCA / UMAP colored by tissue_general

Run locally or on the login node — no GPU required.

Usage:
    python plot_deconv.py [--input PATH] [--output PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── defaults ───────────────────────────────────────────────────────────────────
ROOT           = "/Users/enricoheiss/Thesis/eheiss"
DEFAULT_DIR    = f"{ROOT}/output/deconv/pca_umap"
DEFAULT_INPUT  = f"{DEFAULT_DIR}/deconv_embeddings.npz"
DEFAULT_OUTPUT = f"{DEFAULT_DIR}/deconv_umap.pdf"

COL_ORDER  = ["raw_gex", "random_init", "pretrain_sc", "pretrain_bulk",
              "preadapt_sc", "preadapt_bulk"]
COL_TITLES = ["Raw GEX", "Random init", "Pretrain sc", "Pretrain bulk",
              "Preadapt sc", "Preadapt bulk"]

ROW_LABELS = [
    "PCA\n(cell type)",
    "UMAP\n(cell type)",
    "PCA\n(tissue)",
    "UMAP\n(tissue)",
]

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default=DEFAULT_INPUT)
parser.add_argument("--output", default=DEFAULT_OUTPUT)
args = parser.parse_args()

# ── load ───────────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
data          = np.load(args.input, allow_pickle=True)
tissue_labels = data["tissue_labels"]
proportions   = data["proportions"]        # (n, n_cell_types)
cell_types    = list(data["cell_types"])   # list of str

pca2d  = {name: data[f"pca2d_{name}"]  for name in COL_ORDER}
umap2d = {name: data[f"umap2d_{name}"] for name in COL_ORDER}

# ── derive discrete labels ──────────────────────────────────────────────────────
dominant_idx    = np.argmax(proportions, axis=1)
dominant_labels = np.array([cell_types[i] for i in dominant_idx])

# ── build color palettes (dynamic — unknown count at write time) ───────────────
def make_palette(categories: list[str]) -> dict[str, tuple]:
    n = len(categories)
    if n <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    return dict(zip(categories, colors))

unique_cell_types = sorted(set(dominant_labels))
unique_tissues    = sorted(set(tissue_labels))
ct_colors         = make_palette(unique_cell_types)
tissue_colors     = make_palette(unique_tissues)

print(f"Cell types in dominant set ({len(unique_cell_types)}): {unique_cell_types}")
print(f"Tissues ({len(unique_tissues)}): {unique_tissues}")

# ── figure ─────────────────────────────────────────────────────────────────────
n_cols  = len(COL_ORDER)
n_rows  = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows))

row_configs = [
    (pca2d,  dominant_labels, ct_colors),
    (umap2d, dominant_labels, ct_colors),
    (pca2d,  tissue_labels,   tissue_colors),
    (umap2d, tissue_labels,   tissue_colors),
]

for row, (coords, labels, palette) in enumerate(row_configs):
    for col, (name, title) in enumerate(zip(COL_ORDER, COL_TITLES)):
        ax = axes[row, col]
        xy = coords[name]
        for cat, color in palette.items():
            mask = labels == cat
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=[color], s=4, alpha=0.6, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(ROW_LABELS[row], fontsize=10, fontweight="bold", labelpad=6)

# ── legends ────────────────────────────────────────────────────────────────────
ct_handles = [mpatches.Patch(color=ct_colors[c], label=c) for c in unique_cell_types]
tissue_handles = [mpatches.Patch(color=tissue_colors[t], label=t) for t in unique_tissues]

n_ct_cols     = min(len(unique_cell_types), 6)
n_tissue_cols = min(len(unique_tissues), 6)

fig.text(0.01, 0.505, "Cell type", fontsize=9, fontweight="bold",
         ha="left", va="center", rotation=0)
fig.legend(
    handles=ct_handles, loc="upper center",
    bbox_to_anchor=(0.5, -0.01), ncol=n_ct_cols,
    fontsize=8, framealpha=0.9, handlelength=1.2, columnspacing=0.8,
    title="Dominant cell type",
)

fig.text(0.01, 0.02, "Tissue", fontsize=9, fontweight="bold",
         ha="left", va="center", rotation=0)
fig.legend(
    handles=tissue_handles, loc="lower center",
    bbox_to_anchor=(0.5, -0.08), ncol=n_tissue_cols,
    fontsize=8, framealpha=0.9, handlelength=1.2, columnspacing=0.8,
    title="Tissue (general)",
)

fig.suptitle(
    "Deconvolution — PCA & UMAP of backbone embeddings",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight", dpi=150)
print(f"Saved → {args.output}")
