#!/usr/bin/env python3
"""
Load pre-computed embeddings from embed_drug_resp.py and produce a
2-row × 6-column PDF figure (PCA / UMAP, one column per embedding space).
Each point is one (cell_line, drug) pair, colored by log(1 + IC50).

Run locally or on the login node — no GPU required.

Usage:
    python plot_drug_resp.py [--input PATH] [--output PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── defaults ───────────────────────────────────────────────────────────────────
ROOT           = "/Users/enricoheiss/Thesis/eheiss"
DEFAULT_DIR    = f"{ROOT}/output/drug_resp/pca_umap"
DEFAULT_INPUT  = f"{DEFAULT_DIR}/drug_resp_embeddings.npz"
DEFAULT_OUTPUT = f"{DEFAULT_DIR}/drug_resp_umap.pdf"

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
data     = np.load(args.input, allow_pickle=True)
log_ic50 = data["log_ic50"]   # (n_pairs,)

pca2d  = {name: data[f"pca2d_{name}"]  for name in COL_ORDER}
umap2d = {name: data[f"umap2d_{name}"] for name in COL_ORDER}

print(f"Pairs: {len(log_ic50):,} | log(1+IC50): "
      f"min={log_ic50.min():.2f}  max={log_ic50.max():.2f}  "
      f"median={np.median(log_ic50):.2f}")

# ── shared colormap ────────────────────────────────────────────────────────────
vmin, vmax = np.percentile(log_ic50, [2, 98])   # clip outliers for color range
norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap   = cm.plasma_r   # low IC50 (sensitive) = bright, high IC50 = dark

# ── figure ─────────────────────────────────────────────────────────────────────
n_cols = len(COL_ORDER)
fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 8.0))

for col, (name, title) in enumerate(zip(COL_ORDER, COL_TITLES)):
    for row, (coords, row_label) in enumerate([(pca2d, "PCA"), (umap2d, "UMAP")]):
        ax = axes[row, col]
        xy = coords[name]
        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=log_ic50, cmap=cmap, norm=norm,
            s=1, alpha=0.4, linewidths=0, rasterized=True,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=6)

# ── shared colorbar ────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("log(1 + IC50)  [low = sensitive]", fontsize=10, labelpad=8)

fig.suptitle(
    "Drug Response — PCA & UMAP of [cell_emb ‖ drug_emb] pairs",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.subplots_adjust(right=0.90)
plt.savefig(args.output, bbox_inches="tight", dpi=150)
print(f"Saved → {args.output}")
