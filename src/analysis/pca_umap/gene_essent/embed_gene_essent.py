#!/usr/bin/env python3
"""
Embed DepMap (cell_line, gene) pairs with all pretrained/preadapted backbones.
Each pair embedding is the backbone's per-position output at that gene's token:
  h[cell, gene_pos, :]  →  (dim,) = (200,)

To keep UMAP tractable, N_CELLS_SAMPLE cell lines are randomly drawn.
Each sampled cell line contributes gene_num points → total ~N_CELLS_SAMPLE × gene_num pairs.

Only (cell, gene) pairs where the gene has CRISPR data (valid_gene_mask) are kept.

Coloring: CRISPR essentiality score — negative = essential, ~0 = non-essential.

No raw_gex column: a single gene's expression value is a scalar, not a vector,
so there is no natural per-gene "raw" embedding to compare against.

Intended to run as a SLURM GPU job via embed_gene_essent_job.sh.
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.decomposition import PCA
from umap import UMAP

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = "/cluster/work/boeva/eheiss"
SRC         = f"{ROOT}/scbFM/src"
GENE_LIST   = f"{ROOT}/scbFM/data/gene_list.txt"
DATA_PATH   = f"{ROOT}/datasets/DepMap/depmap.h5ad"
OUTPUT_DIR  = f"{ROOT}/output/gene_essent/pca_umap"
OUTPUT_PATH = f"{OUTPUT_DIR}/gene_essent_embeddings.npz"

CHECKPOINTS = {
    "random_init":   None,
    "pretrain_sc":   f"{ROOT}/output/pretrain_sc/pretrain_sc.pth",
    "pretrain_bulk": f"{ROOT}/output/pretrain_bulk/pretrain_bulk.pth",
    "preadapt_sc":   f"{ROOT}/output/preadapt_sc/preadapt_sc.pth",
    "preadapt_bulk": f"{ROOT}/output/preadapt_bulk/preadapt_bulk.pth",
}

N_CELLS_SAMPLE = 200
RANDOM_SEED    = 42

BIN_NUM    = 10
GENE_NUM   = 11964
DIM        = 200
SEQ_LEN    = GENE_NUM + 1     # 11965
VOCAB_SIZE = BIN_NUM + 2 + 1  # 13
SPECIAL_ID = BIN_NUM + 2      # 12
BATCH_SIZE = 4                # small — (B, 11965, 200) is large on GPU
N_PCA      = 50

# ── imports from scbFM src ─────────────────────────────────────────────────────
sys.path.insert(0, SRC)
from performer_pytorch import PerformerLM
from preprocess import read_gene_list, reindex_to_gene_list, filter_min_genes, quantile_bin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data loading ───────────────────────────────────────────────────────────────
print("Loading DepMap ...")
combined = ad.read_h5ad(DATA_PATH)
print(f"Loaded: {combined.n_obs} cell lines × {combined.n_vars} genes")

# Extract CRISPR targets (n_cells, n_depmap_genes)
essen_X = combined.layers["essen_array"]
if sparse.issparse(essen_X):
    essen_X = essen_X.toarray()
essen_X = np.asarray(essen_X, dtype=np.float32)

# Build expression AnnData from expr_array layer
expr_X = combined.layers["expr_array"]
if sparse.issparse(expr_X):
    expr_X = expr_X.toarray()
expr_adata = ad.AnnData(X=np.asarray(expr_X, dtype=np.float32))
expr_adata.var_names = combined.var_names.copy()
expr_adata.obs_names = combined.obs_names.copy()

# ── preprocess expression ──────────────────────────────────────────────────────
gene_list_txt = read_gene_list(GENE_LIST)
expr_adata, _ = reindex_to_gene_list(expr_adata, gene_list_txt)
expr_adata     = filter_min_genes(expr_adata, min_genes=0)

# ── reindex CRISPR targets to gene_list order ─────────────────────────────────
depmap_gene_to_idx = {str(g): j for j, g in enumerate(combined.var_names)}
gene_list_names    = list(expr_adata.var_names.astype(str))
gene_num           = len(gene_list_names)

valid_mask        = np.zeros(gene_num, dtype=bool)
depmap_src_cols:  list[int] = []
gene_list_tgt_cols: list[int] = []
for i, g in enumerate(gene_list_names):
    if g in depmap_gene_to_idx:
        depmap_src_cols.append(depmap_gene_to_idx[g])
        gene_list_tgt_cols.append(i)
        valid_mask[i] = True

# Align essen_X rows to expr_adata obs order (may differ after filter_min_genes)
combined_obs_to_idx = {str(o): i for i, o in enumerate(combined.obs_names)}
essen_row_order = [combined_obs_to_idx[obs] for obs in expr_adata.obs_names]
essen_X = essen_X[essen_row_order]

# Build aligned targets (n_cells, gene_num), zeros for genes with no CRISPR data
Y = np.zeros((expr_adata.n_obs, gene_num), dtype=np.float32)
Y[:, gene_list_tgt_cols] = essen_X[:, depmap_src_cols]

print(f"{expr_adata.n_obs} cell lines | {gene_num} genes | "
      f"{valid_mask.sum()} genes with CRISPR data")

# ── quantile-bin + special token ──────────────────────────────────────────────
adata_tok = quantile_bin(expr_adata.copy(), bin_num=BIN_NUM)
X_tok = adata_tok.X.toarray() if sparse.issparse(adata_tok.X) else np.asarray(adata_tok.X)
X_tok = np.concatenate(
    [X_tok, np.full((X_tok.shape[0], 1), SPECIAL_ID, dtype=np.int64)], axis=1
).astype(np.int64)

# ── subsample cell lines ───────────────────────────────────────────────────────
rng        = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(expr_adata.n_obs, size=min(N_CELLS_SAMPLE, expr_adata.n_obs),
                        replace=False)
sample_idx = np.sort(sample_idx)
X_tok_sample = X_tok[sample_idx]       # (N_CELLS_SAMPLE, seq_len)
Y_sample     = Y[sample_idx]           # (N_CELLS_SAMPLE, gene_num)
print(f"Subsampled {len(sample_idx)} cell lines → "
      f"{len(sample_idx) * valid_mask.sum():,} valid (cell, gene) pairs")

# Flatten essentiality: only valid genes
flat_essent = Y_sample[:, valid_mask].ravel()   # (N_CELLS_SAMPLE × n_valid,)

# ── model utils ────────────────────────────────────────────────────────────────
def build_backbone(checkpoint_path: str | None) -> nn.Module:
    model = PerformerLM(
        num_tokens=VOCAB_SIZE, max_seq_len=SEQ_LEN, dim=DIM,
        depth=6, heads=10, dim_head=64, ff_mult=4,
        nb_features=None, feature_redraw_interval=1000,
        ff_chunks=1, ff_glu=False, emb_dropout=0., ff_dropout=0.,
        attn_dropout=0., use_scalenorm=False, use_rezero=False,
        no_projection=False, tie_embed=False, g2v_position_emb=False,
        auto_check_redraw=True, qkv_bias=True,
    )
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd   = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(sd)
    model.to_out = nn.Identity()   # → (B, seq_len, dim)
    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_gene_embs(model: nn.Module, X_tokens: np.ndarray) -> np.ndarray:
    """
    Returns per-gene embeddings for all sampled cells, valid genes only.
    Shape: (N_CELLS_SAMPLE × n_valid_genes, dim)
    """
    parts = []
    for i in range(0, len(X_tokens), BATCH_SIZE):
        batch = torch.tensor(X_tokens[i:i+BATCH_SIZE], dtype=torch.long, device=DEVICE)
        h = model(batch)                          # (B, seq_len, dim)
        h = h[:, :gene_num, :]                    # (B, gene_num, dim) — drop special token
        h = h[:, valid_mask, :]                   # (B, n_valid, dim)
        parts.append(h.reshape(-1, DIM).cpu().numpy())  # (B × n_valid, dim)
    return np.concatenate(parts, axis=0)

# ── embedding extraction + PCA + UMAP ─────────────────────────────────────────
save_dict: dict[str, np.ndarray] = {
    "essentiality": flat_essent,
    "valid_mask":   valid_mask,
}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"Extracting gene embs: {name} ...")
    m = build_backbone(ckpt_path)
    emb = extract_gene_embs(m, X_tok_sample)  # (N_CELLS_SAMPLE × n_valid, dim)
    del m
    torch.cuda.empty_cache()

    print(f"  PCA + UMAP: {name}  shape={emb.shape} ...")
    pca50 = PCA(n_components=N_PCA, random_state=42).fit_transform(emb)
    umap2 = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                 random_state=42, low_memory=True).fit_transform(pca50)
    save_dict[f"pca2d_{name}"]  = pca50[:, :2].astype(np.float32)
    save_dict[f"umap2d_{name}"] = umap2.astype(np.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **save_dict)
print(f"Saved → {OUTPUT_PATH}")
