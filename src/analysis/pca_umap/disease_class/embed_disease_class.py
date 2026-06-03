#!/usr/bin/env python3
"""
Embed DiSignAtlas disease classification samples with all pretrained/preadapted
backbones, then reduce to PCA-50 and UMAP-2D. Saves a compressed .npz with
coordinates + labels for the plotting script.

Only "case" samples are used, mirroring the finetuning runner.

Intended to run as a SLURM GPU job via embed_disease_class_job.sh.
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
DATA_PATH   = f"{ROOT}/datasets/DiSignAtlas/disignatlas.h5ad"
OUTPUT_DIR  = f"{ROOT}/output/disease_class/pca_umap"
OUTPUT_PATH = f"{OUTPUT_DIR}/disease_class_embeddings.npz"

CHECKPOINTS = {
    "random_init":   None,
    "pretrain_sc":   f"{ROOT}/output/pretrain_sc/pretrain_sc.pth",
    "pretrain_bulk": f"{ROOT}/output/pretrain_bulk/pretrain_bulk.pth",
    "preadapt_sc":   f"{ROOT}/output/preadapt_sc/preadapt_sc.pth",
    "preadapt_bulk": f"{ROOT}/output/preadapt_bulk/preadapt_bulk.pth",
}

DISEASE_LABEL_COL = "label"
BINARY_LABEL_COL  = "binary_label"

BIN_NUM    = 10
GENE_NUM   = 11964
DIM        = 200
SEQ_LEN    = GENE_NUM + 1     # 11965
VOCAB_SIZE = BIN_NUM + 2 + 1  # 13
SPECIAL_ID = BIN_NUM + 2      # 12
BATCH_SIZE = 8
N_PCA      = 50

# ── imports from scbFM src ─────────────────────────────────────────────────────
sys.path.insert(0, SRC)
from performer_pytorch import PerformerLM
from preprocess import read_gene_list, reindex_to_gene_list, filter_min_genes, quantile_bin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data loading ───────────────────────────────────────────────────────────────
print("Loading DiSignAtlas ...")
adata = ad.read_h5ad(DATA_PATH)
adata = adata[adata.obs[BINARY_LABEL_COL].astype(str) == "case"].copy()
labels = adata.obs[DISEASE_LABEL_COL].astype(str).values
print(f"{adata.n_obs} samples | {adata.n_vars} genes (before reindex)")
print(pd.Series(labels).value_counts().to_string())

gene_list = read_gene_list(GENE_LIST)
adata, _  = reindex_to_gene_list(adata, gene_list)
adata     = filter_min_genes(adata, min_genes=0)   # bulk — no structural zeros

X_raw = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
X_raw = X_raw.astype(np.float32)

adata_tok = quantile_bin(adata.copy(), bin_num=BIN_NUM)
X_tok = adata_tok.X.toarray() if sparse.issparse(adata_tok.X) else np.asarray(adata_tok.X)
X_tok = np.concatenate(
    [X_tok, np.full((X_tok.shape[0], 1), SPECIAL_ID, dtype=np.int64)], axis=1
).astype(np.int64)

print(f"{adata.n_obs} samples after filtering | {adata.n_vars} genes")

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
    model.to_out = nn.Identity()
    return model.to(DEVICE).eval()


torch.manual_seed(0)
_collapse = nn.Sequential(nn.Conv2d(1, 1, (1, DIM)), nn.ReLU()).to(DEVICE)
for p in _collapse.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def extract(model: nn.Module, X_tokens: np.ndarray) -> np.ndarray:
    parts = []
    for i in range(0, len(X_tokens), BATCH_SIZE):
        batch = torch.tensor(X_tokens[i:i+BATCH_SIZE], dtype=torch.long, device=DEVICE)
        h = model(batch)
        h = _collapse(h[:, None])
        h = h.view(h.shape[0], -1)
        parts.append(h.cpu().numpy())
    return np.concatenate(parts, axis=0)

# ── embedding extraction ───────────────────────────────────────────────────────
all_embeddings: dict[str, np.ndarray] = {"raw_gex": X_raw}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"Extracting: {name} ...")
    m = build_backbone(ckpt_path)
    all_embeddings[name] = extract(m, X_tok)
    del m
    torch.cuda.empty_cache()

# ── PCA + UMAP ────────────────────────────────────────────────────────────────
save_dict: dict[str, np.ndarray] = {"labels": labels}

for name, emb in all_embeddings.items():
    print(f"PCA + UMAP: {name} ...")
    pca50 = PCA(n_components=N_PCA, random_state=42).fit_transform(emb)
    umap2 = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                 random_state=42).fit_transform(pca50)
    save_dict[f"pca50_{name}"]  = pca50.astype(np.float32)
    save_dict[f"pca2d_{name}"]  = pca50[:, :2].astype(np.float32)
    save_dict[f"umap2d_{name}"] = umap2.astype(np.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **save_dict)
print(f"Saved → {OUTPUT_PATH}")
