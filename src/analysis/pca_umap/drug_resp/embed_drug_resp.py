#!/usr/bin/env python3
"""
Embed GDSC (cell_line, drug) pairs with all pretrained/preadapted backbones.
Each pair embedding is [cell_emb ‖ drug_emb] where:
  - cell_emb  = mean-pool of backbone output  → (dim,) = (200,)
  - drug_emb  = KPGT drug feature             → (drug_emb_dim,)

Reduces to PCA-50 and UMAP-2D per pair, then saves a compressed .npz with
coordinates and log-IC50 values for the plotting script.

Intended to run as a SLURM GPU job via embed_drug_resp_job.sh.
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
ROOT           = "/cluster/work/boeva/eheiss"
SRC            = f"{ROOT}/scbFM/src"
GENE_LIST      = f"{ROOT}/scbFM/data/gene_list.txt"
EXPR_PATH      = f"{ROOT}/datasets/GDSC/gdsc.h5ad"
IC50_PATH      = f"{ROOT}/datasets/GDSC/drug_response_prediction_IC50.csv"
DRUG_FEAT_PATH = f"{ROOT}/datasets/GDSC/drug_features.npz"
OUTPUT_DIR     = f"{ROOT}/output/drug_resp/pca_umap"
OUTPUT_PATH    = f"{OUTPUT_DIR}/drug_resp_embeddings.npz"

CHECKPOINTS = {
    "random_init":   None,
    "pretrain_sc":   f"{ROOT}/output/pretrain_sc/pretrain_sc.pth",
    "pretrain_bulk": f"{ROOT}/output/pretrain_bulk/pretrain_bulk.pth",
    "preadapt_sc":   f"{ROOT}/output/preadapt_sc/preadapt_sc.pth",
    "preadapt_bulk": f"{ROOT}/output/preadapt_bulk/preadapt_bulk.pth",
}

# IC50 CSV column names (matching runner defaults)
MODEL_ID_COL = "ModelID"
DRUG_ID_COL  = "Drug ID"
IC50_COL     = "IC50"
DATASET_COL  = "Dataset Version"

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

# ── expression data ────────────────────────────────────────────────────────────
print("Loading GDSC expression ...")
adata = ad.read_h5ad(EXPR_PATH)
adata.obs_names = adata.obs_names.astype(str)

gene_list = read_gene_list(GENE_LIST)
adata, _  = reindex_to_gene_list(adata, gene_list)
adata     = filter_min_genes(adata, min_genes=0)   # bulk — no structural zeros

X_tok_raw = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
adata_tok  = quantile_bin(adata.copy(), bin_num=BIN_NUM)
X_tok      = adata_tok.X.toarray() if sparse.issparse(adata_tok.X) else np.asarray(adata_tok.X)
X_tok      = np.concatenate(
    [X_tok, np.full((X_tok.shape[0], 1), SPECIAL_ID, dtype=np.int64)], axis=1
).astype(np.int64)

cell_id_to_row: dict[str, int] = {str(cid): i for i, cid in enumerate(adata.obs_names)}
print(f"{adata.n_obs} cell lines | {adata.n_vars} genes")

# ── drug features ──────────────────────────────────────────────────────────────
print("Loading drug features ...")
drug_data      = np.load(DRUG_FEAT_PATH, allow_pickle=True)
drug_ids_arr   = drug_data["drug_ids"].astype(str)
drug_emb_matrix = drug_data["features"].astype(np.float32)   # (n_drugs, drug_emb_dim)
drug_id_to_idx  = {did: i for i, did in enumerate(drug_ids_arr)}
print(f"{len(drug_ids_arr)} drugs | drug_emb_dim={drug_emb_matrix.shape[1]}")

# ── IC50 pairs ─────────────────────────────────────────────────────────────────
print("Loading IC50 pairs ...")
ic50_df = pd.read_csv(IC50_PATH)
ic50_df[MODEL_ID_COL] = ic50_df[MODEL_ID_COL].astype(str)
ic50_df[DRUG_ID_COL]  = ic50_df[DRUG_ID_COL].astype(str)

# Prefer GDSC2 over GDSC1 for duplicate (cell_line, drug) pairs
ic50_df = ic50_df.sort_values(
    DATASET_COL,
    key=lambda s: s.map({"GDSC1": 0, "GDSC2": 1}).fillna(0),
)
ic50_df = ic50_df.drop_duplicates(subset=[MODEL_ID_COL, DRUG_ID_COL], keep="last")
ic50_df = ic50_df.dropna(subset=[IC50_COL]).reset_index(drop=True)

mask = (
    ic50_df[MODEL_ID_COL].isin(cell_id_to_row) &
    ic50_df[DRUG_ID_COL].isin(drug_id_to_idx)
)
ic50_df = ic50_df[mask].reset_index(drop=True)
print(f"{len(ic50_df)} valid pairs")

cell_idxs  = np.array([cell_id_to_row[m] for m in ic50_df[MODEL_ID_COL]], dtype=np.int64)
drug_idxs  = np.array([drug_id_to_idx[d] for d in ic50_df[DRUG_ID_COL]],  dtype=np.int64)
ic50_vals  = ic50_df[IC50_COL].to_numpy(dtype=np.float32)
log_ic50   = np.log1p(np.clip(ic50_vals, 0, None)).astype(np.float32)

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
def extract_cell_embs(model: nn.Module, X_tokens: np.ndarray) -> np.ndarray:
    """Mean-pool backbone output over seq_len → (n_cells, dim)."""
    parts = []
    for i in range(0, len(X_tokens), BATCH_SIZE):
        batch = torch.tensor(X_tokens[i:i+BATCH_SIZE], dtype=torch.long, device=DEVICE)
        h = model(batch)           # (B, seq_len, dim)
        parts.append(h.mean(dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)  # (n_cells, dim)


def build_pair_embs(cell_embs: np.ndarray) -> np.ndarray:
    """Expand cell embeddings to pairs and concatenate drug embeddings."""
    return np.concatenate(
        [cell_embs[cell_idxs], drug_emb_matrix[drug_idxs]], axis=1
    )  # (n_pairs, dim + drug_emb_dim)

# ── embedding extraction ───────────────────────────────────────────────────────
# Raw GEX baseline: use raw (un-binned) expression directly as cell embedding
X_raw_cell = X_tok_raw.astype(np.float32)   # (n_cells, gene_num)
pair_embs_raw = build_pair_embs(X_raw_cell)

all_pair_embs: dict[str, np.ndarray] = {"raw_gex": pair_embs_raw}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"Extracting cell embs: {name} ...")
    m = build_backbone(ckpt_path)
    cell_embs = extract_cell_embs(m, X_tok)   # (n_cells, 200)
    del m
    torch.cuda.empty_cache()
    all_pair_embs[name] = build_pair_embs(cell_embs)

# ── PCA + UMAP ────────────────────────────────────────────────────────────────
save_dict: dict[str, np.ndarray] = {
    "log_ic50":  log_ic50,
    "ic50_raw":  ic50_vals,
    "cell_idxs": cell_idxs,
    "drug_idxs": drug_idxs,
}

for name, emb in all_pair_embs.items():
    print(f"PCA + UMAP: {name}  shape={emb.shape} ...")
    pca50 = PCA(n_components=N_PCA, random_state=42).fit_transform(emb)
    umap2 = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                 random_state=42, low_memory=False).fit_transform(pca50)
    save_dict[f"pca2d_{name}"]  = pca50[:, :2].astype(np.float32)
    save_dict[f"umap2d_{name}"] = umap2.astype(np.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **save_dict)
print(f"Saved → {OUTPUT_PATH}")
