from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from umap import UMAP


ROOT = Path(os.environ.get("SCBFM_CLUSTER_ROOT", "/cluster/work/boeva/eheiss"))
SRC = ROOT / "scbFM" / "src"
GENE_LIST = ROOT / "scbFM" / "data" / "gene_list.txt"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cancerfoundation_backbone import CancerFoundationBackbone
from preprocess import filter_min_genes, read_gene_list, reindex_to_gene_list


CHECKPOINTS: dict[str, str | None] = {
    "random_init": None,
    "pretrain_sc": str(ROOT / "output" / "pretrain_sc" / "pretrain_sc.pth"),
    "pretrain_bulk": str(ROOT / "output" / "pretrain_bulk" / "pretrain_bulk.pth"),
    "preadapt_sc": str(ROOT / "output" / "preadapt_sc" / "preadapt_sc.pth"),
    "preadapt_bulk": str(ROOT / "output" / "preadapt_bulk" / "preadapt_bulk.pth"),
}

GENE_NUM = 13004
SELECTED_GENE_COUNT = 1199
BIN_NUM = 51
EMBSIZE = 256
NLAYERS = 6
NHEADS = 8
D_HID = 512
DROPOUT = 0.2
VALUE_ENCODER_MAX_VALUE = 512
CLS_GENE_ID = 0
PAD_GENE_ID = 1
GENE_TOKEN_OFFSET = 2
CLS_VALUE = -2.0

BATCH_SIZE = int(os.environ.get("SCBFM_PCA_UMAP_BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("SCBFM_PCA_UMAP_NUM_WORKERS", "2"))
PREFETCH_FACTOR = int(os.environ.get("SCBFM_PCA_UMAP_PREFETCH_FACTOR", "2"))
N_PCA = 50
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dense_matrix(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def load_gene_aligned_adata(
    adata: ad.AnnData,
    *,
    min_genes: int,
) -> ad.AnnData:
    gene_list = read_gene_list(GENE_LIST)
    adata, missing_genes = reindex_to_gene_list(adata, gene_list)
    adata = filter_min_genes(adata, min_genes=min_genes)
    if adata.n_vars != GENE_NUM:
        raise ValueError(f"Expected {GENE_NUM} aligned genes, got {adata.n_vars}.")
    print(
        f"Aligned to gene list: {adata.n_obs} samples | {adata.n_vars} genes "
        f"({len(missing_genes)} missing genes filled with zero)"
    )
    return adata


def select_hvg_indices(
    adata: ad.AnnData,
    *,
    n_top_genes: int = SELECTED_GENE_COUNT,
    flavor: str = "cell_ranger",
) -> np.ndarray:
    if adata.n_vars < n_top_genes:
        raise ValueError(f"Cannot select {n_top_genes} HVGs from {adata.n_vars} genes.")
    hvg_stats = sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        inplace=False,
    )
    selected = np.flatnonzero(hvg_stats["highly_variable"].to_numpy())
    if selected.size > n_top_genes:
        ranking_column = (
            "highly_variable_rank"
            if "highly_variable_rank" in hvg_stats
            else "dispersions_norm"
        )
        scores = hvg_stats[ranking_column].to_numpy()[selected]
        if ranking_column == "highly_variable_rank":
            order = np.lexsort((selected, scores))
        else:
            order = np.lexsort((selected, -scores))
        selected = selected[order[:n_top_genes]]
    if selected.size != n_top_genes:
        raise ValueError(
            f"Scanpy selected {selected.size} HVGs; expected {n_top_genes}."
        )
    selected = np.sort(selected.astype(np.int64, copy=False))
    print(f"Selected {selected.size} HVGs with flavor={flavor}")
    return selected


def _digitize_expression(
    values: np.ndarray,
    bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    left_digits = np.digitize(values, bins)
    right_digits = np.digitize(values, bins, right=True)
    random_offsets = rng.random(len(values))
    digits = random_offsets * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def quantile_bin_expression(
    values: np.ndarray,
    *,
    bin_num: int = BIN_NUM,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    binned = np.zeros(values.shape, dtype=np.int64)
    nonzero = values > 0
    if not nonzero.any():
        return binned

    nonzero_values = values[nonzero]
    bins = np.quantile(nonzero_values, np.linspace(0, 1, bin_num - 1))
    digits = _digitize_expression(nonzero_values, bins, rng)
    binned[nonzero] = np.clip(digits, 1, bin_num - 1)
    return binned


class FixedHVGExpressionDataset(Dataset):
    def __init__(
        self,
        data,
        fixed_gene_indices: np.ndarray,
        *,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.data = data
        self.fixed_gene_indices = np.asarray(fixed_gene_indices, dtype=np.int64)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data[index]
        if sparse.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()

        rng = np.random.default_rng(self.seed + index)
        selected_values = values[self.fixed_gene_indices].astype(np.float32, copy=False)
        selected_values = quantile_bin_expression(selected_values, rng=rng).astype(
            np.float32,
            copy=False,
        )

        gene_ids = torch.from_numpy(
            self.fixed_gene_indices.astype(np.int64, copy=False) + GENE_TOKEN_OFFSET
        )
        gene_ids = torch.cat((torch.tensor([CLS_GENE_ID]), gene_ids))

        expression = torch.from_numpy(selected_values)
        expression = torch.cat((torch.tensor([CLS_VALUE]), expression))
        return {"gene_ids": gene_ids, "expr": expression}


def make_loader(data, fixed_gene_indices: np.ndarray) -> DataLoader:
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    print(
        "DataLoader: "
        f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}, "
        f"prefetch_factor={PREFETCH_FACTOR if NUM_WORKERS > 0 else None}, "
        f"pin_memory={DEVICE.type == 'cuda'}"
    )
    return DataLoader(FixedHVGExpressionDataset(data, fixed_gene_indices), **kwargs)


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return dict(state_dict)


def build_backbone(checkpoint_path: str | None) -> CancerFoundationBackbone:
    torch.manual_seed(0)
    model = CancerFoundationBackbone(
        num_gene_tokens=GENE_NUM + GENE_TOKEN_OFFSET,
        d_model=EMBSIZE,
        nhead=NHEADS,
        d_hid=D_HID,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        pad_gene_id=PAD_GENE_ID,
        max_value=VALUE_ENCODER_MAX_VALUE,
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
    return model.to(DEVICE).eval()


def pool_hidden(hidden: torch.Tensor, *, pooling: str = "mean_cls") -> torch.Tensor:
    pooling = pooling.lower()
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":
        return hidden[:, 1:, :].mean(dim=1)
    if pooling == "mean_cls":
        return torch.cat((hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)), dim=-1)
    raise ValueError(f"Unsupported pooling: {pooling}")


@torch.no_grad()
def extract_sample_embeddings(
    checkpoint_path: str | None,
    adata: ad.AnnData,
    fixed_gene_indices: np.ndarray,
    *,
    pooling: str = "mean_cls",
) -> np.ndarray:
    model = build_backbone(checkpoint_path)
    parts: list[np.ndarray] = []
    for batch in make_loader(adata.X, fixed_gene_indices):
        batch = {
            key: value.to(DEVICE, non_blocking=True)
            for key, value in batch.items()
        }
        hidden = model(batch["gene_ids"], batch["expr"])
        pooled = pool_hidden(hidden, pooling=pooling)
        parts.append(pooled.cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def extract_gene_position_embeddings(
    checkpoint_path: str | None,
    adata: ad.AnnData,
    fixed_gene_indices: np.ndarray,
    *,
    keep_selected_positions: np.ndarray,
) -> np.ndarray:
    model = build_backbone(checkpoint_path)
    keep_selected_positions = np.asarray(keep_selected_positions, dtype=bool)
    parts: list[np.ndarray] = []
    for batch in make_loader(adata.X, fixed_gene_indices):
        batch = {
            key: value.to(DEVICE, non_blocking=True)
            for key, value in batch.items()
        }
        hidden = model(batch["gene_ids"], batch["expr"])
        gene_hidden = hidden[:, 1:, :]
        gene_hidden = gene_hidden[:, keep_selected_positions, :]
        parts.append(gene_hidden.reshape(-1, EMBSIZE).cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0)


def reduce_embeddings(
    all_embeddings: Mapping[str, np.ndarray],
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    low_memory: bool = False,
) -> dict[str, np.ndarray]:
    save_dict: dict[str, np.ndarray] = {}
    for name, emb in all_embeddings.items():
        print(f"PCA + UMAP: {name} shape={emb.shape}")
        n_components = min(N_PCA, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(emb)
        umap2 = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=RANDOM_SEED,
            low_memory=low_memory,
        ).fit_transform(pca)
        save_dict[f"pca50_{name}"] = pca.astype(np.float32)
        save_dict[f"pca2d_{name}"] = pca[:, :2].astype(np.float32)
        save_dict[f"umap2d_{name}"] = umap2.astype(np.float32)
    return save_dict
