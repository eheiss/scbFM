from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENE_LIST_PATH = ROOT / "data" / "gene_list.txt"


def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_reindexer(
    source_gene_ids: list[str] | pd.Index,
    target_gene_list: list[str],
) -> tuple[list[int], list[int], list[str]]:
    first_pos: dict[str, int] = {}
    for i, gene in enumerate(source_gene_ids):
        if gene not in first_pos:
            first_pos[gene] = i

    src_pos = []
    tgt_pos = []
    missing = []
    for j, gene in enumerate(target_gene_list):
        if gene in first_pos:
            src_pos.append(first_pos[gene])
            tgt_pos.append(j)
        else:
            missing.append(gene)
    return src_pos, tgt_pos, missing


def reindex_to_gene_list(adata: ad.AnnData, gene_list: list[str]) -> tuple[ad.AnnData, list[str]]:
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    src_pos, tgt_pos, missing = build_reindexer(adata.var_names.astype(str), gene_list)
    if not src_pos:
        raise ValueError("No input genes matched the requested gene list.")

    if sparse.issparse(adata.X):
        matrix_dtype = adata.X.dtype
        x_present = adata.X[:, src_pos].tocsr().astype(matrix_dtype, copy=False)
        if len(src_pos) == len(gene_list) and tgt_pos == list(range(len(gene_list))):
            x = x_present
        else:
            coo = x_present.tocoo(copy=False)
            target_cols = np.asarray(tgt_pos, dtype=np.int64)[coo.col]
            x = sparse.coo_matrix(
                (coo.data, (coo.row, target_cols)),
                shape=(adata.n_obs, len(gene_list)),
                dtype=matrix_dtype,
            ).tocsr()
    else:
        x_present = np.asarray(adata.X[:, src_pos])
        x = np.zeros((adata.n_obs, len(gene_list)), dtype=x_present.dtype)
        x[:, tgt_pos] = x_present

    var = pd.DataFrame(index=pd.Index(gene_list, name=adata.var_names.name or "ensembl_id"))
    out = ad.AnnData(X=x, obs=adata.obs.copy(), var=var, uns=adata.uns.copy())
    out.obs_names = adata.obs_names.copy()
    out.var_names = pd.Index(gene_list, dtype=str)
    return out, missing


def filter_min_genes(adata: ad.AnnData, min_genes: int) -> ad.AnnData:
    import numpy as np
    from scipy import sparse

    if min_genes <= 0:
        return adata

    if sparse.issparse(adata.X):
        n_genes_by_sample = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    else:
        n_genes_by_sample = (np.asarray(adata.X) > 0).sum(axis=1)

    keep_mask = n_genes_by_sample >= min_genes
    return adata[keep_mask].copy()


def _digitize(x, bins, rng) -> "np.ndarray":
    import numpy as np

    assert x.ndim == 1 and bins.ndim == 1
    left_digits = np.digitize(x, bins)
    right_digits = np.digitize(x, bins, right=True)
    random_offsets = rng.rand(len(x))
    digits = random_offsets * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def _quantile_bin_nonzero_values(values, *, bin_num: int, rng) -> "np.ndarray":
    import numpy as np

    if values.ndim != 1:
        raise ValueError("Expected a 1D array of nonzero expression values.")
    if values.size == 0:
        return np.zeros(0, dtype=np.uint8)

    # Match scGPT: token 0 is reserved for true zeros, and nonzero values are
    # quantile-binned into integer tokens 1..bin_num.
    bins = np.quantile(values, np.linspace(0, 1, bin_num + 1)[1:-1])
    digits = _digitize(values, bins, rng)
    return np.clip(digits + 1, 1, bin_num).astype(np.uint8, copy=False)


def normalize_total_quantile_bin(
    adata: ad.AnnData,
    target_sum: float,
    bin_num: int,
) -> ad.AnnData:
    import numpy as np
    from scipy import sparse

    if target_sum <= 0:
        raise ValueError("--target-sum must be positive.")
    if bin_num < 1:
        raise ValueError("--bin-num must be at least 1.")

    rng = np.random.RandomState(0)

    if sparse.issparse(adata.X):
        x = adata.X.tocsr().astype(np.float32, copy=False)
        libsize = np.asarray(x.sum(axis=1)).ravel().astype(np.float32, copy=False)
        scale = np.zeros_like(libsize, dtype=np.float32)
        nonzero = libsize > 0
        scale[nonzero] = np.float32(target_sum) / libsize[nonzero]

        x = x.multiply(scale[:, None]).tocsr()
        for row_idx in range(x.shape[0]):
            start = x.indptr[row_idx]
            end = x.indptr[row_idx + 1]
            if start == end:
                continue
            x.data[start:end] = _quantile_bin_nonzero_values(
                x.data[start:end],
                bin_num=bin_num,
                rng=rng,
            ).astype(np.float32, copy=False)
        adata.X = x.astype(np.uint8, copy=False)
    else:
        x = np.asarray(adata.X, dtype=np.float32)
        libsize = x.sum(axis=1, keepdims=True)
        nonzero_rows = libsize[:, 0] > 0
        libsize[~nonzero_rows] = 1.0
        x = x / libsize * np.float32(target_sum)

        binned = np.zeros_like(x, dtype=np.uint8)
        for row_idx, row in enumerate(x):
            nonzero = row > 0
            if not np.any(nonzero):
                continue
            binned[row_idx, nonzero] = _quantile_bin_nonzero_values(
                row[nonzero],
                bin_num=bin_num,
                rng=rng,
            )
        adata.X = sparse.csr_matrix(binned)

    return adata


def preprocess_adata_for_tokens(
    adata: ad.AnnData,
    *,
    gene_list_path: Path | None = DEFAULT_GENE_LIST_PATH,
    min_genes: int = 200,
    target_sum: float = 1e4,
    bin_num: int = 10,
    reindex_genes: bool = True,
) -> tuple[ad.AnnData, list[str]]:
    missing: list[str] = []

    if reindex_genes:
        if gene_list_path is None:
            raise ValueError("gene_list_path is required when reindex_genes=True.")
        gene_list = read_gene_list(gene_list_path)
        adata, missing = reindex_to_gene_list(adata, gene_list)
    else:
        adata.var_names_make_unique()

    adata = filter_min_genes(adata, min_genes=min_genes)
    adata = normalize_total_quantile_bin(
        adata,
        target_sum=target_sum,
        bin_num=bin_num,
    )
    return adata, missing


def reindex_adata_genes(
    adata: ad.AnnData,
    *,
    gene_list_path: Path = DEFAULT_GENE_LIST_PATH,
) -> tuple[ad.AnnData, list[str]]:
    gene_list = read_gene_list(gene_list_path)
    return reindex_to_gene_list(adata, gene_list)


def validate_token_matrix(data, *, bin_num: int, name: str = "matrix") -> None:
    import numpy as np
    from scipy import sparse

    if sparse.issparse(data):
        values = data.data
    else:
        values = np.asarray(data).ravel()

    if values.size == 0:
        return

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite token values.")

    min_value = values.min()
    max_value = values.max()
    if min_value < 0 or max_value > bin_num:
        raise ValueError(
            f"{name} must contain preprocessed token IDs in [0, {bin_num}], "
            f"got range [{min_value}, {max_value}]."
        )

    if not np.all(values == np.floor(values)):
        raise ValueError(f"{name} contains non-integer values; run shared preprocessing first.")


def preprocess_raw_h5ad(
    input_path: Path,
    output_path: Path,
    gene_list_path: Path | None,
    min_genes: int,
    target_sum: float,
    bin_num: int,
    overwrite: bool,
) -> None:
    import anndata as ad

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    print(f"Loading {input_path}")
    adata = ad.read_h5ad(input_path)
    original_shape = tuple(adata.shape)

    missing: list[str] = []
    if gene_list_path is not None:
        gene_list = read_gene_list(gene_list_path)
        print(f"Reindexing to {len(gene_list)} genes from {gene_list_path}")
        adata, missing = reindex_to_gene_list(adata, gene_list)
        print(f"Matched {adata.n_vars - len(missing)} / {adata.n_vars} target genes")
        print(f"Missing target genes filled with zeros: {len(missing)}")
    else:
        adata.var_names_make_unique()

    before_filter = adata.n_obs
    adata = filter_min_genes(adata, min_genes=min_genes)
    print(f"Filtered samples by min_genes={min_genes}: {before_filter} -> {adata.n_obs}")

    adata = normalize_total_quantile_bin(
        adata,
        target_sum=target_sum,
        bin_num=bin_num,
    )

    adata.uns["scbfm_preprocess"] = {
        "input_path": str(input_path),
        "original_shape": list(original_shape),
        "output_shape": [int(adata.n_obs), int(adata.n_vars)],
        "gene_list_path": str(gene_list_path) if gene_list_path is not None else None,
        "missing_genes": int(len(missing)),
        "min_genes": int(min_genes),
        "target_sum": float(target_sum),
        "bin_num": int(bin_num),
        "binning_strategy": "scgpt_nonzero_quantile",
        "nonzero_bin_count": int(bin_num),
        "zero_token_reserved": True,
        "output_dtype": "uint8",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(output_path)
    print(f"Saved {output_path}")

    if missing:
        missing_path = output_path.with_suffix(".missing_genes.json")
        with open(missing_path, "w") as f:
            json.dump(missing, f, indent=2)
        print(f"Saved missing gene list to {missing_path}")
