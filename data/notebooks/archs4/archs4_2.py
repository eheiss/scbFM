from __future__ import annotations

from pathlib import Path
import gc
import json
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


ROOT_DIR = Path(os.environ.get("SCBFM_ROOT_DIR", Path(__file__).resolve().parents[4])).expanduser().resolve()
DEFAULT_DATA_DIR = ROOT_DIR / "datasets/ARCHS4"
GENE_LIST_PATH = Path(os.getenv(
    "GENE_LIST_PATH",
    str(Path(__file__).resolve().parents[2] / "gene_list.txt"),
))
ARCHS4_IN = Path(os.getenv(
    "ARCHS4_INPUT_H5AD",
    str(DEFAULT_DATA_DIR / "archs4.h5ad"),
))
ARCHS4_OUT = Path(os.getenv(
    "ARCHS4_OUTPUT_H5AD",
    str(ARCHS4_IN),
))
STATS_OUT = Path(os.getenv(
    "ARCHS4_STATS_OUT",
    str(DEFAULT_DATA_DIR / "archs4_gene_list_stats.json"),
))
ROW_CHUNK_SIZE = int(os.getenv("ARCHS4_ROW_CHUNK_SIZE", "1000"))
COMPRESSION = os.getenv("ARCHS4_H5AD_COMPRESSION", "lzf") or None


def read_gene_list(path: Path) -> list[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def read_h5ad_metadata(
    h5ad_path: Path,
    gene_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int, int]:
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = backed.obs.copy()
        var = backed.var.copy()
        n_obs, n_vars = backed.shape
    finally:
        backed.file.close()

    gene_index = pd.Index(var.index.astype(str))
    if not gene_index.is_unique:
        duplicated = gene_index[gene_index.duplicated()].unique().tolist()
        raise ValueError(
            "ARCHS4 var_names are not unique. First duplicate IDs: "
            f"{duplicated[:20]}"
        )

    reorder_idx = gene_index.get_indexer(gene_list)
    missing = [gene for gene, idx in zip(gene_list, reorder_idx) if idx < 0]
    if missing:
        raise ValueError(
            f"{len(missing)} genes from gene_list are missing in ARCHS4. "
            f"First 20: {missing[:20]}"
        )

    out_var = var.iloc[reorder_idx].copy()
    out_var.index = pd.Index(gene_list, name=var.index.name)
    return obs, out_var, reorder_idx.astype(np.int64), n_obs, n_vars


def create_output_skeleton(
    out_path: Path,
    obs: pd.DataFrame,
    var: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    empty_x = sparse.csr_matrix((obs.shape[0], var.shape[0]), dtype=np.float32)
    adata = ad.AnnData(X=empty_x, obs=obs, var=var)
    adata.write_h5ad(out_path)

    with h5py.File(out_path, "r") as handle:
        x_group = handle["X"]
        x_attrs = dict(x_group.attrs)
        data_attrs = dict(x_group["data"].attrs)
        indices_attrs = dict(x_group["indices"].attrs)
        indptr_attrs = dict(x_group["indptr"].attrs)

    del adata, empty_x
    gc.collect()
    return x_attrs, data_attrs, indices_attrs, indptr_attrs


def require_csr_x(handle: h5py.File) -> h5py.Group:
    x = handle["X"]
    if not isinstance(x, h5py.Group):
        raise ValueError("Expected sparse CSR X group in ARCHS4 h5ad, found dense dataset.")
    required = {"data", "indices", "indptr"}
    missing = sorted(required - set(x.keys()))
    if missing:
        raise ValueError(f"Expected CSR datasets under X; missing: {missing}")
    encoding_type = x.attrs.get("encoding-type", "")
    if hasattr(encoding_type, "decode"):
        encoding_type = encoding_type.decode()
    if encoding_type not in {"csr_matrix", "csc_matrix", ""}:
        raise ValueError(f"Unsupported sparse X encoding-type: {encoding_type!r}")
    if encoding_type == "csc_matrix":
        raise ValueError(
            "ARCHS4 X is CSC, but this streaming filter expects CSR. "
            "Regenerate archs4.h5ad with archs4.py."
        )
    return x


def create_resizable_csr_x(
    out: h5py.File,
    n_obs: int,
    n_vars: int,
    x_attrs: dict,
    data_attrs: dict,
    indices_attrs: dict,
    indptr_attrs: dict,
):
    if "X" in out:
        del out["X"]
    x_group = out.create_group("X")
    for key, value in x_attrs.items():
        x_group.attrs[key] = value
    x_group.attrs["shape"] = np.asarray([n_obs, n_vars], dtype=np.int64)

    indices_dtype = np.int32 if n_vars <= np.iinfo(np.int32).max else np.int64
    data_chunk_len = max(1_000_000, ROW_CHUNK_SIZE * min(n_vars, 1024))

    data_ds = x_group.create_dataset(
        "data",
        shape=(0,),
        maxshape=(None,),
        chunks=(data_chunk_len,),
        dtype=np.float32,
        compression=COMPRESSION,
    )
    indices_ds = x_group.create_dataset(
        "indices",
        shape=(0,),
        maxshape=(None,),
        chunks=(data_chunk_len,),
        dtype=indices_dtype,
        compression=COMPRESSION,
    )
    indptr_ds = x_group.create_dataset(
        "indptr",
        shape=(n_obs + 1,),
        dtype=np.int64,
        compression=COMPRESSION,
    )

    for dataset, attrs in (
        (data_ds, data_attrs),
        (indices_ds, indices_attrs),
        (indptr_ds, indptr_attrs),
    ):
        for key, value in attrs.items():
            dataset.attrs[key] = value

    return data_ds, indices_ds, indptr_ds


def stream_filter_and_compute_stats(
    in_path: Path,
    out_path: Path,
    reorder_idx: np.ndarray,
    n_obs: int,
    n_input_vars: int,
    n_output_vars: int,
    x_attrs: dict,
    data_attrs: dict,
    indices_attrs: dict,
    indptr_attrs: dict,
) -> dict:
    old_to_new = np.full(n_input_vars, -1, dtype=np.int64)
    old_to_new[reorder_idx] = np.arange(n_output_vars, dtype=np.int64)
    in_gene_list = old_to_new >= 0

    sum_frac_nonzero = 0.0
    sum_frac_reads = 0.0
    nnz_total_out = 0

    with h5py.File(in_path, "r") as source, h5py.File(out_path, "r+") as out:
        x_in = require_csr_x(source)
        src_data = x_in["data"]
        src_indices = x_in["indices"]
        src_indptr = x_in["indptr"]

        data_ds, indices_ds, indptr_ds = create_resizable_csr_x(
            out,
            n_obs,
            n_output_vars,
            x_attrs,
            data_attrs,
            indices_attrs,
            indptr_attrs,
        )
        indptr_ds[0] = 0

        for start in range(0, n_obs, ROW_CHUNK_SIZE):
            end = min(start + ROW_CHUNK_SIZE, n_obs)
            src_row_ptr = src_indptr[start:end + 1].astype(np.int64, copy=False)
            base = int(src_row_ptr[0])
            limit = int(src_row_ptr[-1])
            local_ptr = src_row_ptr - base

            block_data = src_data[base:limit].astype(np.float32, copy=False)
            block_indices = src_indices[base:limit].astype(np.int64, copy=False)

            out_data_parts = []
            out_indices_parts = []
            out_indptr = np.zeros(end - start + 1, dtype=np.int64)

            for row in range(end - start):
                row_start = int(local_ptr[row])
                row_end = int(local_ptr[row + 1])
                cols = block_indices[row_start:row_end]
                vals = block_data[row_start:row_end]

                total_nonzero = row_end - row_start
                total_reads = float(vals.sum(dtype=np.float64))

                if total_nonzero:
                    keep = in_gene_list[cols]
                    not_in_list_nonzero = int((~keep).sum())
                    not_in_list_reads = float(vals[~keep].sum(dtype=np.float64))

                    if total_nonzero > 0:
                        sum_frac_nonzero += not_in_list_nonzero / total_nonzero
                    if total_reads > 0:
                        sum_frac_reads += not_in_list_reads / total_reads

                    if keep.any():
                        new_cols = old_to_new[cols[keep]]
                        new_vals = vals[keep]
                        order = np.argsort(new_cols, kind="mergesort")
                        out_indices_parts.append(new_cols[order])
                        out_data_parts.append(new_vals[order])
                        out_indptr[row + 1] = out_indptr[row] + int(keep.sum())
                    else:
                        out_indptr[row + 1] = out_indptr[row]
                else:
                    out_indptr[row + 1] = out_indptr[row]

            out_nnz = int(out_indptr[-1])
            next_nnz_total = nnz_total_out + out_nnz
            data_ds.resize((next_nnz_total,))
            indices_ds.resize((next_nnz_total,))

            if out_nnz:
                out_data = np.concatenate(out_data_parts).astype(np.float32, copy=False)
                out_indices = np.concatenate(out_indices_parts).astype(indices_ds.dtype, copy=False)
                data_ds[nnz_total_out:next_nnz_total] = out_data
                indices_ds[nnz_total_out:next_nnz_total] = out_indices
                del out_data, out_indices

            indptr_ds[start + 1:end + 1] = out_indptr[1:] + nnz_total_out
            nnz_total_out = next_nnz_total
            out.flush()

            print(
                f"Processed {end:,}/{n_obs:,} samples; "
                f"output chunk nnz={out_nnz:,}; output total nnz={nnz_total_out:,}",
                flush=True,
            )

            del block_data, block_indices, out_data_parts, out_indices_parts, out_indptr
            gc.collect()

    return {
        "input_h5ad": str(in_path),
        "output_h5ad": str(ARCHS4_OUT),
        "gene_list_path": str(GENE_LIST_PATH),
        "n_samples": int(n_obs),
        "n_input_genes": int(n_input_vars),
        "n_output_genes": int(n_output_vars),
        "average_portion_nonzero_genes_not_in_gene_list": float(sum_frac_nonzero / n_obs),
        "average_portion_total_reads_not_in_gene_list": float(sum_frac_reads / n_obs),
        "output_nnz": int(nnz_total_out),
    }


def main() -> None:
    if ROW_CHUNK_SIZE <= 0:
        raise ValueError("ARCHS4_ROW_CHUNK_SIZE must be positive.")

    gene_list = read_gene_list(GENE_LIST_PATH)
    if not gene_list:
        raise ValueError(f"gene_list is empty: {GENE_LIST_PATH}")
    gene_list = [str(gene) for gene in gene_list]
    if len(set(gene_list)) != len(gene_list):
        duplicated = pd.Series(gene_list).value_counts()
        duplicated = duplicated[duplicated > 1].index.tolist()
        raise ValueError(f"gene_list contains duplicate genes. First duplicates: {duplicated[:20]}")

    print(f"Input ARCHS4 h5ad: {ARCHS4_IN}", flush=True)
    print(f"Output ARCHS4 h5ad: {ARCHS4_OUT}", flush=True)
    print(f"Gene list: {GENE_LIST_PATH} ({len(gene_list):,} genes)", flush=True)
    print(f"Row chunk size: {ROW_CHUNK_SIZE:,}", flush=True)

    obs, var, reorder_idx, n_obs, n_input_vars = read_h5ad_metadata(ARCHS4_IN, gene_list)
    n_output_vars = len(gene_list)

    tmp_out = ARCHS4_OUT.with_suffix(ARCHS4_OUT.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()
    ARCHS4_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing output skeleton to {tmp_out}", flush=True)
    x_attrs, data_attrs, indices_attrs, indptr_attrs = create_output_skeleton(tmp_out, obs, var)
    del obs, var
    gc.collect()

    print("Streaming statistics + gene-list-aligned CSR matrix", flush=True)
    stats = stream_filter_and_compute_stats(
        ARCHS4_IN,
        tmp_out,
        reorder_idx,
        n_obs,
        n_input_vars,
        n_output_vars,
        x_attrs,
        data_attrs,
        indices_attrs,
        indptr_attrs,
    )

    print(
        "Average portion of non-zero genes NOT in gene_list: "
        f"{stats['average_portion_nonzero_genes_not_in_gene_list']}",
        flush=True,
    )
    print(
        "Average portion of total reads NOT in gene_list: "
        f"{stats['average_portion_total_reads_not_in_gene_list']}",
        flush=True,
    )

    backed = ad.read_h5ad(tmp_out, backed="r")
    try:
        print(f"Validation read before replace: {backed}", flush=True)
    finally:
        backed.file.close()

    tmp_out.replace(ARCHS4_OUT)
    with open(STATS_OUT, "w") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Wrote: {ARCHS4_OUT}", flush=True)
    print(f"Wrote stats: {STATS_OUT}", flush=True)


if __name__ == "__main__":
    main()
