from __future__ import annotations

from pathlib import Path
import gc
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


ARCHS4_IN = Path(os.getenv(
    "ARCHS4_INPUT",
    "/cluster/work/boeva/eheiss/datasets/ARCHS4/human_gene_v2.latest.h5",
))
ARCHS4_OUT = Path(os.getenv(
    "ARCHS4_OUTPUT",
    "/cluster/work/boeva/eheiss/datasets/ARCHS4/archs4.h5ad",
))
SAMPLE_CHUNK_SIZE = int(os.getenv("ARCHS4_SAMPLE_CHUNK_SIZE", "1000"))
COMPRESSION = os.getenv("ARCHS4_H5AD_COMPRESSION", "lzf") or None


def decode_hdf5_strings(dataset: h5py.Dataset) -> list[str]:
    try:
        values = dataset.asstr()[:]
    except (AttributeError, TypeError):
        values = dataset[:]
    return [
        value.decode("latin-1", errors="ignore") if hasattr(value, "decode") else str(value)
        for value in values
    ]


def expression_orientation(
    expression: h5py.Dataset,
    n_genes: int,
    n_samples: int,
) -> str:
    if expression.shape == (n_genes, n_samples):
        return "genes_by_samples"
    if expression.shape == (n_samples, n_genes):
        return "samples_by_genes"
    raise ValueError(
        "Could not infer ARCHS4 expression orientation: "
        f"expression shape={expression.shape}, genes={n_genes}, samples={n_samples}."
    )


def read_expression_block(
    expression: h5py.Dataset,
    orientation: str,
    start: int,
    end: int,
) -> np.ndarray:
    if orientation == "genes_by_samples":
        return np.asarray(expression[:, start:end], dtype=np.float32).T
    if orientation == "samples_by_genes":
        return np.asarray(expression[start:end, :], dtype=np.float32)
    raise ValueError(f"Unknown expression orientation: {orientation}")


def create_h5ad_skeleton(
    out_path: Path,
    obs: pd.DataFrame,
    var: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    empty_x = sparse.csr_matrix((obs.shape[0], var.shape[0]), dtype=np.float32)
    adata = ad.AnnData(X=empty_x, obs=obs, var=var)
    adata.write_h5ad(out_path)

    with h5py.File(out_path, "r") as handle:
        x_group = handle["X"]
        attrs = dict(x_group.attrs)
        data_attrs = dict(x_group["data"].attrs)
        indices_attrs = dict(x_group["indices"].attrs)
        indptr_attrs = dict(x_group["indptr"].attrs)

    del adata, empty_x
    gc.collect()
    return attrs, data_attrs, indices_attrs, indptr_attrs


def replace_x_with_streamed_csr(
    out_path: Path,
    source_path: Path,
    n_samples: int,
    n_genes: int,
    orientation: str,
    x_attrs: dict,
    data_attrs: dict,
    indices_attrs: dict,
    indptr_attrs: dict,
) -> None:
    indices_dtype = np.int32 if n_genes <= np.iinfo(np.int32).max else np.int64
    data_chunk_len = max(1_000_000, SAMPLE_CHUNK_SIZE * min(n_genes, 1024))

    with h5py.File(source_path, "r") as source, h5py.File(out_path, "r+") as out:
        expression = source["data/expression"]

        del out["X"]
        x_group = out.create_group("X")
        for key, value in x_attrs.items():
            x_group.attrs[key] = value
        x_group.attrs["shape"] = np.asarray([n_samples, n_genes], dtype=np.int64)

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
            shape=(n_samples + 1,),
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

        nnz_total = 0
        indptr_ds[0] = 0

        for start in range(0, n_samples, SAMPLE_CHUNK_SIZE):
            end = min(start + SAMPLE_CHUNK_SIZE, n_samples)
            dense_block = read_expression_block(expression, orientation, start, end)
            csr_block = sparse.csr_matrix(dense_block)
            csr_block.eliminate_zeros()

            nnz = int(csr_block.nnz)
            next_nnz_total = nnz_total + nnz
            data_ds.resize((next_nnz_total,))
            indices_ds.resize((next_nnz_total,))

            data_ds[nnz_total:next_nnz_total] = csr_block.data.astype(np.float32, copy=False)
            indices_ds[nnz_total:next_nnz_total] = csr_block.indices.astype(indices_dtype, copy=False)
            indptr_ds[start + 1:end + 1] = csr_block.indptr[1:].astype(np.int64, copy=False) + nnz_total

            nnz_total = next_nnz_total
            out.flush()

            print(
                f"Wrote samples {end}/{n_samples}; "
                f"chunk nnz={nnz:,}; total nnz={nnz_total:,}",
                flush=True,
            )

            del dense_block, csr_block
            gc.collect()


def main() -> None:
    ARCHS4_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = ARCHS4_OUT.with_suffix(ARCHS4_OUT.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()

    with h5py.File(ARCHS4_IN, "r") as source:
        genes = decode_hdf5_strings(source["meta/genes/symbol"])
        ensembl = decode_hdf5_strings(source["meta/genes/ensembl_gene"])
        geo_accession = decode_hdf5_strings(source["meta/samples/geo_accession"])
        characteristics = decode_hdf5_strings(source["meta/samples/characteristics_ch1"])
        sample = decode_hdf5_strings(source["meta/samples/sample"])
        series_id = decode_hdf5_strings(source["meta/samples/series_id"])
        source_name = decode_hdf5_strings(source["meta/samples/source_name_ch1"])
        organism = decode_hdf5_strings(source["meta/samples/organism_ch1"])

        n_genes = len(ensembl)
        n_samples = len(geo_accession)
        orientation = expression_orientation(source["data/expression"], n_genes, n_samples)

        print(
            "ARCHS4 input: "
            f"{n_samples:,} samples x {n_genes:,} genes; "
            f"expression shape={source['data/expression'].shape}; "
            f"orientation={orientation}",
            flush=True,
        )

    obs = pd.DataFrame(
        {
            "characteristics": characteristics,
            "sample": sample,
            "series_id": series_id,
            "source": source_name,
            "organism": organism,
        },
        index=pd.Index(geo_accession, name="geo_accession"),
    )
    var = pd.DataFrame(
        {"symbol": genes},
        index=pd.Index(ensembl, name="ensembl_id"),
    )

    print(f"Writing h5ad skeleton to {tmp_out}", flush=True)
    x_attrs, data_attrs, indices_attrs, indptr_attrs = create_h5ad_skeleton(tmp_out, obs, var)
    del obs, var, genes, ensembl, geo_accession, characteristics, sample, series_id, source_name, organism
    gc.collect()

    print("Streaming ARCHS4 expression matrix into h5ad CSR storage", flush=True)
    replace_x_with_streamed_csr(
        tmp_out,
        ARCHS4_IN,
        n_samples,
        n_genes,
        orientation,
        x_attrs,
        data_attrs,
        indices_attrs,
        indptr_attrs,
    )

    tmp_out.replace(ARCHS4_OUT)
    print(f"Saved {ARCHS4_OUT}", flush=True)

    backed = ad.read_h5ad(ARCHS4_OUT, backed="r")
    try:
        print(f"Validation read: {backed}", flush=True)
    finally:
        backed.file.close()


if __name__ == "__main__":
    main()
