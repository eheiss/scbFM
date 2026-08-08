from __future__ import annotations

from pathlib import Path
import gc
import json
import os
import re

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# =========================
# Paths
# =========================

GENE_LIST_PATH = Path(os.getenv(
    "SCBFM_GENE_LIST_PATH",
    str(Path(__file__).resolve().parent / "gene_list.txt"),
))
GTEX_PATH = Path(os.getenv(
    "SCBFM_GTEX_H5AD",
    "/cluster/work/boeva/eheiss/datasets/GTEx/gtex.h5ad",
))
ARCHS4_PATH = Path(os.getenv(
    "SCBFM_ARCHS4_H5AD",
    "/cluster/work/boeva/eheiss/datasets/ARCHS4/archs4.h5ad",
))
ARCHS4_METADATA_PATH = Path(os.getenv(
    "SCBFM_ARCHS4_METADATA_H5",
    "/cluster/work/boeva/eheiss/datasets/ARCHS4/human_gene_v2.latest.h5",
))

OUT_DIR = Path(os.getenv(
    "SCBFM_BULK_OUT_DIR",
    "/cluster/work/boeva/eheiss/datasets/bulk",
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_OUT = OUT_DIR / "pretraining_bulk_RAW.h5ad"
PREADAPT_OUT = OUT_DIR / "preadapt_bulk_RAW.h5ad"
SUMMARY_OUT = OUT_DIR / "bulk_summary_RAW.json"
ARCHS4_GTEX_DONOR_HITS_OUT = OUT_DIR / "archs4_gtex_donor_hits_RAW.csv"
ARCHS4_DOWNSTREAM_HITS_OUT = OUT_DIR / "archs4_downstream_hits_RAW.csv"


# =========================
# Settings
# =========================

MIN_GENES = int(os.getenv("SCBFM_BULK_MIN_GENES", "200"))
ROW_CHUNK_SIZE = int(os.getenv("SCBFM_BULK_ROW_CHUNK_SIZE", "1000"))
RANDOM_SEED = int(os.getenv("SCBFM_BULK_RANDOM_SEED", "42"))
PRETRAIN_FRACTION = float(os.getenv("SCBFM_BULK_PRETRAIN_FRACTION", "0.9"))
EXPECTED_PRETRAIN_SAMPLES = int(os.getenv(
    "SCBFM_BULK_EXPECTED_PRETRAIN_SAMPLES",
    "642406",
))
COMPRESSION = os.getenv("SCBFM_BULK_H5AD_COMPRESSION", "lzf") or None

GTEX_DONOR_PATTERN = re.compile(r"GTEX-[A-Z0-9]+")
METADATA_TOKEN_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9._:-]{2,}")

DOWNSTREAM_DATASET_PATHS = {
    "TCGA": Path("/cluster/work/boeva/eheiss/datasets/TCGA/tcga.h5ad"),
    "DepMap": Path("/cluster/work/boeva/eheiss/datasets/DepMap/depmap.h5ad"),
    "GDSC": Path("/cluster/work/boeva/eheiss/datasets/GDSC/gdsc.h5ad"),
    "DiSignAtlas": Path("/cluster/work/boeva/eheiss/datasets/DiSignAtlas/disignatlas.h5ad"),
}

DOWNSTREAM_GCTX_PATHS = {
    "LINCS": Path("/cluster/work/boeva/eheiss/datasets/LINCS/level5_beta_all_n1201944x12328.gctx"),
}

DOWNSTREAM_DATASET_TERMS = {
    "TCGA": (
        "TCGA",
        "The Cancer Genome Atlas",
        "Cancer Genome Atlas",
    ),
    "DepMap": (
        "DepMap",
        "CCLE",
        "Cancer Cell Line Encyclopedia",
    ),
    "GDSC": (
        "GDSC",
        "Genomics of Drug Sensitivity in Cancer",
        "CancerRxGene",
        "Cancer RX Gene",
    ),
    "DiSignAtlas": (
        "DiSignAtlas",
        "Disease Signature Atlas",
    ),
    "LINCS": (
        "LINCS",
        "L1000",
        "Connectivity Map",
        "CMAP",
    ),
}

DOWNSTREAM_ID_PATTERNS = {
    "TCGA": (
        re.compile(r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9]{2,4}){0,4}\b"),
    ),
    "DepMap": (
        re.compile(r"\bACH-\d{6}\b"),
    ),
    "LINCS": (
        re.compile(r"\b(?:LINCS|L1000)[-_:][A-Z0-9._:-]+\b"),
    ),
}


# =========================
# Generic helpers
# =========================

def read_gene_list(path: Path) -> list[str]:
    with open(path) as handle:
        genes = [line.strip() for line in handle if line.strip()]
    if not genes:
        raise ValueError(f"Gene list is empty: {path}")
    duplicated = pd.Series(genes).value_counts()
    duplicated = duplicated[duplicated > 1]
    if not duplicated.empty:
        raise ValueError(f"gene_list contains duplicate genes. First duplicates: {duplicated.index[:20].tolist()}")
    return [str(gene) for gene in genes]


def decode_value(value) -> str:
    if hasattr(value, "decode"):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def hdf5_string_values(dataset: h5py.Dataset) -> list[str]:
    try:
        values = dataset.asstr()[:]
    except (AttributeError, TypeError):
        values = dataset[:]
    return [decode_value(value) for value in values]


def gtex_donor_id(sample_id: str) -> str:
    parts = sample_id.upper().split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else sample_id.upper()


def extract_gtex_donors(value: str, valid_donors: set[str]) -> set[str]:
    return {match for match in GTEX_DONOR_PATTERN.findall(value.upper()) if match in valid_donors}


def normalize_metadata_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def term_in_metadata_value(value_upper: str, term: str) -> bool:
    term_upper = term.upper()
    if len(term_upper) <= 5 and term_upper.replace("-", "").isalnum():
        return re.search(rf"(?<![A-Z0-9]){re.escape(term_upper)}(?![A-Z0-9])", value_upper) is not None
    return term_upper in value_upper


def validate_h5ad_gene_order(path: Path, gene_list: list[str]) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(path)

    backed = ad.read_h5ad(path, backed="r")
    try:
        var_names = backed.var_names.astype(str).tolist()
        shape = backed.shape
    finally:
        backed.file.close()

    if var_names != gene_list:
        first_mismatch = next(
            (
                i
                for i, (observed, expected) in enumerate(zip(var_names, gene_list))
                if observed != expected
            ),
            None,
        )
        if len(var_names) != len(gene_list):
            detail = f"n_vars={len(var_names)}, gene_list={len(gene_list)}"
        elif first_mismatch is not None:
            detail = (
                f"first mismatch at position {first_mismatch}: "
                f"observed={var_names[first_mismatch]!r}, expected={gene_list[first_mismatch]!r}"
            )
        else:
            detail = "gene order mismatch"
        raise ValueError(
            f"{path} is not aligned to gene_list.txt ({detail}). "
            "Run the notebook/script filtering steps first: GTEx Part III and ARCHS4 archs4_2.py."
        )

    return int(shape[0]), int(shape[1])


def metadata_tokens_from_adata(path: Path) -> set[str]:
    if not path.exists():
        print(f"Downstream dataset not found, skipping ID extraction: {path}")
        return set()

    print(f"Loading downstream sample IDs from {path}...")
    adata = ad.read_h5ad(path, backed="r")
    try:
        tokens = {normalize_metadata_token(str(idx)) for idx in adata.obs_names}
    finally:
        adata.file.close()
    tokens = {token for token in tokens if len(token) >= 4}
    print(f"Loaded {len(tokens)} downstream sample IDs from {path.name}")
    return tokens


def metadata_tokens_from_gctx(path: Path) -> set[str]:
    if not path.exists():
        print(f"Downstream GCTX not found, skipping ID extraction: {path}")
        return set()

    print(f"Loading downstream sample IDs from {path}...")
    candidate_names = {"ID", "SIG_ID", "SAMPLE_ID", "SAMPLE", "GEO_ID", "DISTIL_ID"}
    tokens: set[str] = set()

    with h5py.File(path, "r") as handle:
        dataset_names: list[str] = []

        def collect_candidate(name: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            upper_name = name.upper()
            base_name = upper_name.rsplit("/", 1)[-1]
            if "META/COL" in upper_name and base_name in candidate_names and obj.ndim == 1:
                dataset_names.append(name)

        handle.visititems(collect_candidate)

        for dataset_name in dataset_names:
            values = hdf5_string_values(handle[dataset_name])
            for value in values:
                token = normalize_metadata_token(value)
                if len(token) >= 4:
                    tokens.add(token)

    print(f"Loaded {len(tokens)} downstream sample IDs from {path.name}")
    return tokens


def build_downstream_id_tokens() -> dict[str, set[str]]:
    tokens = {
        dataset: metadata_tokens_from_adata(path)
        for dataset, path in DOWNSTREAM_DATASET_PATHS.items()
    }
    for dataset, path in DOWNSTREAM_GCTX_PATHS.items():
        tokens.setdefault(dataset, set()).update(metadata_tokens_from_gctx(path))
    return tokens


def extract_downstream_matches(
    value: str,
    downstream_id_tokens: dict[str, set[str]],
) -> dict[str, dict[str, set[str]]]:
    value_upper = value.upper()
    value_tokens = {normalize_metadata_token(token) for token in METADATA_TOKEN_PATTERN.findall(value_upper)}
    value_tokens = {token for token in value_tokens if len(token) >= 4}
    matches: dict[str, dict[str, set[str]]] = {}

    for dataset, terms in DOWNSTREAM_DATASET_TERMS.items():
        matched_terms = {term for term in terms if term_in_metadata_value(value_upper, term)}
        matched_ids = set()

        for pattern in DOWNSTREAM_ID_PATTERNS.get(dataset, ()):
            matched_ids.update(pattern.findall(value_upper))

        dataset_tokens = downstream_id_tokens.get(dataset, set())
        if dataset_tokens and value_tokens:
            matched_ids.update(sorted(value_tokens.intersection(dataset_tokens)))

        if matched_terms or matched_ids:
            matches[dataset] = {
                "matched_terms": matched_terms,
                "matched_ids": matched_ids,
            }

    return matches


# =========================
# Filter planning
# =========================

def row_nonzero_counts(x_chunk) -> np.ndarray:
    if sparse.issparse(x_chunk):
        return np.asarray(x_chunk.getnnz(axis=1)).ravel()
    return np.asarray(np.asarray(x_chunk) > 0).sum(axis=1)


def collect_min_gene_keep_mask(path: Path, min_genes: int) -> np.ndarray:
    backed = ad.read_h5ad(path, backed="r")
    try:
        keep = np.zeros(backed.n_obs, dtype=bool)
        for start in range(0, backed.n_obs, ROW_CHUNK_SIZE):
            end = min(start + ROW_CHUNK_SIZE, backed.n_obs)
            counts = row_nonzero_counts(backed.X[start:end])
            keep[start:end] = counts >= min_genes
            print(
                f"{path.name}: min_genes scan {end:,}/{backed.n_obs:,} "
                f"(kept so far {int(keep[:end].sum()):,})",
                flush=True,
            )
    finally:
        backed.file.close()
    return keep


def load_archs4_singlecell_probability(n_obs: int) -> np.ndarray | None:
    if not ARCHS4_METADATA_PATH.exists():
        print(
            "ARCHS4 metadata HDF5 not found; skipping singlecellprobability filter: "
            f"{ARCHS4_METADATA_PATH}"
        )
        return None

    with h5py.File(ARCHS4_METADATA_PATH, "r") as handle:
        if "meta/samples/singlecellprobability" not in handle:
            print("ARCHS4 metadata HDF5 has no meta/samples/singlecellprobability; skipping filter.")
            return None
        sc_prob = np.asarray(handle["meta/samples/singlecellprobability"][:], dtype=np.float32)

    if len(sc_prob) != n_obs:
        print(
            "ARCHS4 singlecellprobability length does not match archs4.h5ad rows; "
            f"skipping filter ({len(sc_prob)} vs {n_obs})."
        )
        return None
    return sc_prob


def scan_archs4_metadata_hits(
    obs: pd.DataFrame,
    gtex_donors: set[str],
    downstream_id_tokens: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray]:
    print("Scanning ARCHS4 h5ad obs metadata for GTEx donor/downstream leakage...")
    gtex_hit = np.zeros(obs.shape[0], dtype=bool)
    downstream_hit = np.zeros(obs.shape[0], dtype=bool)
    gtex_rows: dict[int, dict[str, set[str]]] = {}
    downstream_rows: dict[int, dict[str, set[str]]] = {}

    obs_str = obs.astype("string").fillna("").astype(str)
    for field in obs_str.columns:
        values = obs_str[field].tolist()
        for i, value in enumerate(values):
            if gtex_donors and "GTEX" in value.upper():
                matched_donors = extract_gtex_donors(value, gtex_donors)
                if matched_donors:
                    gtex_hit[i] = True
                    record = gtex_rows.setdefault(i, {"matched_fields": set(), "matched_donors": set()})
                    record["matched_fields"].add(field)
                    record["matched_donors"].update(matched_donors)

            matches = extract_downstream_matches(value, downstream_id_tokens)
            if matches:
                downstream_hit[i] = True
                record = downstream_rows.setdefault(
                    i,
                    {
                        "matched_fields": set(),
                        "matched_datasets": set(),
                        "matched_terms": set(),
                        "matched_ids": set(),
                    },
                )
                record["matched_fields"].add(field)
                for dataset, match in matches.items():
                    record["matched_datasets"].add(dataset)
                    record["matched_terms"].update(
                        f"{dataset}:{term}" for term in match["matched_terms"]
                    )
                    record["matched_ids"].update(
                        f"{dataset}:{matched_id}" for matched_id in match["matched_ids"]
                    )

    gtex_out_rows = []
    for i, record in sorted(gtex_rows.items()):
        row = {
            "archs4_row": int(i),
            "obs_name": str(obs.index[i]),
            "matched_fields": ";".join(sorted(record["matched_fields"])),
            "matched_donors": ";".join(sorted(record["matched_donors"])),
        }
        for field in obs.columns:
            row[field] = obs.iloc[i][field]
        gtex_out_rows.append(row)
    pd.DataFrame(gtex_out_rows).to_csv(ARCHS4_GTEX_DONOR_HITS_OUT, index=False)

    downstream_out_rows = []
    for i, record in sorted(downstream_rows.items()):
        row = {
            "archs4_row": int(i),
            "obs_name": str(obs.index[i]),
            "matched_fields": ";".join(sorted(record["matched_fields"])),
            "matched_datasets": ";".join(sorted(record["matched_datasets"])),
            "matched_terms": ";".join(sorted(record["matched_terms"])),
            "matched_ids": ";".join(sorted(record["matched_ids"])),
        }
        for field in obs.columns:
            row[field] = obs.iloc[i][field]
        downstream_out_rows.append(row)
    pd.DataFrame(downstream_out_rows).to_csv(ARCHS4_DOWNSTREAM_HITS_OUT, index=False)

    print(
        f"ARCHS4 GTEx donor metadata hits: {int(gtex_hit.sum()):,} / {obs.shape[0]:,}; "
        f"details saved to {ARCHS4_GTEX_DONOR_HITS_OUT}"
    )
    print(
        f"ARCHS4 downstream metadata hits: {int(downstream_hit.sum()):,} / {obs.shape[0]:,}; "
        f"details saved to {ARCHS4_DOWNSTREAM_HITS_OUT}"
    )
    return gtex_hit, downstream_hit


def build_filtered_records(gene_list: list[str]) -> tuple[pd.DataFrame, dict]:
    gtex_shape = validate_h5ad_gene_order(GTEX_PATH, gene_list)
    archs4_shape = validate_h5ad_gene_order(ARCHS4_PATH, gene_list)
    print(f"GTEx input: {gtex_shape[0]:,} samples x {gtex_shape[1]:,} genes")
    print(f"ARCHS4 input: {archs4_shape[0]:,} samples x {archs4_shape[1]:,} genes")

    gtex_backed = ad.read_h5ad(GTEX_PATH, backed="r")
    try:
        gtex_obs_names = gtex_backed.obs_names.astype(str).tolist()
    finally:
        gtex_backed.file.close()
    gtex_donors = {gtex_donor_id(sample_id) for sample_id in gtex_obs_names}
    print(f"GTEx donor IDs: {len(gtex_donors):,}")

    downstream_id_tokens = build_downstream_id_tokens()

    gtex_min_gene_keep = collect_min_gene_keep_mask(GTEX_PATH, MIN_GENES)

    archs4_backed = ad.read_h5ad(ARCHS4_PATH, backed="r")
    try:
        archs4_obs = archs4_backed.obs.copy()
        archs4_n_obs = archs4_backed.n_obs
    finally:
        archs4_backed.file.close()

    archs4_min_gene_keep = collect_min_gene_keep_mask(ARCHS4_PATH, MIN_GENES)
    sc_prob = load_archs4_singlecell_probability(archs4_n_obs)
    if sc_prob is None:
        archs4_bulk_like_keep = np.ones(archs4_n_obs, dtype=bool)
        excluded_single_cell_like = 0
    else:
        archs4_bulk_like_keep = sc_prob < 0.5
        excluded_single_cell_like = int((~archs4_bulk_like_keep).sum())

    archs4_gtex_hit, archs4_downstream_hit = scan_archs4_metadata_hits(
        archs4_obs,
        gtex_donors,
        downstream_id_tokens,
    )

    archs4_keep = (
        archs4_min_gene_keep
        & archs4_bulk_like_keep
        & ~archs4_gtex_hit
        & ~archs4_downstream_hit
    )

    gtex_idx = np.flatnonzero(gtex_min_gene_keep)
    archs4_idx = np.flatnonzero(archs4_keep)

    records = pd.concat(
        [
            pd.DataFrame({"source": "GTEx", "source_order": 0, "row_idx": gtex_idx}),
            pd.DataFrame({"source": "ARCHS4", "source_order": 1, "row_idx": archs4_idx}),
        ],
        ignore_index=True,
    )

    summary = {
        "gene_list_path": str(GENE_LIST_PATH),
        "n_genes": len(gene_list),
        "min_genes": MIN_GENES,
        "random_seed": RANDOM_SEED,
        "inputs": {
            "GTEx": str(GTEX_PATH),
            "ARCHS4": str(ARCHS4_PATH),
            "ARCHS4_metadata": str(ARCHS4_METADATA_PATH),
        },
        "filters": {
            "GTEx": {
                "input_samples": int(gtex_shape[0]),
                "kept_min_genes": int(gtex_min_gene_keep.sum()),
                "final_kept": int(len(gtex_idx)),
            },
            "ARCHS4": {
                "input_samples": int(archs4_shape[0]),
                "kept_min_genes": int(archs4_min_gene_keep.sum()),
                "excluded_single_cell_like": int(excluded_single_cell_like),
                "excluded_gtex_donor_hits": int(archs4_gtex_hit.sum()),
                "excluded_downstream_hits": int(archs4_downstream_hit.sum()),
                "final_kept": int(len(archs4_idx)),
            },
        },
        "post_filter_samples": int(len(records)),
    }
    print(f"Post-filter bulk samples: {len(records):,}")
    return records, summary


def split_pretrain_preadapt_records(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < PRETRAIN_FRACTION < 1:
        raise ValueError("SCBFM_BULK_PRETRAIN_FRACTION must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("Need at least two post-filter bulk samples to create pretraining/preadapt datasets.")

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_positions = rng.permutation(len(records))
    n_pretrain = int(round(len(records) * PRETRAIN_FRACTION))
    n_pretrain = min(max(n_pretrain, 1), len(records) - 1)

    pretrain_positions = np.sort(shuffled_positions[:n_pretrain])
    preadapt_positions = np.sort(shuffled_positions[n_pretrain:])

    pretrain_records = records.iloc[pretrain_positions].copy().reset_index(drop=True)
    preadapt_records = records.iloc[preadapt_positions].copy().reset_index(drop=True)
    return pretrain_records, preadapt_records


# =========================
# Streaming h5ad writing
# =========================

def read_obs_for_records(records: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    source_paths = {"GTEx": GTEX_PATH, "ARCHS4": ARCHS4_PATH}
    obs_parts = []

    for source in ("GTEx", "ARCHS4"):
        source_records = records[records["source"] == source]
        if source_records.empty:
            continue
        row_idx = source_records["row_idx"].to_numpy(dtype=np.int64)
        backed = ad.read_h5ad(source_paths[source], backed="r")
        try:
            obs = backed.obs.iloc[row_idx].copy()
        finally:
            backed.file.close()
        obs["dataset"] = source
        obs["bulk_source"] = source
        obs["bulk_dataset"] = dataset_label
        obs_parts.append(obs)

    if not obs_parts:
        raise ValueError(f"No rows selected for dataset {dataset_label!r}.")

    obs = pd.concat(obs_parts, axis=0)
    obs.index = pd.Index(obs.index.astype(str), name=obs.index.name)
    return obs


def create_output_skeleton(out_path: Path, obs: pd.DataFrame, gene_list: list[str]) -> tuple[dict, dict, dict, dict]:
    empty_x = sparse.csr_matrix((obs.shape[0], len(gene_list)), dtype=np.float32)
    var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))
    adata = ad.AnnData(X=empty_x, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.write_h5ad(out_path)

    with h5py.File(out_path, "r") as handle:
        x_group = handle["X"]
        x_attrs = dict(x_group.attrs)
        data_attrs = dict(x_group["data"].attrs)
        indices_attrs = dict(x_group["indices"].attrs)
        indptr_attrs = dict(x_group["indptr"].attrs)

    del adata, empty_x, var
    gc.collect()
    return x_attrs, data_attrs, indices_attrs, indptr_attrs


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

    data_chunk_len = max(1_000_000, ROW_CHUNK_SIZE * min(n_vars, 1024))
    indices_dtype = np.int32 if n_vars <= np.iinfo(np.int32).max else np.int64

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


def append_csr_block(
    data_ds,
    indices_ds,
    indptr_ds,
    csr_block: sparse.csr_matrix,
    output_row_offset: int,
    nnz_total: int,
) -> int:
    csr_block = csr_block.astype(np.float32, copy=False).tocsr()
    csr_block.eliminate_zeros()

    nnz = int(csr_block.nnz)
    next_nnz_total = nnz_total + nnz
    data_ds.resize((next_nnz_total,))
    indices_ds.resize((next_nnz_total,))

    if nnz:
        data_ds[nnz_total:next_nnz_total] = csr_block.data.astype(np.float32, copy=False)
        indices_ds[nnz_total:next_nnz_total] = csr_block.indices.astype(indices_ds.dtype, copy=False)

    indptr_ds[output_row_offset + 1:output_row_offset + csr_block.shape[0] + 1] = (
        csr_block.indptr[1:].astype(np.int64, copy=False) + nnz_total
    )
    return next_nnz_total


def stream_source_rows_to_output(
    source: str,
    selected_row_idx: np.ndarray,
    out_handles,
    output_row_offset: int,
    nnz_total: int,
) -> tuple[int, int]:
    if len(selected_row_idx) == 0:
        return output_row_offset, nnz_total

    source_path = GTEX_PATH if source == "GTEx" else ARCHS4_PATH
    data_ds, indices_ds, indptr_ds = out_handles
    backed = ad.read_h5ad(source_path, backed="r")
    try:
        selected = np.zeros(backed.n_obs, dtype=bool)
        selected[selected_row_idx] = True

        for start in range(0, backed.n_obs, ROW_CHUNK_SIZE):
            end = min(start + ROW_CHUNK_SIZE, backed.n_obs)
            row_mask = selected[start:end]
            if not row_mask.any():
                continue

            x_chunk = backed.X[start:end]
            if sparse.issparse(x_chunk):
                csr_block = x_chunk.tocsr()[row_mask]
            else:
                csr_block = sparse.csr_matrix(np.asarray(x_chunk)[row_mask])

            nnz_total = append_csr_block(
                data_ds,
                indices_ds,
                indptr_ds,
                csr_block,
                output_row_offset,
                nnz_total,
            )
            output_row_offset += csr_block.shape[0]
            print(
                f"{source}: wrote {output_row_offset:,} output rows "
                f"(source scan {end:,}/{backed.n_obs:,}; total nnz={nnz_total:,})",
                flush=True,
            )
            del x_chunk, csr_block
            gc.collect()
    finally:
        backed.file.close()

    return output_row_offset, nnz_total


def write_bulk_h5ad(
    records: pd.DataFrame,
    out_path: Path,
    dataset_label: str,
    gene_list: list[str],
) -> dict:
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()

    records = records.sort_values(["source_order", "row_idx"]).reset_index(drop=True)
    obs = read_obs_for_records(records, dataset_label=dataset_label)
    print(f"Writing bulk dataset skeleton: {tmp_out} ({len(obs):,} samples)")
    x_attrs, data_attrs, indices_attrs, indptr_attrs = create_output_skeleton(tmp_out, obs, gene_list)
    del obs
    gc.collect()

    with h5py.File(tmp_out, "r+") as out:
        out_handles = create_resizable_csr_x(
            out,
            n_obs=len(records),
            n_vars=len(gene_list),
            x_attrs=x_attrs,
            data_attrs=data_attrs,
            indices_attrs=indices_attrs,
            indptr_attrs=indptr_attrs,
        )
        out_handles[2][0] = 0

        output_row_offset = 0
        nnz_total = 0
        for source in ("GTEx", "ARCHS4"):
            source_records = records[records["source"] == source]
            row_idx = source_records["row_idx"].to_numpy(dtype=np.int64)
            output_row_offset, nnz_total = stream_source_rows_to_output(
                source,
                row_idx,
                out_handles,
                output_row_offset,
                nnz_total,
            )
        out.flush()

    if output_row_offset != len(records):
        raise ValueError(
            "Internal row count mismatch while writing bulk dataset: "
            f"wrote {output_row_offset}, expected {len(records)}"
        )

    backed = ad.read_h5ad(tmp_out, backed="r")
    try:
        print(f"Validation read bulk dataset: {backed}")
    finally:
        backed.file.close()

    tmp_out.replace(out_path)
    print(f"Saved bulk dataset: {out_path}")
    return {
        "path": str(out_path),
        "samples": int(len(records)),
        "genes": int(len(gene_list)),
        "shape": [int(len(records)), int(len(gene_list))],
        "source_counts": {
            source: int((records["source"] == source).sum())
            for source in ("GTEx", "ARCHS4")
        },
    }


# =========================
# Main
# =========================

def main() -> None:
    if ROW_CHUNK_SIZE <= 0:
        raise ValueError("SCBFM_BULK_ROW_CHUNK_SIZE must be positive.")
    if EXPECTED_PRETRAIN_SAMPLES <= 0:
        raise ValueError("SCBFM_BULK_EXPECTED_PRETRAIN_SAMPLES must be positive.")

    gene_list = read_gene_list(GENE_LIST_PATH)
    print(f"Gene list length: {len(gene_list):,}")

    records, summary = build_filtered_records(gene_list)
    pretrain_records, preadapt_records = split_pretrain_preadapt_records(records)
    if len(pretrain_records) != EXPECTED_PRETRAIN_SAMPLES:
        raise ValueError(
            "The controlled benchmark requires exactly "
            f"{EXPECTED_PRETRAIN_SAMPLES:,} bulk pretraining profiles, but the "
            f"current filtering and split produced {len(pretrain_records):,}. "
            "Resolve the source-data or filtering discrepancy before pretraining."
        )
    print(
        "Writing post-filter bulk datasets: "
        f"pretraining={len(pretrain_records):,} samples, "
        f"preadapt={len(preadapt_records):,} samples "
        f"(fraction={PRETRAIN_FRACTION:.3f}, seed={RANDOM_SEED})."
    )

    summary["pretrain_fraction"] = PRETRAIN_FRACTION
    summary["preadapt_fraction"] = 1 - PRETRAIN_FRACTION
    summary["expected_pretraining_samples"] = EXPECTED_PRETRAIN_SAMPLES

    summary["pretraining_dataset"] = write_bulk_h5ad(
        pretrain_records,
        PRETRAIN_OUT,
        dataset_label="pretraining_bulk_RAW",
        gene_list=gene_list,
    )
    summary["preadapt_dataset"] = write_bulk_h5ad(
        preadapt_records,
        PREADAPT_OUT,
        dataset_label="preadapt_bulk_RAW",
        gene_list=gene_list,
    )
    summary["runner_split_note"] = (
        "This script creates pretraining/preadapt datasets only. "
        "The pretrain runner separately performs its own train/validation split "
        "inside whichever dataset is passed through pretrain.data_path, using "
        "pretrain.validation_split."
    )

    with open(SUMMARY_OUT, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote summary: {SUMMARY_OUT}")
    print("Final bulk dataset dimensions:")
    print(
        "  pretraining_bulk_RAW.h5ad: "
        f"{summary['pretraining_dataset']['samples']:,} samples x "
        f"{summary['pretraining_dataset']['genes']:,} genes"
    )
    print(
        "  preadapt_bulk_RAW.h5ad: "
        f"{summary['preadapt_dataset']['samples']:,} samples x "
        f"{summary['preadapt_dataset']['genes']:,} genes"
    )


if __name__ == "__main__":
    main()
