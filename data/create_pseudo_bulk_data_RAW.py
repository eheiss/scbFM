from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import re
import time
import urllib.request

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

cellxgene_census = None


# =========================
# Paths
# =========================

ROOT_DIR = Path(os.environ.get("SCBFM_ROOT_DIR", Path(__file__).resolve().parents[2])).expanduser().resolve()

GENE_LIST_PATH = Path(os.getenv(
    "SCBFM_GENE_LIST_PATH",
    str(Path(__file__).resolve().parent / "gene_list.txt"),
))

OUT_DIR = Path(os.getenv(
    "SCBFM_PSEUDO_OUT_DIR",
    str(ROOT_DIR / "datasets/pseudo_bulk"),
))
CHUNK_DIR = OUT_DIR / "pseudo_bulk_RAW_chunks"
SOURCE_CHUNK_DIR = OUT_DIR / "source_cell_chunks"
MERGE_TMP_DIR = OUT_DIR / "RAW_merge_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT = OUT_DIR / "pseudo_bulk_RAW.h5ad"
PLAN_OUT = OUT_DIR / "pseudo_bulk_sampling_plan.csv"
SUMMARY_OUT = OUT_DIR / "pseudo_bulk_summary_RAW.json"
MISSING_GENES_OUT = OUT_DIR / "cellxgene_missing_genes.json"
ELIGIBLE_CONTEXTS_OUT = OUT_DIR / "eligible_contexts.csv"
SOURCE_POOL_QUOTAS_OUT = OUT_DIR / "source_cell_pool_quotas.csv"
SAMPLED_SOURCE_CELLS_OUT = OUT_DIR / "sampled_source_cells.csv"
SOURCE_ADATA_OUT = OUT_DIR / "sampled_source_cells_aligned.h5ad"
SOURCE_CHUNK_MANIFEST_OUT = OUT_DIR / "source_cell_chunk_manifest.csv"
SOURCE_DOWNLOAD_SUMMARY_OUT = OUT_DIR / "source_cell_download_summary.json"
CELL_TYPE_MAPPING_OUT = OUT_DIR / "cell_type_ontology_mapping.csv"
TARGET_AUDIT_OUT = OUT_DIR / "broad_cell_type_audit.csv"
DATASET_AUDIT_OUT = OUT_DIR / "pseudo_bulk_dataset_audit.json"
METADATA_MANIFEST_OUT = OUT_DIR / "metadata_audit_manifest.json"
SAMPLING_MANIFEST_OUT = OUT_DIR / "sampling_plan_manifest.json"


# =========================
# Settings
# =========================

ORGANISM = os.getenv("SCBFM_PSEUDO_ORGANISM", "Homo sapiens")
CENSUS_ORGANISM_KEY = os.getenv("SCBFM_PSEUDO_CENSUS_ORGANISM_KEY", "homo_sapiens")
CENSUS_VERSION = os.getenv(
    "SCBFM_PSEUDO_CENSUS_VERSION",
    os.getenv("SCBFM_CENSUS_VERSION", "2025-11-08"),
)
MIN_GENES = int(os.getenv("SCBFM_PSEUDO_MIN_GENES", "200"))
TARGET_PSEUDO_BULKS = int(os.getenv(
    "SCBFM_PSEUDO_TARGET_PSEUDO_BULKS",
    os.getenv("SCBFM_TARGET_PSEUDO_BULKS", "20000"),
))
CELLS_PER_PSEUDO_BULK = int(os.getenv(
    "SCBFM_PSEUDO_CELLS_PER_PSEUDO_BULK",
    os.getenv("SCBFM_CELLS_PER_PSEUDO_BULK", "1000"),
))
DOWNLOAD_CHUNK_SIZE = int(os.getenv("SCBFM_PSEUDO_DOWNLOAD_CHUNK_SIZE", "5000"))
DOWNLOAD_MAX_ATTEMPTS = int(os.getenv("SCBFM_PSEUDO_DOWNLOAD_MAX_ATTEMPTS", "8"))
DOWNLOAD_RETRY_BASE_SECONDS = float(
    os.getenv("SCBFM_PSEUDO_DOWNLOAD_RETRY_BASE_SECONDS", "15")
)
DOWNLOAD_RETRY_MAX_SECONDS = float(
    os.getenv("SCBFM_PSEUDO_DOWNLOAD_RETRY_MAX_SECONDS", "120")
)
WRITE_CHUNK_SIZE = int(os.getenv("SCBFM_PSEUDO_WRITE_CHUNK_SIZE", "500"))
MERGE_BATCH_SIZE = int(os.getenv("SCBFM_PSEUDO_MERGE_BATCH_SIZE", "8"))
RANDOM_SEED = int(os.getenv("SCBFM_PSEUDO_RANDOM_SEED", "2021"))
GENERATOR_SCHEMA_VERSION = 2

DEFAULT_CELL_ONTOLOGY_RELEASE = "2026-06-08"
CELL_ONTOLOGY_RELEASE = os.getenv(
    "SCBFM_CELL_ONTOLOGY_RELEASE",
    DEFAULT_CELL_ONTOLOGY_RELEASE,
)
CELL_ONTOLOGY_PATH = Path(os.getenv(
    "SCBFM_CELL_ONTOLOGY_PATH",
    str(OUT_DIR / f"cl-basic-{CELL_ONTOLOGY_RELEASE}.obo"),
))
CELL_ONTOLOGY_URL = os.getenv(
    "SCBFM_CELL_ONTOLOGY_URL",
    (
        "https://purl.obolibrary.org/obo/cl/releases/"
        f"{CELL_ONTOLOGY_RELEASE}/cl-basic.obo"
    ),
)
BROAD_CELL_TYPE_CONFIG_PATH = Path(os.getenv(
    "SCBFM_BROAD_CELL_TYPE_CONFIG_PATH",
    str(Path(__file__).resolve().parent / "deconv_broad_cell_types.csv"),
))

MIN_CONTEXT_CELL_TYPES = int(os.getenv("SCBFM_MIN_CONTEXT_CELL_TYPES", "2"))
MIN_AVAILABLE_CELLS_PER_CELLTYPE = int(
    os.getenv("SCBFM_MIN_AVAILABLE_CELLS_PER_CELLTYPE", "20")
)
MIN_CONTEXT_TOTAL_CELLS = int(os.getenv("SCBFM_MIN_CONTEXT_TOTAL_CELLS", "100"))
MIN_TARGET_CONTEXTS = int(os.getenv("SCBFM_MIN_TARGET_CONTEXTS", "20"))
MIN_TARGET_SOURCE_CELLS = int(os.getenv("SCBFM_MIN_TARGET_SOURCE_CELLS", "1000"))
MIN_EXPECTED_TARGETS = int(os.getenv("SCBFM_MIN_EXPECTED_TARGETS", "15"))
MAX_EXPECTED_TARGETS = int(os.getenv("SCBFM_MAX_EXPECTED_TARGETS", "30"))
MIN_ACTIVE_CELL_TYPES = int(os.getenv("SCBFM_MIN_ACTIVE_CELL_TYPES", "2"))
MAX_ACTIVE_CELL_TYPES = int(os.getenv("SCBFM_MAX_ACTIVE_CELL_TYPES", "8"))
DIRICHLET_ALPHA = float(os.getenv("SCBFM_DIRICHLET_ALPHA", "1.0"))
MIN_REALIZED_CELLS_PER_ACTIVE_TYPE = int(
    os.getenv("SCBFM_MIN_REALIZED_CELLS_PER_ACTIVE_TYPE", "5")
)
MAX_MIXTURE_DRAW_ATTEMPTS = int(os.getenv("SCBFM_MAX_MIXTURE_DRAW_ATTEMPTS", "1000"))
MIN_SOURCE_POOL_PER_CELLTYPE = int(
    os.getenv("SCBFM_MIN_SOURCE_POOL_PER_CELLTYPE", "32")
)
MAX_SOURCE_POOL_PER_CELLTYPE = int(
    os.getenv("SCBFM_MAX_SOURCE_POOL_PER_CELLTYPE", "256")
)
TISSUE_COLUMN = os.getenv("SCBFM_TISSUE_COLUMN", "tissue_general")
RESUME = os.getenv("SCBFM_PSEUDO_RESUME", "1") != "0"
OFFLINE = os.getenv("SCBFM_PSEUDO_OFFLINE", "0") == "1"
AUDIT_ONLY = os.getenv("SCBFM_PSEUDO_AUDIT_ONLY", "0") == "1"
DOWNLOAD_ONLY = os.getenv("SCBFM_PSEUDO_DOWNLOAD_ONLY", "0") == "1"
VALIDATE_TRANSFER_ONLY = (
    os.getenv("SCBFM_PSEUDO_VALIDATE_TRANSFER_ONLY", "0") == "1"
)
VERIFY_SOURCE_CHUNK_HASHES = (
    os.getenv("SCBFM_PSEUDO_VERIFY_SOURCE_CHUNK_HASHES", "1") != "0"
)

TILEDB_CONFIG = {
    "py.init_buffer_bytes": 256 * 1024**2,
    "soma.init_buffer_bytes": 256 * 1024**2,
}

OBS_CONTEXT_COLUMNS = [
    "soma_joinid",
    "dataset_id",
    "donor_id",
    TISSUE_COLUMN,
    "cell_type",
    "cell_type_ontology_term_id",
]


# =========================
# Helpers
# =========================

def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_json(data, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_cell_ontology_file() -> Path:
    if CELL_ONTOLOGY_PATH.exists():
        return CELL_ONTOLOGY_PATH
    if OFFLINE:
        raise FileNotFoundError(
            f"Offline generation requires the pinned Cell Ontology file at "
            f"{CELL_ONTOLOGY_PATH}."
        )
    CELL_ONTOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = CELL_ONTOLOGY_PATH.with_suffix(".downloading.obo")
    print(f"Downloading Cell Ontology {CELL_ONTOLOGY_RELEASE} from {CELL_ONTOLOGY_URL}")
    try:
        urllib.request.urlretrieve(CELL_ONTOLOGY_URL, temporary_path)
        os.replace(temporary_path, CELL_ONTOLOGY_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    return CELL_ONTOLOGY_PATH


def parse_cell_ontology(
    path: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]], dict[str, str], str]:
    names: dict[str, str] = {}
    parents: dict[str, tuple[str, ...]] = {}
    alt_to_primary: dict[str, str] = {}
    data_version = "unknown"
    current: dict[str, list[str]] | None = None

    def commit(term: dict[str, list[str]] | None) -> None:
        if not term or "id" not in term or "name" not in term:
            return
        if term.get("is_obsolete", ["false"])[0].lower() == "true":
            return
        term_id = term["id"][0]
        names[term_id] = term["name"][0]
        parents[term_id] = tuple(
            value.split(None, 1)[0].strip()
            for value in term.get("is_a", [])
        )
        for alt_id in term.get("alt_id", []):
            alt_to_primary[alt_id] = term_id

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if current is None and line.startswith("data-version: "):
                data_version = line.split(": ", 1)[1]
            if line == "[Term]":
                commit(current)
                current = {}
                continue
            if line.startswith("["):
                commit(current)
                current = None
                continue
            if current is None or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            current.setdefault(key, []).append(value)
    commit(current)
    if not names:
        raise ValueError(f"No Cell Ontology terms could be parsed from {path}.")
    return names, parents, alt_to_primary, data_version


def load_broad_cell_type_specs(
    path: Path,
    ontology_names: dict[str, str],
) -> list[dict[str, object]]:
    specs_df = pd.read_csv(path)
    required = {"priority", "target_cell_type", "ontology_term_id"}
    missing = required.difference(specs_df.columns)
    if missing:
        raise ValueError(f"Broad cell-type config {path} is missing columns: {sorted(missing)}")
    specs_df = specs_df.sort_values("priority", kind="stable")
    if specs_df["priority"].duplicated().any():
        raise ValueError("Broad cell-type priorities must be unique.")
    if specs_df["target_cell_type"].duplicated().any():
        raise ValueError("Broad target cell-type names must be unique.")

    specs: list[dict[str, object]] = []
    for row in specs_df.itertuples(index=False):
        term_id = str(row.ontology_term_id)
        if term_id not in ontology_names:
            raise ValueError(
                f"Broad target {row.target_cell_type!r} references missing ontology term {term_id}."
            )
        specs.append(
            {
                "priority": int(row.priority),
                "target_cell_type": str(row.target_cell_type),
                "ontology_term_id": term_id,
                "ontology_name": ontology_names[term_id],
            }
        )
    return specs


def split_ontology_term_ids(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [
        term_id.strip()
        for term_id in re.split(r"[,|]", str(value))
        if term_id.strip().startswith("CL:")
    ]


def map_ontology_term_to_broad_target(
    raw_term_id: object,
    *,
    parents: dict[str, tuple[str, ...]],
    alt_to_primary: dict[str, str],
    broad_specs: list[dict[str, object]],
) -> tuple[str | None, str | None, int | None]:
    roots = {
        str(spec["ontology_term_id"]): (
            str(spec["target_cell_type"]),
            int(spec["priority"]),
        )
        for spec in broad_specs
    }
    candidates: list[tuple[int, int, str, str]] = []
    for source_id in split_ontology_term_ids(raw_term_id):
        source_id = alt_to_primary.get(source_id, source_id)
        queue = deque([(source_id, 0)])
        visited: set[str] = set()
        while queue:
            term_id, distance = queue.popleft()
            if term_id in visited:
                continue
            visited.add(term_id)
            if term_id in roots:
                target_name, priority = roots[term_id]
                candidates.append((distance, priority, target_name, term_id))
            queue.extend((parent, distance + 1) for parent in parents.get(term_id, ()))
    if not candidates:
        return None, None, None
    distance, _priority, target_name, root_id = min(candidates)
    return target_name, root_id, distance


def map_obs_to_broad_cell_types(
    df: pd.DataFrame,
    *,
    parents: dict[str, tuple[str, ...]],
    alt_to_primary: dict[str, str],
    broad_specs: list[dict[str, object]],
    mapping_cache: dict[str, tuple[str | None, str | None, int | None]],
    mapping_counts: dict[tuple[str, str, str, str, int], int] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    broad_targets: list[str | None] = []
    for row in df.itertuples(index=False):
        raw_term_id = str(row.cell_type_ontology_term_id)
        if raw_term_id not in mapping_cache:
            mapping_cache[raw_term_id] = map_ontology_term_to_broad_target(
                raw_term_id,
                parents=parents,
                alt_to_primary=alt_to_primary,
                broad_specs=broad_specs,
            )
        target, root_id, distance = mapping_cache[raw_term_id]
        broad_targets.append(target)
        if mapping_counts is not None:
            mapping_counts[
                (
                    str(row.cell_type),
                    raw_term_id,
                    target or "unmapped",
                    root_id or "",
                    int(distance) if distance is not None else -1,
                )
            ] += 1
    df["source_cell_type"] = df["cell_type"].astype(str)
    df["cell_type"] = pd.Series(broad_targets, index=df.index, dtype="string")
    return df


def write_cell_type_mapping_audit(
    mapping_counts: dict[tuple[str, str, str, str, int], int],
) -> None:
    records = [
        {
            "source_cell_type": key[0],
            "source_ontology_term_id": key[1],
            "target_cell_type": key[2],
            "target_ontology_term_id": key[3],
            "ontology_distance": key[4],
            "source_cells": count,
        }
        for key, count in mapping_counts.items()
    ]
    mapping_df = pd.DataFrame(records)
    if not mapping_df.empty:
        mapping_df = mapping_df.sort_values(
            ["target_cell_type", "source_cells", "source_cell_type"],
            ascending=[True, False, True],
        )
    mapping_df.to_csv(CELL_TYPE_MAPPING_OUT, index=False)


def metadata_manifest() -> dict[str, object]:
    return {
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "census_version": CENSUS_VERSION,
        "organism": ORGANISM,
        "tissue_column": TISSUE_COLUMN,
        "cell_ontology_release": CELL_ONTOLOGY_RELEASE,
        "cell_ontology_sha256": sha256_file(CELL_ONTOLOGY_PATH),
        "broad_cell_type_config_sha256": sha256_file(BROAD_CELL_TYPE_CONFIG_PATH),
        "min_available_cells_per_celltype": MIN_AVAILABLE_CELLS_PER_CELLTYPE,
        "min_context_cell_types": MIN_CONTEXT_CELL_TYPES,
        "min_context_total_cells": MIN_CONTEXT_TOTAL_CELLS,
        "min_target_contexts": MIN_TARGET_CONTEXTS,
        "min_target_source_cells": MIN_TARGET_SOURCE_CELLS,
    }


def sampling_manifest() -> dict[str, object]:
    return {
        **metadata_manifest(),
        "target_pseudo_bulks": TARGET_PSEUDO_BULKS,
        "cells_per_pseudo_bulk": CELLS_PER_PSEUDO_BULK,
        "minimum_active_cell_types": MIN_ACTIVE_CELL_TYPES,
        "maximum_active_cell_types": MAX_ACTIVE_CELL_TYPES,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "minimum_realized_cells_per_active_type": MIN_REALIZED_CELLS_PER_ACTIVE_TYPE,
        "random_seed": RANDOM_SEED,
    }


def manifest_matches(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        return read_json(path) == expected
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def close_backed_adata(adata: ad.AnnData) -> None:
    file_obj = getattr(adata, "file", None)
    if file_obj is not None:
        file_obj.close()


def inspect_cached_h5ad(path: Path) -> tuple[tuple[int, int], list[str]]:
    cached = ad.read_h5ad(path, backed="r")
    try:
        shape = tuple(map(int, cached.shape))
        var_names = list(cached.var_names.astype(str))
    finally:
        close_backed_adata(cached)
    return shape, var_names


def cached_h5ad_matches(
    path: Path,
    gene_list: list[str],
    *,
    expected_n_obs: int | None = None,
) -> tuple[bool, str, tuple[int, int] | None]:
    try:
        shape, cached_gene_list = inspect_cached_h5ad(path)
    except Exception as exc:
        return False, f"cannot be opened as h5ad ({exc})", None

    if cached_gene_list != gene_list:
        return (
            False,
            f"gene order/count mismatch (found {shape[1]} genes, expected {len(gene_list)})",
            shape,
        )
    if expected_n_obs is not None and shape[0] != expected_n_obs:
        return (
            False,
            f"row count mismatch (found {shape[0]}, expected {expected_n_obs})",
            shape,
        )
    return True, f"{shape[0]} samples x {shape[1]} genes", shape


def source_chunk_matches(
    path: Path,
    gene_list: list[str],
    expected_meta: pd.DataFrame,
) -> tuple[bool, str, tuple[int, int] | None]:
    ok, reason, shape = cached_h5ad_matches(
        path,
        gene_list,
        expected_n_obs=int(expected_meta.shape[0]),
    )
    if not ok:
        return ok, reason, shape
    try:
        cached = ad.read_h5ad(path, backed="r")
        try:
            if "soma_joinid" not in cached.obs:
                return False, "missing soma_joinid metadata", shape
            observed_ids = cached.obs["soma_joinid"].to_numpy(dtype=np.int64)
        finally:
            close_backed_adata(cached)
    except Exception as exc:
        return False, f"cannot read source-cell metadata ({exc})", shape
    expected_ids = expected_meta["soma_joinid"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_ids, expected_ids):
        return False, "soma_joinid sequence differs from the current sampling plan", shape
    return True, reason, shape


def validate_cached_h5ad(
    path: Path,
    gene_list: list[str],
    *,
    label: str,
    expected_n_obs: int | None = None,
    allow_recompute: bool = False,
) -> bool:
    ok, reason, _shape = cached_h5ad_matches(
        path,
        gene_list,
        expected_n_obs=expected_n_obs,
    )
    if ok:
        return True
    if allow_recompute:
        print(f"Ignoring cached {label} at {path}: {reason}")
        return False
    raise ValueError(
        f"Cached {label} at {path} is not compatible with the current "
        f"gene list/setup: {reason}"
    )


def validate_cached_final(gene_list: list[str]) -> bool:
    if not FINAL_OUT.exists():
        return False

    ok, reason, shape = cached_h5ad_matches(FINAL_OUT, gene_list)
    if not ok:
        print(f"Ignoring cached pseudo-bulk final at {FINAL_OUT}: {reason}")
        return False

    if not SUMMARY_OUT.exists():
        print(
            f"Ignoring cached pseudo-bulk final at {FINAL_OUT}: "
            f"missing run summary {SUMMARY_OUT}."
        )
        return False

    summary = read_json(SUMMARY_OUT)
    summary_target = int(summary.get("target_pseudo_bulks", -1))
    summary_cells = int(summary.get("cells_per_pseudo_bulk", -1))
    summary_genes = int(summary.get("target_gene_count", -1))
    summary_schema = int(summary.get("generator_schema_version", -1))
    summary_ontology = str(summary.get("cell_ontology_sha256", ""))
    summary_targets = str(summary.get("broad_cell_type_config_sha256", ""))
    if (
        summary_target != TARGET_PSEUDO_BULKS
        or summary_cells != CELLS_PER_PSEUDO_BULK
        or summary_genes != len(gene_list)
        or summary_schema != GENERATOR_SCHEMA_VERSION
        or summary_ontology != sha256_file(CELL_ONTOLOGY_PATH)
        or summary_targets != sha256_file(BROAD_CELL_TYPE_CONFIG_PATH)
    ):
        print(
            f"Ignoring cached pseudo-bulk final at {FINAL_OUT}: summary "
            "does not match the current target/cell/gene settings."
        )
        return False

    assert shape is not None
    print(
        f"Reusing existing pseudo-bulk file at {FINAL_OUT}: "
        f"{shape[0]} samples x {shape[1]} genes"
    )
    return True


def require_cellxgene_census():
    global cellxgene_census
    if cellxgene_census is None:
        try:
            import cellxgene_census as census_module
        except ImportError as exc:
            raise ImportError(
                "cellxgene_census is required unless SCBFM_PSEUDO_OFFLINE=1 and all "
                "metadata/source chunks are already cached."
            ) from exc
        cellxgene_census = census_module
    return cellxgene_census


def normalize_obs_chunk(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col == "soma_joinid":
            df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype("string").fillna("unknown").replace("<NA>", "unknown")
            df[col] = df[col].astype(str)
    return df


def filter_raw_dense_block(
    x: np.ndarray,
    min_genes: int = MIN_GENES,
) -> tuple[np.ndarray, np.ndarray]:
    n_genes_by_sample = (x > 0).sum(axis=1)
    keep_mask = n_genes_by_sample >= min_genes
    x = x[keep_mask]

    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32), keep_mask

    return x.astype(np.float32, copy=False), keep_mask


def merge_h5ad_group(paths: list[Path], out_path: Path, gene_list: list[str]) -> Path:
    for path in paths:
        validate_cached_h5ad(path, gene_list, label="merge input chunk")
    adatas = [ad.read_h5ad(p) for p in paths]
    merged = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    merged = merged[:, gene_list].copy()
    merged.var_names = pd.Index(gene_list, dtype=str)
    write_h5ad_compat(merged, out_path)

    del adatas, merged
    gc.collect()
    return out_path


def iter_obs_tables(census, column_names: list[str]):
    exp = census["census_data"][CENSUS_ORGANISM_KEY]
    return exp.obs.read(
        column_names=column_names,
        value_filter="is_primary_data == True",
    )


def build_var_coords(census, gene_list: list[str]) -> tuple[list[int], list[str]]:
    census_api = require_cellxgene_census()
    var_df = census_api.get_var(
        census=census,
        organism=ORGANISM,
        column_names=["soma_joinid", "feature_id"],
    )
    var_df["feature_id"] = var_df["feature_id"].astype(str)
    var_df = var_df.drop_duplicates("feature_id", keep="first")
    joinid_by_feature = dict(zip(var_df["feature_id"], var_df["soma_joinid"].astype(int)))

    missing = [g for g in gene_list if g not in joinid_by_feature]
    var_coords = [int(joinid_by_feature[g]) for g in gene_list if g in joinid_by_feature]
    return var_coords, missing


def make_context_key(row) -> tuple[str, str, str]:
    return (row.dataset_id, row.donor_id, getattr(row, TISSUE_COLUMN))


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip().lower()).strip("_")
    return sanitized or "cell_type"


def build_proportion_column_map(cell_types: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for cell_type in sorted(cell_types):
        base = f"prop__{sanitize_name(cell_type)}"
        column = base
        suffix = 2
        while column in used:
            column = f"{base}_{suffix}"
            suffix += 1
        mapping[cell_type] = column
        used.add(column)

    return mapping


def write_h5ad_compat(adata: ad.AnnData, path: Path) -> None:
    """Write h5ad with HDF5 libver='earliest' so cluster HDF5 < 1.10 can read it."""
    tmp = path.with_suffix(".writing.h5ad")
    ready_tmp = path.with_suffix(".ready.h5ad")
    try:
        # AnnData 0.10 cannot serialize Arrow-backed pandas string arrays.
        for frame_name in ("obs", "var"):
            frame = getattr(adata, frame_name).copy()
            if isinstance(frame.index.dtype, pd.StringDtype):
                frame.index = pd.Index(
                    frame.index.to_numpy(dtype=object),
                    dtype=object,
                    name=frame.index.name,
                )
            for column in frame.columns:
                if isinstance(frame[column].dtype, pd.StringDtype):
                    frame[column] = frame[column].astype(object)
            setattr(adata, frame_name, frame)
        adata.write(tmp)
        with h5py.File(tmp, "r") as f_in:
            with h5py.File(ready_tmp, "w", libver="earliest") as f_out:
                for key in f_in.keys():
                    f_in.copy(key, f_out)
                for k, v in f_in.attrs.items():
                    f_out.attrs[k] = v
        os.replace(ready_tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
        ready_tmp.unlink(missing_ok=True)


def store_proportion_metadata(adata: ad.AnnData, proportion_column_map: dict[str, str]) -> None:
    cell_types = sorted(proportion_column_map)
    adata.uns["cell_type_proportion_cell_types"] = np.asarray(cell_types, dtype=object)
    adata.uns["cell_type_proportion_obs_columns"] = np.asarray(
        [proportion_column_map[cell_type] for cell_type in cell_types],
        dtype=object,
    )
    adata.uns["cell_type_proportion_columns"] = proportion_column_map
    adata.uns["deconv_generator_schema_version"] = GENERATOR_SCHEMA_VERSION
    adata.uns["cell_ontology_release"] = CELL_ONTOLOGY_RELEASE
    adata.uns["cell_ontology_sha256"] = sha256_file(CELL_ONTOLOGY_PATH)
    adata.uns["broad_cell_type_config_sha256"] = sha256_file(
        BROAD_CELL_TYPE_CONFIG_PATH
    )
    adata.uns["mixture_distribution"] = "dirichlet_multinomial"
    adata.uns["dirichlet_alpha"] = DIRICHLET_ALPHA
    adata.uns["minimum_active_cell_types"] = MIN_ACTIVE_CELL_TYPES
    adata.uns["maximum_active_cell_types"] = MAX_ACTIVE_CELL_TYPES
    adata.uns["minimum_realized_cells_per_active_type"] = (
        MIN_REALIZED_CELLS_PER_ACTIVE_TYPE
    )


def parse_json_dict_column(value: object, cast):
    if isinstance(value, dict):
        return {str(k): cast(v) for k, v in value.items()}
    if pd.isna(value):
        return {}
    parsed = json.loads(value)
    return {str(k): cast(v) for k, v in parsed.items()}


def load_eligible_contexts_from_csv(
    path: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, int]]]:
    eligible_df = pd.read_csv(path)
    if eligible_df.empty:
        return eligible_df, {}

    eligible_df = normalize_obs_chunk(
        eligible_df,
        [col for col in ["dataset_id", "donor_id", TISSUE_COLUMN] if col in eligible_df.columns],
    )

    eligible_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in eligible_df.itertuples(index=False):
        context_key = (row.dataset_id, row.donor_id, getattr(row, TISSUE_COLUMN))
        eligible_counts[context_key] = parse_json_dict_column(row.cell_type_counts, int)
    return eligible_df, eligible_counts


def load_sample_plan_from_csv(
    plan_path: Path,
    quota_path: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[tuple[tuple[str, str, str], str], int], list[str]]:
    plan_df = pd.read_csv(plan_path)
    plan_df = normalize_obs_chunk(
        plan_df,
        [col for col in ["sample_id", "dataset_id", "donor_id", TISSUE_COLUMN] if col in plan_df.columns],
    )

    plan_rows: list[dict[str, object]] = []
    all_cell_types: set[str] = set()
    for row in plan_df.itertuples(index=False):
        cell_type_counts = parse_json_dict_column(row.cell_type_counts, int)
        intended_cell_type_proportions = parse_json_dict_column(
            getattr(row, "intended_cell_type_proportions", row.cell_type_proportions),
            float,
        )
        cell_type_proportions = parse_json_dict_column(row.cell_type_proportions, float)
        plan_rows.append(
            {
                "sample_id": row.sample_id,
                "dataset_id": row.dataset_id,
                "donor_id": row.donor_id,
                TISSUE_COLUMN: getattr(row, TISSUE_COLUMN),
                "total_cells": int(row.total_cells),
                "n_cell_types": int(row.n_cell_types),
                "cell_type_counts": cell_type_counts,
                "intended_cell_type_proportions": intended_cell_type_proportions,
                "cell_type_proportions": cell_type_proportions,
            }
        )
        all_cell_types.update(cell_type_counts)

    quota_df = pd.read_csv(quota_path)
    quota_df = normalize_obs_chunk(
        quota_df,
        [col for col in ["dataset_id", "donor_id", TISSUE_COLUMN, "cell_type"] if col in quota_df.columns],
    )
    reservoir_quotas: dict[tuple[tuple[str, str, str], str], int] = {}
    for row in quota_df.itertuples(index=False):
        context_key = (row.dataset_id, row.donor_id, getattr(row, TISSUE_COLUMN))
        reservoir_quotas[(context_key, row.cell_type)] = int(row.reservoir_quota)
        all_cell_types.add(row.cell_type)

    return plan_rows, plan_df, reservoir_quotas, sorted(all_cell_types)


def load_sampled_source_meta(path: Path) -> pd.DataFrame:
    sampled_meta = pd.read_csv(path)
    return normalize_obs_chunk(sampled_meta, OBS_CONTEXT_COLUMNS)


RETRYABLE_CENSUS_ERROR_MARKERS = (
    "couldn't resolve host name",
    "temporary failure in name resolution",
    "failed to read s3 object",
    "vfs parallel read error",
    "connection reset",
    "connection aborted",
    "connection refused",
    "operation timed out",
    "request timeout",
    "requesttimeout",
    "slowdown",
    "curlcode:",
    "http response code: 429",
    "http response code: 500",
    "http response code: 502",
    "http response code: 503",
    "http response code: 504",
)


def is_retryable_census_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    messages: list[str] = []
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    message = " ".join(messages)
    return isinstance(exc, (ConnectionError, TimeoutError)) or any(
        marker in message for marker in RETRYABLE_CENSUS_ERROR_MARKERS
    )


def get_anndata_with_retry(census_api, *, chunk_id: int, **kwargs) -> ad.AnnData:
    if DOWNLOAD_MAX_ATTEMPTS < 1:
        raise ValueError("SCBFM_PSEUDO_DOWNLOAD_MAX_ATTEMPTS must be at least 1.")

    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            return census_api.get_anndata(**kwargs)
        except Exception as exc:
            retryable = is_retryable_census_error(exc)
            if not retryable or attempt == DOWNLOAD_MAX_ATTEMPTS:
                raise
            delay = min(
                DOWNLOAD_RETRY_MAX_SECONDS,
                DOWNLOAD_RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
            )
            print(
                f"Source chunk {chunk_id}: transient Census read failure on "
                f"attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS}; retrying in "
                f"{delay:g}s ({type(exc).__name__}: {exc})",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable Census retry state.")


def count_context_cell_types(
    census,
    *,
    parents: dict[str, tuple[str, ...]],
    alt_to_primary: dict[str, str],
    broad_specs: list[dict[str, object]],
    mapping_cache: dict[str, tuple[str | None, str | None, int | None]],
    mapping_counts: dict[tuple[str, str, str, str, int], int],
) -> dict[tuple[str, str, str], dict[str, int]]:
    counts_by_context: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for table in iter_obs_tables(census, OBS_CONTEXT_COLUMNS):
        df = normalize_obs_chunk(table.to_pandas(), OBS_CONTEXT_COLUMNS)
        df = map_obs_to_broad_cell_types(
            df,
            parents=parents,
            alt_to_primary=alt_to_primary,
            broad_specs=broad_specs,
            mapping_cache=mapping_cache,
            mapping_counts=mapping_counts,
        )
        df = df[
            (df["donor_id"] != "unknown")
            & (df[TISSUE_COLUMN] != "unknown")
            & df["cell_type"].notna()
        ]
        for row in df.itertuples(index=False):
            context_key = make_context_key(row)
            counts_by_context[context_key][str(row.cell_type)] += 1

        del df
        gc.collect()

    return counts_by_context


def select_supported_target_cell_types(
    counts_by_context: dict[tuple[str, str, str], dict[str, int]],
    broad_specs: list[dict[str, object]],
) -> set[str]:
    records: list[dict[str, object]] = []
    supported: set[str] = set()
    for spec in broad_specs:
        target = str(spec["target_cell_type"])
        context_counts = [
            int(counts.get(target, 0))
            for counts in counts_by_context.values()
        ]
        total_cells = int(sum(context_counts))
        qualifying_contexts = int(
            sum(count >= MIN_AVAILABLE_CELLS_PER_CELLTYPE for count in context_counts)
        )
        keep = (
            total_cells >= MIN_TARGET_SOURCE_CELLS
            and qualifying_contexts >= MIN_TARGET_CONTEXTS
        )
        if keep:
            supported.add(target)
        records.append(
            {
                **spec,
                "source_cells": total_cells,
                "qualifying_contexts": qualifying_contexts,
                "selected": bool(keep),
                "minimum_source_cells": MIN_TARGET_SOURCE_CELLS,
                "minimum_qualifying_contexts": MIN_TARGET_CONTEXTS,
            }
        )

    audit_df = pd.DataFrame(records).sort_values("priority")
    audit_df.to_csv(TARGET_AUDIT_OUT, index=False)
    if not MIN_EXPECTED_TARGETS <= len(supported) <= MAX_EXPECTED_TARGETS:
        raise ValueError(
            f"Ontology/frequency filtering retained {len(supported)} broad cell types; "
            f"expected {MIN_EXPECTED_TARGETS}..{MAX_EXPECTED_TARGETS}. Review {TARGET_AUDIT_OUT}."
        )
    return supported


def build_eligible_contexts(
    counts_by_context: dict[tuple[str, str, str], dict[str, int]],
    supported_targets: set[str],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, int]]]:
    records: list[dict[str, object]] = []
    filtered_counts: dict[tuple[str, str, str], dict[str, int]] = {}

    for context_key, cell_type_counts in counts_by_context.items():
        filtered_cell_types = {
            cell_type: count
            for cell_type, count in cell_type_counts.items()
            if (
                cell_type in supported_targets
                and count >= MIN_AVAILABLE_CELLS_PER_CELLTYPE
            )
        }
        total_cells = int(sum(filtered_cell_types.values()))
        if len(filtered_cell_types) < MIN_CONTEXT_CELL_TYPES:
            continue
        if total_cells < MIN_CONTEXT_TOTAL_CELLS:
            continue

        dataset_id, donor_id, tissue = context_key
        filtered_counts[context_key] = filtered_cell_types
        records.append(
            {
                "dataset_id": dataset_id,
                "donor_id": donor_id,
                TISSUE_COLUMN: tissue,
                "n_cell_types": int(len(filtered_cell_types)),
                "n_cells": total_cells,
                "cell_types": json.dumps(sorted(filtered_cell_types)),
                "cell_type_counts": json.dumps(filtered_cell_types, sort_keys=True),
            }
        )

    eligible_df = pd.DataFrame(
        records,
        columns=[
            "dataset_id",
            "donor_id",
            TISSUE_COLUMN,
            "n_cell_types",
            "n_cells",
            "cell_types",
            "cell_type_counts",
        ],
    )
    if not eligible_df.empty:
        eligible_df = eligible_df.sort_values(
            ["n_cells", "n_cell_types", "dataset_id", "donor_id", TISSUE_COLUMN],
            ascending=[False, False, True, True, True],
        )
    return eligible_df, filtered_counts


def simulate_sample_plan(
    eligible_counts: dict[tuple[str, str, str], dict[str, int]],
    target_samples: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[tuple[tuple[str, str, str], str], int], list[str]]:
    if not eligible_counts:
        raise ValueError("No eligible contexts found for pseudo-bulk generation.")

    rng = np.random.default_rng(RANDOM_SEED)
    context_keys = list(eligible_counts)
    context_weights = np.array(
        [math.sqrt(sum(eligible_counts[key].values())) for key in context_keys],
        dtype=np.float64,
    )
    context_weights /= context_weights.sum()

    plan_rows: list[dict[str, object]] = []
    reservoir_demands: dict[tuple[tuple[str, str, str], str], int] = {}
    all_cell_types: set[str] = {
        cell_type
        for counts in eligible_counts.values()
        for cell_type in counts
    }
    if not MIN_EXPECTED_TARGETS <= len(all_cell_types) <= MAX_EXPECTED_TARGETS:
        raise ValueError(
            f"Eligible mixture contexts expose {len(all_cell_types)} target cell types; "
            f"expected {MIN_EXPECTED_TARGETS}..{MAX_EXPECTED_TARGETS}."
        )

    for sample_idx in range(target_samples):
        context_key = context_keys[int(rng.choice(len(context_keys), p=context_weights))]
        cell_type_counts = eligible_counts[context_key]
        available_cell_types = sorted(cell_type_counts)
        maximum_active = min(MAX_ACTIVE_CELL_TYPES, len(available_cell_types))
        minimum_active = min(MIN_ACTIVE_CELL_TYPES, maximum_active)
        n_active = int(rng.integers(minimum_active, maximum_active + 1))
        active_cell_types = sorted(
            rng.choice(available_cell_types, size=n_active, replace=False).tolist()
        )

        realized_counts = None
        intended_weights = None
        for _ in range(MAX_MIXTURE_DRAW_ATTEMPTS):
            intended_weights = rng.dirichlet(
                np.full(n_active, DIRICHLET_ALPHA, dtype=np.float64)
            )
            candidate_counts = rng.multinomial(
                CELLS_PER_PSEUDO_BULK,
                intended_weights,
            )
            if np.all(candidate_counts >= MIN_REALIZED_CELLS_PER_ACTIVE_TYPE):
                realized_counts = candidate_counts
                break
        if realized_counts is None or intended_weights is None:
            raise RuntimeError(
                "Could not draw a valid Dirichlet-multinomial mixture after "
                f"{MAX_MIXTURE_DRAW_ATTEMPTS} attempts."
            )

        realized_cell_type_counts = {
            cell_type: int(count)
            for cell_type, count in zip(active_cell_types, realized_counts.tolist())
            if count > 0
        }
        intended_cell_type_props = {
            cell_type: float(weight)
            for cell_type, weight in zip(active_cell_types, intended_weights.tolist())
        }
        realized_cell_type_props = {
            cell_type: count / CELLS_PER_PSEUDO_BULK
            for cell_type, count in realized_cell_type_counts.items()
        }

        for cell_type, count in realized_cell_type_counts.items():
            key = (context_key, cell_type)
            current = reservoir_demands.get(key, 0)
            reservoir_demands[key] = max(current, count)

        dataset_id, donor_id, tissue = context_key
        plan_rows.append(
            {
                "sample_id": f"pseudo_bulk:{sample_idx:06d}",
                "dataset_id": dataset_id,
                "donor_id": donor_id,
                TISSUE_COLUMN: tissue,
                "total_cells": CELLS_PER_PSEUDO_BULK,
                "n_cell_types": int(len(realized_cell_type_counts)),
                "cell_type_counts": realized_cell_type_counts,
                "intended_cell_type_proportions": intended_cell_type_props,
                "cell_type_proportions": realized_cell_type_props,
            }
        )

    quota_records: list[dict[str, object]] = []
    for (context_key, cell_type), max_count in reservoir_demands.items():
        available = eligible_counts[context_key][cell_type]
        quota = min(
            available,
            max(
                min(available, MIN_SOURCE_POOL_PER_CELLTYPE),
                min(MAX_SOURCE_POOL_PER_CELLTYPE, int(math.ceil(math.sqrt(max_count * target_samples / max(1, len(context_keys)))))),
                max_count,
            ),
        )
        reservoir_demands[(context_key, cell_type)] = int(quota)
        dataset_id, donor_id, tissue = context_key
        quota_records.append(
            {
                "dataset_id": dataset_id,
                "donor_id": donor_id,
                TISSUE_COLUMN: tissue,
                "cell_type": cell_type,
                "available_cells": int(available),
                "reservoir_quota": int(quota),
            }
        )

    plan_df = pd.DataFrame(
        [
            {
                "sample_id": row["sample_id"],
                "dataset_id": row["dataset_id"],
                "donor_id": row["donor_id"],
                TISSUE_COLUMN: row[TISSUE_COLUMN],
                "total_cells": row["total_cells"],
                "n_cell_types": row["n_cell_types"],
                "cell_type_counts": json.dumps(row["cell_type_counts"], sort_keys=True),
                "intended_cell_type_proportions": json.dumps(
                    row["intended_cell_type_proportions"],
                    sort_keys=True,
                ),
                "cell_type_proportions": json.dumps(row["cell_type_proportions"], sort_keys=True),
            }
            for row in plan_rows
        ]
    )
    if quota_records:
        quota_df = pd.DataFrame(quota_records).sort_values(
            ["dataset_id", "donor_id", TISSUE_COLUMN, "cell_type"]
        )
        quota_df.to_csv(SOURCE_POOL_QUOTAS_OUT, index=False)

    return plan_rows, plan_df, reservoir_demands, sorted(all_cell_types)


def reservoir_sample_source_cells(
    census,
    reservoir_quotas: dict[tuple[tuple[str, str, str], str], int],
    *,
    parents: dict[str, tuple[str, ...]],
    alt_to_primary: dict[str, str],
    broad_specs: list[dict[str, object]],
    mapping_cache: dict[str, tuple[str | None, str | None, int | None]],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    seen_counts: dict[tuple[tuple[str, str, str], str], int] = defaultdict(int)
    reservoirs: dict[tuple[tuple[str, str, str], str], list[dict[str, object]]] = {
        key: [] for key, quota in reservoir_quotas.items() if quota > 0
    }

    for table in iter_obs_tables(census, OBS_CONTEXT_COLUMNS):
        df = normalize_obs_chunk(table.to_pandas(), OBS_CONTEXT_COLUMNS)
        df = map_obs_to_broad_cell_types(
            df,
            parents=parents,
            alt_to_primary=alt_to_primary,
            broad_specs=broad_specs,
            mapping_cache=mapping_cache,
        )
        df = df[
            (df["donor_id"] != "unknown")
            & (df[TISSUE_COLUMN] != "unknown")
            & df["cell_type"].notna()
        ]
        for row in df.itertuples(index=False):
            context_key = make_context_key(row)
            quota_key = (context_key, row.cell_type)
            quota = reservoir_quotas.get(quota_key, 0)
            if quota <= 0:
                continue

            seen_counts[quota_key] += 1
            sample_row = {
                "soma_joinid": int(row.soma_joinid),
                "dataset_id": row.dataset_id,
                "donor_id": row.donor_id,
                TISSUE_COLUMN: getattr(row, TISSUE_COLUMN),
                "cell_type": str(row.cell_type),
                "source_cell_type": str(row.source_cell_type),
                "cell_type_ontology_term_id": str(row.cell_type_ontology_term_id),
            }
            reservoir = reservoirs[quota_key]
            if len(reservoir) < quota:
                reservoir.append(sample_row)
            else:
                j = int(rng.integers(0, seen_counts[quota_key]))
                if j < quota:
                    reservoir[j] = sample_row

        del df
        gc.collect()

    sampled_rows = [row for reservoir in reservoirs.values() for row in reservoir]
    sampled_meta = pd.DataFrame(sampled_rows).drop_duplicates("soma_joinid")
    sampled_meta = sampled_meta.sort_values("soma_joinid").reset_index(drop=True)
    return sampled_meta


def download_source_cells(
    census,
    sampled_meta: pd.DataFrame,
    var_coords: list[int],
    gene_list: list[str],
    *,
    combine_chunks: bool = True,
) -> ad.AnnData | None:
    census_api = require_cellxgene_census()
    chunk_paths: list[Path] = []

    for chunk_id, start in enumerate(range(0, sampled_meta.shape[0], DOWNLOAD_CHUNK_SIZE)):
        end = min(start + DOWNLOAD_CHUNK_SIZE, sampled_meta.shape[0])
        chunk_path = SOURCE_CHUNK_DIR / f"source_cells_chunk_{chunk_id:05d}.h5ad"
        meta_chunk = sampled_meta.iloc[start:end].copy()
        if RESUME and chunk_path.exists():
            valid, reason, _shape = source_chunk_matches(
                chunk_path,
                gene_list,
                meta_chunk,
            )
            if valid:
                print(
                    f"Reusing downloaded source CELLxGENE chunk {chunk_id}: "
                    f"cells {start}:{end}"
                )
                chunk_paths.append(chunk_path)
                continue
            print(
                f"Redownloading source CELLxGENE chunk {chunk_id}: cached file "
                f"is incompatible ({reason})"
            )

        obs_coords = meta_chunk["soma_joinid"].astype(np.int64).tolist()

        print(f"Downloading source CELLxGENE chunk {chunk_id}: cells {start}:{end}")
        t0 = time.perf_counter()
        adata = get_anndata_with_retry(
            census_api,
            chunk_id=chunk_id,
            census=census,
            organism=ORGANISM,
            obs_coords=obs_coords,
            var_coords=var_coords,
            obs_column_names=OBS_CONTEXT_COLUMNS,
            var_column_names=["feature_id"],
        )
        t1 = time.perf_counter()
        print(
            f"Source chunk {chunk_id}: download finished in {t1 - t0:.1f}s "
            f"with shape {adata.n_obs} x {adata.n_vars}"
        )

        feature_ids = adata.var["feature_id"].astype(str).tolist()
        reorder_idx = pd.Index(feature_ids).get_indexer(gene_list)
        if (reorder_idx < 0).any():
            raise ValueError("Downloaded source chunk is missing requested genes.")

        adata = adata[:, reorder_idx].copy()
        adata.var_names = pd.Index(gene_list, dtype=str)
        adata.obs = normalize_obs_chunk(adata.obs.reset_index(drop=True), OBS_CONTEXT_COLUMNS)
        broad_by_joinid = meta_chunk.set_index("soma_joinid")["cell_type"].astype(str)
        source_by_joinid = meta_chunk.set_index("soma_joinid")["source_cell_type"].astype(str)
        adata.obs["source_cell_type"] = adata.obs["soma_joinid"].map(source_by_joinid)
        adata.obs["cell_type"] = adata.obs["soma_joinid"].map(broad_by_joinid)
        if adata.obs["cell_type"].isna().any():
            raise ValueError("Downloaded source cells could not be mapped back to broad cell types.")
        adata.obs.index = pd.Index(
            [f"cellxgene:{sid}" for sid in adata.obs["soma_joinid"].astype(str)],
            name="cell_id",
        )
        adata.var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        if sparse.issparse(adata.X):
            adata.X = adata.X.tocsr()
        else:
            adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))

        write_h5ad_compat(adata, chunk_path)
        chunk_paths.append(chunk_path)

        del adata
        gc.collect()

    if not chunk_paths:
        raise ValueError("No source cell chunks are available.")

    write_source_chunk_manifest(chunk_paths, sampled_meta, gene_list)
    if not combine_chunks:
        return None

    source_chunks = [ad.read_h5ad(path) for path in chunk_paths]
    source_adata = ad.concat(
        source_chunks,
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )
    source_adata.obs_names_make_unique()
    source_adata.var_names = pd.Index(gene_list, dtype=str)

    del source_chunks
    gc.collect()
    return source_adata


def build_source_row_index(
    source_adata: ad.AnnData,
) -> dict[tuple[tuple[str, str, str], str], np.ndarray]:
    row_index_by_context_cell_type: dict[tuple[tuple[str, str, str], str], list[int]] = defaultdict(list)

    obs = source_adata.obs.reset_index(drop=True)
    for row_idx, row in enumerate(obs.itertuples(index=False)):
        context_key = make_context_key(row)
        row_index_by_context_cell_type[(context_key, row.cell_type)].append(row_idx)

    return {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in row_index_by_context_cell_type.items()
    }


def write_source_chunk_manifest(
    chunk_paths: list[Path],
    sampled_meta: pd.DataFrame,
    gene_list: list[str],
) -> None:
    records: list[dict[str, object]] = []
    for chunk_id, path in enumerate(chunk_paths):
        start = chunk_id * DOWNLOAD_CHUNK_SIZE
        end = min(start + DOWNLOAD_CHUNK_SIZE, sampled_meta.shape[0])
        expected_meta = sampled_meta.iloc[start:end]
        ok, reason, shape = source_chunk_matches(
            path,
            gene_list,
            expected_meta,
        )
        if not ok or shape is None:
            raise ValueError(f"Cannot include source chunk {path} in transfer manifest: {reason}")
        records.append(
            {
                "chunk_id": chunk_id,
                "relative_path": str(path.relative_to(OUT_DIR)),
                "n_obs": int(shape[0]),
                "n_vars": int(shape[1]),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(records).to_csv(SOURCE_CHUNK_MANIFEST_OUT, index=False)


def validate_source_chunk_transfer(
    sampled_meta: pd.DataFrame,
    gene_list: list[str],
) -> list[Path]:
    if not SOURCE_CHUNK_MANIFEST_OUT.exists():
        raise FileNotFoundError(
            f"Missing source chunk transfer manifest: {SOURCE_CHUNK_MANIFEST_OUT}"
        )
    manifest = pd.read_csv(SOURCE_CHUNK_MANIFEST_OUT).sort_values("chunk_id")
    expected_chunks = math.ceil(sampled_meta.shape[0] / DOWNLOAD_CHUNK_SIZE)
    if manifest.shape[0] != expected_chunks:
        raise ValueError(
            f"Source chunk manifest contains {manifest.shape[0]} chunks; "
            f"expected {expected_chunks}."
        )

    chunk_paths: list[Path] = []
    for expected_chunk_id, row in enumerate(manifest.itertuples(index=False)):
        if int(row.chunk_id) != expected_chunk_id:
            raise ValueError("Source chunk manifest IDs must be contiguous and zero-based.")
        expected_relative = Path("source_cell_chunks") / (
            f"source_cells_chunk_{expected_chunk_id:05d}.h5ad"
        )
        if Path(str(row.relative_path)) != expected_relative:
            raise ValueError(
                f"Unexpected path in source chunk manifest: {row.relative_path!r}; "
                f"expected {expected_relative}."
            )
        path = OUT_DIR / expected_relative
        if not path.exists():
            raise FileNotFoundError(f"Uploaded source chunk is missing: {path}")
        if path.stat().st_size != int(row.size_bytes):
            raise ValueError(f"Uploaded source chunk size differs from manifest: {path}")
        if VERIFY_SOURCE_CHUNK_HASHES:
            observed_sha256 = sha256_file(path)
            if observed_sha256 != str(row.sha256):
                raise ValueError(f"Uploaded source chunk checksum differs from manifest: {path}")
        start = expected_chunk_id * DOWNLOAD_CHUNK_SIZE
        end = min(start + DOWNLOAD_CHUNK_SIZE, sampled_meta.shape[0])
        valid, reason, _shape = source_chunk_matches(
            path,
            gene_list,
            sampled_meta.iloc[start:end],
        )
        if not valid:
            raise ValueError(
                f"Transferred source CELLxGENE chunk {expected_chunk_id} is invalid: {reason}"
            )
        chunk_paths.append(path)
    return chunk_paths


def build_source_chunk_index(
    sampled_meta: pd.DataFrame,
    source_chunk_paths: list[Path],
) -> dict[tuple[tuple[str, str, str], str], list[tuple[int, int]]]:
    index: dict[tuple[tuple[str, str, str], str], list[tuple[int, int]]] = defaultdict(list)

    expected_chunks = math.ceil(sampled_meta.shape[0] / DOWNLOAD_CHUNK_SIZE)
    if expected_chunks != len(source_chunk_paths):
        raise ValueError(
            "Cached source metadata and chunk files are inconsistent: "
            f"metadata implies {expected_chunks} chunks, found {len(source_chunk_paths)} files."
        )

    # Cached source chunks are written chunk-by-chunk from sampled_meta in order,
    # so each metadata row maps directly to a (chunk_id, row_idx) location.
    for absolute_row_idx, row in enumerate(sampled_meta.itertuples(index=False)):
        chunk_id = absolute_row_idx // DOWNLOAD_CHUNK_SIZE
        row_idx = absolute_row_idx % DOWNLOAD_CHUNK_SIZE
        context_key = make_context_key(row)
        index[(context_key, row.cell_type)].append((chunk_id, row_idx))

    return index


def aggregate_selected_cached_cells(
    source_chunk_paths: list[Path],
    selected_locations: list[tuple[int, int]],
) -> sparse.csr_matrix:
    rows_by_chunk: dict[int, list[int]] = defaultdict(list)
    for chunk_id, row_idx in selected_locations:
        rows_by_chunk[int(chunk_id)].append(int(row_idx))

    aggregated = None
    for chunk_id, row_indices in rows_by_chunk.items():
        adata = ad.read_h5ad(source_chunk_paths[chunk_id])
        selected = adata.X[np.asarray(row_indices, dtype=np.int64)]
        chunk_sum = selected.sum(axis=0)
        chunk_sum = sparse.csr_matrix(chunk_sum)
        aggregated = chunk_sum if aggregated is None else aggregated + chunk_sum
        del adata, selected, chunk_sum
        gc.collect()

    if aggregated is None:
        raise ValueError("No source cells were selected for aggregation.")
    return sparse.csr_matrix(aggregated)


def generate_pseudo_bulk_chunks(
    source_adata: ad.AnnData,
    plan_rows: list[dict[str, object]],
    proportion_column_map: dict[str, str],
    gene_list: list[str],
) -> list[Path]:
    source_index = build_source_row_index(source_adata)
    chunk_paths: list[Path] = []

    for chunk_id, start in enumerate(range(0, len(plan_rows), WRITE_CHUNK_SIZE)):
        end = min(start + WRITE_CHUNK_SIZE, len(plan_rows))
        chunk_plan = plan_rows[start:end]
        out_path = CHUNK_DIR / f"pseudo_bulk_RAW_chunk_{chunk_id:05d}.h5ad"
        if RESUME and out_path.exists():
            if validate_cached_h5ad(
                out_path,
                gene_list,
                label=f"pseudo-bulk chunk {chunk_id}",
                allow_recompute=True,
            ):
                print(f"Pseudo-bulk chunk {chunk_id}: reusing existing {out_path.name}")
                chunk_paths.append(out_path)
                continue

        rng = np.random.default_rng(RANDOM_SEED + chunk_id)

        aggregated_rows: list[sparse.csr_matrix] = []
        obs_records: list[dict[str, object]] = []

        for row in chunk_plan:
            context_key = (row["dataset_id"], row["donor_id"], row[TISSUE_COLUMN])
            selected_row_indices: list[np.ndarray] = []

            for cell_type, count in row["cell_type_counts"].items():
                pool = source_index.get((context_key, cell_type))
                if pool is None or pool.size == 0:
                    raise ValueError(
                        f"No sampled source cells available for context={context_key}, cell_type={cell_type}."
                    )
                chosen = rng.choice(pool, size=int(count), replace=pool.size < int(count))
                selected_row_indices.append(np.asarray(chosen, dtype=np.int64))

            selected_indices = np.concatenate(selected_row_indices)
            aggregated = source_adata.X[selected_indices].sum(axis=0)
            aggregated_rows.append(sparse.csr_matrix(aggregated))

            obs_record = {
                "dataset": "CELLxGENE_Census_pseudobulk",
                "dataset_id": row["dataset_id"],
                "donor_id": row["donor_id"],
                TISSUE_COLUMN: row[TISSUE_COLUMN],
                "total_cells": int(row["total_cells"]),
                "n_cell_types": int(row["n_cell_types"]),
            }
            for column in proportion_column_map.values():
                obs_record[column] = np.float32(0.0)
            for cell_type, proportion in row["cell_type_proportions"].items():
                obs_record[proportion_column_map[cell_type]] = np.float32(proportion)
            obs_records.append(obs_record)

        raw_chunk = sparse.vstack(aggregated_rows, format="csr")
        x_dense = raw_chunk.toarray().astype(np.float32, copy=False)
        x_raw, keep_mask = filter_raw_dense_block(x_dense)
        if x_raw.shape[0] == 0:
            continue

        obs = pd.DataFrame(obs_records, index=pd.Index([row["sample_id"] for row in chunk_plan], name="sample_id"))
        obs = obs.iloc[np.where(keep_mask)[0]].copy()
        var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        out = ad.AnnData(X=sparse.csr_matrix(x_raw), obs=obs, var=var)
        out.var_names = pd.Index(gene_list, dtype=str)
        store_proportion_metadata(out, proportion_column_map)

        write_h5ad_compat(out, out_path)
        chunk_paths.append(out_path)

        del raw_chunk, x_dense, x_raw, obs, var, out
        gc.collect()

        print(
            f"Pseudo-bulk chunk {chunk_id}: wrote {chunk_paths[-1].name} "
            f"for samples {start}:{end}"
        )

    return chunk_paths


def generate_pseudo_bulk_chunks_from_cached_sources(
    sampled_meta: pd.DataFrame,
    source_chunk_paths: list[Path],
    plan_rows: list[dict[str, object]],
    proportion_column_map: dict[str, str],
    gene_list: list[str],
) -> list[Path]:
    source_index = build_source_chunk_index(sampled_meta, source_chunk_paths)
    chunk_paths: list[Path] = []

    for chunk_id, start in enumerate(range(0, len(plan_rows), WRITE_CHUNK_SIZE)):
        end = min(start + WRITE_CHUNK_SIZE, len(plan_rows))
        chunk_plan = plan_rows[start:end]
        out_path = CHUNK_DIR / f"pseudo_bulk_RAW_chunk_{chunk_id:05d}.h5ad"
        if RESUME and out_path.exists():
            if validate_cached_h5ad(
                out_path,
                gene_list,
                label=f"pseudo-bulk chunk {chunk_id}",
                allow_recompute=True,
            ):
                print(f"Pseudo-bulk chunk {chunk_id}: reusing existing {out_path.name}")
                chunk_paths.append(out_path)
                continue

        rng = np.random.default_rng(RANDOM_SEED + chunk_id)
        aggregated_rows: list[sparse.csr_matrix] = []
        obs_records: list[dict[str, object]] = []

        for row in chunk_plan:
            context_key = (row["dataset_id"], row["donor_id"], row[TISSUE_COLUMN])
            selected_locations: list[tuple[int, int]] = []

            for cell_type, count in row["cell_type_counts"].items():
                pool = source_index.get((context_key, cell_type))
                if not pool:
                    raise ValueError(
                        f"No cached source cells available for context={context_key}, cell_type={cell_type}."
                    )
                chosen_idx = rng.choice(len(pool), size=int(count), replace=len(pool) < int(count))
                selected_locations.extend(pool[int(idx)] for idx in np.asarray(chosen_idx).ravel())

            aggregated_rows.append(
                aggregate_selected_cached_cells(source_chunk_paths, selected_locations)
            )

            obs_record = {
                "dataset": "CELLxGENE_Census_pseudobulk",
                "dataset_id": row["dataset_id"],
                "donor_id": row["donor_id"],
                TISSUE_COLUMN: row[TISSUE_COLUMN],
                "total_cells": int(row["total_cells"]),
                "n_cell_types": int(row["n_cell_types"]),
            }
            for column in proportion_column_map.values():
                obs_record[column] = np.float32(0.0)
            for cell_type, proportion in row["cell_type_proportions"].items():
                obs_record[proportion_column_map[cell_type]] = np.float32(proportion)
            obs_records.append(obs_record)

        raw_chunk = sparse.vstack(aggregated_rows, format="csr")
        x_dense = raw_chunk.toarray().astype(np.float32, copy=False)
        x_raw, keep_mask = filter_raw_dense_block(x_dense)
        if x_raw.shape[0] == 0:
            continue

        obs = pd.DataFrame(
            obs_records,
            index=pd.Index([row["sample_id"] for row in chunk_plan], name="sample_id"),
        )
        obs = obs.iloc[np.where(keep_mask)[0]].copy()
        var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

        out = ad.AnnData(X=sparse.csr_matrix(x_raw), obs=obs, var=var)
        out.var_names = pd.Index(gene_list, dtype=str)
        store_proportion_metadata(out, proportion_column_map)

        write_h5ad_compat(out, out_path)
        chunk_paths.append(out_path)

        del raw_chunk, x_dense, x_raw, obs, var, out
        gc.collect()

        print(
            f"Pseudo-bulk chunk {chunk_id}: wrote {chunk_paths[-1].name} "
            f"for samples {start}:{end}"
        )

    return chunk_paths


def merge_chunks(
    chunk_paths: list[Path],
    proportion_column_map: dict[str, str],
    gene_list: list[str],
) -> Path:
    if len(chunk_paths) == 0:
        raise ValueError("No pseudo-bulk chunk files were created.")

    if RESUME and validate_cached_final(gene_list):
        return FINAL_OUT

    for path in chunk_paths:
        validate_cached_h5ad(path, gene_list, label="pseudo-bulk chunk")

    current_paths = list(chunk_paths)
    round_id = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for batch_id, start in enumerate(range(0, len(current_paths), MERGE_BATCH_SIZE)):
            batch_paths = current_paths[start:start + MERGE_BATCH_SIZE]
            out_path = MERGE_TMP_DIR / f"RAW_merge_r{round_id:02d}_b{batch_id:04d}.h5ad"
            print(
                f"Merging pseudo-bulk batch round {round_id}, batch {batch_id}: "
                f"{len(batch_paths)} files"
            )
            next_paths.append(merge_h5ad_group(batch_paths, out_path, gene_list))
        current_paths = next_paths
        round_id += 1

    final_merged = ad.read_h5ad(current_paths[0])
    final_merged.obs_names_make_unique()
    final_merged = final_merged[:, gene_list].copy()
    final_merged.var_names = pd.Index(gene_list, dtype=str)
    store_proportion_metadata(final_merged, proportion_column_map)
    print(
        f"Pseudo-bulk merged: {final_merged.n_obs} samples x "
        f"{final_merged.n_vars} genes"
    )
    write_h5ad_compat(final_merged, FINAL_OUT)

    del final_merged
    gc.collect()
    print(f"Saved {FINAL_OUT}")
    return FINAL_OUT


def audit_pseudobulk_dataset(
    path: Path,
    proportion_column_map: dict[str, str],
) -> dict[str, object]:
    backed = ad.read_h5ad(path, backed="r")
    try:
        targets = backed.obs[
            [proportion_column_map[cell_type] for cell_type in sorted(proportion_column_map)]
        ].to_numpy(dtype=np.float64)
        context_columns = ["dataset_id", "donor_id", TISSUE_COLUMN]
        contexts = backed.obs[context_columns].astype(str).agg("||".join, axis=1)
        n_obs = int(backed.n_obs)
        n_vars = int(backed.n_vars)
    finally:
        close_backed_adata(backed)

    if n_obs != TARGET_PSEUDO_BULKS:
        raise ValueError(
            f"Final pseudobulk contains {n_obs} samples; expected exactly {TARGET_PSEUDO_BULKS}."
        )
    if not np.all(np.isfinite(targets)) or np.any(targets < 0):
        raise ValueError("Final pseudobulk target proportions must be finite and non-negative.")
    row_sums = targets.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            "Final pseudobulk target rows do not sum to one: "
            f"min={row_sums.min():.8f}, max={row_sums.max():.8f}."
        )
    active_counts = np.count_nonzero(targets > 0, axis=1)
    if active_counts.min() < MIN_ACTIVE_CELL_TYPES or active_counts.max() > MAX_ACTIVE_CELL_TYPES:
        raise ValueError(
            "Final pseudobulk active-cell-type counts violate the configured range: "
            f"observed {active_counts.min()}..{active_counts.max()}, expected "
            f"{MIN_ACTIVE_CELL_TYPES}..{MAX_ACTIVE_CELL_TYPES}."
        )

    cell_types = sorted(proportion_column_map)
    positive_samples = np.count_nonzero(targets > 0, axis=0)
    context_array = contexts.to_numpy()
    positive_contexts = {
        cell_type: int(np.unique(context_array[targets[:, index] > 0]).size)
        for index, cell_type in enumerate(cell_types)
    }
    missing_targets = [
        cell_type
        for index, cell_type in enumerate(cell_types)
        if positive_samples[index] == 0
    ]
    if missing_targets:
        raise ValueError(f"Generated targets never used in a mixture: {missing_targets}")

    audit = {
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "samples": n_obs,
        "genes": n_vars,
        "target_cell_types": len(cell_types),
        "cell_types": cell_types,
        "target_row_sum_min": float(row_sums.min()),
        "target_row_sum_max": float(row_sums.max()),
        "active_cell_types_min": int(active_counts.min()),
        "active_cell_types_mean": float(active_counts.mean()),
        "active_cell_types_median": float(np.median(active_counts)),
        "active_cell_types_max": int(active_counts.max()),
        "zero_target_fraction": float(np.mean(targets == 0)),
        "positive_samples_by_cell_type": dict(
            zip(cell_types, positive_samples.astype(int).tolist())
        ),
        "positive_contexts_by_cell_type": positive_contexts,
        "cell_ontology_release": CELL_ONTOLOGY_RELEASE,
        "cell_ontology_sha256": sha256_file(CELL_ONTOLOGY_PATH),
        "broad_cell_type_config_sha256": sha256_file(BROAD_CELL_TYPE_CONFIG_PATH),
    }
    write_json(audit, DATASET_AUDIT_OUT)
    return audit


def main() -> None:
    if AUDIT_ONLY and DOWNLOAD_ONLY:
        raise ValueError("Audit-only and download-only modes are mutually exclusive.")
    if VALIDATE_TRANSFER_ONLY and not OFFLINE:
        raise ValueError("Transfer-only validation must run with SCBFM_PSEUDO_OFFLINE=1.")
    if DOWNLOAD_ONLY and OFFLINE:
        raise ValueError("Download-only mode cannot run with SCBFM_PSEUDO_OFFLINE=1.")

    gene_list = read_gene_list(GENE_LIST_PATH)
    if len(set(gene_list)) != len(gene_list):
        raise ValueError(f"Gene list contains duplicates: {GENE_LIST_PATH}")

    ontology_path = ensure_cell_ontology_file()
    ontology_names, ontology_parents, alt_to_primary, ontology_data_version = (
        parse_cell_ontology(ontology_path)
    )
    broad_specs = load_broad_cell_type_specs(
        BROAD_CELL_TYPE_CONFIG_PATH,
        ontology_names,
    )
    mapping_cache: dict[str, tuple[str | None, str | None, int | None]] = {}
    mapping_counts: dict[tuple[str, str, str, str, int], int] = defaultdict(int)

    print(f"Target pseudo-bulk sample count: {TARGET_PSEUDO_BULKS}")
    print(f"Target gene count: {len(gene_list)}")
    print(f"Gene list: {GENE_LIST_PATH}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Census version: {CENSUS_VERSION}")
    print(f"Cell Ontology: {ontology_data_version} ({ontology_path})")
    print(f"Broad cell-type config: {BROAD_CELL_TYPE_CONFIG_PATH}")
    print(
        "Mixture distribution: "
        f"Dirichlet(alpha={DIRICHLET_ALPHA}) + Multinomial({CELLS_PER_PSEUDO_BULK})"
    )
    print(f"Resume enabled: {RESUME}")
    print(f"Offline mode: {OFFLINE}")
    print(f"Audit-only mode: {AUDIT_ONLY}")
    print(f"Download-only mode: {DOWNLOAD_ONLY}")
    print(f"Transfer-validation-only mode: {VALIDATE_TRANSFER_ONLY}")

    if RESUME and validate_cached_final(gene_list):
        return

    missing_genes = read_json(MISSING_GENES_OUT) if RESUME and MISSING_GENES_OUT.exists() else None

    if (
        RESUME
        and ELIGIBLE_CONTEXTS_OUT.exists()
        and manifest_matches(METADATA_MANIFEST_OUT, metadata_manifest())
    ):
        print(f"Reusing eligible contexts from {ELIGIBLE_CONTEXTS_OUT}")
        eligible_df, eligible_counts = load_eligible_contexts_from_csv(ELIGIBLE_CONTEXTS_OUT)
    else:
        if RESUME and ELIGIBLE_CONTEXTS_OUT.exists():
            print("Ignoring cached eligible contexts because their audit manifest differs.")
        eligible_df = pd.DataFrame()
        eligible_counts = {}

    if (
        RESUME
        and PLAN_OUT.exists()
        and SOURCE_POOL_QUOTAS_OUT.exists()
        and manifest_matches(SAMPLING_MANIFEST_OUT, sampling_manifest())
    ):
        print(f"Reusing pseudo-bulk plan from {PLAN_OUT}")
        plan_rows, plan_df, reservoir_quotas, all_cell_types = load_sample_plan_from_csv(
            PLAN_OUT,
            SOURCE_POOL_QUOTAS_OUT,
        )
    else:
        if RESUME and PLAN_OUT.exists():
            print("Ignoring cached sampling plan because its manifest differs.")
        plan_rows = []
        plan_df = pd.DataFrame()
        reservoir_quotas = {}
        all_cell_types = []

    if (
        not OFFLINE
        and not DOWNLOAD_ONLY
        and RESUME
        and SOURCE_ADATA_OUT.exists()
    ):
        if validate_cached_h5ad(
            SOURCE_ADATA_OUT,
            gene_list,
            label="aligned source cells",
            allow_recompute=True,
        ):
            print(f"Reusing downloaded source cells from {SOURCE_ADATA_OUT}")
            source_adata = ad.read_h5ad(SOURCE_ADATA_OUT)
        else:
            source_adata = None
    else:
        source_adata = None

    if not plan_rows and eligible_counts:
        plan_rows, plan_df, reservoir_quotas, all_cell_types = simulate_sample_plan(
            eligible_counts,
            TARGET_PSEUDO_BULKS,
        )
        plan_df.to_csv(PLAN_OUT, index=False)
        write_json(sampling_manifest(), SAMPLING_MANIFEST_OUT)

    if missing_genes is None and (source_adata is not None or OFFLINE):
        missing_genes = []

    if OFFLINE and not eligible_counts:
        raise ValueError(
            f"SCBFM_PSEUDO_OFFLINE=1 requires cached eligible contexts at {ELIGIBLE_CONTEXTS_OUT}."
        )
    if OFFLINE and not plan_rows:
        raise ValueError(
            f"SCBFM_PSEUDO_OFFLINE=1 requires cached sampling plan at {PLAN_OUT} "
            f"and quotas at {SOURCE_POOL_QUOTAS_OUT}."
        )
    if OFFLINE and not SAMPLED_SOURCE_CELLS_OUT.exists():
        raise ValueError(
            f"SCBFM_PSEUDO_OFFLINE=1 requires cached source metadata at {SAMPLED_SOURCE_CELLS_OUT}."
        )

    if VALIDATE_TRANSFER_ONLY:
        sampled_source_meta = load_sampled_source_meta(SAMPLED_SOURCE_CELLS_OUT)
        transferred_paths = validate_source_chunk_transfer(
            sampled_source_meta,
            gene_list,
        )
        print(
            f"Validated {len(transferred_paths)} uploaded source chunks for "
            f"{sampled_source_meta.shape[0]} source cells."
        )
        return

    if AUDIT_ONLY and eligible_counts:
        print(f"Audit artifacts are ready under {OUT_DIR}")
        return

    need_census = (not OFFLINE) and (
        (not eligible_counts) or (source_adata is None and not AUDIT_ONLY)
    )

    if need_census:
        census_api = require_cellxgene_census()
        with census_api.open_soma(
            census_version=CENSUS_VERSION,
            tiledb_config=TILEDB_CONFIG,
        ) as census:
            if missing_genes is None and not AUDIT_ONLY:
                var_coords, missing_genes = build_var_coords(census, gene_list)
                write_json(missing_genes, MISSING_GENES_OUT)
            else:
                var_coords = None

            if missing_genes:
                raise ValueError(
                    f"CELLxGENE Census is missing {len(missing_genes)} genes from gene_list.txt. "
                    f"See {MISSING_GENES_OUT}."
                )

            if not eligible_counts:
                counts_by_context = count_context_cell_types(
                    census,
                    parents=ontology_parents,
                    alt_to_primary=alt_to_primary,
                    broad_specs=broad_specs,
                    mapping_cache=mapping_cache,
                    mapping_counts=mapping_counts,
                )
                write_cell_type_mapping_audit(mapping_counts)
                supported_targets = select_supported_target_cell_types(
                    counts_by_context,
                    broad_specs,
                )
                eligible_df, eligible_counts = build_eligible_contexts(
                    counts_by_context,
                    supported_targets,
                )
                if eligible_df.empty:
                    raise ValueError("No biologically feasible contexts were found.")
                eligible_df.to_csv(ELIGIBLE_CONTEXTS_OUT, index=False)
                write_json(metadata_manifest(), METADATA_MANIFEST_OUT)

            if AUDIT_ONLY:
                print(
                    f"Metadata audit complete: {len(eligible_counts)} eligible contexts, "
                    f"artifacts written under {OUT_DIR}"
                )
                return

            if not plan_rows:
                plan_rows, plan_df, reservoir_quotas, all_cell_types = simulate_sample_plan(
                    eligible_counts,
                    TARGET_PSEUDO_BULKS,
                )
                plan_df.to_csv(PLAN_OUT, index=False)
                write_json(sampling_manifest(), SAMPLING_MANIFEST_OUT)

            if source_adata is None:
                if RESUME and SAMPLED_SOURCE_CELLS_OUT.exists():
                    print(f"Reusing sampled source cell metadata from {SAMPLED_SOURCE_CELLS_OUT}")
                    sampled_source_meta = load_sampled_source_meta(SAMPLED_SOURCE_CELLS_OUT)
                else:
                    sampled_source_meta = reservoir_sample_source_cells(
                        census,
                        reservoir_quotas,
                        parents=ontology_parents,
                        alt_to_primary=alt_to_primary,
                        broad_specs=broad_specs,
                        mapping_cache=mapping_cache,
                    )
                    sampled_source_meta.to_csv(SAMPLED_SOURCE_CELLS_OUT, index=False)
                print(
                    f"Sampled {sampled_source_meta.shape[0]} source cells for pseudo-bulk generation"
                )

                if var_coords is None:
                    var_coords, _ = build_var_coords(census, gene_list)
                source_adata = download_source_cells(
                    census,
                    sampled_source_meta,
                    var_coords,
                    gene_list,
                    combine_chunks=not DOWNLOAD_ONLY,
                )
                if DOWNLOAD_ONLY:
                    download_summary = {
                        **sampling_manifest(),
                        "sampled_source_cells": int(sampled_source_meta.shape[0]),
                        "source_chunks": int(
                            math.ceil(sampled_source_meta.shape[0] / DOWNLOAD_CHUNK_SIZE)
                        ),
                        "source_chunk_manifest": str(SOURCE_CHUNK_MANIFEST_OUT),
                        "ready_for_offline_transfer": True,
                    }
                    write_json(download_summary, SOURCE_DOWNLOAD_SUMMARY_OUT)
                    print(
                        "Source-cell download complete. Upload the entire output "
                        f"directory to the cluster: {OUT_DIR}"
                    )
                    return
                if source_adata is None:
                    raise RuntimeError("Source-cell download did not return an aligned AnnData.")
                write_h5ad_compat(source_adata, SOURCE_ADATA_OUT)
                print(f"Cached aligned source cells at {SOURCE_ADATA_OUT}")

    if eligible_df.empty:
        raise ValueError("No biologically feasible contexts were found.")
    if not plan_rows:
        raise ValueError("No pseudo-bulk plan is available.")

    proportion_column_map = build_proportion_column_map(all_cell_types)
    if source_adata is None:
        sampled_source_meta = load_sampled_source_meta(SAMPLED_SOURCE_CELLS_OUT)
        source_chunk_paths = validate_source_chunk_transfer(
            sampled_source_meta,
            gene_list,
        )
        chunk_paths = generate_pseudo_bulk_chunks_from_cached_sources(
            sampled_source_meta,
            source_chunk_paths,
            plan_rows,
            proportion_column_map,
            gene_list,
        )
        sampled_source_cells = int(sampled_source_meta.shape[0])
        source_mode = "cached_source_chunks"
    else:
        chunk_paths = generate_pseudo_bulk_chunks(
            source_adata,
            plan_rows,
            proportion_column_map,
            gene_list,
        )
        sampled_source_cells = int(source_adata.n_obs)
        source_mode = "aligned_source_adata"
    final_path = merge_chunks(chunk_paths, proportion_column_map, gene_list)
    dataset_audit = audit_pseudobulk_dataset(final_path, proportion_column_map)

    final = ad.read_h5ad(final_path, backed="r")
    try:
        print(f"Final pseudo-bulk dataset dimensions: {final.n_obs} samples x {final.n_vars} genes")
    finally:
        close_backed_adata(final)

    summary = {
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "census_version": CENSUS_VERSION,
        "organism": ORGANISM,
        "cell_ontology_release": CELL_ONTOLOGY_RELEASE,
        "cell_ontology_data_version": ontology_data_version,
        "cell_ontology_sha256": sha256_file(CELL_ONTOLOGY_PATH),
        "broad_cell_type_config_sha256": sha256_file(BROAD_CELL_TYPE_CONFIG_PATH),
        "target_pseudo_bulks": TARGET_PSEUDO_BULKS,
        "cells_per_pseudo_bulk": CELLS_PER_PSEUDO_BULK,
        "target_gene_count": len(gene_list),
        "eligible_contexts": int(len(eligible_counts)),
        "sampled_source_cells": sampled_source_cells,
        "source_mode": source_mode,
        "distinct_cell_types": int(len(all_cell_types)),
        "tissue_column": TISSUE_COLUMN,
        "min_context_cell_types": MIN_CONTEXT_CELL_TYPES,
        "min_available_cells_per_celltype": MIN_AVAILABLE_CELLS_PER_CELLTYPE,
        "min_context_total_cells": MIN_CONTEXT_TOTAL_CELLS,
        "min_target_contexts": MIN_TARGET_CONTEXTS,
        "min_target_source_cells": MIN_TARGET_SOURCE_CELLS,
        "minimum_active_cell_types": MIN_ACTIVE_CELL_TYPES,
        "maximum_active_cell_types": MAX_ACTIVE_CELL_TYPES,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "minimum_realized_cells_per_active_type": MIN_REALIZED_CELLS_PER_ACTIVE_TYPE,
        "dataset_audit": dataset_audit,
        "missing_genes": int(len(missing_genes)),
        "outputs": {
            "final_h5ad": str(FINAL_OUT),
            "sampling_plan_csv": str(PLAN_OUT),
            "eligible_contexts_csv": str(ELIGIBLE_CONTEXTS_OUT),
            "source_pool_quotas_csv": str(SOURCE_POOL_QUOTAS_OUT),
            "sampled_source_cells_csv": str(SAMPLED_SOURCE_CELLS_OUT),
            "source_cell_chunks_dir": str(SOURCE_CHUNK_DIR),
            "source_chunk_manifest_csv": str(SOURCE_CHUNK_MANIFEST_OUT),
            "source_download_summary_json": str(SOURCE_DOWNLOAD_SUMMARY_OUT),
            "aligned_source_cells_h5ad": str(SOURCE_ADATA_OUT),
            "missing_genes_json": str(MISSING_GENES_OUT),
            "cell_type_mapping_csv": str(CELL_TYPE_MAPPING_OUT),
            "broad_cell_type_audit_csv": str(TARGET_AUDIT_OUT),
            "dataset_audit_json": str(DATASET_AUDIT_OUT),
            "cell_ontology_obo": str(CELL_ONTOLOGY_PATH),
            "metadata_manifest_json": str(METADATA_MANIFEST_OUT),
            "sampling_manifest_json": str(SAMPLING_MANIFEST_OUT),
        },
        "resume_enabled": RESUME,
        "offline_enabled": OFFLINE,
    }
    write_json(summary, SUMMARY_OUT)


if __name__ == "__main__":
    main()
