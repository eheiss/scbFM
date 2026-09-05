from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse


FORMAT_VERSION = 1
EXPECTED_GENE_COUNT = 13_004
MODALITY_KEYS = ("sc", "bulk")
STATISTICS_FILENAME = "pretraining_distribution_statistics.csv"
ARCHIVE_FILENAME = "pretraining_distribution_histograms.npz"
METADATA_FILENAME = "pretraining_distribution_metadata.json"
STATISTIC_FIELDS = (
    "modality",
    "profiles",
    "genes",
    "entries",
    "nonzero_entries",
    "zero_fraction",
    "minimum_nonzero_count",
    "median_nonzero_count",
    "maximum_nonzero_count",
    "fraction_nonzero_equal_one",
    "mean_total_count_per_profile",
    "median_total_count_per_profile",
    "mean_nonzero_genes_per_profile",
    "median_nonzero_genes_per_profile",
    "minimum_nonzero_genes_per_profile",
    "maximum_nonzero_genes_per_profile",
)


def default_log_count_edges() -> np.ndarray:
    return np.linspace(0.0, 20.0, 101, dtype=np.float64)


def default_nonzero_gene_edges() -> np.ndarray:
    return np.linspace(0.0, float(EXPECTED_GENE_COUNT), 61, dtype=np.float64)


def summarize_expression_matrix(
    adata,
    *,
    modality: str,
    log_count_edges: np.ndarray,
    nonzero_gene_edges: np.ndarray,
    chunk_size: int = 5_000_000,
) -> dict[str, object]:
    """Compute exact corpus statistics and compact histograms for one modality."""
    if modality not in MODALITY_KEYS:
        raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITY_KEYS}.")
    if int(adata.n_vars) != EXPECTED_GENE_COUNT:
        raise ValueError(
            f"{modality} matrix has {adata.n_vars} genes; expected {EXPECTED_GENE_COUNT}."
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    matrix = adata.X
    if sparse.issparse(matrix):
        values = matrix.data
        if int(matrix.count_nonzero()) != int(values.size):
            raise ValueError(f"{modality} sparse matrix contains explicitly stored zeros.")
        nonzero_genes = np.asarray(matrix.getnnz(axis=1)).ravel()
        profile_totals = np.asarray(
            matrix.sum(axis=1, dtype=np.float64)
        ).ravel()
    else:
        dense = np.asarray(matrix)
        values = dense[dense != 0]
        nonzero_genes = np.count_nonzero(dense, axis=1)
        profile_totals = np.sum(dense, axis=1, dtype=np.float64)

    if values.size == 0:
        raise ValueError(f"{modality} expression matrix contains no non-zero values.")
    minimum_nonzero_count = float(values.min())
    maximum_nonzero_count = float(values.max())
    if minimum_nonzero_count < 0:
        raise ValueError(f"{modality} expression matrix contains negative raw counts.")

    maximum_log_count = float(np.log1p(maximum_nonzero_count))
    if maximum_log_count > float(log_count_edges[-1]):
        raise ValueError(
            f"{modality} maximum log1p count {maximum_log_count:.3f} exceeds "
            f"the shared histogram limit {log_count_edges[-1]:.3f}."
        )

    log_count_histogram = np.zeros(log_count_edges.size - 1, dtype=np.int64)
    exactly_one = 0
    for start in range(0, values.size, chunk_size):
        chunk = values[start : start + chunk_size]
        log_count_histogram += np.histogram(
            np.log1p(chunk), bins=log_count_edges
        )[0]
        exactly_one += int(np.count_nonzero(chunk == 1))

    nonzero_gene_histogram = np.histogram(
        nonzero_genes, bins=nonzero_gene_edges
    )[0].astype(np.int64, copy=False)
    entries = int(adata.n_obs) * int(adata.n_vars)
    statistics = {
        "modality": modality,
        "profiles": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "entries": entries,
        "nonzero_entries": int(values.size),
        "zero_fraction": 1.0 - (values.size / entries),
        "minimum_nonzero_count": minimum_nonzero_count,
        "median_nonzero_count": float(np.median(values)),
        "maximum_nonzero_count": maximum_nonzero_count,
        "fraction_nonzero_equal_one": exactly_one / values.size,
        "mean_total_count_per_profile": float(np.mean(profile_totals)),
        "median_total_count_per_profile": float(np.median(profile_totals)),
        "mean_nonzero_genes_per_profile": float(np.mean(nonzero_genes)),
        "median_nonzero_genes_per_profile": float(np.median(nonzero_genes)),
        "minimum_nonzero_genes_per_profile": int(np.min(nonzero_genes)),
        "maximum_nonzero_genes_per_profile": int(np.max(nonzero_genes)),
    }
    return {
        "statistics": statistics,
        "log_count_histogram": log_count_histogram,
        "nonzero_gene_histogram": nonzero_gene_histogram,
    }


def save_distribution_summary(
    output_dir: Path,
    *,
    summaries: dict[str, dict[str, object]],
    log_count_edges: np.ndarray,
    nonzero_gene_edges: np.ndarray,
    metadata: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    if tuple(summaries) != MODALITY_KEYS:
        raise ValueError(
            f"Summaries must use modality order {MODALITY_KEYS}; got {tuple(summaries)}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / ARCHIVE_FILENAME
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp.npz")
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray([FORMAT_VERSION], dtype=np.int64),
        "modality_keys": np.asarray(MODALITY_KEYS),
        "log_count_edges": np.asarray(log_count_edges, dtype=np.float64),
        "nonzero_gene_edges": np.asarray(nonzero_gene_edges, dtype=np.float64),
    }
    for modality in MODALITY_KEYS:
        payload[f"log_count_histogram__{modality}"] = np.asarray(
            summaries[modality]["log_count_histogram"], dtype=np.int64
        )
        payload[f"nonzero_gene_histogram__{modality}"] = np.asarray(
            summaries[modality]["nonzero_gene_histogram"], dtype=np.int64
        )
    np.savez_compressed(temporary_archive, **payload)
    os.replace(temporary_archive, archive_path)

    statistics_path = output_dir / STATISTICS_FILENAME
    temporary_statistics = statistics_path.with_name(f".{statistics_path.name}.tmp")
    with temporary_statistics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATISTIC_FIELDS)
        writer.writeheader()
        for modality in MODALITY_KEYS:
            writer.writerow(summaries[modality]["statistics"])
    os.replace(temporary_statistics, statistics_path)

    metadata_path = output_dir / METADATA_FILENAME
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_metadata.write_text(
        json.dumps(
            {"format_version": FORMAT_VERSION, **metadata},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    validate_distribution_summary(output_dir)


def validate_distribution_summary(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    archive_path = output_dir / ARCHIVE_FILENAME
    statistics_path = output_dir / STATISTICS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    for path in (archive_path, statistics_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing distribution-summary output: {path}")

    with statistics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(row["modality"] for row in rows) != MODALITY_KEYS:
        raise ValueError("Statistics CSV has an unexpected modality order.")
    rows_by_modality = {row["modality"]: row for row in rows}

    with np.load(archive_path, allow_pickle=False) as archive:
        if int(archive["format_version"][0]) != FORMAT_VERSION:
            raise ValueError("Histogram archive has an unsupported format version.")
        if tuple(archive["modality_keys"].astype(str)) != MODALITY_KEYS:
            raise ValueError("Histogram archive has an unexpected modality order.")
        log_edges = archive["log_count_edges"]
        gene_edges = archive["nonzero_gene_edges"]
        if np.any(np.diff(log_edges) <= 0) or np.any(np.diff(gene_edges) <= 0):
            raise ValueError("Histogram edges must be strictly increasing.")
        for modality in MODALITY_KEYS:
            log_histogram = archive[f"log_count_histogram__{modality}"]
            gene_histogram = archive[f"nonzero_gene_histogram__{modality}"]
            if log_histogram.shape != (log_edges.size - 1,):
                raise ValueError(f"Invalid log-count histogram for {modality}.")
            if gene_histogram.shape != (gene_edges.size - 1,):
                raise ValueError(f"Invalid non-zero-gene histogram for {modality}.")
            if np.any(log_histogram < 0) or np.any(gene_histogram < 0):
                raise ValueError(f"Negative histogram count for {modality}.")
            row = rows_by_modality[modality]
            if int(log_histogram.sum()) != int(row["nonzero_entries"]):
                raise ValueError(f"Log-count histogram total does not match {modality} CSV.")
            if int(gene_histogram.sum()) != int(row["profiles"]):
                raise ValueError(f"Gene-count histogram total does not match {modality} CSV.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Metadata has an unsupported format version.")
