from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from analysis.distributions.common import (
    ARCHIVE_FILENAME,
    MODALITY_KEYS,
    STATISTICS_FILENAME,
    default_log_count_edges,
    default_nonzero_gene_edges,
    save_distribution_summary,
    summarize_expression_matrix,
    validate_distribution_summary,
)


def _adata(values: np.ndarray) -> SimpleNamespace:
    matrix = sparse.csr_matrix(values)
    return SimpleNamespace(X=matrix, n_obs=matrix.shape[0], n_vars=matrix.shape[1])


def _summary(modality: str, offset: int = 0) -> dict[str, object]:
    values = np.zeros((3, 13_004), dtype=np.float32)
    values[0, :2] = [1 + offset, 2 + offset]
    values[1, :3] = [3 + offset, 4 + offset, 5 + offset]
    values[2, 0] = 1 + offset
    return summarize_expression_matrix(
        _adata(values),
        modality=modality,
        log_count_edges=default_log_count_edges(),
        nonzero_gene_edges=default_nonzero_gene_edges(),
        chunk_size=2,
    )


def test_summary_statistics_and_histograms_are_exact() -> None:
    summary = _summary("sc")
    statistics = summary["statistics"]

    assert statistics["profiles"] == 3
    assert statistics["nonzero_entries"] == 6
    assert statistics["median_nonzero_count"] == 2.5
    assert statistics["median_total_count_per_profile"] == 3.0
    assert statistics["median_nonzero_genes_per_profile"] == 2.0
    assert summary["log_count_histogram"].sum() == 6
    assert summary["nonzero_gene_histogram"].sum() == 3


def test_summary_bundle_round_trip(tmp_path) -> None:
    summaries = {"sc": _summary("sc"), "bulk": _summary("bulk", offset=2)}
    save_distribution_summary(
        tmp_path,
        summaries=summaries,
        log_count_edges=default_log_count_edges(),
        nonzero_gene_edges=default_nonzero_gene_edges(),
        metadata={"test": True},
    )

    validate_distribution_summary(tmp_path)
    with np.load(tmp_path / ARCHIVE_FILENAME, allow_pickle=False) as archive:
        assert tuple(archive["modality_keys"].astype(str)) == MODALITY_KEYS
        assert archive["log_count_histogram__bulk"].sum() == 6
    with (tmp_path / STATISTICS_FILENAME).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["modality"] for row in rows] == list(MODALITY_KEYS)
