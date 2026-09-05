#!/usr/bin/env python3
"""Generate compact pretraining-corpus summaries for local thesis plots."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad


ROOT = Path(os.environ.get("SCBFM_CLUSTER_ROOT", "/cluster/work/boeva/eheiss"))
SCBFM_ROOT = ROOT / "scbFM"
SRC = SCBFM_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.distributions.common import (
    MODALITY_KEYS,
    default_log_count_edges,
    default_nonzero_gene_edges,
    save_distribution_summary,
    summarize_expression_matrix,
    validate_distribution_summary,
)


log = logging.getLogger("distribution_summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bulk-path",
        type=Path,
        default=ROOT / "datasets" / "bulk" / "pretraining_bulk_RAW.h5ad",
    )
    parser.add_argument(
        "--sc-path",
        type=Path,
        default=ROOT / "datasets" / "sc" / "pretraining_sc_RAW.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "distributions",
    )
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.validate_only:
        validate_distribution_summary(args.output_dir)
        log.info("Validated distribution summaries in %s", args.output_dir)
        return

    paths = {"bulk": args.bulk_path, "sc": args.sc_path}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pretraining matrices: {missing}")

    log_count_edges = default_log_count_edges()
    nonzero_gene_edges = default_nonzero_gene_edges()
    generated: dict[str, dict[str, object]] = {}
    # Process bulk first and release it before loading the single-cell corpus.
    for modality in ("bulk", "sc"):
        path = paths[modality]
        log.info("Loading %s matrix from %s", modality, path)
        adata = ad.read_h5ad(path)
        log.info("Summarizing %s matrix with shape %s", modality, adata.shape)
        generated[modality] = summarize_expression_matrix(
            adata,
            modality=modality,
            log_count_edges=log_count_edges,
            nonzero_gene_edges=nonzero_gene_edges,
            chunk_size=args.chunk_size,
        )
        log.info(
            "%s statistics:\n%s",
            modality,
            json.dumps(generated[modality]["statistics"], indent=2),
        )
        del adata
        gc.collect()

    summaries = {modality: generated[modality] for modality in MODALITY_KEYS}
    save_distribution_summary(
        args.output_dir,
        summaries=summaries,
        log_count_edges=log_count_edges,
        nonzero_gene_edges=nonzero_gene_edges,
        metadata={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_paths": {key: str(value) for key, value in paths.items()},
            "histogram_chunk_size": args.chunk_size,
            "histograms_contain_exact_counts": True,
        },
    )
    log.info("Wrote and validated distribution summaries in %s", args.output_dir)


if __name__ == "__main__":
    main()
