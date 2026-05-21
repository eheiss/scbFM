from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import DEFAULT_GENE_LIST_PATH, preprocess_raw_h5ad  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw-count h5ad files into scbFM token input by applying "
            "gene reindexing, min-gene filtering, and scGPT-style nonzero quantile binning."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw input .h5ad file.")
    parser.add_argument("--output", required=True, type=Path, help="Preprocessed output .h5ad file.")
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=DEFAULT_GENE_LIST_PATH,
        help="Gene list used to reindex columns. Defaults to data/gene_list.txt.",
    )
    parser.add_argument(
        "--skip-gene-reindex",
        action="store_true",
        help="Keep the input gene order instead of reindexing to --gene-list.",
    )
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--bin-num", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_raw_h5ad(
        input_path=args.input,
        output_path=args.output,
        gene_list_path=None if args.skip_gene_reindex else args.gene_list,
        min_genes=args.min_genes,
        bin_num=args.bin_num,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
