#!/bin/bash

set -euo pipefail

trap 'echo "Download interrupted. Rerun the same command to validate and resume existing chunks."; exit 130' INT TERM

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
OUT_DIR=${1:-/Users/enricoheiss/Downloads/pseudo_bulk}
MODE=${2:-download}
VENV_DIR=${SCBFM_CENSUS_VENV:-${HOME}/.venvs/scbfm-census-2025-11-08}
PYTHON_BIN=${PYTHON_BIN:-${VENV_DIR}/bin/python}

if [ "${MODE}" != "audit" ] && [ "${MODE}" != "download" ]; then
  echo "Usage: $0 [output-directory] [audit|download]"
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python is not executable: ${PYTHON_BIN}"
  echo "Create the compatible Census environment first:"
  echo "  ${SCRIPT_DIR}/setup_pseudo_bulk_download_env.sh"
  exit 1
fi

"${PYTHON_BIN}" -c '
import sys
import anndata
import cellxgene_census
import pandas
import scipy
import tiledbsoma

expected_census = "1.17.0"
expected_soma = "1.15.5"
expected_pandas = "2.2.3"
if cellxgene_census.__version__ != expected_census:
    raise RuntimeError(
        f"Expected cellxgene-census {expected_census}, got "
        f"{cellxgene_census.__version__}. Run data/setup_pseudo_bulk_download_env.sh."
    )
if tiledbsoma.__version__ != expected_soma:
    raise RuntimeError(
        f"Expected tiledbsoma {expected_soma}, got {tiledbsoma.__version__}. "
        "Run data/setup_pseudo_bulk_download_env.sh."
    )
if pandas.__version__ != expected_pandas:
    raise RuntimeError(
        f"Expected pandas {expected_pandas}, got {pandas.__version__}. "
        "Run data/setup_pseudo_bulk_download_env.sh."
    )
print(
    f"Download environment: Python {sys.version.split()[0]}, "
    f"cellxgene-census {cellxgene_census.__version__}, "
    f"tiledbsoma {tiledbsoma.__version__}, anndata {anndata.__version__}, "
    f"pandas {pandas.__version__}"
)
'
mkdir -p "${OUT_DIR}/source_cell_chunks"
existing_chunks=$(find "${OUT_DIR}/source_cell_chunks" -maxdepth 1 -name 'source_cells_chunk_*.h5ad' 2>/dev/null | wc -l | tr -d ' ')
echo "Found ${existing_chunks} existing source chunks; each will be validated before reuse."

export SCBFM_GENE_LIST_PATH="${REPO_DIR}/data/gene_list.txt"
export SCBFM_BROAD_CELL_TYPE_CONFIG_PATH="${REPO_DIR}/data/deconv_broad_cell_types.csv"
export SCBFM_PSEUDO_OUT_DIR="${OUT_DIR}"
export SCBFM_PSEUDO_CENSUS_VERSION=2025-11-08
export SCBFM_CELL_ONTOLOGY_RELEASE=2026-06-08
if [ "${MODE}" = "audit" ]; then
  export SCBFM_PSEUDO_AUDIT_ONLY=1
  export SCBFM_PSEUDO_DOWNLOAD_ONLY=0
else
  export SCBFM_PSEUDO_AUDIT_ONLY=0
  export SCBFM_PSEUDO_DOWNLOAD_ONLY=1
fi
export SCBFM_PSEUDO_VALIDATE_TRANSFER_ONLY=0
export SCBFM_PSEUDO_OFFLINE=0
export SCBFM_PSEUDO_RESUME=1
export SCBFM_PSEUDO_VERIFY_SOURCE_CHUNK_HASHES=1

cd "${REPO_DIR}"
"${PYTHON_BIN}" data/create_pseudo_bulk_data_RAW.py

if [ "${MODE}" = "audit" ]; then
  for path in \
    "${OUT_DIR}/broad_cell_type_audit.csv" \
    "${OUT_DIR}/cell_type_ontology_mapping.csv" \
    "${OUT_DIR}/eligible_contexts.csv" \
    "${OUT_DIR}/metadata_audit_manifest.json"; do
    if [ ! -s "${path}" ]; then
      echo "Local audit did not create required artifact: ${path}"
      exit 1
    fi
  done
  echo "Local metadata audit is ready: ${OUT_DIR}/broad_cell_type_audit.csv"
  exit 0
fi

required_files=(
  "${OUT_DIR}/broad_cell_type_audit.csv"
  "${OUT_DIR}/cell_type_ontology_mapping.csv"
  "${OUT_DIR}/eligible_contexts.csv"
  "${OUT_DIR}/metadata_audit_manifest.json"
  "${OUT_DIR}/pseudo_bulk_sampling_plan.csv"
  "${OUT_DIR}/sampling_plan_manifest.json"
  "${OUT_DIR}/source_cell_pool_quotas.csv"
  "${OUT_DIR}/sampled_source_cells.csv"
  "${OUT_DIR}/cellxgene_missing_genes.json"
  "${OUT_DIR}/source_cell_chunk_manifest.csv"
  "${OUT_DIR}/source_cell_download_summary.json"
  "${OUT_DIR}/cl-basic-2026-06-08.obo"
)

for path in "${required_files[@]}"; do
  if [ ! -s "${path}" ]; then
    echo "Local download stage did not create required artifact: ${path}"
    exit 1
  fi
done

if [ ! -d "${OUT_DIR}/source_cell_chunks" ]; then
  echo "Local download stage did not create source chunks."
  exit 1
fi

echo "Local source bundle is ready: ${OUT_DIR}"
echo "Upload this directory to: /cluster/work/boeva/eheiss/datasets/pseudo_bulk/"
