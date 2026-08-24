#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR=${SCBFM_CENSUS_VENV:-${HOME}/.venvs/scbfm-census-2025-11-08}

if [ -n "${PYTHON_BOOTSTRAP:-}" ]; then
  bootstrap_python=${PYTHON_BOOTSTRAP}
elif command -v python3.11 >/dev/null 2>&1; then
  bootstrap_python=$(command -v python3.11)
elif command -v python3.12 >/dev/null 2>&1; then
  bootstrap_python=$(command -v python3.12)
elif command -v python3.10 >/dev/null 2>&1; then
  bootstrap_python=$(command -v python3.10)
else
  echo "Python 3.10, 3.11, or 3.12 is required for the CELLxGENE Census client."
  exit 1
fi

"${bootstrap_python}" -c \
  'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'

echo "Creating or updating Census environment: ${VENV_DIR}"
echo "Bootstrap Python: ${bootstrap_python}"
"${bootstrap_python}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --upgrade \
  --requirement "${SCRIPT_DIR}/requirements-pseudo-bulk-download.txt"

"${VENV_DIR}/bin/python" -c \
  'import anndata, cellxgene_census, pandas, tiledbsoma; print(f"Ready: cellxgene-census {cellxgene_census.__version__}, tiledbsoma {tiledbsoma.__version__}, anndata {anndata.__version__}, pandas {pandas.__version__}")'

echo "The pseudo-bulk audit/download script will use this environment automatically."
