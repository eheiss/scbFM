#!/bin/bash
#SBATCH --job-name=archs4_2
#SBATCH --time=06:00:00
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source ~/.bashrc || true
if command -v module >/dev/null 2>&1; then
  module load singularity || true
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONNOUSERSITE=1
export ARCHS4_ROW_CHUNK_SIZE=${ARCHS4_ROW_CHUNK_SIZE:-1000}

: "${SCBFM_ROOT_DIR:?Set SCBFM_ROOT_DIR to your experiment workspace}"
ROOT=${SCBFM_ROOT_DIR}
REPO=${SCBFM_REPO_DIR:-${ROOT}/scbFM}
SIF=${SCBFM_SIF:-${ROOT}/singularity/scbfm.sif}

if ! command -v singularity >/dev/null 2>&1; then
  echo "ERROR: singularity command not found." >&2
  exit 1
fi

exec singularity exec \
  -B "${ROOT}:${ROOT}" -B "${REPO}:${REPO}" \
  "${SIF}" python "${REPO}/data/notebooks/archs4/archs4_2.py"
