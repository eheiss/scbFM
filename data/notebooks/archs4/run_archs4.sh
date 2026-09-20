#!/bin/bash
#SBATCH --job-name=archs4_convert
#SBATCH --time=12:00:00
#SBATCH --mem=600G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source ~/.bashrc
module load singularity

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONNOUSERSITE=1
export ARCHS4_SAMPLE_CHUNK_SIZE=${ARCHS4_SAMPLE_CHUNK_SIZE:-1000}

: "${SCBFM_ROOT_DIR:?Set SCBFM_ROOT_DIR to your experiment workspace}"
ROOT=${SCBFM_ROOT_DIR}
REPO=${SCBFM_REPO_DIR:-${ROOT}/scbFM}
SIF=${SCBFM_SIF:-${ROOT}/singularity/scbfm.sif}

exec singularity exec \
  -B "${ROOT}:${ROOT}" -B "${REPO}:${REPO}" \
  "${SIF}" python "${REPO}/data/notebooks/archs4/archs4.py"
