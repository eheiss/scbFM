#!/bin/bash
#SBATCH --job-name=archs4_2
#SBATCH --time=06:00:00
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --output=/cluster/work/boeva/eheiss/job_output/output.%x.%J.out
#SBATCH --error=/cluster/work/boeva/eheiss/job_output/output.%x.%J.err

source ~/.bashrc || true
if command -v module >/dev/null 2>&1; then
  module load singularity || true
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONNOUSERSITE=1
export ARCHS4_ROW_CHUNK_SIZE=${ARCHS4_ROW_CHUNK_SIZE:-1000}

SIF=/cluster/customapps/biomed/boeva/eheiss/singularity/scbfm.sif
ROOT=/cluster/work/boeva/eheiss

if ! command -v singularity >/dev/null 2>&1; then
  echo "ERROR: singularity command not found." >&2
  exit 1
fi

singularity exec --nv \
  -B ${ROOT}:${ROOT} \
  ${SIF} \
  bash -lc "
    cd ${ROOT}/scbFM/data/notebooks/archs4
    python archs4_2.py
  "
