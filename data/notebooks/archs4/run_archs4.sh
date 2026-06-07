#!/bin/bash
#SBATCH --job-name=archs4_convert
#SBATCH --time=12:00:00
#SBATCH --mem=600G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --output=/cluster/work/boeva/eheiss/job_output/output.%x.%J.out
#SBATCH --error=/cluster/work/boeva/eheiss/job_output/output.%x.%J.err

source ~/.bashrc
module load singularity

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONNOUSERSITE=1
export ARCHS4_SAMPLE_CHUNK_SIZE=${ARCHS4_SAMPLE_CHUNK_SIZE:-1000}

SIF=/cluster/customapps/biomed/boeva/eheiss/singularity/scbfm.sif
ROOT=/cluster/work/boeva/eheiss

singularity exec --nv \
  -B ${ROOT}:${ROOT} \
  ${SIF} \
  bash -lc "
    cd ${ROOT}/scbFM/data/notebooks/archs4
    python archs4.py
  "
