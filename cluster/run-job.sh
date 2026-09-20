#!/usr/bin/env bash
#SBATCH --job-name=scbfm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

: "${SCBFM_ROOT_DIR:?Set SCBFM_ROOT_DIR to the experiment workspace}"
: "${SCBFM_REPO_DIR:?Use cluster/submit.sh or set SCBFM_REPO_DIR to the checkout}"
SIF=${SCBFM_SIF:-${SCBFM_ROOT_DIR}/singularity/scbfm.sif}
ENTRYPOINT=${SCBFM_ENTRYPOINT:-${SCBFM_REPO_DIR}/src/main.py}
NPROC=${SCBFM_NPROC:-4}

for file in "$SIF" "$ENTRYPOINT"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing required file: $file" >&2
        exit 1
    fi
done
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
    echo 'SCBFM_NPROC must be a positive integer.' >&2
    exit 1
fi
if command -v apptainer >/dev/null 2>&1; then
    engine=apptainer
elif command -v singularity >/dev/null 2>&1; then
    engine=singularity
else
    echo 'Load Apptainer or Singularity before submitting.' >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
# Explicit container variables also work when the site uses a clean environment.
export SINGULARITYENV_SCBFM_ROOT_DIR="$SCBFM_ROOT_DIR"
export APPTAINERENV_SCBFM_ROOT_DIR="$SCBFM_ROOT_DIR"
gpu_args=()
if [[ ${SCBFM_USE_GPU:-1} == 1 ]]; then gpu_args=(--nv); fi

cd "$SCBFM_REPO_DIR"
exec "$engine" exec "${gpu_args[@]}" \
    --bind "$SCBFM_ROOT_DIR:$SCBFM_ROOT_DIR" \
    --bind "$SCBFM_REPO_DIR:$SCBFM_REPO_DIR" \
    "$SIF" python -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" "$ENTRYPOINT" "$@"
