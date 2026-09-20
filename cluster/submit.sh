#!/usr/bin/env bash
set -euo pipefail

: "${SCBFM_ROOT_DIR:?Set SCBFM_ROOT_DIR to the experiment workspace}"
export SCBFM_ROOT_DIR
export SCBFM_REPO_DIR
SCBFM_REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$SCBFM_ROOT_DIR/job_output"

exec sbatch --export=ALL \
    --output="$SCBFM_ROOT_DIR/job_output/%x.%j.out" \
    --error="$SCBFM_ROOT_DIR/job_output/%x.%j.err" \
    "$SCBFM_REPO_DIR/cluster/run-job.sh" "$@"
