# Pretraining-corpus distributions

The cluster job reads the bulkRNA-seq and scRNA-seq pretraining matrices
sequentially and writes compact, validated histogram and statistics files. The
local `src/analysis/distributions.ipynb` notebook renders the thesis figures
without accessing either expression matrix.

From the checkout, after configuring the cluster environment, submit a
single-process CPU job. This example reserves 512 GB because one complete
matrix is loaded at a time; adjust to the actual inputs:

```bash
export SCBFM_REPO_DIR="$PWD"
export SCBFM_ENTRYPOINT="$PWD/src/analysis/distributions/generate_summary.py"
export SCBFM_NPROC=1
export SCBFM_USE_GPU=0
mkdir -p "$SCBFM_ROOT_DIR/job_output"
sbatch --gres=none --mem=512G \
  --output="$SCBFM_ROOT_DIR/job_output/%x.%j.out" \
  --error="$SCBFM_ROOT_DIR/job_output/%x.%j.err" cluster/run-job.sh
```

The job writes these files under `output/distributions`:

- `pretraining_distribution_histograms.npz`
- `pretraining_distribution_statistics.csv`
- `pretraining_distribution_metadata.json`

After the job completes, sync that directory to the local `output/` directory
and run `src/analysis/distributions.ipynb`. The summaries include median total
counts per profile. `--bulk-path`, `--sc-path`, and `--output-dir` override the
default paths; `--validate-only` checks existing summaries without reloading
the matrices. See [cluster setup](../../../cluster/README.md) for image settings.
