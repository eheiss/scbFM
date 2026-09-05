# Pretraining-corpus distributions

The cluster job reads the bulkRNA-seq and scRNA-seq pretraining matrices
sequentially and writes compact, validated histogram and statistics files. The
local `src/analysis/distributions.ipynb` notebook renders the thesis figures
without accessing either expression matrix.

From the cluster work directory, submit:

```bash
cd /cluster/work/boeva/eheiss
sbatch job_files/analysis/distributions_summary-job.sh
```

The job writes these files under `output/distributions`:

- `pretraining_distribution_histograms.npz`
- `pretraining_distribution_statistics.csv`
- `pretraining_distribution_metadata.json`

After the job completes, sync that directory to the local `eheiss/output`
directory and run `src/analysis/distributions.ipynb` locally.
