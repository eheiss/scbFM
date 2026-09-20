# Analysis and Figures

The analysis notebooks read completed benchmark outputs; they do not replace
training jobs. Start Jupyter from inside this checkout after setting
`SCBFM_ROOT_DIR` to your local experiment workspace. If Jupyter starts elsewhere,
also set `SCBFM_REPO_DIR` to the checkout. `SCBFM_OUTPUT_DIR` can point to a
separate, locally synced results directory.

| Notebook | Inputs | Purpose |
| --- | --- | --- |
| `task_performances.ipynb` | Downstream metrics, provenance, and epoch curves | Task comparisons, pretraining and convergence figures |
| `distributions.ipynb` | `output/distributions/` | Expression histograms, nonzero-gene counts, and corpus statistics |
| `umap_pca.ipynb` | `output/umap_pca/` | Frozen/fine-tuned classification representations |

Generate [distribution summaries](distributions/README.md) and
[PCA/UMAP coordinates](umap_pca/README.md) on the cluster, then transfer the
compact output directories. The local plotting notebooks do not need the
large source matrices or full fine-tuned checkpoints.

Figures are written under `output/figures/` by default. Set
`SCBFM_FIGURE_DIR=/path/to/figures` before starting the notebook kernel to use a
different destination. Nothing is written into a thesis repository implicitly.
Restart the kernel after changing environment variables.

Run notebook cells in order. Missing or incompatible benchmark runs can prevent
a comparison from being plotted: retain associated metadata and fold manifests,
not only the aggregate CSVs. Saved cell outputs are cleared for publication;
regenerate them locally when the corresponding results are available.
