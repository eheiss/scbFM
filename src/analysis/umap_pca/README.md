# Downstream PCA and UMAP analysis

This analysis produces one full-page PCA figure and one full-page UMAP figure
for each of the following tasks:

- five-type cancer classification;
- 33-type cancer classification;
- disease classification.

## Figure layout

Each figure has six rows and two columns. The first five rows contain random
initialization, PT-sc, PA-sc, PT-bulk, and PA-bulk. Their columns compare the
initial frozen representation used by head-only training with the representation
after full fine-tuning. The final row contains raw expression using all 13,004
genes and raw expression using the 1,199 training-selected MAD genes.

## Protocol

- All analyses use held-out observations from fold 1 of the existing canonical
  five-fold split.
- The dedicated job trains only fold 1; it does not rerun or overwrite the
  complete downstream benchmark.
- Full fine-tuning uses the benchmark optimizer, burn-in, warm-up, 20-epoch
  schedule, and four-GPU global effective batch size of 64.
- Final fold-1 backbones are retained under `output/umap_pca_backbones`. A
  resubmitted job validates and reuses each completed model before continuing.
- MAD selection uses the complete training fold. Feature standardization, PCA,
  and UMAP are fitted without held-out observations. PCA retains up to 50
  components; UMAP is fitted on the resulting training PCA scores.
- At most 10,000 training observations fit each reducer and at most 5,000
  held-out observations are plotted. Subsampling is deterministic and
  approximately class-stratified.

## Cluster execution

From the cluster repository root:

```bash
export SCBFM_ROOT_DIR=/path/to/experiment-workspace
export SCBFM_SIF="$SCBFM_ROOT_DIR/singularity/scbfm.sif"
export SCBFM_ENTRYPOINT="$PWD/src/analysis/umap_pca/generate_coordinates.py"
export SCBFM_NPROC=4
for task in canc_type_class canc_type_class_33 disease_class; do
  bash cluster/submit.sh --task "$task"
done
```

The three jobs are independent. Compact coordinate archives, provenance JSON,
and fold-1 training curves are written to `output/umap_pca`. Only that directory
needs to be synced to the local machine; retained backbones do not.

After syncing, run `src/analysis/umap_pca.ipynb` locally. It writes PDF and PNG
versions of all six figures to `output/figures`, or `SCBFM_FIGURE_DIR` when set.
The core environment also needs `umap-learn` (included in the development
requirements). Set Slurm memory/time limits for the dataset sizes in use; the
launcher defaults are not a guarantee that every analysis fits.
