# scGPT single-cell benchmarks

The single-cell benchmark uses the official scGPT zero-shot reference-mapping
and integration datasets instead of the CancerFoundation malignant-cell-only
experiment. It evaluates whether bulk pre-adaptation changes frozen single-cell
representations without allowing target-dataset fine-tuning to relearn them.

## Datasets

The official processed files come from the scGPT repository's
[`data/README.md`](https://github.com/bowang-lab/scGPT/blob/main/data/README.md)
and zero-shot tutorials.

COVID-19, derived from Lotfollahi et al., contains 18 batches. Its official
reference/query split contains 15,997 and 4,003 cells:

- Google Drive folder:
  <https://drive.google.com/drive/folders/1jSPoPunGQOmd71vDsK0FS7UvmDhGdhQS>
- `batch_covid_subsampled_train.h5ad`
- `batch_covid_subsampled_test.h5ad`

The official Lung-Kim reference/query split contains 30,472 cells from 14
primary lung adenocarcinoma samples:

- Google Drive folder:
  <https://drive.google.com/drive/folders/1gbfO7VqxCOkfzgHAih6hO88zFv6pd8wO>
- `sample_proc_lung_train.h5ad`
- `sample_proc_lung_test.h5ad`

On the cluster, the required layout is:

```text
/cluster/work/boeva/eheiss/datasets/scgpt_single_cell/
├── covid/
│   ├── batch_covid_subsampled_train.h5ad
│   └── batch_covid_subsampled_test.h5ad
└── lung/
    ├── sample_proc_lung_train.h5ad
    └── sample_proc_lung_test.h5ad
```

The preparation script reproduces this layout:

```bash
python3 -m pip install --user gdown
cd /cluster/work/boeva/eheiss/job_files/finetune/batch_integration
bash prepare_scgpt_single_cell_data.sh
```

Manual downloads are also valid as long as the four files are stored at the
paths above. The jobs validate every path before allocating model runtime.

## Shared protocol

For each dataset, the runner:

1. maps official `gene_name` symbols to the fixed 13,004-gene scbFM vocabulary;
2. computes MAD using reference cells only and selects one fixed set of 1,199
   token positions shared by every model and both tasks;
3. applies the same 51-bin expression encoding and `[CLS]` input used by scbFM;
4. extracts L2-normalized frozen final `[CLS]` embeddings; and
5. evaluates random initialization, the four controlled scbFM checkpoints,
   downloaded scGPT, preadapted scGPT, and a raw-expression PCA baseline.

COVID-19 is already processed in the official split. Lung-Kim receives the
official reference-mapping normalization (`normalize_total(1e4)` followed by
`log1p`). Query labels never influence gene selection or any fitted method.

The official COVID-19 split supplies 1,200 preselected genes and the Lung-Kim
split supplies 3,000. If fewer than 1,199 of those genes map to the scbFM
vocabulary, the runner fills the remaining fixed input positions with zero
expression. It records the source-present and zero-filled counts in both the
selected-gene manifest and run metadata. External scGPT must match at least 90%
of the source-present genes or the run fails instead of reporting an
under-matched comparison.

The reference-only MAD rule is an intentional adaptation to this thesis
benchmark. The scGPT tutorial uses its own model-specific tokenization/HVG
choices, while the controlled scbFM checkpoints require exactly 1,199 genes.
Using one reference-derived set prevents target leakage and keeps the model
comparison controlled.

## Cell-type annotation

`finetune.cell_type_annotation` follows scGPT's zero-shot reference mapping:

- ten Euclidean nearest neighbours in the frozen embedding space;
- majority-vote transfer from labelled reference cells to query cells;
- Macro F1 as the primary scGPT metric;
- accuracy, macro precision, macro recall, and weighted F1 as additional
  benchmark metrics.

There are no trained heads, folds, adapters, full fine-tuning, validation
epochs, or checkpoint selection. Outputs are under
`output/cell_type_annotation/zero_shot_scgpt/`.

## Batch integration

`finetune.batch_integration` combines the reference and query cells after gene
selection and reports scGPT's exact score definitions:

```text
AvgBIO   = mean(NMI-cell, ARI-cell, ASW-cell)
AvgBATCH = mean(ASW-batch, GraphConn)
Overall  = 0.6 * AvgBIO + 0.4 * AvgBATCH
```

Individual metrics, embeddings, UMAP coordinates, plots, and pre-adaptation
deltas are retained. Outputs are under
`output/batch_integration/zero_shot_scgpt/`.

## Environment and submission

Build the separate image once from the definition file. This does not modify
`scbfm.sif` or `scgpt.sif`:

```bash
cd /cluster/work/boeva/eheiss/scbFM
singularity build --fakeroot \
  /cluster/customapps/biomed/boeva/eheiss/singularity/scbfm_single_cell.sif \
  cluster/scbfm_single_cell.def
```

The cluster image used by the submission files is:

```text
/cluster/customapps/biomed/boeva/eheiss/singularity/scbfm_single_cell.sif
```

Submit the two independent tasks:

```bash
sbatch cell_type_annotation-job.sh
sbatch batch_integration-job.sh
```

The primary catastrophic-forgetting quantity in both tasks is
`preadapt_sc - pretrain_sc`. Negative annotation deltas indicate worse
performance. Integration must be interpreted from `AvgBIO` and `AvgBATCH`
together because batch mixing alone can reward biologically collapsed
embeddings.
