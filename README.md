# scbFM

Code for **Single-Cell Foundation Models for Bulk Transcriptomics: Evaluation
and Adaptation**.

The benchmark compares otherwise identical transformers pretrained on scRNA-seq
or bulkRNA-seq, with and without bulkRNA-seq pre-adaptation. It includes random
initialization, direct-expression MLP and PCA+random-forest baselines, frozen
scGPT and BulkFormer evaluations, and complementary single-cell analyses.
Datasets, pretrained weights, result files, and the thesis itself are not bundled.

## Setup

Use Python 3.11. From this checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`requirements.txt` contains the core benchmark dependencies;
`requirements-dev.txt` also includes testing and local plotting tools. These
files describe the public setup, not a complete lockfile of the original runs.
Large experiments use CUDA and distributed PyTorch. Separate container
definitions are provided for scGPT, BulkFormer, single-cell metrics, and KPGT
drug features; see [cluster/README.md](cluster/README.md).

## Configure Paths

Set one storage root containing the datasets, generated outputs, and external
model repositories:

```bash
export SCBFM_ROOT_DIR=/path/to/experiment-workspace
```

The default is the parent of this checkout, preserving the original layout:

```text
experiment-workspace/
  scbFM/                  # this repository; may also be stored elsewhere
  datasets/               # prepared inputs
  output/                 # checkpoints, metrics, analysis summaries
  other/                  # external scGPT/BulkFormer/KPGT code and weights
  singularity/            # locally built .sif images
  job_output/             # scheduler logs
```

The `root_dir` variable in [`src/main.py`](src/main.py) is the configuration
point. It can also be overridden for one invocation:

```bash
python src/main.py root_dir=/another/workspace --cfg job --resolve
```

This prints the resolved configuration without running an experiment. Bundled
gene lists and mapping files always resolve relative to the checkout. Override
individual dataset/checkpoint paths through Hydra when your layout differs.
`output_dir=/path/to/results` or `SCBFM_OUTPUT_DIR` changes the result location.
`SCBFM_FIGURE_DIR` changes the local figure destination, which defaults to
`$SCBFM_ROOT_DIR/output/figures`. Absolute storage paths are recommended.

## Experiment Order

1. Prepare the data following [data/README.md](data/README.md).
2. Pretrain the controlled scRNA-seq and bulkRNA-seq models independently.
3. Pre-adapt each checkpoint on the same held-out bulkRNA-seq corpus.
4. Run downstream tasks and direct-expression baselines. The baselines can run
   independently of pretraining.
5. Optionally run published-model comparisons and single-cell retention tasks.
6. Generate compact analysis files on the cluster and render figures locally.

Controlled pretraining, using the thesis's four-process setup:

```bash
torchrun --standalone --nproc_per_node=4 src/main.py \
  task=pretrain pretrain.model_name=pretrain_sc \
  pretrain.data_path="$SCBFM_ROOT_DIR/datasets/sc/pretraining_sc_RAW.h5ad"

torchrun --standalone --nproc_per_node=4 src/main.py \
  task=pretrain pretrain.model_name=pretrain_bulk \
  pretrain.data_path="$SCBFM_ROOT_DIR/datasets/bulk/pretraining_bulk_RAW.h5ad"
```

Pre-adaptation starts a new optimizer schedule from the pretrained weights:

```bash
torchrun --standalone --nproc_per_node=4 src/main.py \
  task=pretrain pretrain.model_name=preadapt_sc \
  pretrain.data_path="$SCBFM_ROOT_DIR/datasets/bulk/preadapt_bulk_RAW.h5ad" \
  pretrain.resume_checkpoint="$SCBFM_ROOT_DIR/output/pretrain_sc/pretrain_sc.pth" \
  pretrain.resume_optimizer_state=false
```

For the bulk control, replace `preadapt_sc` and `pretrain_sc` with `preadapt_bulk`
and `pretrain_bulk`. When resuming an interrupted stage, supply that stage's
checkpoint and use `pretrain.resume_optimizer_state=true`.

## Downstream Benchmark

| Task key | Endpoint |
| --- | --- |
| `canc_type_class` | Five-type cancer classification |
| `canc_type_class_33` | 33-type cancer classification |
| `disease_class` | Disease classification |
| `drug_resp` | Drug-response prediction |
| `gene_essent` | Gene-essentiality prediction |
| `surv_pred` | Pan-cancer time-to-event prediction |
| `surv_pred_survboard` | Cohort-specific SurvBoard survival prediction |
| `surv_pred_binary` | Binary vital-status prediction |
| `deconv` | Cell-type deconvolution |

Each task has a configuration in `src/configs/finetune/`. The neural regimes are
`head_only`, `adapters`, and `full_ft`. For example:

```bash
torchrun --standalone --nproc_per_node=4 src/main.py \
  task=finetune.canc_type_class \
  finetune.canc_type_class.finetune_mode=full_ft
```

By default, controlled runners evaluate the four pretrained/pre-adapted states
and random initialization. Restrict a run with
`'finetune.canc_type_class.model_keys=[random_init]'`. Baseline task names append
`_raw_mlp`, `_raw_pca_rf`, `_pca_rf`, `_scgpt_pca_rf`, or `_bulkformer_pca_rf`:

```bash
torchrun --standalone --nproc_per_node=4 src/main.py \
  task=finetune.canc_type_class_raw_mlp \
  finetune.canc_type_class.raw_mlp_feature_mode=all_genes

python src/main.py task=finetune.canc_type_class_raw_pca_rf \
  finetune.canc_type_class.raw_pca_rf_feature_mode=hvg1199
```

The historical `hvg1199` option denotes the training-fold MAD-selected gene set.
For SurvBoard, also set `finetune.surv_pred_survboard.cancer=BRCA` (or another
eligible cohort); each cohort uses its supplied 25 outer splits. See
[src/README.md](src/README.md) for outputs and external-model examples.

## Analysis and Repository Layout

| Location | Contents |
| --- | --- |
| [data/](data/README.md) | Source-data preparation, resumable Census downloads, gene mappings |
| [src/](src/README.md) | Backbone, Hydra entry point, training and downstream runners |
| [src/analysis/](src/analysis/README.md) | Local figures and cluster-side summary/coordinate generation |
| [cluster/](cluster/README.md) | Container definitions and portable Slurm launcher |
| [tests/](tests/README.md) | Synthetic-data regression and configuration checks |

Fold manifests, training curves, and run metadata are part of the experimental
outputs. Keep them together when transferring results. Training and extraction
are substantial jobs; the test suite does not download corpora or run the full
benchmark. Notebook outputs and Python caches are deliberately excluded from
the repository. External data and model artifacts retain their upstream terms.
