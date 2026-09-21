# scbFM

Code for *Single-Cell Foundation Models for Bulk Transcriptomics: Evaluation
and Adaptation*. The benchmark compares otherwise identical transformers
pretrained on scRNA-seq or bulkRNA-seq, before and after bulkRNA-seq
pre-adaptation. It also evaluates direct-expression baselines and the published
scGPT and BulkFormer models.

- [Processed data and source-data instructions](https://huggingface.co/datasets/eheiss/scbFM_data)
- [Pretrained and pre-adapted checkpoints](https://huggingface.co/eheiss/scbFM_models)
- [Data preparation in this repository](data/README.md)

## Setup

Clone the repository and define a workspace containing datasets, checkpoints,
external repositories, and job outputs:

```bash
git clone https://github.com/eheiss/scbFM.git
cd scbFM
export SCBFM_ROOT_DIR=/path/to/experiment-workspace
```

The default layout is:

```text
$SCBFM_ROOT_DIR/
|-- datasets/
|-- output/
|-- other/
|   |-- scGPT/
|   |-- BulkFormer/
|   |-- KPGT/
|   `-- models/
|-- singularity/
`-- job_output/
```

Download the harmonized data and scbFM checkpoints directly into the paths
expected by the configurations:

```bash
python -m pip install --upgrade huggingface_hub
hf download eheiss/scbFM_data --repo-type dataset \
  --local-dir "$SCBFM_ROOT_DIR/datasets"
hf download eheiss/scbFM_models \
  --local-dir "$SCBFM_ROOT_DIR/output"
```

The data and model cards list additional files that must be obtained from their
original providers, including SurvBoard, the published scGPT whole-human model,
and BulkFormer. Clone the corresponding upstream repositories under `other/`.
Store the original scGPT files as `other/models/{args.json,vocab.json,best_model.pt}`
and the BulkFormer checkpoint as `other/models/BulkFormer_147M.pt`.

### Local environment

Python 3.11 is used for tests and local figure generation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Training and large preprocessing jobs use the container definitions in
`cluster/`. Build the required images on a machine with internet access and
transfer them to an offline cluster. For example:

```bash
mkdir -p "$SCBFM_ROOT_DIR/singularity"
singularity build --fakeroot "$SCBFM_ROOT_DIR/singularity/scbfm.sif" \
  cluster/scbfm.def
singularity build --fakeroot "$SCBFM_ROOT_DIR/singularity/scgpt.sif" \
  cluster/scgpt.def
singularity build --fakeroot "$SCBFM_ROOT_DIR/singularity/bulkformer.sif" \
  cluster/bulkformer.def
singularity build --fakeroot "$SCBFM_ROOT_DIR/singularity/kpgt.sif" \
  cluster/kpgt.def
```

`scbfm_single_cell.def` extends the locally built `scgpt.sif`:

```bash
export SCBFM_REPO_DIR="$PWD"
cd "$SCBFM_ROOT_DIR/singularity"
singularity build --fakeroot scbfm_single_cell.sif \
  "$SCBFM_REPO_DIR/cluster/scbfm_single_cell.def"
cd "$SCBFM_REPO_DIR"
```

The examples below use Slurm through `cluster/submit.sh`. Set `SCBFM_SIF` to the
appropriate image and `SCBFM_NPROC` to the number of workers. The launcher
defaults to four workers and writes logs under `$SCBFM_ROOT_DIR/job_output/`.
Site-specific time, memory, partition, and GPU requirements can be supplied to
`sbatch` by adapting `cluster/submit.sh` or invoking `cluster/run-job.sh`
directly.

## Controlled pretraining

The scRNA-seq and bulkRNA-seq models are trained independently with the same
architecture and self-supervised objective:

```bash
export SCBFM_SIF="$SCBFM_ROOT_DIR/singularity/scbfm.sif"

bash cluster/submit.sh \
  task=pretrain pretrain.model_name=pretrain_sc \
  pretrain.data_path="$SCBFM_ROOT_DIR/datasets/sc/pretraining_sc_RAW.h5ad"

bash cluster/submit.sh \
  task=pretrain pretrain.model_name=pretrain_bulk \
  pretrain.data_path="$SCBFM_ROOT_DIR/datasets/bulk/pretraining_bulk_RAW.h5ad"
```

Checkpoints and diagnostics are written to
`output/<model_name>/<model_name>.pth` and adjacent metric files.

## Controlled pre-adaptation

Each pretrained model is continued on the same held-out bulkRNA-seq corpus with
a reset optimizer schedule:

```bash
bash cluster/submit.sh \
  task=pretrain pretrain.model_name=preadapt_sc \
  pretrain.data_path="$SCBFM_ROOT_DIR/datasets/bulk/preadapt_bulk_RAW.h5ad" \
  pretrain.resume_checkpoint="$SCBFM_ROOT_DIR/output/pretrain_sc/pretrain_sc.pth" \
  pretrain.resume_optimizer_state=false

bash cluster/submit.sh \
  task=pretrain pretrain.model_name=preadapt_bulk \
  pretrain.data_path="$SCBFM_ROOT_DIR/datasets/bulk/preadapt_bulk_RAW.h5ad" \
  pretrain.resume_checkpoint="$SCBFM_ROOT_DIR/output/pretrain_bulk/pretrain_bulk.pth" \
  pretrain.resume_optimizer_state=false
```

For an interrupted stage, resume its own checkpoint with
`pretrain.resume_optimizer_state=true`.

## Downstream fine-tuning

The controlled neural runners support `head_only`, `adapters`, and `full_ft`.
They evaluate random initialization and the four controlled model states by
default. For example:

```bash
bash cluster/submit.sh \
  task=finetune.canc_type_class \
  finetune.canc_type_class.finetune_mode=head_only

bash cluster/submit.sh \
  task=finetune.canc_type_class \
  finetune.canc_type_class.finetune_mode=adapters

bash cluster/submit.sh \
  task=finetune.canc_type_class \
  finetune.canc_type_class.finetune_mode=full_ft
```

Restrict a run with, for example,
`'finetune.canc_type_class.model_keys=[pretrain_sc,preadapt_sc]'`.

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

Replace `canc_type_class` in both the task and override key for another
endpoint. SurvBoard additionally requires a cohort, for example
`finetune.surv_pred_survboard.cancer=BRCA`.

## Baselines

Each bulk task exposes the same suffix-based baseline interface:

| Task suffix | Evaluation |
| --- | --- |
| `_raw_mlp` | MLP trained directly on expression values |
| `_raw_pca_rf` | PCA+RF trained directly on expression values |
| `_pca_rf` | PCA+RF trained on frozen controlled-model representations |
| `_scgpt_pca_rf` | PCA+RF trained on frozen scGPT representations |
| `_bulkformer_pca_rf` | PCA+RF trained on frozen BulkFormer representations |

Examples for the five-type cancer task are:

```bash
# All-gene and 1,199-gene direct-expression MLPs
bash cluster/submit.sh task=finetune.canc_type_class_raw_mlp \
  finetune.canc_type_class.raw_mlp_feature_mode=all_genes
bash cluster/submit.sh task=finetune.canc_type_class_raw_mlp \
  finetune.canc_type_class.raw_mlp_feature_mode=hvg1199

# Direct-expression and controlled-representation PCA+RF
SCBFM_NPROC=1 bash cluster/submit.sh \
  task=finetune.canc_type_class_raw_pca_rf \
  finetune.canc_type_class.raw_pca_rf_feature_mode=all_genes
SCBFM_NPROC=1 bash cluster/submit.sh \
  task=finetune.canc_type_class_pca_rf
```

The historical configuration name `hvg1199` denotes 1,199 genes selected by
training-fold median absolute deviation. Available feature modes and
task-specific paths are defined in `src/configs/finetune/`.

## Published models

Use the downloaded whole-human scGPT model with the scGPT container:

```bash
export SCBFM_SIF="$SCBFM_ROOT_DIR/singularity/scgpt.sif"
SCBFM_NPROC=1 bash cluster/submit.sh \
  task=finetune.canc_type_class_scgpt_pca_rf
```

The bulkRNA-seq-pre-adapted scGPT checkpoint is available from the model
repository. Its original `args.json` and `vocab.json` are still required:

```bash
cp "$SCBFM_ROOT_DIR/other/models/args.json" \
  "$SCBFM_ROOT_DIR/output/scgpt_preadapt/args.json"
cp "$SCBFM_ROOT_DIR/other/models/vocab.json" \
  "$SCBFM_ROOT_DIR/output/scgpt_preadapt/vocab.json"

SCBFM_NPROC=1 bash cluster/submit.sh \
  task=finetune.canc_type_class_scgpt_pca_rf \
  finetune.canc_type_class.scgpt_model_dir="$SCBFM_ROOT_DIR/output/scgpt_preadapt" \
  finetune.canc_type_class.scgpt_args_filename=args.json \
  finetune.canc_type_class.scgpt_vocab_filename=vocab.json \
  finetune.canc_type_class.scgpt_checkpoint_filename=last_model.pt \
  finetune.canc_type_class.scgpt_model_key=scgpt_preadapt \
  finetune.canc_type_class.scgpt_variant=scgpt_preadapt_pca_rf
```

To reproduce native scGPT pre-adaptation instead of downloading its checkpoint:

```bash
export SCBFM_SIF="$SCBFM_ROOT_DIR/singularity/scgpt.sif"
bash cluster/submit.sh \
  pretrain=scgpt_preadapt task=pretrain.scgpt_preadapt \
  "pretrain.data_paths=[$SCBFM_ROOT_DIR/datasets/bulk/pretraining_bulk_RAW.h5ad,$SCBFM_ROOT_DIR/datasets/bulk/preadapt_bulk_RAW.h5ad]" \
  pretrain.audit_only=true
```

After inspecting the audit, rerun with `pretrain.audit_only=false`. Interrupted
runs accept `pretrain.resume_state_path=<training_state.pt>`.

Use the BulkFormer container for the corresponding published bulk model:

```bash
export SCBFM_SIF="$SCBFM_ROOT_DIR/singularity/bulkformer.sif"
SCBFM_NPROC=1 bash cluster/submit.sh \
  task=finetune.canc_type_class_bulkformer_pca_rf
```

The zero-shot single-cell retention tasks use the dedicated image and the
datasets documented in `data/README.md`:

```bash
export SCBFM_SIF="$SCBFM_ROOT_DIR/singularity/scbfm_single_cell.sif"
bash cluster/submit.sh task=finetune.cell_type_annotation
bash cluster/submit.sh task=finetune.batch_integration
```

## Outputs and analysis

Downstream runs write metrics, fold results, training curves, and run metadata
under `output/<task>/<variant>/`. SurvBoard inserts the cohort before the
variant. Keep metadata and fold manifests with aggregate results; they record
completion status and the data/fold fingerprints required for valid
comparisons.

The notebooks in `src/analysis/` render benchmark, distribution, and PCA/UMAP
figures from completed outputs. Cluster-side generators under
`src/analysis/distributions/` and `src/analysis/umap_pca/` create compact
summaries that can be transferred to a local machine before plotting.
