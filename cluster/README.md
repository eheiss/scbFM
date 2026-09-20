# Cluster Execution

No cluster account, username, or storage mount is built into the launcher.
Set `SCBFM_ROOT_DIR` to the workspace containing `datasets/`, `output/`, and
`other/`. Existing site-specific submission scripts outside this repository
are not required by the public entry point.

## Containers

| Definition | Purpose |
| --- | --- |
| `scbfm.def` | Controlled pretraining, fine-tuning, direct-expression baselines |
| `scgpt.def` | Native scGPT pre-adaptation and frozen scGPT extraction |
| `scbfm_single_cell.def` | Single-cell metrics and PCA/UMAP analysis |
| `bulkformer.def` | BulkFormer extraction with PyTorch Geometric |
| `kpgt.def` | Drug feature generation |

Build on a machine with internet access, then transfer the image to an offline
cluster. These commands use Singularity; sites using Apptainer can substitute
`apptainer`. Run the first commands from this repository:

```bash
mkdir -p "$SCBFM_ROOT_DIR/singularity"
singularity build --fakeroot "$SCBFM_ROOT_DIR/singularity/scbfm.sif" cluster/scbfm.def
singularity build --fakeroot "$SCBFM_ROOT_DIR/singularity/scgpt.sif" cluster/scgpt.def
```

The single-cell definition extends a local `scgpt.sif`. Build it from the
directory containing that image; point the definition argument at this checkout:

```bash
export SCBFM_REPO_DIR="$PWD"
cd "$SCBFM_ROOT_DIR/singularity"
singularity build --fakeroot scbfm_single_cell.sif \
  "$SCBFM_REPO_DIR/cluster/scbfm_single_cell.def"
```

Building a new image does not modify the base image. Existing compatible images
can be reused. Some sites use a remote builder or another build privilege model
instead of `--fakeroot`. External model repositories and weights must also be
transferred; container definitions only install dependencies.

## Slurm

From the repository root, after loading the site's container module:

```bash
export SCBFM_ROOT_DIR=/path/to/workspace
export SCBFM_SIF=/path/to/scbfm.sif
bash cluster/submit.sh task=finetune.canc_type_class \
  finetune.canc_type_class.finetune_mode=full_ft
```

`submit.sh` exports the actual checkout path and creates the log directory
before submitting `run-job.sh`. Arguments are passed unchanged to Hydra.
No jobs are submitted by installing or importing the repository.

The default request is one node, four GPUs, 16 CPUs, 64 GB RAM, and 12 hours.
Adjust resources for the task and site. Historical deconvolution jobs requested
at least 128 GB and BulkFormer survival jobs 192 GB; source-data generation can
require 512 GB or more. To override Slurm directives explicitly:

```bash
export SCBFM_REPO_DIR="$PWD"
mkdir -p "$SCBFM_ROOT_DIR/job_output"
sbatch --partition=gpu --mem=192G --export=ALL \
  --output="$SCBFM_ROOT_DIR/job_output/%x.%j.out" \
  --error="$SCBFM_ROOT_DIR/job_output/%x.%j.err" \
  cluster/run-job.sh task=finetune.deconv
```

`SCBFM_NPROC` controls distributed worker count (default 4). Match it to the GPU
request; changing it changes the global effective batch size unless training
settings are adjusted too. PCA+RF on raw expression uses one process. For a CPU
allocation, request no GPUs and set `SCBFM_USE_GPU=0` and `SCBFM_NPROC=1`.

`SCBFM_ENTRYPOINT` can select a different Python script, for example a coordinate
generator. The launcher binds both the checkout and workspace into the image.
Keep other required artifacts under those roots or add explicit container binds.

Submit the two controlled pretraining jobs independently. Each pre-adaptation
job depends on its own pretraining checkpoint. Full controlled downstream runs
need all four checkpoints; raw baselines need none. SurvBoard additionally needs
one job per eligible cohort. There is no automatic resubmission or scheduler
dependency graph in this generic launcher.
