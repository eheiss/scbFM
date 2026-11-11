# Data preparation

Ready-to-use harmonized datasets, the required directory structure, and links
to the original data providers are maintained in the
[scbFM data repository](https://huggingface.co/datasets/eheiss/scbFM_data).
Download all hosted files with:

```bash
python -m pip install --upgrade huggingface_hub
hf download eheiss/scbFM_data --repo-type dataset \
  --local-dir "$SCBFM_ROOT_DIR/datasets"
```

The files in this directory reproduce the preprocessing when starting from the
primary sources instead. Set `SCBFM_ROOT_DIR` to the workspace containing
`datasets/`, run commands from the scbFM checkout, and place each source file at
the path specified by the data repository. Large conversions and corpus builds
are intended for the cluster; CELLxGENE downloads require an internet-connected
machine and can subsequently be transferred to an offline cluster.

The bundled `gene_list.txt` defines the controlled 13,004-gene vocabulary, and
`bulkformer_gene_info.csv` provides gene-identifier mappings. Changing the
contents or order of the controlled vocabulary breaks checkpoint compatibility.

## Preparation components

| Path | Purpose |
| --- | --- |
| `notebooks/tcga.ipynb` | Harmonize TCGA expression and phenotype data |
| `notebooks/disignatlas.ipynb` | Harmonize disease-classification data |
| `notebooks/depmap.ipynb` | Harmonize gene-essentiality data |
| `notebooks/gdsc.ipynb` | Harmonize drug-response expression and targets |
| `notebooks/gtex.ipynb` | Convert GTEx source data to the bulk-corpus intermediate |
| `notebooks/create_gene_list.ipynb` | Audit source vocabularies and regenerate the shared gene list |
| `notebooks/archs4/archs4.py` and `archs4_2.py` | Convert ARCHS4 HDF5 data to the required AnnData intermediate |
| `preprocess_raw_h5ad.py` | Apply gene reindexing, filtering, and expression binning to a raw-count AnnData file |
| `create_pretrain_data_bulk_RAW.py` | Build the (optionally leakage-filtered) bulk pretraining corpus |
| `create_pretrain_data_sc_RAW.py` | Sample the matched scRNA-seq pretraining corpus from CELLxGENE Census |
| `create_pseudo_bulk_data_RAW.py` | Generate the cell-type deconvolution dataset |
| `generate_drug_features.py` | Generate KPGT molecular representations for GDSC drugs |

The notebooks use the same `SCBFM_ROOT_DIR` workspace as the runners. Start
Jupyter inside the checkout, or additionally set `SCBFM_REPO_DIR` when starting
it elsewhere. Run notebook cells in order.

### Generic AnnData preprocessing

For a raw-count `.h5ad` whose variables use the controlled gene identifiers:

```bash
python data/preprocess_raw_h5ad.py \
  --input /path/to/raw.h5ad \
  --output /path/to/preprocessed.h5ad
```

The command reindexes to `data/gene_list.txt`, filters profiles with too few
detected genes, and applies the configured scGPT-style expression binning. Run
`python data/preprocess_raw_h5ad.py --help` for overrides.

### ARCHS4

Place the upstream ARCHS4 HDF5 file under `datasets/ARCHS4/`. The two scripts in
`data/notebooks/archs4/` perform the extraction and subsequent harmonization;
their Slurm wrappers are `run_archs4.sh` and `run_archs4_2.sh`. Run the second
stage only after the first has completed. The scripts read `SCBFM_ROOT_DIR`,
`SCBFM_REPO_DIR`, and `SCBFM_SIF`.

## Controlled pretraining corpora

### BulkRNA-seq

After preparing GTEx, ARCHS4, TCGA, DepMap, GDSC, and DiSignAtlas at the paths
listed in the data repository, run:

```bash
python data/create_pretrain_data_bulk_RAW.py
```

The builder combines GTEx and ARCHS4, excludes GTEx donors and downstream
dataset identifiers from the pretraining source, and writes a single file (no
pretrain/preadapt split):

```text
datasets/bulk/pretraining_bulk_RAW.h5ad
```

ARCHS4 rows carry `instrument_model` from the ARCHS4 metadata HDF5. Options:

| Variable | Default | Effect |
|---|---|---|
| `SCBFM_BULK_SOURCES` | `GTEx,ARCHS4` | Set to `ARCHS4` to build an ARCHS4-only corpus |
| `SCBFM_BULK_LEAKAGE_FILTER` | `1` | Set to `0` to keep GTEx-donor and TCGA/DepMap/GDSC/DiSignAtlas matches |
| `SCBFM_BULK_EXPECTED_SAMPLES` | unset | If set, fail unless exactly this many profiles survive |

With the leakage filter on, it also writes leakage-audit artifacts.
Input and output paths can be overridden through the `SCBFM_GTEX_H5AD`,
`SCBFM_ARCHS4_H5AD`, `SCBFM_ARCHS4_METADATA_H5`, and `SCBFM_BULK_OUT_DIR`
environment variables defined near the top of the script.

### scRNA-seq

`create_pretrain_data_sc_RAW.py` samples the pinned CELLxGENE Census release to
match the controlled bulk pretraining size:

```bash
python data/create_pretrain_data_sc_RAW.py
```

The script writes validated sampling plans and source chunks below
`datasets/sc/` and resumes them by default. Download missing chunks on a machine
with internet access. After transferring the complete directory to the cluster,
set `SCBFM_SC_OFFLINE=1` to prohibit network access while assembling
`datasets/sc/pretraining_sc_RAW.h5ad`.

## Pseudo-bulk deconvolution data

Create the pinned Census environment and audit the available ontology groups on
an internet-connected machine:

```bash
bash data/setup_pseudo_bulk_download_env.sh
bash data/download_pseudo_bulk_broad_sources.sh \
  "$HOME/Downloads/pseudo_bulk" audit
```

Download the source cells in resumable chunks:

```bash
bash data/download_pseudo_bulk_broad_sources.sh \
  "$HOME/Downloads/pseudo_bulk" download
```

Repeating the command validates and reuses completed chunks. Transfer the
entire staging directory, including the ontology file, manifests, sampling
plan, metadata, and `source_cell_chunks/`, to
`$SCBFM_ROOT_DIR/datasets/pseudo_bulk/`. Validate the transfer on the cluster:

```bash
SCBFM_PSEUDO_OFFLINE=1 \
SCBFM_PSEUDO_VALIDATE_TRANSFER_ONLY=1 \
python data/create_pseudo_bulk_data_RAW.py
```

Then generate the final matrix without internet access:

```bash
SCBFM_PSEUDO_OFFLINE=1 python data/create_pseudo_bulk_data_RAW.py
```

The default generator creates 20,000 profiles from 1,000 cells each, with 2--8
active broad cell types. Candidate ontology mappings are defined in
`deconv_broad_cell_types.csv`; support filters and the generated manifest define
the final target set. Mixture proportions use a symmetric Dirichlet draw,
followed by multinomial cell sampling and minimum-support checks.

Each pseudobulk has a 1-based `pseudobulk_id`. The cells summed into each one are
listed in `pseudo_bulk_source_membership.csv` (`pseudobulk_id`, `soma_joinid`,
`n_times`), and `sampled_source_cells_aligned.h5ad` gets the per-cell columns
`pseudobulk_ids` (";"-joined), `pseudobulk_id` (smallest ID, 0 if unused),
`n_pseudobulks` and `is_pseudobulk_source`. Source cells are reused across
pseudobulks, so one cell usually has several IDs.

To also merge the source cells into the single-cell corpus, set
`SCBFM_PSEUDO_MERGE_INTO_SC=1`. This reads `SCBFM_SC_FINAL_H5AD` (default
`datasets/sc/pretraining_sc_RAW.h5ad`) and writes `SCBFM_PSEUDO_MERGED_SC_OUT`
(default `datasets/sc/pretraining_sc_with_pb_sources_RAW.h5ad`). Rows are
deduplicated by `soma_joinid` with the source row winning, and non-source cells
get `pseudobulk_id = 0`.

## Drug features

Clone KPGT under `$SCBFM_ROOT_DIR/other/KPGT`, download its pretrained base
checkpoint, and use the `kpgt.sif` image to generate the molecular feature file:

```bash
python data/generate_drug_features.py \
  --ic50_path "$SCBFM_ROOT_DIR/datasets/GDSC/drug_response_prediction_IC50.csv" \
  --output_dir "$SCBFM_ROOT_DIR/datasets/GDSC" \
  --kpgt_root "$SCBFM_ROOT_DIR/other/KPGT" \
  --model_path "$SCBFM_ROOT_DIR/other/KPGT/pretrained/base/base.pth"
```

## Published-model and single-cell inputs

Full-vocabulary BulkFormer evaluations use the primary TCGA, DiSignAtlas,
DepMap, and GDSC files listed in the data repository rather than only the
13,004-gene harmonized matrices.

The zero-shot single-cell tasks use the official scGPT reference/query splits:

- [COVID-19 files](https://drive.google.com/drive/folders/1jSPoPunGQOmd71vDsK0FS7UvmDhGdhQS):
  `batch_covid_subsampled_train.h5ad` and
  `batch_covid_subsampled_test.h5ad`.
- [Lung-Kim files](https://drive.google.com/drive/folders/1gbfO7VqxCOkfzgHAih6hO88zFv6pd8wO):
  `sample_proc_lung_train.h5ad` and `sample_proc_lung_test.h5ad`.

Store them as:

```text
$SCBFM_ROOT_DIR/datasets/scgpt_single_cell/
|-- covid/
|   |-- batch_covid_subsampled_train.h5ad
|   `-- batch_covid_subsampled_test.h5ad
`-- lung/
    |-- sample_proc_lung_train.h5ad
    `-- sample_proc_lung_test.h5ad
```