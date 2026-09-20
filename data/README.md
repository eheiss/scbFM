# Data Preparation

Set `SCBFM_ROOT_DIR` before running scripts or starting a notebook kernel. Inputs
and generated matrices live under `$SCBFM_ROOT_DIR/datasets`, outside this
repository. No expression matrices, clinical records, or weights are bundled.
Obtain source data under the respective providers' access and reuse terms.

## Prepared Inputs

The Hydra defaults expect the following relative paths under `datasets/`.
Preparation notebooks document their source-file names and transformations;
they are not an automatic download service.

| Resource | Prepared input | Preparation |
| --- | --- | --- |
| TCGA | `TCGA/tcga.h5ad` | `notebooks/tcga.ipynb` |
| DiSignAtlas | `DiSignAtlas/disignatlas.h5ad` | `notebooks/disignatlas.ipynb` |
| DepMap | `DepMap/depmap.h5ad` | `notebooks/depmap.ipynb` |
| GDSC | `GDSC/gdsc.h5ad`, `drug_response_prediction_IC50.csv`, `drug_features.npz` | `notebooks/gdsc.ipynb`, `generate_drug_features.py` |
| SurvBoard | `SurvBoard/data_reproduced/` | Upstream preprocessed cohort data and published split files |
| GTEx | `GTEx/gtex.h5ad` | `notebooks/gtex.ipynb` |
| ARCHS4 | `ARCHS4/archs4.h5ad`, `human_gene_v2.latest.h5` | [notebooks/archs4](notebooks/archs4/README.md) |
| Single-cell tasks | `scgpt_single_cell/{covid,lung}/` | [Dataset files and protocol](../src/finetune/batch_integration/README.md) |
| Deconvolution | `pseudo_bulk/pseudo_bulk_RAW.h5ad` | Local download and offline generation below |

GDSC's three files belong in the same `GDSC/` directory. SurvBoard also requires
the matching train/test split CSVs; a harmonized expression matrix alone is not
sufficient. Task-specific column names and file overrides are declared in
`src/configs/finetune/`. Preserve labels, sample identifiers, and target arrays
when transferring `.h5ad` files.

The bundled `gene_list.txt` defines the controlled 13,004-gene vocabulary;
`bulkformer_gene_info.csv` supplies identifier mappings. Regenerating or
reordering a vocabulary changes checkpoint compatibility. Published-model
evaluations can require the original, larger-vocabulary input files as well as
the harmonized matrices. See [external-model setup](../src/README.md).

## Pretraining Corpora

Run preparation from the repository root in a suitable scientific Python
environment. The bulk builder reads GTEx and ARCHS4, applies the recorded
leakage filters, and writes the controlled pretraining/pre-adaptation split:

```bash
python data/create_pretrain_data_bulk_RAW.py
```

Prepare the downstream metadata first: TCGA, DepMap, GDSC, DiSignAtlas, and the
LINCS metadata source are used for exclusions. Missing required inputs fail
validation rather than silently disabling filtering. The builder records its
audit outputs under `datasets/bulk/` and checks the expected pretraining count;
different source releases may therefore require an explicit, justified config
change. `SCBFM_GTEX_H5AD`, `SCBFM_ARCHS4_H5AD`, and the other environment settings
at the top of the script allow input overrides.

`create_pretrain_data_sc_RAW.py` samples the pinned CELLxGENE Census release,
matches the controlled bulk pretraining size, and writes
`datasets/sc/pretraining_sc_RAW.h5ad`. It requires network access for missing
chunks and resumes from validated cached plans and chunks. Use the Census
environment below; `SCBFM_SC_OFFLINE=1` is only valid once all required source
chunks are present.

## Resumable Deconvolution Download

On an internet-connected machine, from this checkout:

```bash
bash data/setup_pseudo_bulk_download_env.sh
bash data/download_pseudo_bulk_broad_sources.sh "$HOME/Downloads/pseudo_bulk" audit
bash data/download_pseudo_bulk_broad_sources.sh "$HOME/Downloads/pseudo_bulk" download
```

The audit checks ontology mappings and eligible source contexts. Download mode
saves source chunks and manifests, not the final pseudo-bulk matrix. Repeating
the same command validates and reuses completed chunks after interruption.
Do not change sampling settings partway through a download.

Transfer the **entire** staging directory, including metadata, the ontology
file, manifests, and chunks, to `$SCBFM_ROOT_DIR/datasets/pseudo_bulk/` on the
cluster. Then validate the transfer in the core benchmark environment:

```bash
SCBFM_PSEUDO_OFFLINE=1 SCBFM_PSEUDO_VALIDATE_TRANSFER_ONLY=1 \
  python data/create_pseudo_bulk_data_RAW.py
```

After successful validation, generate the matrix without internet access:

```bash
SCBFM_PSEUDO_OFFLINE=1 python data/create_pseudo_bulk_data_RAW.py
```

For Slurm, set `SCBFM_ENTRYPOINT` to this script and use the portable launcher
with one process and sufficient CPU memory; see [cluster setup](../cluster/README.md).

The default generator targets 20,000 profiles of 1,000 cells, with 2--8 active
types per mixture. `deconv_broad_cell_types.csv` specifies 27 candidate ontology
groups; support filters determine the retained target set. Composition draws
use a symmetric Dirichlet distribution followed by multinomial sampling and
minimum-support checks. The generated manifests, rather than the candidate
count alone, define the labels of a particular dataset. Keep these files with
the final matrix; deconvolution runners validate its provenance.
