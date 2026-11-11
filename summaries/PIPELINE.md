# scbFM Pretraining Data Pipeline

This document summarizes how the three pretraining data modalities (**single-cell**,
**pseudobulk**, and **real bulk**) are generated, and what fields to expect in the
resulting `.h5ad` files.

All three producer scripts emit **raw counts**. Tokenization (gene reindexing,
min-gene filtering, scGPT-style quantile binning) is a separate downstream step
(`preprocess_raw_h5ad.py`), described at the end.

---

## Common foundations

Every pipeline shares the same spine:

| Aspect | Value |
|---|---|
| **Gene axis** | `data/gene_list.txt` (ordered Ensembl IDs). All outputs reindex/align to this exact list, so `var_names` is identical across every file. |
| **`var`** | Empty `DataFrame` whose index is named `ensembl_id`. |
| **Quality filter** | `MIN_GENES = 200`: a sample is dropped unless ≥200 genes are non-zero. |
| **`X`** | Raw counts, sparse CSR, `float32`. No normalization/binning at this stage. |
| **Sampling weighting** | `√(availability)` weighting to avoid large datasets/contexts dominating; reservoir sampling for a fixed memory footprint. |

---

## 1. Single-cell

**Script:** `create_pretrain_data_sc_RAW.py`
**Output:** `pretraining_sc_RAW.h5ad`
**Source:** CELLxGENE Census (`2025-11-08`), human, `is_primary_data == True` cells only.

### Pipeline
1. **Count** cells per `(dataset_id, tissue_general, cell_type)` group.
2. **Allocate quotas** toward `TARGET_TOTAL_CELLS` (default 642,406): split across datasets
   by √-weighting, then across tissue×cell_type groups within each dataset.
3. **Reservoir-sample** cell IDs (`soma_joinid`) to fill each group quota → `sampling_plan_RAW.csv`.
4. **Download** expression in chunks of 5,000 cells, reorder genes to `gene_list`,
   re-apply the ≥200-gene filter per batch, truncate to the exact target.
5. **Merge** chunks pairwise into the final file.
6. **Overdraw logic:** samples 1.10× the target and doubles the factor across up to 4
   attempts, since the min-gene filter drops cells after download.

On resume, cached chunks and final files without an `assay` column are re-downloaded or
rebuilt. The sampling plan is reused, so the cells are the same.

### `obs` fields
`soma_joinid`, `dataset_id`, `cell_type`, `tissue_general`, `donor_id`, `assay`,
`dataset` (= `"CELLxGENE_Census"`).
Index (`cell_id`) = `cellxgene:<soma_joinid>`.

---

## 2. Pseudobulk

**Script:** `create_pseudo_bulk_data_RAW.py`
**Output:** `pseudo_bulk/pseudo_bulk_RAW.h5ad`
**Source:** CELLxGENE Census, with single cells **aggregated (summed) into synthetic bulk samples**.

### Pipeline
1. **Define contexts** = `(dataset_id, donor_id, tissue_general)`. Map cell types to broad
   Cell Ontology targets; keep eligible contexts (≥2 supported cell types with ≥20 cells
   each, ≥100 total cells) → `eligible_contexts.csv`.
2. **Simulate a plan** for `TARGET_PSEUDO_BULKS` (default 20,000) samples of
   `CELLS_PER_PSEUDO_BULK` (default 1,000) cells each: pick a context (√-weighted),
   pick 2 to 8 active cell types, draw Dirichlet proportions, then multinomial counts
   → `pseudo_bulk_sampling_plan.csv`.
3. **Compute source-cell pool quotas** per (context, cell_type) and **reservoir-sample**
   real cells to fill them (`source_cell_pool_quotas.csv`, `sampled_source_cells.csv`).
4. **Download** those source cells once, aligned to `gene_list` (`source_cell_chunks/`).
5. **Generate** each pseudobulk: randomly select the required cells per cell type
   (pools are small, so cells are **reused across pseudobulks** and sometimes within one)
   and **sum their raw counts** into one row. Apply the ≥200-gene filter, then merge.
6. **Track sources:** write the long-format membership table and annotate the source cells
   (see below).
7. *(Optional)* **Merge source cells into the single-cell file** (see below).

### `obs` fields
`pseudobulk_id` (int, **1-based** = plan index + 1; not renumbered if the min-gene filter
drops a row), `dataset` (= `"CELLxGENE_Census_pseudobulk"`), `dataset_id`, `donor_id`,
`tissue_general`, `assay` (`";"`-joined sorted set of the assays of the summed cells),
`total_cells`, `n_cell_types`, and **one `prop__<cell_type>` column per cell type**
(`float32` proportion, `0.0` where absent).
Index (`sample_id`) = `pseudo_bulk:000123` (→ `pseudobulk_id` 124).

### `uns` fields (unique to this file)
- `cell_type_proportion_cell_types`: sorted list of all cell types
- `cell_type_proportion_obs_columns`: matching `prop__*` column names
- `cell_type_proportion_columns`: dict mapping cell_type → column name
- `pseudobulk_id_base` (= 1), `pseudobulk_source_membership_csv`, `output_schema_version`

### Source-cell tracking
- **`pseudo_bulk_source_membership.csv`**: one row per (pseudobulk, cell) with columns
  `pseudobulk_id`, `soma_joinid`, `n_times` (how often the cell was drawn into that
  pseudobulk). `Σ n_times × X[cell]` reproduces the pseudobulk row exactly.
- **`sampled_source_cells_aligned.h5ad`** (all sampled source cells) gets these obs columns:
  - `pseudobulk_ids`: `";"`-joined ascending IDs of every pseudobulk the cell contributed to (`""` if none)
  - `pseudobulk_id`: smallest such ID, `0` if the cell was sampled but never used
  - `n_pseudobulks`: number of pseudobulks the cell contributed to
  - `is_pseudobulk_source`: `True`

  Other source obs: `soma_joinid`, `dataset_id`, `donor_id`, `tissue_general`,
  `cell_type` (broad target), `source_cell_type` (original label),
  `cell_type_ontology_term_id`, `assay`.

### Merging source cells into the single-cell file
Enabled with `SCBFM_PSEUDO_MERGE_INTO_SC=1`. Reads `SCBFM_SC_FINAL_H5AD` (default
`datasets/sc/pretraining_sc_RAW.h5ad`) and writes a **new** file,
`SCBFM_PSEUDO_MERGED_SC_OUT` (default `datasets/sc/pretraining_sc_with_pb_sources_RAW.h5ad`):
- Source rows use the sc schema: `cell_type` = original label, plus `broad_cell_type`.
- Rows are **deduplicated by `soma_joinid`**; when a cell is in both, the source row wins.
- Non-source rows get `pseudobulk_id = 0`, `pseudobulk_ids = ""`, `n_pseudobulks = 0`,
  `is_pseudobulk_source = False`, `broad_cell_type = ""`.
- The counts (kept, replaced, added) are logged under `merged_single_cell` in the summary JSON.

> Written with HDF5 `libver="earliest"` for old-cluster HDF5 (<1.10) compatibility.
> `OUTPUT_SCHEMA_VERSION` invalidates cached pseudobulk chunks/finals without discarding
> cached plans or source downloads.

---

## 3. Real bulk

**Script:** `create_pretrain_data_bulk_RAW.py`
**Output:** `pretraining_bulk_RAW.h5ad` (single file, **no pretrain/preadapt split**)
**Source:** GTEx + ARCHS4 (pre-aligned to `gene_list` by upstream scripts; the loader
validates gene order and errors if not).

### Options
| Variable | Default | Effect |
|---|---|---|
| `SCBFM_BULK_SOURCES` | `GTEx,ARCHS4` | `ARCHS4` → ARCHS4-only corpus (GTEx expression is not loaded) |
| `SCBFM_BULK_LEAKAGE_FILTER` | `1` | `0` → skip GTEx-donor and downstream-benchmark removal |
| `SCBFM_BULK_EXPECTED_SAMPLES` | unset | If set, fail unless exactly this many profiles survive |

### Pipeline
1. **Validate** the selected inputs are already gene-aligned.
2. Apply the ≥200-gene filter.
3. **ARCHS4-specific filters:**
   - Drop single-cell-like samples (`singlecellprobability ≥ 0.5` from the metadata HDF5).
     This always applies, regardless of the leakage option.
   - **Leakage removal** (only if `SCBFM_BULK_LEAKAGE_FILTER=1`): scan all obs metadata text
     for GTEx donor IDs (`GTEX-xxxx`, read from `gtex.h5ad` even in ARCHS4-only mode if it
     exists) and for downstream-benchmark identifiers (TCGA, DepMap/CCLE, GDSC, DiSignAtlas).
     Matched rows are excluded. Hits are logged to `archs4_gtex_donor_hits_RAW.csv` /
     `archs4_downstream_hits_RAW.csv`.
4. **Stream** all surviving rows into resizable CSR datasets (memory-bounded, `lzf` compression).

### `obs` fields
The original GTEx/ARCHS4 obs columns (heterogeneous: outer-joined, so source-specific
columns are `NaN` for the other source), plus:
- `instrument_model` (ARCHS4 rows; from `meta/samples/instrument_model` in the ARCHS4
  metadata HDF5, `"unknown"` if unavailable; `NaN` for GTEx rows)
- `dataset` (= `"GTEx"` or `"ARCHS4"`)
- `bulk_source` (same as `dataset`)
- `bulk_dataset` (= `"pretraining_bulk_RAW"`, kept for compatibility)

Original sample IDs are kept as the index.

> **No tissue harmonization.** Unlike the single-cell/pseudobulk pipelines, there is no
> `tissue_general` field. GTEx rows carry GTEx's native tissue columns (e.g. `SMTS`/`SMTSD`)
> if the upstream `gtex.h5ad` preserved them; ARCHS4 rows carry unharmonized free-text
> metadata.

> ⚠️ `src/pretrain/scgpt_preadapt_runner.py` still expects a `preadapt_bulk_RAW.h5ad`,
> which this script no longer produces.

---

## Tokenization (downstream, shared)

**Script:** `preprocess_raw_h5ad.py` → `src/preprocess.py::preprocess_raw_h5ad`

The `*_RAW.h5ad` files hold raw counts. Before pretraining, each is passed through:
1. **Reindex** to `gene_list` (missing genes zero-filled; `.missing_genes.json` sidecar written).
2. Re-apply the **min-genes** filter.
3. **scGPT-style quantile binning:** token `0` reserved for true zeros; nonzero values
   quantile-binned into `1..bin_num` (default 10). `X` becomes **`uint8` tokens**.

Adds `uns["scbfm_preprocess"]` recording: `input_path`, `original_shape`, `output_shape`,
`gene_list_path`, `missing_genes`, `min_genes`, `bin_num`,
`binning_strategy = "scgpt_nonzero_quantile"`, `zero_token_reserved = True`,
`output_dtype = "uint8"`.

---

## Quick comparison

| | Single-cell | Pseudobulk | Bulk |
|---|---|---|---|
| **Source** | CELLxGENE Census | CELLxGENE (summed cells) | GTEx + ARCHS4 (or ARCHS4 only) |
| **Output** | `pretraining_sc_RAW.h5ad` (+ optional `pretraining_sc_with_pb_sources_RAW.h5ad`) | `pseudo_bulk_RAW.h5ad`, `sampled_source_cells_aligned.h5ad`, `pseudo_bulk_source_membership.csv` | `pretraining_bulk_RAW.h5ad` |
| **`X`** | raw counts, CSR f32 | summed raw counts, CSR f32 | raw counts, CSR f32 |
| **Key `obs`** | dataset_id, cell_type, tissue_general, donor_id, assay | pseudobulk_id, dataset_id, donor_id, tissue_general, assay, total_cells, n_cell_types, `prop__*` | original GTEx/ARCHS4 cols + instrument_model, dataset, bulk_source, bulk_dataset |
| **Technical covariate** | `assay` | `assay` (set of summed cells) | `instrument_model` (ARCHS4) |
| **Tissue field** | `tissue_general` (CELLxGENE ontology) | `tissue_general` (context key) | none (native GTEx/ARCHS4 columns only) |
| **`uns`** | none | cell-type proportion maps, pseudobulk ID metadata | none |
| **Leakage filtering** | `is_primary_data` only | `is_primary_data` only | optional TCGA/DepMap/GDSC/DiSignAtlas + GTEx-donor removal |
| **`var`** | gene_list (`ensembl_id`) | same | same |
