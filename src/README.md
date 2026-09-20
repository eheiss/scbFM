# Training and Evaluation

`main.py` selects a runner using `task=...`. `configs/config.yaml` composes the
default backbone, task configurations, and storage paths. Configuration keys
can be inspected with `python src/main.py --cfg job --resolve` from the repo root.

## Outputs

Controlled pretraining writes
`output/<model_name>/<model_name>.pth`, epoch metrics, and bin diagnostics.
Native scGPT pre-adaptation writes `output/scgpt_preadapt/last_model.pt`,
`args.json`, `vocab.json`, an audit, and a resumable `training_state.pt`.

Downstream results use `output/<task>/<variant>/`; SurvBoard adds a cohort
directory between task and variant. Files include evaluation metrics, per-fold
results/training curves where applicable, and run metadata. The metadata records
completion and data/fold fingerprints. Match those fingerprints when comparing
runs. A file's existence alone does not establish that the run completed.
Result directories are reused by some runners on resubmission; use an explicit
output suffix or separate `output_dir` for a distinct experiment.

The canonical fold manifests live beside their datasets. Reuse them across
model variants. The drug-response default explicitly names the historical
manifest; for a newly prepared dataset, `finetune.drug_resp.cv_fold_manifest_path=auto`
generates/reuses a manifest for that dataset instead.

## Published Models

Use the corresponding container in [../cluster](../cluster/README.md).
External repositories and weights are obtained separately. Default locations:

```text
$SCBFM_ROOT_DIR/other/
  scGPT/                         # upstream repository
  BulkFormer/                    # upstream repository, including data/ assets
  models/
    scgpt_args.json
    scgpt_vocab.json
    scgpt_best_model.pt
    BulkFormer_147M.pt
```

The scGPT files are the published human checkpoint, vocabulary, and arguments,
renamed as above. BulkFormer also requires its gene metadata, graph, graph
weights, ESM2 features, and interested-gene list. Their paths are exposed in
each task YAML. Full-vocabulary BulkFormer inputs are kept alongside the
harmonized inputs; see [../data](../data/README.md).

Native scGPT pre-adaptation uses both leakage-filtered bulk partitions:

```bash
torchrun --standalone --nproc_per_node=4 src/main.py \
  pretrain=scgpt_preadapt task=pretrain.scgpt_preadapt \
  "pretrain.data_paths=[$SCBFM_ROOT_DIR/datasets/bulk/pretraining_bulk_RAW.h5ad,$SCBFM_ROOT_DIR/datasets/bulk/preadapt_bulk_RAW.h5ad]" \
  pretrain.audit_only=true
```

After auditing, set `pretrain.audit_only=false` to train. For an interrupted
run, set `pretrain.resume_state_path` to its `training_state.pt`.

Frozen scGPT and BulkFormer use PCA+RF downstream evaluation:

```bash
python src/main.py task=finetune.canc_type_class_scgpt_pca_rf
python src/main.py task=finetune.canc_type_class_bulkformer_pca_rf
```

To use pre-adapted scGPT, retain the `_scgpt_pca_rf` task and change its model
files and output identity together:

```bash
python src/main.py task=finetune.canc_type_class_scgpt_pca_rf \
  finetune.canc_type_class.scgpt_model_dir="$SCBFM_ROOT_DIR/output/scgpt_preadapt" \
  finetune.canc_type_class.scgpt_args_filename=args.json \
  finetune.canc_type_class.scgpt_vocab_filename=vocab.json \
  finetune.canc_type_class.scgpt_checkpoint_filename=last_model.pt \
  finetune.canc_type_class.scgpt_model_key=scgpt_preadapt \
  finetune.canc_type_class.scgpt_variant=scgpt_preadapt_pca_rf
```

Use the appropriate task prefix for other endpoints. The two single-cell tasks
are documented in [finetune/batch_integration](finetune/batch_integration/README.md).

## Protocol Notes

Training-fold preprocessing, fold reuse, and checkpoint compatibility checks
are implemented in the runners. The reconstruction objective uses masked binned
expression values; it is distinct from downstream supervised objectives.
Deconvolution uses MSE for training and MAE as its primary evaluation metric.
Time-to-event neural models use Cox losses; the ordinary PCA+RF survival
baseline uses a documented censoring-weighted regression approximation.
The single-cell tasks evaluate frozen embeddings without supervised fine-tuning.
