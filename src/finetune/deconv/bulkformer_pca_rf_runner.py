from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from finetune.canc_type_class.bulkformer_pca_rf_runner import (
    CancTypeClassBulkFormerPCARFRunner,
)
from finetune.deconv.pca_rf_runner import DeconvPCARFRunner
from run_provenance import complete_run_metadata
from utils import seed_all

log = logging.getLogger(__name__)


class DeconvBulkFormerPCARFRunner(DeconvPCARFRunner):
    """Frozen BulkFormer max-pooled embeddings followed by PCA and RF."""

    _required_path = CancTypeClassBulkFormerPCARFRunner._required_path
    _strip_ensembl_version = staticmethod(
        CancTypeClassBulkFormerPCARFRunner._strip_ensembl_version
    )
    _adata_to_gene_frame = staticmethod(
        CancTypeClassBulkFormerPCARFRunner._adata_to_gene_frame
    )
    _align_to_bulkformer_genes = staticmethod(
        CancTypeClassBulkFormerPCARFRunner._align_to_bulkformer_genes
    )
    _build_bulkformer_model = (
        CancTypeClassBulkFormerPCARFRunner._build_bulkformer_model
    )
    _extract_bulkformer_embeddings = (
        CancTypeClassBulkFormerPCARFRunner._extract_bulkformer_embeddings
    )
    _gather_bulkformer_embeddings = (
        CancTypeClassBulkFormerPCARFRunner._gather_bulkformer_embeddings
    )

    def _finetune_mode(self) -> str:
        variant = str(
            getattr(self.task_cfg, "bulkformer_variant", "") or ""
        ).strip()
        return variant or "bulkformer_pca_rf"

    def _bulkformer_paths(self) -> dict[str, Path]:
        paths = {
            "repo_dir": self._required_path("bulkformer_repo_dir"),
            "checkpoint": self._required_path("bulkformer_checkpoint_path"),
            "gene_info": self._required_path("bulkformer_gene_info_path"),
            "graph": self._required_path("bulkformer_graph_path"),
            "graph_weight": self._required_path("bulkformer_graph_weight_path"),
            "gene_emb": self._required_path("bulkformer_gene_emb_path"),
        }
        interested_path = getattr(
            self.task_cfg,
            "bulkformer_interested_gene_list_path",
            None,
        )
        if interested_path:
            path = Path(hydra.utils.to_absolute_path(str(interested_path)))
            if not path.exists():
                raise FileNotFoundError(f"Missing BulkFormer interested-gene list: {path}")
            paths["interested_gene_list"] = path
        return paths

    def _prepare_bulkformer_expression(
        self,
        adata,
        paths: dict[str, Path],
    ) -> tuple[np.ndarray, float]:
        gene_info = pd.read_csv(paths["gene_info"])
        if "ensg_id" not in gene_info:
            raise ValueError("BulkFormer gene info must contain ensg_id.")
        gene_list = self._strip_ensembl_version(
            gene_info["ensg_id"].astype(str)
        ).tolist()
        expected = int(getattr(self.task_cfg, "bulkformer_expected_gene_count", 20010))
        if len(gene_list) != expected or len(set(gene_list)) != expected:
            raise ValueError(
                f"BulkFormer gene vocabulary must contain {expected} unique Ensembl IDs."
            )
        expr_df = self._adata_to_gene_frame(adata)
        values = expr_df.to_numpy(dtype=np.float32, copy=False)
        if not np.all(np.isfinite(values)) or np.min(values) < 0:
            raise ValueError("BulkFormer deconvolution expression must be finite and non-negative.")
        maximum = float(np.max(values))
        configured_maximum = float(
            getattr(self.task_cfg, "bulkformer_max_expected_expression", 30.0)
        )
        if maximum > configured_maximum:
            raise ValueError(
                f"BulkFormer input maximum {maximum:.6g} exceeds {configured_maximum:.6g}."
            )
        missing_mask = ~np.isin(gene_list, expr_df.columns.astype(str))
        missing_fraction = float(np.mean(missing_mask))
        self._bulkformer_input_gene_count = int(len(gene_list))
        self._bulkformer_matched_gene_count = int((~missing_mask).sum())
        self._bulkformer_missing_gene_count = int(missing_mask.sum())
        self._bulkformer_missing_fraction = missing_fraction
        self._bulkformer_expression_min = float(np.min(values))
        self._bulkformer_expression_max = maximum
        log.info(
            "BulkFormer deconvolution mapping: matched=%d, missing=%d, fraction=%.4f",
            self._bulkformer_matched_gene_count,
            self._bulkformer_missing_gene_count,
            missing_fraction,
        )
        return self._align_to_bulkformer_genes(expr_df, gene_list), missing_fraction

    def _save_external_metadata(self, checkpoint_path: str) -> None:
        self._save_run_metadata({"bulkformer_147m": checkpoint_path})
        if not self.is_master:
            return
        with self._run_metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(
            {
                "bulkformer_expected_gene_count": self._bulkformer_input_gene_count,
                "bulkformer_matched_gene_count": self._bulkformer_matched_gene_count,
                "bulkformer_missing_gene_count": self._bulkformer_missing_gene_count,
                "bulkformer_missing_vocab_fraction": self._bulkformer_missing_fraction,
                "bulkformer_aggregate_type": str(
                    getattr(self.task_cfg, "bulkformer_aggregate_type", "max")
                ),
                "bulkformer_input_limitation": (
                    "The canonical pseudobulk contains the 13,004-gene thesis vocabulary; "
                    "unavailable BulkFormer genes are sentinel-filled."
                ),
            }
        )
        with self._run_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def run(self) -> dict:
        try:
            self._setup_runtime()
            paths = self._bulkformer_paths()
            adata, targets, groups = self._prepare_cv_data()
            splits = self._build_or_load_cv_splits(adata, groups)
            expression, missing_fraction = self._prepare_bulkformer_expression(
                adata,
                paths,
            )
            checkpoint_path = str(paths["checkpoint"])
            self._save_external_metadata(checkpoint_path)
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model = self._build_bulkformer_model(paths)
            embeddings, sample_indices = self._extract_bulkformer_embeddings(
                model,
                expression,
                paths,
                missing_fraction,
            )
            embeddings, sample_indices = self._gather_bulkformer_embeddings(
                embeddings,
                sample_indices,
                expression.shape[0],
            )
            if not np.array_equal(sample_indices, np.arange(expression.shape[0])):
                raise RuntimeError("Distributed BulkFormer extraction changed sample order.")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}

            fold_rows = []
            prediction_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                train_y = targets[train_idx]
                test_y = targets[test_idx]
                train_y = train_y / train_y.sum(axis=1, keepdims=True)
                test_y = test_y / test_y.sum(axis=1, keepdims=True)
                self.fold_train_target_mean = train_y.mean(axis=0)
                train_metrics, test_metrics = self._fit_predict_features(
                    embeddings[train_idx],
                    train_y,
                    embeddings[test_idx],
                    test_y,
                    prefix="bulkformer_pca",
                    rf_prefix="bulkformer_rf",
                )
                test_adata = adata[test_idx].copy()
                fold_rows.append(
                    self._flatten_fold_metrics(
                        "bulkformer_147m",
                        fold_idx,
                        len(splits),
                        checkpoint_path,
                        train_metrics,
                        test_metrics,
                    )
                )
                prediction_rows.extend(
                    self._prediction_rows(
                        "bulkformer_147m",
                        fold_idx,
                        checkpoint_path,
                        test_adata,
                        test_metrics,
                    )
                )
                self.fold_train_target_mean = None
            aggregate = self._write_model_results(
                checkpoint_path,
                fold_rows,
                prediction_rows,
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
