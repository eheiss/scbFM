from __future__ import annotations

import numpy as np
import torch.distributed as dist

from finetune.deconv.pca_rf_runner import DeconvPCARFRunner
from run_provenance import complete_run_metadata
from utils import seed_all


class DeconvRawPCARFRunner(DeconvPCARFRunner):
    """Raw log1p expression followed by PCA and multi-output RF."""

    def _finetune_mode(self) -> str:
        variant = str(
            getattr(self.task_cfg, "raw_pca_rf_variant", "") or ""
        ).strip()
        if variant:
            return variant
        feature_mode = str(
            getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes")
        )
        return f"raw_pca_rf_{feature_mode}"

    def _get_checkpoint_paths(self) -> dict[str, str]:
        return {"raw_pca_rf": ""}

    def _select_raw_features(self, train_adata) -> np.ndarray:
        mode = str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes"))
        if mode == "all_genes":
            return np.arange(train_adata.n_vars, dtype=np.int64)
        if mode == "hvg1199":
            return self._select_training_hvg_indices(train_adata)
        raise ValueError("raw_pca_rf_feature_mode must be all_genes or hvg1199.")

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("Raw deconvolution PCA+RF must use one process.")
            adata, targets, groups = self._prepare_cv_data()
            splits = self._build_or_load_cv_splits(adata, groups)
            self._save_run_metadata({"raw_pca_rf": ""})
            fold_rows = []
            prediction_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + fold_idx)
                train_adata = adata[train_idx].copy()
                test_adata = adata[test_idx].copy()
                feature_indices = self._select_raw_features(train_adata)
                train_y = targets[train_idx]
                test_y = targets[test_idx]
                train_y = train_y / train_y.sum(axis=1, keepdims=True)
                test_y = test_y / test_y.sum(axis=1, keepdims=True)
                self.fold_train_target_mean = train_y.mean(axis=0)
                train_metrics, test_metrics = self._fit_predict_features(
                    train_adata.X[:, feature_indices],
                    train_y,
                    test_adata.X[:, feature_indices],
                    test_y,
                    prefix="raw_pca_rf",
                )
                fold_rows.append(
                    self._flatten_fold_metrics(
                        "raw_pca_rf",
                        fold_idx,
                        len(splits),
                        "",
                        train_metrics,
                        test_metrics,
                    )
                )
                prediction_rows.extend(
                    self._prediction_rows(
                        "raw_pca_rf",
                        fold_idx,
                        "",
                        test_adata,
                        test_metrics,
                    )
                )
                self._cleanup_fold_state()
            aggregate = self._write_model_results("", fold_rows, prediction_rows)
            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
