from __future__ import annotations

import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig
from scipy import sparse

from finetune.surv_pred.pca_rf_runner import SurvPredPCARFRunner
from finetune.surv_pred.runner import SurvPredRunner
from run_provenance import complete_run_metadata
from utils import seed_all


class SurvPredRawPCARFRunner(SurvPredPCARFRunner):
    """Raw expression followed by train-only PCA and random survival forest."""

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredRunner._resolve_task_cfg(cfg)

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "raw_pca_rf_variant", "") or "")
        mode = str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes"))
        return variant or f"raw_pca_rf_{mode}"

    def _select_feature_indices(self, train_adata):
        mode = str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes"))
        if mode == "all_genes":
            return np.arange(train_adata.n_vars, dtype=np.int64)
        if mode == "hvg1199":
            return self._select_training_hvg_indices(train_adata)
        raise ValueError("raw_pca_rf_feature_mode must be all_genes or hvg1199.")

    @staticmethod
    def _array(matrix, feature_indices):
        matrix = matrix[:, feature_indices]
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("Raw PCA+RF must be launched with one process.")
            adata, times, events, cohorts, groups = self._load_tcga_survival_data()
            strat_labels = self._survival_stratification_labels(events, cohorts)
            splits = self._build_or_load_cv_splits(adata, strat_labels, groups)
            self._save_run_metadata({})
            fold_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + fold_idx)
                features = self._select_feature_indices(adata[train_idx].copy())
                result = self._fit_survival_regressor(
                    self._array(adata[train_idx].X, features),
                    times[train_idx],
                    events[train_idx],
                    self._array(adata[test_idx].X, features),
                    times[test_idx],
                    prefix="raw_pca_rf",
                )
                metrics = self._forest_metrics(
                    result,
                    times[train_idx],
                    events[train_idx],
                    times[test_idx],
                    events[test_idx],
                    cohorts[test_idx],
                )
                fold_rows.append(
                    self._flatten_fold_metrics(
                        "raw_pca_rf",
                        fold_idx,
                        len(splits),
                        "",
                        "pan_cancer",
                        "TCGA",
                        {"loss": float("nan")},
                        metrics,
                    )
                )
            aggregate = self._write_model_results("", fold_rows, [])
            output_path = self._task_output_dir() / f"{self._output_prefix()}_evaluation_metrics.csv"
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
