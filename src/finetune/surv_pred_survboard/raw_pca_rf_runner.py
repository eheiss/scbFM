from __future__ import annotations

import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig
from scipy import sparse

from finetune.surv_pred_survboard.pca_rf_runner import SurvPredSurvBoardPCARFRunner
from finetune.surv_pred_survboard.runner import SurvPredSurvBoardRunner
from run_provenance import complete_run_metadata
from utils import seed_all


class SurvPredSurvBoardRawPCARFRunner(SurvPredSurvBoardPCARFRunner):
    """Raw expression followed by PCA and RSF on official SurvBoard splits."""

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredSurvBoardRunner._resolve_task_cfg(cfg)

    def _finetune_mode(self):
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
    def _array(matrix, indices):
        matrix = matrix[:, indices]
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("Raw PCA+RF must be launched with one process.")
            cancer = str(getattr(self.task_cfg, "cancer", "")).upper()
            project = str(getattr(self.task_cfg, "project", "TCGA"))
            adata, times, events = self._load_survboard_data()
            train_splits, test_splits = self._load_splits(adata.n_obs)
            outer_splits = list(getattr(self.task_cfg, "outer_splits", range(len(train_splits))))
            self._save_run_metadata({})
            fold_rows = []
            for position, split_idx in enumerate(outer_splits):
                train_idx, test_idx = train_splits[split_idx], test_splits[split_idx]
                seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + split_idx)
                features = self._select_feature_indices(adata[train_idx].copy())
                result = self._fit_survival_regressor(
                    self._array(adata[train_idx].X, features),
                    times[train_idx],
                    events[train_idx],
                    self._array(adata[test_idx].X, features),
                    times[test_idx],
                    prefix="raw_pca_rf",
                )
                metrics = self._survboard_metrics(
                    result,
                    times[train_idx],
                    events[train_idx],
                    times[test_idx],
                    events[test_idx],
                    checkpoint_path="",
                    model_key="raw_pca_rf",
                )
                fold_rows.append(
                    self._flatten_fold_metrics(
                        "raw_pca_rf",
                        split_idx,
                        len(outer_splits),
                        "",
                        cancer,
                        project,
                        {"loss": float("nan")},
                        metrics,
                    )
                )
                self._save_survboard_predictions(
                    "raw_pca_rf", split_idx, result.test_survival, result.time_points
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
