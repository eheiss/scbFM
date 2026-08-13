from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from finetune.canc_type_class.bulkformer_pca_rf_runner import (
    CancTypeClassBulkFormerPCARFRunner,
)
from finetune.surv_pred.pca_rf_runner import SurvPredPCARFRunner
from finetune.surv_pred.runner import SurvPredRunner
from finetune.survival_regression import fit_pca_random_forest_regressor
from run_provenance import complete_run_metadata, update_run_metadata
from utils import seed_all


class SurvPredBulkFormerPCARFRunner(CancTypeClassBulkFormerPCARFRunner):
    """Frozen BulkFormer embeddings followed by PCA and random-forest regression."""

    task_name = "surv_pred"
    config_node = "surv_pred"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredRunner._resolve_task_cfg(cfg)

    def _load_canonical_tcga(self):
        adata, times, events, cohorts, _ = SurvPredRunner._load_tcga_survival_data(self)
        adata.obs["survival_time"] = times
        adata.obs["survival_event"] = events
        adata.obs["survival_cohort"] = cohorts
        adata.obs["cancer_type"] = SurvPredRunner._survival_stratification_labels(
            self, events, cohorts
        )
        return adata

    def _survival_stratification_labels(self, events, cohorts):
        return SurvPredRunner._survival_stratification_labels(self, events, cohorts)

    def _build_cv_splits(self, labels, groups=None):
        return SurvPredRunner._build_cv_splits(self, labels, groups)

    def _build_or_load_cv_splits(self, adata, labels, groups=None):
        return SurvPredRunner._build_or_load_cv_splits(self, adata, labels, groups)

    def _resolve_cv_fold_manifest_path(self, source_rows):
        return SurvPredRunner._resolve_cv_fold_manifest_path(self, source_rows)

    def _forest_metrics(self, *args, **kwargs):
        return SurvPredPCARFRunner._forest_metrics(self, *args, **kwargs)

    def _cohort_c_index_metrics(self, *args, **kwargs):
        return SurvPredRunner._cohort_c_index_metrics(*args, **kwargs)

    def _save_run_metadata(self) -> None:
        super()._save_run_metadata()
        if self.is_master:
            update_run_metadata(
                self._run_metadata_path,
                {
                    "baseline": "bulkformer_147m_embedding_pca_random_forest_regressor",
                    "evaluation_protocol": "stratified_five_fold_cross_validation",
                    "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                    "cv_repeats": None,
                    "cv_test_size": None,
                },
            )

    def run(self) -> dict:
        try:
            self._setup_runtime()
            paths = self._bulkformer_paths()
            expr_array, _, groups, adata, missing_fraction = self._prepare_bulkformer_data(paths)
            times = adata.obs["survival_time"].to_numpy(dtype=float)
            events = adata.obs["survival_event"].to_numpy(dtype=float)
            cohorts = adata.obs["survival_cohort"].astype(str).to_numpy()
            strat_labels = self._survival_stratification_labels(events, cohorts)
            splits = self._build_or_load_cv_splits(adata, strat_labels, groups)
            self._save_run_metadata()

            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model = self._build_bulkformer_model(paths)
            embeddings, sample_indices = self._extract_bulkformer_embeddings(
                model, expr_array, paths, missing_fraction
            )
            embeddings, sample_indices = self._gather_bulkformer_embeddings(
                embeddings, sample_indices, expr_array.shape[0]
            )
            if not np.array_equal(sample_indices, np.arange(expr_array.shape[0])):
                raise RuntimeError("Distributed BulkFormer extraction changed sample order.")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            output_path = self._task_output_dir() / f"{self._output_prefix()}_evaluation_metrics.csv"
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}
            fold_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                result = fit_pca_random_forest_regressor(
                    self.task_cfg,
                    embeddings[train_idx],
                    times[train_idx],
                    events[train_idx],
                    embeddings[test_idx],
                    times[test_idx],
                    prefix="bulkformer_pca",
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
                    SurvPredRunner._flatten_fold_metrics(
                        self,
                        "bulkformer_147m",
                        fold_idx,
                        len(splits),
                        str(paths["checkpoint"]),
                        "pan_cancer",
                        "TCGA",
                        {"loss": float("nan")},
                        metrics,
                    )
                )
            aggregate = SurvPredRunner._write_model_results(
                self, str(paths["checkpoint"]), fold_rows, []
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
