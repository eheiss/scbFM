from __future__ import annotations

import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig

from finetune.canc_type_class.scgpt_pca_rf_runner import (
    CancTypeClassScGPTPCARFRunner,
)
from finetune.surv_pred.pca_rf_runner import SurvPredPCARFRunner
from finetune.surv_pred.runner import SurvPredRunner
from finetune.survival_regression import fit_pca_random_forest_regressor
from run_provenance import complete_run_metadata, update_run_metadata
from utils import seed_all


class SurvPredScGPTPCARFRunner(CancTypeClassScGPTPCARFRunner):
    """Frozen scGPT CLS embeddings followed by PCA and random-forest regression."""

    task_name = "surv_pred"
    config_node = "surv_pred"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredRunner._resolve_task_cfg(cfg)

    def _load_tcga_survival_data(self):
        return SurvPredRunner._load_tcga_survival_data(self)

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
                    "baseline": "scgpt_frozen_cls_embedding_pca_random_forest_regressor",
                    "evaluation_protocol": "stratified_five_fold_cross_validation",
                    "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                    "cv_repeats": None,
                    "cv_test_size": None,
                },
            )

    def run(self) -> dict:
        try:
            self._setup_runtime()
            paths = self._scgpt_paths()
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model, vocab, model_configs = self._build_scgpt_model(paths)
            adata, times, events, cohorts, groups = self._load_tcga_survival_data()
            adata = self._add_scgpt_gene_symbols(adata, paths["gene_info"], vocab)
            strat_labels = self._survival_stratification_labels(events, cohorts)
            splits = self._build_or_load_cv_splits(adata, strat_labels, groups)
            sample_indices = np.arange(adata.n_obs, dtype=np.int64)
            self._save_run_metadata()

            fold_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                hvg_idx = self._select_training_hvg_indices(adata[train_idx].copy())
                fold_adata = adata[:, hvg_idx].copy()
                gene_ids = np.asarray(
                    vocab(fold_adata.var["scgpt_gene_symbol"].astype(str).tolist()),
                    dtype=np.int64,
                )
                train_loader = self._make_scgpt_loader(
                    fold_adata[train_idx].copy(),
                    sample_indices[train_idx],
                    gene_ids,
                    vocab,
                    model_configs,
                    stage=f"fold {fold_idx} training split",
                )
                test_loader = self._make_scgpt_loader(
                    fold_adata[test_idx].copy(),
                    sample_indices[test_idx],
                    gene_ids,
                    vocab,
                    model_configs,
                    stage=f"fold {fold_idx} validation split",
                )
                train_x, train_order = self._extract_scgpt_embeddings_safely(
                    model, train_loader, stage=f"fold {fold_idx} training split"
                )
                test_x, test_order = self._extract_scgpt_embeddings_safely(
                    model, test_loader, stage=f"fold {fold_idx} validation split"
                )
                train_x, train_order = self._gather_scgpt_embeddings(
                    train_x, train_order, len(train_idx)
                )
                test_x, test_order = self._gather_scgpt_embeddings(
                    test_x, test_order, len(test_idx)
                )
                if not np.array_equal(train_order, train_idx) or not np.array_equal(
                    test_order, test_idx
                ):
                    raise RuntimeError("Distributed scGPT extraction changed sample order.")

                if self.is_master:
                    result = fit_pca_random_forest_regressor(
                        self.task_cfg,
                        train_x,
                        times[train_idx],
                        events[train_idx],
                        test_x,
                        times[test_idx],
                        prefix="scgpt_pca",
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
                            self._model_key(),
                            fold_idx,
                            len(splits),
                            str(paths["checkpoint"]),
                            "pan_cancer",
                            "TCGA",
                            {"loss": float("nan")},
                            metrics,
                        )
                    )
                if self.is_distributed:
                    dist.barrier()

            output_path = self._task_output_dir() / f"{self._output_prefix()}_evaluation_metrics.csv"
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}
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
