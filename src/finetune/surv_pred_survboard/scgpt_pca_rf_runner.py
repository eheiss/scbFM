from __future__ import annotations

import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig

from finetune.surv_pred.scgpt_pca_rf_runner import SurvPredScGPTPCARFRunner
from finetune.surv_pred_survboard.pca_rf_runner import SurvPredSurvBoardPCARFRunner
from finetune.surv_pred_survboard.runner import SurvPredSurvBoardRunner
from finetune.survival_regression import fit_pca_random_forest_regressor
from run_provenance import complete_run_metadata, update_run_metadata
from utils import seed_all


class SurvPredSurvBoardScGPTPCARFRunner(SurvPredScGPTPCARFRunner):
    """Frozen scGPT CLS embeddings with PCA+RF on official SurvBoard splits."""

    task_name = "surv_pred_survboard"
    config_node = "surv_pred_survboard"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredSurvBoardRunner._resolve_task_cfg(cfg)

    def _task_output_dir(self):
        return SurvPredSurvBoardRunner._task_output_dir(self)

    def _output_prefix(self):
        return SurvPredSurvBoardRunner._output_prefix(self)

    def _survboard_metrics(self, *args, **kwargs):
        return SurvPredSurvBoardPCARFRunner._survboard_metrics(*args, **kwargs)

    def _save_run_metadata(self) -> None:
        super()._save_run_metadata()
        if self.is_master:
            update_run_metadata(
                self._run_metadata_path,
                {
                    "baseline": "scgpt_frozen_cls_embedding_pca_random_forest_regressor",
                    "cancer": str(getattr(self.task_cfg, "cancer", "")).upper(),
                    "project": str(getattr(self.task_cfg, "project", "TCGA")),
                    "evaluation_protocol": "survboard_repeated_five_fold_cross_validation",
                    "cv_folds": 5,
                    "cv_repetitions": 5,
                    "cv_repeats": None,
                    "cv_test_size": None,
                    "n_outer_splits": int(
                        getattr(self.task_cfg, "expected_outer_splits", 25)
                    ),
                    "survboard_split_fingerprint": str(
                        getattr(self, "_survboard_split_fingerprint", "")
                    ),
                },
            )

    def run(self) -> dict:
        try:
            self._setup_runtime()
            paths = self._scgpt_paths()
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model, vocab, model_configs = self._build_scgpt_model(paths)
            adata, times, events = SurvPredSurvBoardRunner._load_survboard_data(self)
            adata = self._add_scgpt_gene_symbols(adata, paths["gene_info"], vocab)
            train_splits, test_splits = SurvPredSurvBoardRunner._load_splits(
                self, adata.n_obs
            )
            outer_splits = list(getattr(self.task_cfg, "outer_splits", range(len(train_splits))))
            sample_indices = np.arange(adata.n_obs, dtype=np.int64)
            self._save_run_metadata()
            cancer = str(getattr(self.task_cfg, "cancer", "")).upper()
            project = str(getattr(self.task_cfg, "project", "TCGA"))

            fold_rows = []
            for position, split_idx in enumerate(outer_splits):
                train_idx, test_idx = train_splits[split_idx], test_splits[split_idx]
                hvg_idx = self._select_training_hvg_indices(adata[train_idx].copy())
                fold_adata = adata[:, hvg_idx].copy()
                gene_ids = np.asarray(
                    vocab(fold_adata.var["scgpt_gene_symbol"].astype(str).tolist()),
                    dtype=np.int64,
                )
                train_loader = self._make_scgpt_loader(
                    fold_adata[train_idx].copy(), sample_indices[train_idx], gene_ids,
                    vocab, model_configs, stage=f"split {split_idx} training split"
                )
                test_loader = self._make_scgpt_loader(
                    fold_adata[test_idx].copy(), sample_indices[test_idx], gene_ids,
                    vocab, model_configs, stage=f"split {split_idx} validation split"
                )
                train_x, train_order = self._extract_scgpt_embeddings_safely(
                    model, train_loader, stage=f"split {split_idx} training split"
                )
                test_x, test_order = self._extract_scgpt_embeddings_safely(
                    model, test_loader, stage=f"split {split_idx} validation split"
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
                        self.task_cfg, train_x, times[train_idx], events[train_idx],
                        test_x, times[test_idx], prefix="scgpt_pca"
                    )
                    metrics = self._survboard_metrics(
                        result, times[train_idx], events[train_idx],
                        times[test_idx], events[test_idx],
                        checkpoint_path=str(paths["checkpoint"]),
                        model_key=self._model_key(),
                    )
                    fold_rows.append(
                        SurvPredSurvBoardRunner._flatten_fold_metrics(
                            self, self._model_key(), split_idx, len(outer_splits),
                            str(paths["checkpoint"]), cancer, project,
                            {"loss": float("nan")}, metrics
                        )
                    )
                    SurvPredSurvBoardRunner._save_survboard_predictions(
                        self, self._model_key(), split_idx,
                        result.test_survival, result.time_points
                    )
                if self.is_distributed:
                    dist.barrier()

            output_path = self._task_output_dir() / f"{self._output_prefix()}_evaluation_metrics.csv"
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}
            aggregate = SurvPredSurvBoardRunner._write_model_results(
                self, str(paths["checkpoint"]), fold_rows, []
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
