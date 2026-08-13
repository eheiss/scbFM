from __future__ import annotations

import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig

from finetune.surv_pred.pca_rf_runner import SurvPredPCARFRunner
from finetune.surv_pred_survboard.runner import (
    SurvPredSurvBoardRunner,
    survival_metrics,
)
from run_provenance import complete_run_metadata
from utils import seed_all


class SurvPredSurvBoardPCARFRunner(SurvPredPCARFRunner):
    """Frozen scbFM CLS embeddings followed by PCA and RF on SurvBoard splits."""

    task_name = "surv_pred_survboard"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredSurvBoardRunner._resolve_task_cfg(cfg)

    def _task_output_dir(self):
        return SurvPredSurvBoardRunner._task_output_dir(self)

    def _output_prefix(self):
        return SurvPredSurvBoardRunner._output_prefix(self)

    def _get_checkpoint_paths(self):
        return SurvPredSurvBoardRunner._get_checkpoint_paths(self)

    def _save_run_metadata(self, checkpoint_paths=None) -> None:
        SurvPredSurvBoardRunner._save_run_metadata(
            self,
            checkpoint_paths or {},
        )

    @staticmethod
    def _survboard_metrics(
        result, train_time, train_event, test_time, test_event, **_unused
    ):
        metrics = survival_metrics(
            survival_probs=result.test_survival,
            time_points=result.time_points,
            test_time=test_time,
            test_event=test_event,
            test_log_hazard=result.test_risk,
            train_time=train_time,
            train_event=train_event,
        )
        metrics.update(
            {
                "pca_components": result.pca_components,
                "embedding_dim": result.embedding_dim,
            }
        )
        return metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("scbFM PCA+RF must be launched with one process.")
            cancer = str(getattr(self.task_cfg, "cancer", "")).upper()
            project = str(getattr(self.task_cfg, "project", "TCGA"))
            adata, times, events = self._load_survboard_data()
            train_splits, test_splits = self._load_splits(adata.n_obs)
            outer_splits = list(getattr(self.task_cfg, "outer_splits", range(len(train_splits))))
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            aggregate_rows = []
            for model_idx, (model_key, checkpoint_path) in enumerate(checkpoint_paths.items()):
                fold_rows = []
                for position, split_idx in enumerate(outer_splits):
                    train_idx, test_idx = train_splits[split_idx], test_splits[split_idx]
                    seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + model_idx * 10000 + split_idx)
                    self.fold_gene_indices = self._select_training_hvg_indices(
                        adata[train_idx].copy()
                    )
                    self._build_loaders(
                        adata[train_idx].X,
                        times[train_idx],
                        events[train_idx],
                        adata[test_idx].X,
                        times[test_idx],
                        events[test_idx],
                    )
                    backbone = self._build_backbone(checkpoint_path)
                    train_x, train_time, train_event = self._extract_embeddings(
                        backbone, self.train_infer_loader
                    )
                    test_x, test_time, test_event = self._extract_embeddings(
                        backbone, self.test_loader
                    )
                    result = self._fit_survival_regressor(
                        train_x, train_time, train_event, test_x, test_time
                    )
                    metrics = self._survboard_metrics(
                        result,
                        train_time,
                        train_event,
                        test_time,
                        test_event,
                        checkpoint_path=checkpoint_path,
                        model_key=model_key,
                    )
                    fold_rows.append(
                        self._flatten_fold_metrics(
                            model_key,
                            split_idx,
                            len(outer_splits),
                            checkpoint_path,
                            cancer,
                            project,
                            {"loss": float("nan")},
                            metrics,
                        )
                    )
                    self._save_survboard_predictions(
                        model_key, split_idx, result.test_survival, result.time_points
                    )
                    del backbone
                    self._cleanup_fold_state()
                aggregate_rows.append(self._write_model_results(checkpoint_path, fold_rows, []))

            output_path = self._task_output_dir() / f"{self._output_prefix()}_evaluation_metrics.csv"
            self._write_csv(output_path, aggregate_rows)
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": aggregate_rows}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
