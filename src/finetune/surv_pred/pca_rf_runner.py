from __future__ import annotations

import logging
from pathlib import Path
from paths import output_root

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from finetune.canc_type_class.pca_rf_runner import CancTypeClassPCARFRunner
from finetune.canc_type_class.runner import CancTypeClassRunner
from finetune.surv_pred.runner import SurvPredRunner
from finetune.surv_pred_survboard.runner import harrell_c_index, ipcw_weighted_c_index
from finetune.survival_regression import fit_pca_random_forest_regressor
from utils import seed_all

log = logging.getLogger(__name__)


class SurvPredPCARFRunner(SurvPredRunner):
    """Frozen CLS embeddings followed by PCA and random-forest regression."""

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredRunner._resolve_task_cfg(cfg)

    def _finetune_mode(self) -> str:
        return str(getattr(self.task_cfg, "pca_rf_variant", "") or "pca_rf")

    def _task_output_dir(self) -> Path:
        return output_root(self.cfg) / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _extract_embeddings(self, backbone, loader):
        embeddings, times, events = [], [], []
        with torch.no_grad():
            for batch, batch_time, batch_event in loader:
                model_batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                hidden = backbone(
                    model_batch["gene_ids"],
                    model_batch["expr"],
                    src_key_padding_mask=model_batch.get("attention_key_padding_mask"),
                )
                embeddings.append(hidden[:, 0, :].cpu().numpy())
                times.append(batch_time.numpy())
                events.append(batch_event.numpy())
        return np.vstack(embeddings), np.concatenate(times), np.concatenate(events)

    def _build_backbone(self, checkpoint_path):
        return CancTypeClassPCARFRunner._build_backbone(self, checkpoint_path)

    def _validate_backbone_checkpoint(self, checkpoint, checkpoint_path):
        return CancTypeClassRunner._validate_backbone_checkpoint(
            self, checkpoint, checkpoint_path
        )

    def _fit_survival_regressor(
        self, train_x, train_time, train_event, test_x, test_time, *, prefix="pca_rf"
    ):
        return fit_pca_random_forest_regressor(
            self.task_cfg,
            train_x,
            train_time,
            train_event,
            test_x,
            test_time,
            prefix=prefix,
        )

    def _forest_metrics(
        self, result, train_time, train_event, test_time, test_event, test_cohorts
    ):
        metrics = {
            "test_c_index": harrell_c_index(result.test_risk, test_time, test_event),
            "test_ipcw_c_index": ipcw_weighted_c_index(
                result.test_risk,
                test_time,
                test_event,
                train_time,
                train_event,
            ),
            "pca_components": result.pca_components,
            "embedding_dim": result.embedding_dim,
        }
        metrics.update(
            self._cohort_c_index_metrics(
                result.test_risk, test_time, test_event, test_cohorts
            )
        )
        return metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("scbFM PCA+RF must be launched with one process.")
            adata, times, events, cohorts, groups = self._load_tcga_survival_data()
            strat_labels = self._survival_stratification_labels(events, cohorts)
            splits = self._build_or_load_cv_splits(adata, strat_labels, groups)
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            aggregate_rows = []
            for model_idx, (model_key, checkpoint_path) in enumerate(checkpoint_paths.items()):
                fold_rows = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + model_idx * 10000 + fold_idx)
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
                    test_metrics = self._forest_metrics(
                        result,
                        train_time,
                        train_event,
                        test_time,
                        test_event,
                        cohorts[test_idx],
                    )
                    fold_rows.append(
                        self._flatten_fold_metrics(
                            model_key,
                            fold_idx,
                            len(splits),
                            checkpoint_path,
                            "pan_cancer",
                            "TCGA",
                            {"loss": float("nan")},
                            test_metrics,
                        )
                    )
                    del backbone
                    self._cleanup_fold_state()
                aggregate_rows.append(self._write_model_results(checkpoint_path, fold_rows, []))

            output_path = self._task_output_dir() / f"{self._output_prefix()}_evaluation_metrics.csv"
            self._write_csv(output_path, aggregate_rows)
            from run_provenance import complete_run_metadata

            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": aggregate_rows}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
