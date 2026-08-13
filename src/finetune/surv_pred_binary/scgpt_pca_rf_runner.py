from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from finetune.canc_type_class.scgpt_pca_rf_runner import (
    CancTypeClassScGPTPCARFRunner,
)
from finetune.surv_pred_binary.pca_rf_runner import BinarySurvivalRFMetricMixin
from finetune.surv_pred_binary.runner import SurvPredBinaryRunner


class SurvPredBinaryScGPTPCARFRunner(
    BinarySurvivalRFMetricMixin,
    CancTypeClassScGPTPCARFRunner,
):
    task_name = "surv_pred_binary"
    config_node = "surv_pred_binary"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredBinaryRunner._resolve_task_cfg(cfg)

    def _prepare_raw_tcga_data(self):
        adata = SurvPredBinaryRunner._load_tcga(self)
        labels = SurvPredBinaryRunner._derive_binary_survival_labels(self, adata)
        adata.obs["cancer_type"] = labels
        self.label_dict = np.asarray(["0", "1"])
        patient_ids = adata.obs["patient_id"].astype(str).to_numpy()
        groups = patient_ids if len(np.unique(patient_ids)) < len(patient_ids) else None
        return adata, labels, groups

    def _fit_predict_embeddings(self, train_x, train_y, test_x, test_y):
        return self._fit_binary_rf(
            train_x, train_y, test_x, test_y, prefix="scgpt_pca"
        )

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return SurvPredBinaryRunner._prediction_rows(
            self, model_key, fold, checkpoint_path, test_adata, test_metrics
        )
