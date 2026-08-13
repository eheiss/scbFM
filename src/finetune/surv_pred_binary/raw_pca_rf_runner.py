from __future__ import annotations

from omegaconf import DictConfig

from finetune.canc_type_class.raw_pca_rf_runner import CancTypeClassRawPCARFRunner
from finetune.surv_pred_binary.pca_rf_runner import BinarySurvivalRFMetricMixin
from finetune.surv_pred_binary.runner import SurvPredBinaryRunner


class SurvPredBinaryRawPCARFRunner(
    BinarySurvivalRFMetricMixin,
    CancTypeClassRawPCARFRunner,
):
    task_name = "surv_pred_binary"
    config_node = "surv_pred_binary"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredBinaryRunner._resolve_task_cfg(cfg)

    def _prepare_cv_data(self):
        return SurvPredBinaryRunner._prepare_cv_data(self)

    def _fit_predict_raw(
        self, train_adata, train_y, test_adata, test_y, feature_indices
    ):
        train_x = self._matrix_to_array(train_adata.X, feature_indices)
        test_x = self._matrix_to_array(test_adata.X, feature_indices)
        return self._fit_binary_rf(
            train_x, train_y, test_x, test_y, prefix="raw_pca_rf"
        )

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return SurvPredBinaryRunner._prediction_rows(
            self, model_key, fold, checkpoint_path, test_adata, test_metrics
        )
