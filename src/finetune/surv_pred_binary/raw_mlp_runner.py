from __future__ import annotations

from omegaconf import DictConfig

from finetune.canc_type_class.raw_mlp_runner import CancTypeClassRawMLPRunner
from finetune.surv_pred_binary.runner import SurvPredBinaryRunner


class SurvPredBinaryRawMLPRunner(CancTypeClassRawMLPRunner):
    """Raw-expression MLP baseline for BulkFormer-style binary survival prediction."""

    task_name = "surv_pred_binary"
    config_node = "surv_pred_binary"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "surv_pred_binary" in cfg.finetune:
            return cfg.finetune.surv_pred_binary
        raise ValueError(
            "Could not find a binary survival prediction config. "
            "Expected cfg.finetune.surv_pred_binary."
        )

    def _load_tcga(self):
        return SurvPredBinaryRunner._load_tcga(self)

    def _load_input_adata(self):
        return SurvPredBinaryRunner._load_input_adata(self)

    def _numeric_obs(self, adata, column):
        return SurvPredBinaryRunner._numeric_obs(adata, column)

    def _derive_binary_survival_labels(self, adata):
        return SurvPredBinaryRunner._derive_binary_survival_labels(self, adata)

    def _prepare_cv_data(self):
        return SurvPredBinaryRunner._prepare_cv_data(self)

    def _evaluate(self):
        return SurvPredBinaryRunner._evaluate(self)

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return SurvPredBinaryRunner._prediction_rows(
            self,
            model_key,
            fold,
            checkpoint_path,
            test_adata,
            test_metrics,
        )
