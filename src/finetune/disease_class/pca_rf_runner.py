from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from finetune.canc_type_class.pca_rf_runner import CancTypeClassPCARFRunner
from finetune.disease_class.runner import DiseaseClassRunner


class DiseaseClassPCARFRunner(CancTypeClassPCARFRunner):
    """Frozen-backbone embedding PCA + random-forest baseline for DiSignAtlas disease classification."""

    task_name = "disease_class"
    config_node = "disease_class"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "disease_class" in cfg.finetune:
            return cfg.finetune.disease_class
        raise ValueError(
            "Could not find a disease classification config. "
            "Expected cfg.finetune.disease_class."
        )

    def _load_input_adata(self):
        return DiseaseClassRunner._load_input_adata(self)

    def _read_disignatlas(self, data_path):
        return DiseaseClassRunner._read_disignatlas(self, data_path)

    def _load_disignatlas(self):
        return DiseaseClassRunner._load_disignatlas(self)

    def _preprocess_adata(self, adata):
        return DiseaseClassRunner._preprocess_adata(self, adata)

    def _prepare_cv_data(self):
        return DiseaseClassRunner._prepare_cv_data(self)

    def _labels(self, adata, label_col: str = "cancer_type") -> np.ndarray:
        label_col = str(getattr(self.task_cfg, "disease_label_col", "label"))
        return super()._labels(adata, label_col=label_col)

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return DiseaseClassRunner._prediction_rows(
            self,
            model_key,
            fold,
            checkpoint_path,
            test_adata,
            test_metrics,
        )
