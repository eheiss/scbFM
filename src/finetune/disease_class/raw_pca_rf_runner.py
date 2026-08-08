from __future__ import annotations

from omegaconf import DictConfig

from finetune.canc_type_class.raw_pca_rf_runner import CancTypeClassRawPCARFRunner
from finetune.disease_class.runner import DiseaseClassRunner


class DiseaseClassRawPCARFRunner(CancTypeClassRawPCARFRunner):
    """Raw-expression PCA + random-forest baseline for DiSignAtlas."""

    task_name = "disease_class"
    config_node = "disease_class"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return DiseaseClassRunner._resolve_task_cfg(cfg)

    def _load_input_adata(self):
        return DiseaseClassRunner._load_input_adata(self)

    def _load_disignatlas(self):
        return DiseaseClassRunner._load_disignatlas(self)

    def _read_disignatlas(self, data_path):
        return DiseaseClassRunner._read_disignatlas(self, data_path)

    def _preprocess_adata(self, adata):
        return DiseaseClassRunner._preprocess_adata(self, adata)

    def _prepare_cv_data(self):
        return DiseaseClassRunner._prepare_cv_data(self)

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return DiseaseClassRunner._prediction_rows(
            self,
            model_key,
            fold,
            checkpoint_path,
            test_adata,
            test_metrics,
        )
