from __future__ import annotations

from omegaconf import DictConfig

from finetune.canc_type_class.scgpt_pca_rf_runner import (
    CancTypeClassScGPTPCARFRunner,
)
from finetune.disease_class.runner import DiseaseClassRunner


class DiseaseClassScGPTPCARFRunner(CancTypeClassScGPTPCARFRunner):
    """Frozen scGPT CLS embeddings with PCA + RF for DiSignAtlas."""

    task_name = "disease_class"
    config_node = "disease_class"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return DiseaseClassRunner._resolve_task_cfg(cfg)

    def _read_disignatlas(self, data_path):
        return DiseaseClassRunner._read_disignatlas(self, data_path)

    def _load_input_adata(self):
        return DiseaseClassRunner._load_input_adata(self)

    def _load_disignatlas(self):
        return DiseaseClassRunner._load_disignatlas(self)

    def _prepare_raw_tcga_data(self):
        adata, labels, groups = super()._prepare_raw_tcga_data()
        expected_classes = int(getattr(self.task_cfg, "expected_disease_classes", 23))
        if len(self.label_dict) != expected_classes:
            raise ValueError(
                f"scGPT DiSignAtlas input contains {len(self.label_dict)} disease classes; "
                f"expected {expected_classes}."
            )
        return adata, labels, groups

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return DiseaseClassRunner._prediction_rows(
            self,
            model_key,
            fold,
            checkpoint_path,
            test_adata,
            test_metrics,
        )
