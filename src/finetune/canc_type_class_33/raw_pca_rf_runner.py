from __future__ import annotations

from omegaconf import DictConfig

from finetune.canc_type_class.raw_pca_rf_runner import (
    CancTypeClassRawPCARFRunner,
)
from finetune.canc_type_class_33.runner import DEFAULT_COHORTS, CancTypeClass33Runner


class CancTypeClass33RawPCARFRunner(CancTypeClassRawPCARFRunner):
    """Raw-expression PCA + random-forest baseline for TCGA 33-type classification."""

    task_name = "canc_type_class_33"
    config_node = "canc_type_class_33"
    default_cohorts = DEFAULT_COHORTS

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "canc_type_class_33" in cfg.finetune:
            return cfg.finetune.canc_type_class_33
        raise ValueError(
            "Could not find a 33-class cancer type classification config. "
            "Expected cfg.finetune.canc_type_class_33."
        )

    def _load_tcga(self):
        return CancTypeClass33Runner._load_tcga(self)
