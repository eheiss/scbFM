from __future__ import annotations

from pathlib import Path

import anndata as ad
import hydra
from omegaconf import DictConfig

from finetune.canc_type_class.runner import (
    CHECKPOINT_MODEL_KEYS as BASE_CHECKPOINT_MODEL_KEYS,
    RANDOM_INIT_MODEL_KEY as BASE_RANDOM_INIT_MODEL_KEY,
    CancTypeClassRunner,
)

DEFAULT_COHORTS = [
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
]


class CancTypeClass33Runner(CancTypeClassRunner):
    task_name = "canc_type_class_33"
    config_node = "canc_type_class_33"
    checkpoint_model_keys = BASE_CHECKPOINT_MODEL_KEYS
    random_init_model_key = BASE_RANDOM_INIT_MODEL_KEY
    default_cohorts = DEFAULT_COHORTS

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "canc_type_class_33" in cfg.finetune:
            return cfg.finetune.canc_type_class_33
        raise ValueError(
            "Could not find a 33-class cancer type classification config. "
            "Expected cfg.finetune.canc_type_class_33."
        )

    def _load_tcga(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "tcga_data_dir", None)
        if not configured_path:
            raise ValueError(f"finetune.{self.config_node}.tcga_data_dir must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"TCGA h5ad file not found: {data_path}")

        adata = ad.read_h5ad(data_path)
        required_obs = {"sample_id", "patient_id", "project"}
        missing_obs = sorted(required_obs.difference(adata.obs.columns))
        if missing_obs:
            raise ValueError(f"TCGA AnnData is missing required obs columns: {missing_obs}.")

        adata.obs["project"] = adata.obs["project"].astype(str).str.strip().str.upper()

        cohorts = list(getattr(self.task_cfg, "cohorts", self.default_cohorts))
        selected_cancer_types = {str(cohort).upper() for cohort in cohorts}
        keep_mask = adata.obs["project"].isin(selected_cancer_types).to_numpy()
        adata = adata[keep_mask].copy()

        if adata.n_obs == 0:
            raise ValueError(
                f"No TCGA samples matched cohorts {cohorts} in the project metadata."
            )

        adata.obs["cancer_type"] = adata.obs["project"].astype(str)
        adata.obs_names = adata.obs["sample_id"].astype(str)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        return adata
