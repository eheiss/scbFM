from __future__ import annotations

from pathlib import Path

import anndata as ad
import hydra
from omegaconf import DictConfig, OmegaConf

from finetune.canc_type_class.runner import (
    CHECKPOINT_MODEL_KEYS as BASE_CHECKPOINT_MODEL_KEYS,
    RANDOM_INIT_MODEL_KEY as BASE_RANDOM_INIT_MODEL_KEY,
    ROOT,
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

    def _save_run_metadata(self, checkpoint_paths: dict[str, str]) -> None:
        if not self.is_master:
            return
        out_dir = self._task_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_prefix()
        (out_dir / f"{prefix}_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        self._write_json(
            out_dir / f"{prefix}_run_metadata.json",
            {
                "task": self.task_name,
                "finetune_mode": self._finetune_mode(),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 10)),
                "git_commit": self._get_git_commit(),
                "checkpoint_paths": checkpoint_paths,
            },
        )

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                f"finetune.{self.config_node}.pretrained_model_paths must define "
                f"{', '.join(self.checkpoint_model_keys)}."
            )

        checkpoint_paths: dict[str, str] = {}
        missing = []
        for key in self.checkpoint_model_keys:
            value = paths_cfg.get(key)
            if value:
                checkpoint_paths[key] = str(Path(hydra.utils.to_absolute_path(str(value))))
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f"Missing checkpoint paths in finetune.{self.config_node}.pretrained_model_paths: "
                f"{missing}"
            )
        checkpoint_paths[self.random_init_model_key] = ""
        return checkpoint_paths

    def _load_tcga(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "tcga_data_dir", None)
        if not configured_path:
            raise ValueError(f"finetune.{self.config_node}.tcga_data_dir must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"TCGA h5ad file not found: {data_path}")

        adata = ad.read_h5ad(data_path)
        if "project_id" not in adata.obs:
            raise ValueError("TCGA AnnData must contain obs['project_id'] to filter cohorts.")

        cohorts = list(getattr(self.task_cfg, "cohorts", self.default_cohorts))
        selected_projects = {f"TCGA-{cohort}" for cohort in cohorts}
        keep_mask = adata.obs["project_id"].astype(str).isin(selected_projects).to_numpy()
        adata = adata[keep_mask].copy()

        if adata.n_obs == 0:
            raise ValueError(
                f"No TCGA samples matched cohorts {cohorts} in obs['project_id']."
            )

        adata.obs["cancer_type"] = adata.obs["project_id"].astype(str).str.removeprefix("TCGA-")
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        return adata

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / self.task_name / self._finetune_mode()
