from __future__ import annotations

import logging

import anndata as ad
import numpy as np
from omegaconf import DictConfig

from finetune.canc_type_class.bulkformer_pca_rf_runner import (
    CancTypeClassBulkFormerPCARFRunner,
)
from finetune.disease_class.runner import DiseaseClassRunner

log = logging.getLogger(__name__)


class DiseaseClassBulkFormerPCARFRunner(CancTypeClassBulkFormerPCARFRunner):
    """Frozen BulkFormer embeddings with PCA + RF for DiSignAtlas."""

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

    def _load_canonical_tcga(self) -> ad.AnnData:
        return self._load_input_adata()

    def _load_bulkformer_tcga(self, paths) -> ad.AnnData:
        canonical = self._load_canonical_tcga()
        adata = self._read_disignatlas(paths["tcga_data"])

        canonical_ids = canonical.obs["sample_id"].astype(str).tolist()
        missing_samples = [
            sample_id for sample_id in canonical_ids if sample_id not in adata.obs_names
        ]
        if missing_samples:
            raise ValueError(
                f"Full-gene DiSignAtlas data is missing {len(missing_samples)} canonical "
                f"case samples. First missing IDs: {missing_samples[:10]}"
            )
        extra_samples = int(adata.n_obs - len(canonical_ids))
        adata = adata[canonical_ids].copy()

        observed_labels = adata.obs["disease_label"].astype(str).to_numpy()
        expected_labels = canonical.obs["disease_label"].astype(str).to_numpy()
        if not np.array_equal(observed_labels, expected_labels):
            mismatches = np.flatnonzero(observed_labels != expected_labels)[:10]
            details = [
                {
                    "sample_id": canonical_ids[index],
                    "canonical": expected_labels[index],
                    "bulkformer": observed_labels[index],
                }
                for index in mismatches
            ]
            raise ValueError(
                "Full-gene and canonical DiSignAtlas disease labels disagree: "
                f"{details}"
            )

        adata.obs = canonical.obs.copy()
        adata.obs_names = canonical.obs_names.copy()
        adata.var_names_make_unique()
        log.info(
            "Aligned full-gene DiSignAtlas expression to canonical cases: "
            "samples=%d, extra_case_samples_dropped=%d",
            adata.n_obs,
            max(0, extra_samples),
        )
        return adata

    def _prepare_bulkformer_data(self, paths):
        result = super()._prepare_bulkformer_data(paths)
        expected_classes = int(getattr(self.task_cfg, "expected_disease_classes", 23))
        if len(self.label_dict) != expected_classes:
            raise ValueError(
                f"BulkFormer DiSignAtlas input contains {len(self.label_dict)} disease "
                f"classes; expected {expected_classes}."
            )
        return result

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return DiseaseClassRunner._prediction_rows(
            self,
            model_key,
            fold,
            checkpoint_path,
            test_adata,
            test_metrics,
        )
