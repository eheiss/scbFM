from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
from omegaconf import DictConfig

from finetune.canc_type_class.runner import CancTypeClassRunner
from preprocess import filter_min_genes, reindex_adata_genes, validate_token_matrix

log = logging.getLogger(__name__)


class DiseaseClassRunner(CancTypeClassRunner):
    """DiSignAtlas classification using the shared CancerFoundation fine-tuning stack."""

    task_name = "disease_class"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "disease_class" in cfg.finetune:
            return cfg.finetune.disease_class
        raise ValueError(
            "Could not find a disease classification config. "
            "Expected cfg.finetune.disease_class."
        )

    def _load_disignatlas(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "disignatlas_data_path", None)
        if not configured_path:
            raise ValueError("finetune.disease_class.disignatlas_data_path must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"DiSignAtlas h5ad file not found: {data_path}")

        adata = ad.read_h5ad(data_path)
        disease_label_col = str(getattr(self.task_cfg, "disease_label_col", "disease"))
        if disease_label_col not in adata.obs:
            raise ValueError(
                f"obs column '{disease_label_col}' not found in DiSignAtlas h5ad. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        binary_label_col = str(getattr(self.task_cfg, "binary_label_col", "binary_label"))
        if binary_label_col not in adata.obs:
            raise ValueError(
                f"obs column '{binary_label_col}' not found in DiSignAtlas h5ad. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        n_before = adata.n_obs
        adata = adata[adata.obs[binary_label_col].astype(str) == "case"].copy()
        if adata.n_obs == 0:
            raise ValueError(
                f"No case samples found in obs['{binary_label_col}']."
            )

        adata.obs["disease_label"] = adata.obs[disease_label_col].astype(str)
        # The shared classification runner consistently consumes this internal
        # label column; output files still use the original disease names.
        adata.obs["cancer_type"] = adata.obs["disease_label"]
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        log.info("Filtered DiSignAtlas to cases: %d -> %d samples", n_before, adata.n_obs)
        return adata

    def _load_input_adata(self) -> ad.AnnData:
        log.info("Loading DiSignAtlas cases for disease classification")
        return self._load_disignatlas()

    def _preprocess_adata(self, adata: ad.AnnData) -> ad.AnnData:
        gene_list_path = self._resolve_gene_list_path()
        adata, missing_genes = reindex_adata_genes(
            adata,
            gene_list_path=gene_list_path,
        )
        if self._should_preprocess_input():
            adata = filter_min_genes(
                adata,
                min_genes=int(getattr(self.task_cfg, "min_genes", 200)),
            )
            log.info(
                "Aligned raw DiSignAtlas input for on-the-fly sequence binning: "
                "%d target genes missing, output shape %s",
                len(missing_genes),
                adata.shape,
            )
        else:
            validate_token_matrix(
                adata.X,
                bin_num=int(self.model_cfg.bin_num),
                name="disease classification input data",
            )
            log.info(
                "Reindexed preprocessed DiSignAtlas input: "
                "%d target genes missing, output shape %s",
                len(missing_genes),
                adata.shape,
            )

        self._missing_genes_note = (
            "Model genes missing from DiSignAtlas and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
            if missing_genes
            else ""
        )

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(
                f"Expected {expected_gene_num} genes for the scbFM backbone, "
                f"got {adata.n_vars}."
            )
        return adata

    def _prepare_cv_data(
        self,
    ) -> tuple[ad.AnnData, np.ndarray, np.ndarray | None]:
        adata = self._preprocess_adata(self._load_input_adata())
        labels = adata.obs["disease_label"].astype(str).to_numpy()
        adata.obs["cancer_type"] = labels
        self.label_dict = np.unique(labels)
        log.info(
            "Prepared DiSignAtlas data: samples=%d, genes=%d, diseases=%d",
            adata.n_obs,
            adata.n_vars,
            len(self.label_dict),
        )
        return adata, labels, None

    def _prediction_rows(
        self,
        model_key: str,
        fold: int,
        checkpoint_path: str,
        test_adata: ad.AnnData,
        test_metrics: dict[str, object],
    ) -> list[dict[str, object]]:
        truth_indices = np.asarray(test_metrics["truth_indices"], dtype=int)
        prediction_indices = np.asarray(test_metrics["prediction_indices"], dtype=int)
        labels = self.label_dict.tolist()
        dataset_ids = (
            test_adata.obs["dataset"].astype(str).to_numpy()
            if "dataset" in test_adata.obs
            else np.asarray([""] * test_adata.n_obs)
        )

        rows: list[dict[str, object]] = []
        for idx, (truth_idx, pred_idx) in enumerate(
            zip(truth_indices, prediction_indices)
        ):
            rows.append(
                {
                    "model": model_key,
                    "fold": fold,
                    "finetune_mode": self._finetune_mode(),
                    "checkpoint_path": checkpoint_path,
                    "sample_id": str(test_adata.obs_names[idx]),
                    "dataset_id": dataset_ids[idx],
                    "true_idx": int(truth_idx),
                    "pred_idx": int(pred_idx),
                    "true_label": labels[int(truth_idx)],
                    "pred_label": labels[int(pred_idx)],
                    "correct": int(truth_idx == pred_idx),
                }
            )
        return rows
