from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from finetune.drug_resp.pca_rf_runner import DrugRespPCARFRunner
from paths import output_root


class DrugRespRawPCARFRunner(DrugRespPCARFRunner):
    """Raw GDSC expression + KPGT drug features -> PCA -> RF regression."""

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "raw_pca_rf_variant", "") or "").strip()
        if not variant:
            feature_mode = str(
                getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes")
            )
            variant = f"raw_pca_rf_{feature_mode}"
        return variant

    def _task_output_dir(self) -> Path:
        return output_root(self.cfg) / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _pca_prefix(self) -> str:
        return "raw_pca_rf"

    def _rf_prefix(self) -> str:
        return "raw_pca_rf"

    def _get_checkpoint_paths(self) -> dict[str, str]:
        return {"raw_pca_rf": ""}

    def _select_training_hvg_indices(
        self,
        X_cell,
        training_cell_idxs: np.ndarray,
    ) -> np.ndarray:
        feature_mode = str(
            getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes")
        )
        if feature_mode == "all_genes":
            return np.arange(X_cell.shape[1], dtype=np.int64)
        if feature_mode == "hvg1199":
            return super()._select_training_hvg_indices(X_cell, training_cell_idxs)
        raise ValueError(
            "raw_pca_rf_feature_mode must be one of: all_genes, hvg1199."
        )

    def _extract_cell_features(self, X_cell, checkpoint_path: str) -> np.ndarray:
        if self.fold_gene_indices is None:
            raise RuntimeError("Raw PCA feature indices have not been selected.")
        selected = X_cell[:, self.fold_gene_indices]
        if sparse.issparse(selected):
            selected = selected.toarray()
        return np.asarray(selected, dtype=np.float32)
