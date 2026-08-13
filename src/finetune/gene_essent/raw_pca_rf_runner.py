from __future__ import annotations

import logging

import numpy as np
import torch.distributed as dist
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from finetune.gene_essent.pca_rf_runner import GeneEssentPCARFRunner
from run_provenance import complete_run_metadata
from utils import seed_all

log = logging.getLogger(__name__)


class GeneEssentRawPCARFRunner(GeneEssentPCARFRunner):
    """Raw log1p expression followed by PCA and multi-output RF."""

    def _finetune_mode(self) -> str:
        variant = str(
            getattr(self.task_cfg, "raw_pca_rf_variant", "") or ""
        ).strip()
        if variant:
            return variant
        mode = str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes"))
        return f"raw_pca_rf_{mode}"

    def _get_checkpoint_paths(self) -> dict[str, str]:
        return {"raw_pca_rf": ""}

    def _select_raw_features(self) -> np.ndarray:
        mode = str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes"))
        if mode == "all_genes":
            return np.arange(int(self.model_cfg.gene_num), dtype=np.int64)
        if mode == "hvg1199":
            if self.fold_gene_indices is None:
                raise RuntimeError("Fold-level MAD genes are unavailable.")
            return self.fold_gene_indices.astype(np.int64, copy=False)
        raise ValueError("raw_pca_rf_feature_mode must be all_genes or hvg1199.")

    def _fit_predict_raw_features(
        self,
        train_x,
        train_y: np.ndarray,
        test_x,
        test_y: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, object]]:
        if sparse.issparse(train_x):
            train_x = train_x.toarray()
        if sparse.issparse(test_x):
            test_x = test_x.toarray()
        train_x = np.asarray(train_x, dtype=np.float32)
        test_x = np.asarray(test_x, dtype=np.float32)

        if bool(getattr(self.task_cfg, "raw_pca_rf_standardize", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
        requested_components = int(
            getattr(self.task_cfg, "raw_pca_rf_components", 256)
        )
        n_components = min(
            requested_components,
            train_x.shape[0] - 1,
            train_x.shape[1],
        )
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than one component.")
        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, "raw_pca_rf_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(train_x)
        test_z = pca.transform(test_x)

        finite = np.isfinite(train_y)
        if not np.all(finite):
            observed_counts = finite.sum(axis=0)
            if np.any(observed_counts == 0):
                raise ValueError(
                    "At least one selected target gene has no observed training scores."
                )
            target_means = np.nanmean(train_y, axis=0)
            train_y_fit = np.where(finite, train_y, target_means[None, :])
            log.warning(
                "Imputed %d missing training gene-effect targets with training-fold "
                "gene means for the multi-output random forest.",
                int((~finite).sum()),
            )
        else:
            train_y_fit = train_y

        rf = RandomForestRegressor(
            n_estimators=int(
                getattr(self.task_cfg, "raw_pca_rf_n_estimators", 500)
            ),
            max_depth=getattr(self.task_cfg, "raw_pca_rf_max_depth", None),
            min_samples_leaf=int(
                getattr(self.task_cfg, "raw_pca_rf_min_samples_leaf", 1)
            ),
            min_samples_split=int(
                getattr(self.task_cfg, "raw_pca_rf_min_samples_split", 2)
            ),
            max_features=str(
                getattr(self.task_cfg, "raw_pca_rf_max_features", "sqrt")
            ),
            bootstrap=bool(getattr(self.task_cfg, "raw_pca_rf_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, "raw_pca_rf_n_jobs", -1)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        rf.fit(train_z, train_y_fit)
        train_prediction = rf.predict(train_z)
        test_prediction = rf.predict(test_z)
        train_metrics_full = self._metrics_from_matrices(train_prediction, train_y)
        test_metrics = self._metrics_from_matrices(test_prediction, test_y)
        test_metrics.update(
            {
                "pca_components": int(n_components),
                "input_feature_dim": int(train_x.shape[1]),
                "imputed_training_targets": int((~finite).sum()),
            }
        )
        return {"loss": float(train_metrics_full["loss"])}, test_metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("Raw gene-essentiality PCA+RF must use one process.")
            X_expression, Y_targets, cell_ids = self._load_depmap_data()
            splits = self._build_or_load_cv_splits(cell_ids)
            fold_gene_indices = [
                self._select_training_hvg_indices(X_expression, train_idx)
                for train_idx, _ in splits
            ]
            self._save_run_metadata({"raw_pca_rf": ""})

            fold_rows = []
            cell_line_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + fold_idx)
                self.fold_gene_indices = fold_gene_indices[fold_idx - 1]
                self.fold_valid_gene_mask = self.valid_gene_mask[
                    self.fold_gene_indices
                ]
                self.fold_n_valid_genes = int(self.fold_valid_gene_mask.sum())
                feature_indices = self._select_raw_features()
                target_indices = self.fold_gene_indices[self.fold_valid_gene_mask]
                train_metrics, test_metrics = self._fit_predict_raw_features(
                    X_expression[train_idx][:, feature_indices],
                    Y_targets[train_idx][:, target_indices],
                    X_expression[test_idx][:, feature_indices],
                    Y_targets[test_idx][:, target_indices],
                )
                fold_rows.append(
                    self._flatten_fold_metrics(
                        "raw_pca_rf",
                        fold_idx,
                        len(splits),
                        "",
                        train_metrics,
                        test_metrics,
                    )
                )
                cell_line_rows.extend(
                    self._per_cell_line_rows(
                        "raw_pca_rf",
                        fold_idx,
                        "",
                        [cell_ids[index] for index in test_idx],
                        test_metrics,
                    )
                )
                self._cleanup_fold_state()

            aggregate = self._write_model_results("", fold_rows, cell_line_rows)
            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
