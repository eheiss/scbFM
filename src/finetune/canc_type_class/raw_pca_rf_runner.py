from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from finetune.canc_type_class.runner import ROOT, CancTypeClassRunner
from utils import seed_all

log = logging.getLogger(__name__)


class CancTypeClassRawPCARFRunner(CancTypeClassRunner):
    """Raw-expression baseline: expression matrix -> PCA -> random forest."""

    task_name = "canc_type_class"
    config_node = "canc_type_class"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "canc_type_class" in cfg.finetune:
            return cfg.finetune.canc_type_class
        raise ValueError(
            "Could not find cancer type classification config. "
            "Expected cfg.finetune.canc_type_class."
        )

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "raw_pca_rf_variant", "") or "").strip()
        if not variant:
            feature_mode = str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes"))
            variant = f"raw_pca_rf_{feature_mode}"
        return variant

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _save_run_metadata(self, checkpoint_paths: dict[str, str] | None = None) -> None:
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
                "task": "finetune.canc_type_class_raw_pca_rf",
                "baseline": "raw_expression_pca_random_forest",
                "variant": self._finetune_mode(),
                "feature_mode": str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes")),
                "hvg_selection_method": str(getattr(self.task_cfg, "hvg_selection_method", "mad")),
                "pca_components": int(getattr(self.task_cfg, "raw_pca_rf_components", 256)),
                "rf_n_estimators": int(getattr(self.task_cfg, "raw_pca_rf_n_estimators", 500)),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "git_commit": self._get_git_commit(),
            },
        )

    def _select_feature_indices(self, train_adata: ad.AnnData) -> np.ndarray:
        feature_mode = str(getattr(self.task_cfg, "raw_pca_rf_feature_mode", "all_genes"))
        if feature_mode == "all_genes":
            return np.arange(train_adata.n_vars, dtype=np.int64)
        if feature_mode == "hvg1199":
            return self._select_training_hvg_indices(train_adata)
        raise ValueError(
            "finetune.canc_type_class.raw_pca_rf_feature_mode must be one of: "
            "all_genes, hvg1199."
        )

    @staticmethod
    def _matrix_to_array(data, feature_indices: np.ndarray) -> np.ndarray:
        selected = data[:, feature_indices]
        if sparse.issparse(selected):
            selected = selected.toarray()
        return np.asarray(selected, dtype=np.float32)

    def _fit_predict_raw(
        self,
        train_adata: ad.AnnData,
        train_y: np.ndarray,
        test_adata: ad.AnnData,
        test_y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, object]]:
        train_x = self._matrix_to_array(train_adata.X, feature_indices)
        test_x = self._matrix_to_array(test_adata.X, feature_indices)

        if bool(getattr(self.task_cfg, "raw_pca_rf_standardize", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        requested_components = int(getattr(self.task_cfg, "raw_pca_rf_components", 256))
        n_components = min(requested_components, train_x.shape[0] - 1, train_x.shape[1])
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than 1 component.")

        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, "raw_pca_rf_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(train_x)
        test_z = pca.transform(test_x)

        rf = RandomForestClassifier(
            n_estimators=int(getattr(self.task_cfg, "raw_pca_rf_n_estimators", 500)),
            max_depth=getattr(self.task_cfg, "raw_pca_rf_max_depth", None),
            min_samples_leaf=int(getattr(self.task_cfg, "raw_pca_rf_min_samples_leaf", 1)),
            min_samples_split=int(getattr(self.task_cfg, "raw_pca_rf_min_samples_split", 2)),
            max_features=str(getattr(self.task_cfg, "raw_pca_rf_max_features", "sqrt")),
            class_weight=str(getattr(self.task_cfg, "raw_pca_rf_class_weight", "balanced")),
            bootstrap=bool(getattr(self.task_cfg, "raw_pca_rf_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, "raw_pca_rf_n_jobs", -1)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        rf.fit(train_z, train_y)

        train_pred = rf.predict(train_z)
        pred = rf.predict(test_z)
        precision, recall, fscore, support = precision_recall_fscore_support(
            test_y,
            pred,
            labels=np.arange(len(self.label_dict)),
            zero_division=0,
        )
        train_metrics = {
            "loss": float("nan"),
            "accuracy": 100.0 * float(accuracy_score(train_y, train_pred)),
        }
        test_metrics = {
            "loss": float("nan"),
            "accuracy": float(accuracy_score(test_y, pred)),
            "f1_macro": float(
                f1_score(
                    test_y,
                    pred,
                    labels=np.arange(len(self.label_dict)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    test_y,
                    pred,
                    labels=np.arange(len(self.label_dict)),
                    average="weighted",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                test_y,
                pred,
                labels=np.arange(len(self.label_dict)),
            ),
            "classification_report": classification_report(
                test_y,
                pred,
                labels=np.arange(len(self.label_dict)),
                target_names=self.label_dict.tolist(),
                digits=4,
                zero_division=0,
            ),
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": fscore,
            "support_per_class": support,
            "label_dict": self.label_dict.tolist(),
            "n_test_samples": int(test_y.size),
            "truth_indices": test_y,
            "prediction_indices": pred,
            "pca_components": int(n_components),
            "embedding_dim": int(train_x.shape[1]),
        }
        return train_metrics, test_metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("Raw PCA+RF baseline should be launched with one process.")

            adata, labels_str, groups = self._prepare_cv_data()
            splits = self._build_cv_splits(labels_str, groups)
            self._save_run_metadata({})
            label_to_idx = {label: idx for idx, label in enumerate(self.label_dict.tolist())}
            labels = np.array([label_to_idx[label] for label in labels_str], dtype=np.int64)

            if self.is_master:
                log.info(
                    "Prepared raw PCA+RF TCGA 5-type classification data: "
                    "samples=%d, genes=%d, folds=%d",
                    adata.n_obs,
                    adata.n_vars,
                    len(splits),
                )

            model_key = "raw_pca_rf"
            fold_rows: list[dict[str, object]] = []
            prediction_rows: list[dict[str, object]] = []
            confusion_matrices: list[np.ndarray] = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + fold_idx)
                train_adata = adata[train_idx].copy()
                test_adata = adata[test_idx].copy()
                train_y = labels[train_idx]
                test_y = labels[test_idx]
                feature_indices = self._select_feature_indices(train_adata)

                if self.is_master:
                    log.info(
                        "Raw PCA+RF | Fold %d/%d | train=%d, test=%d, features=%d",
                        fold_idx,
                        len(splits),
                        train_adata.n_obs,
                        test_adata.n_obs,
                        feature_indices.size,
                    )

                train_metrics, test_metrics = self._fit_predict_raw(
                    train_adata=train_adata,
                    train_y=train_y,
                    test_adata=test_adata,
                    test_y=test_y,
                    feature_indices=feature_indices,
                )

                if self.is_master:
                    log.info(
                        "Raw PCA+RF | Fold %d/%d | PCA=%d | Accuracy: %.4f | Weighted F1: %.4f",
                        fold_idx,
                        len(splits),
                        int(test_metrics["pca_components"]),
                        float(test_metrics["accuracy"]),
                        float(test_metrics["f1_weighted"]),
                    )
                    fold_rows.append(
                        self._flatten_fold_metrics(
                            model_key=model_key,
                            fold=fold_idx,
                            n_folds=len(splits),
                            checkpoint_path="",
                            train_metrics=train_metrics,
                            test_metrics=test_metrics,
                        )
                    )
                    prediction_rows.extend(
                        self._prediction_rows(
                            model_key=model_key,
                            fold=fold_idx,
                            checkpoint_path="",
                            test_adata=test_adata,
                            test_metrics=test_metrics,
                        )
                    )
                    confusion_matrices.append(np.asarray(test_metrics["confusion_matrix"]))

                self._cleanup_fold_state()
                if self.is_distributed:
                    dist.barrier()

            if self.is_master:
                aggregate = self._write_model_results(
                    "",
                    fold_rows,
                    prediction_rows,
                    confusion_matrices,
                )
                out_dir = self._task_output_dir()
                output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
                self._write_csv(output_path, [aggregate])
                return {"results_path": str(output_path), "results": [aggregate]}
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
