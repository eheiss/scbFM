from __future__ import annotations

import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from finetune.canc_type_class.pca_rf_runner import CancTypeClassPCARFRunner
from finetune.surv_pred_binary.runner import SurvPredBinaryRunner


class BinarySurvivalRFMetricMixin:
    """Random-forest fitting and probability metrics for the OS-status task."""

    def _fit_binary_rf(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        *,
        prefix: str,
    ) -> tuple[dict[str, float], dict[str, object]]:
        if bool(getattr(self.task_cfg, f"{prefix}_standardize", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        requested = int(getattr(self.task_cfg, f"{prefix}_components", 256))
        n_components = min(requested, train_x.shape[0] - 1, train_x.shape[1])
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than one component.")
        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, f"{prefix}_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(train_x)
        test_z = pca.transform(test_x)

        rf_prefix = {
            "pca_rf": "pca_rf",
            "raw_pca_rf": "raw_pca_rf",
            "scgpt_pca": "scgpt_rf",
            "bulkformer_pca": "bulkformer_rf",
        }.get(prefix, prefix)
        rf = RandomForestClassifier(
            n_estimators=int(getattr(self.task_cfg, f"{rf_prefix}_n_estimators", 500)),
            max_depth=getattr(self.task_cfg, f"{rf_prefix}_max_depth", None),
            min_samples_leaf=int(
                getattr(self.task_cfg, f"{rf_prefix}_min_samples_leaf", 1)
            ),
            min_samples_split=int(
                getattr(self.task_cfg, f"{rf_prefix}_min_samples_split", 2)
            ),
            max_features=str(
                getattr(self.task_cfg, f"{rf_prefix}_max_features", "sqrt")
            ),
            class_weight=str(
                getattr(self.task_cfg, f"{rf_prefix}_class_weight", "balanced")
            ),
            bootstrap=bool(getattr(self.task_cfg, f"{rf_prefix}_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, f"{rf_prefix}_n_jobs", -1)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        rf.fit(train_z, train_y)
        train_pred = rf.predict(train_z)
        pred = rf.predict(test_z)
        positive_column = int(np.where(rf.classes_ == 1)[0][0])
        probability = rf.predict_proba(test_z)[:, positive_column]
        precision, recall, fscore, support = precision_recall_fscore_support(
            test_y, pred, labels=np.arange(2), zero_division=0
        )
        try:
            auroc = float(roc_auc_score(test_y, probability))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(test_y, probability))
        except ValueError:
            auprc = float("nan")

        train_metrics = {
            "loss": float("nan"),
            "accuracy": 100.0 * float(accuracy_score(train_y, train_pred)),
        }
        test_metrics = {
            "loss": float("nan"),
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": float(accuracy_score(test_y, pred)),
            "f1_macro": float(f1_score(test_y, pred, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(test_y, pred, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(test_y, pred, labels=np.arange(2)),
            "classification_report": classification_report(
                test_y,
                pred,
                labels=np.arange(2),
                target_names=["alive_or_censored", "event"],
                digits=4,
                zero_division=0,
            ),
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": fscore,
            "support_per_class": support,
            "label_dict": ["0", "1"],
            "n_test_samples": int(test_y.size),
            "truth_indices": test_y,
            "prediction_indices": pred,
            "positive_probabilities": probability,
            "pca_components": int(n_components),
            "embedding_dim": int(train_x.shape[1]),
        }
        return train_metrics, test_metrics


class SurvPredBinaryPCARFRunner(BinarySurvivalRFMetricMixin, CancTypeClassPCARFRunner):
    task_name = "surv_pred_binary"
    config_node = "surv_pred_binary"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredBinaryRunner._resolve_task_cfg(cfg)

    def _prepare_cv_data(self):
        return SurvPredBinaryRunner._prepare_cv_data(self)

    def _get_checkpoint_paths(self):
        return SurvPredBinaryRunner._get_checkpoint_paths(self)

    def _fit_predict_embeddings(self, train_x, train_y, test_x, test_y):
        return self._fit_binary_rf(
            train_x, train_y, test_x, test_y, prefix="pca_rf"
        )

    def _prediction_rows(self, model_key, fold, checkpoint_path, test_adata, test_metrics):
        return SurvPredBinaryRunner._prediction_rows(
            self, model_key, fold, checkpoint_path, test_adata, test_metrics
        )
