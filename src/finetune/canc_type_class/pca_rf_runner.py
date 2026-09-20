from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from paths import output_root
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

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.canc_type_class.runner import (
    CHECKPOINT_MODEL_KEYS,
    RANDOM_INIT_MODEL_KEY,
    CancTypeClassRunner,
)
from utils import seed_all

log = logging.getLogger(__name__)


class CancTypeClassPCARFRunner(CancTypeClassRunner):
    """BulkFormer-style baseline: frozen backbone embeddings -> PCA -> random forest."""

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
        variant = str(getattr(self.task_cfg, "pca_rf_variant", "") or "").strip()
        return variant or "pca_rf"

    def _task_output_dir(self) -> Path:
        return output_root(self.cfg) / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _save_run_metadata(self, checkpoint_paths: dict[str, str] | None = None) -> None:
        super()._save_run_metadata(checkpoint_paths or {})

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                f"finetune.{self.config_node}.pretrained_model_paths must define "
                f"{', '.join(CHECKPOINT_MODEL_KEYS)}."
            )

        checkpoint_paths: dict[str, str] = {}
        missing = []
        for key in CHECKPOINT_MODEL_KEYS:
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
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
        return checkpoint_paths

    def _build_backbone(self, checkpoint_path: str) -> CancerFoundationBackbone:
        backbone = CancerFoundationBackbone(
            num_gene_tokens=self.num_gene_tokens,
            d_model=int(self.model_cfg.embsize),
            nhead=int(self.model_cfg.nheads),
            d_hid=int(self.model_cfg.d_hid),
            nlayers=int(self.model_cfg.nlayers),
            dropout=float(self.model_cfg.dropout),
            pad_gene_id=self.pad_gene_id,
            max_value=int(getattr(self.model_cfg, "value_encoder_max_value", 512)),
        )
        if checkpoint_path:
            resolved_path = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved_path, map_location="cpu")
            self._validate_backbone_checkpoint(checkpoint, resolved_path)
            state_dict = self._strip_module_prefix(checkpoint["model_state_dict"])
            backbone.load_state_dict(state_dict)
            log.info("Loaded pretrained checkpoint from %s", resolved_path)
        else:
            log.info("Using randomly initialized backbone for PCA+RF")
        backbone = backbone.to(self.device)
        backbone.eval()
        return backbone

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0, :]

    def _extract_embeddings(
        self,
        backbone: CancerFoundationBackbone,
        loader,
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch, batch_labels in loader:
                model_batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                hidden = backbone(
                    model_batch["gene_ids"],
                    model_batch["expr"],
                    src_key_padding_mask=model_batch.get("attention_key_padding_mask"),
                )
                embeddings.append(self._pool_hidden(hidden).detach().cpu().numpy())
                labels.append(batch_labels.numpy())
        return np.vstack(embeddings), np.concatenate(labels)

    def _fit_predict_embeddings(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, object]]:
        if bool(getattr(self.task_cfg, "pca_rf_standardize", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        requested_components = int(getattr(self.task_cfg, "pca_rf_components", 256))
        n_components = min(requested_components, train_x.shape[0] - 1, train_x.shape[1])
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than 1 component.")

        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, "pca_rf_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(train_x)
        test_z = pca.transform(test_x)

        rf = RandomForestClassifier(
            n_estimators=int(getattr(self.task_cfg, "pca_rf_n_estimators", 500)),
            max_depth=getattr(self.task_cfg, "pca_rf_max_depth", None),
            min_samples_leaf=int(getattr(self.task_cfg, "pca_rf_min_samples_leaf", 1)),
            min_samples_split=int(getattr(self.task_cfg, "pca_rf_min_samples_split", 2)),
            max_features=str(getattr(self.task_cfg, "pca_rf_max_features", "sqrt")),
            class_weight=str(getattr(self.task_cfg, "pca_rf_class_weight", "balanced")),
            bootstrap=bool(getattr(self.task_cfg, "pca_rf_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, "pca_rf_n_jobs", -1)),
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
                raise ValueError("Backbone embedding PCA+RF baseline should be launched with one process.")

            adata, labels, groups = self._prepare_cv_data()
            splits = self._build_or_load_cv_splits(adata, labels, groups)
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            if self.is_master:
                log.info(
                    "Prepared backbone PCA+RF classification data: samples=%d, genes=%d, folds=%d",
                    adata.n_obs,
                    adata.n_vars,
                    len(splits),
                )

            aggregate_rows: list[dict[str, object]] = []
            for model_idx, (model_key, checkpoint_path) in enumerate(checkpoint_paths.items()):
                fold_rows: list[dict[str, object]] = []
                prediction_rows: list[dict[str, object]] = []
                confusion_matrices: list[np.ndarray] = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + model_idx * 10000
                        + fold_idx
                    )
                    train_adata = adata[train_idx].copy()
                    test_adata = adata[test_idx].copy()
                    if self.is_master:
                        log.info(
                            "PCA+RF %s | Fold %d/%d | train=%d, test=%d",
                            model_key,
                            fold_idx,
                            len(splits),
                            train_adata.n_obs,
                            test_adata.n_obs,
                        )

                    self._build_loaders(train_adata, test_adata)
                    backbone = self._build_backbone(checkpoint_path)
                    train_x, train_y = self._extract_embeddings(backbone, self.train_loader)
                    test_x, test_y = self._extract_embeddings(backbone, self.test_loader)
                    train_metrics, test_metrics = self._fit_predict_embeddings(
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                    )

                    fold_rows.append(
                        self._flatten_fold_metrics(
                            model_key=model_key,
                            fold=fold_idx,
                            n_folds=len(splits),
                            checkpoint_path=checkpoint_path,
                            train_metrics=train_metrics,
                            test_metrics=test_metrics,
                        )
                    )
                    prediction_rows.extend(
                        self._prediction_rows(
                            model_key=model_key,
                            fold=fold_idx,
                            checkpoint_path=checkpoint_path,
                            test_adata=test_adata,
                            test_metrics=test_metrics,
                        )
                    )
                    confusion_matrices.append(np.asarray(test_metrics["confusion_matrix"]))
                    self._cleanup_fold_state()

                aggregate_rows.append(
                    self._write_model_results(
                        checkpoint_path,
                        fold_rows,
                        prediction_rows,
                        confusion_matrices,
                    )
                )

            out_dir = self._task_output_dir()
            output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
            self._write_csv(output_path, aggregate_rows)
            from run_provenance import complete_run_metadata

            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": aggregate_rows}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
