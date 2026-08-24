from __future__ import annotations

import logging

import hydra
import numpy as np
import torch
import torch.distributed as dist
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.deconv.runner import DeconvRunner
from finetune.training_correctness import validate_backbone_checkpoint
from run_provenance import complete_run_metadata
from utils import seed_all

log = logging.getLogger(__name__)


class DeconvPCARFRunner(DeconvRunner):
    """Frozen scbFM CLS embeddings followed by PCA and multi-output RF."""

    def _training_objective(self) -> str:
        return "random_forest_squared_error"

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "pca_rf_variant", "") or "").strip()
        return variant or "pca_rf"

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
            validate_backbone_checkpoint(
                checkpoint,
                resolved_path,
                gene_num=int(self.model_cfg.gene_num),
                selected_gene_count=self.selected_gene_count,
                max_seq_len=self.max_seq_len,
                bin_num=int(self.model_cfg.bin_num),
            )
            backbone.load_state_dict(
                self._strip_module_prefix(checkpoint["model_state_dict"])
            )
        backbone = backbone.to(self.device)
        backbone.eval()
        return backbone

    def _extract_embeddings(self, backbone, loader) -> tuple[np.ndarray, np.ndarray]:
        embeddings = []
        targets = []
        with torch.no_grad():
            for batch, batch_targets in loader:
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                hidden = backbone(
                    batch["gene_ids"],
                    batch["expr"],
                    src_key_padding_mask=batch.get("attention_key_padding_mask"),
                )
                embeddings.append(hidden[:, 0, :].detach().cpu().numpy())
                targets.append(batch_targets.numpy())
        return np.vstack(embeddings), np.vstack(targets)

    def _fit_predict_features(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        *,
        prefix: str,
        rf_prefix: str | None = None,
    ) -> tuple[dict[str, float], dict[str, object]]:
        rf_prefix = prefix if rf_prefix is None else rf_prefix
        if sparse.issparse(train_x):
            train_x = train_x.toarray()
        if sparse.issparse(test_x):
            test_x = test_x.toarray()
        train_x = np.asarray(train_x, dtype=np.float32)
        test_x = np.asarray(test_x, dtype=np.float32)

        if bool(getattr(self.task_cfg, f"{prefix}_standardize", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        requested_components = int(getattr(self.task_cfg, f"{prefix}_components", 256))
        n_components = min(requested_components, train_x.shape[0] - 1, train_x.shape[1])
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than one component.")
        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, f"{prefix}_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(train_x)
        test_z = pca.transform(test_x)

        rf = RandomForestRegressor(
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
            bootstrap=bool(getattr(self.task_cfg, f"{rf_prefix}_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, f"{rf_prefix}_n_jobs", -1)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        rf.fit(train_z, train_y)
        if self.fold_train_target_mean is None:
            raise RuntimeError("Training-fold target mean is unavailable for RF normalization.")
        train_pred = self._normalize_composition_predictions(
            rf.predict(train_z),
            self.fold_train_target_mean,
        )
        test_pred = self._normalize_composition_predictions(
            rf.predict(test_z),
            self.fold_train_target_mean,
        )
        train_loss = self._distribution_metrics(train_pred, train_y)[0]
        test_loss = self._distribution_metrics(test_pred, test_y)[0]
        test_metrics = self._evaluation_metrics_from_arrays(
            test_pred,
            test_y,
            test_loss=test_loss,
        )
        test_metrics.update(
            {
                "pca_components": int(n_components),
                "input_feature_dim": int(train_x.shape[1]),
            }
        )
        return {"loss": float(train_loss)}, test_metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("Deconvolution scbFM PCA+RF must use one process.")
            adata, targets, groups = self._prepare_cv_data()
            splits = self._build_or_load_cv_splits(adata, groups)
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            aggregate_rows = []
            for model_idx, (model_key, checkpoint_path) in enumerate(
                checkpoint_paths.items()
            ):
                fold_rows = []
                prediction_rows = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + model_idx * 10000
                        + fold_idx
                    )
                    train_adata = adata[train_idx].copy()
                    test_adata = adata[test_idx].copy()
                    self._build_loaders(
                        train_adata,
                        test_adata,
                        targets[train_idx],
                        targets[test_idx],
                    )
                    backbone = self._build_backbone(checkpoint_path)
                    train_x, train_y = self._extract_embeddings(
                        backbone,
                        self.train_loader,
                    )
                    test_x, test_y = self._extract_embeddings(
                        backbone,
                        self.test_loader,
                    )
                    train_metrics, test_metrics = self._fit_predict_features(
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        prefix="pca_rf",
                    )
                    fold_rows.append(
                        self._flatten_fold_metrics(
                            model_key,
                            fold_idx,
                            len(splits),
                            checkpoint_path,
                            train_metrics,
                            test_metrics,
                        )
                    )
                    prediction_rows.extend(
                        self._prediction_rows(
                            model_key,
                            fold_idx,
                            checkpoint_path,
                            test_adata,
                            test_metrics,
                        )
                    )
                    del backbone
                    self._cleanup_fold_state()
                    del train_adata, test_adata, train_x, train_y, test_x, test_y
                aggregate_rows.append(
                    self._write_model_results(
                        checkpoint_path,
                        fold_rows,
                        prediction_rows,
                    )
                )

            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            self._write_csv(output_path, aggregate_rows)
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": aggregate_rows}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
