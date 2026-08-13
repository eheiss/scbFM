from __future__ import annotations

import logging

import hydra
import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.gene_essent.runner import GeneEssentRunner
from finetune.training_correctness import validate_backbone_checkpoint
from run_provenance import complete_run_metadata
from utils import seed_all

log = logging.getLogger(__name__)


class GeneEssentPCARFRunner(GeneEssentRunner):
    """Frozen contextualized gene states followed by PCA and a shared RF."""

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

    def _extract_gene_embeddings(
        self,
        backbone: CancerFoundationBackbone,
        loader,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.fold_valid_gene_mask is None:
            raise RuntimeError("Fold-level valid-gene mask is unavailable.")
        valid_mask = torch.as_tensor(self.fold_valid_gene_mask, device=self.device)
        embeddings = []
        targets = []
        with torch.no_grad():
            for batch, batch_targets in loader:
                batch = self._move_batch_to_device(batch)
                hidden = backbone(
                    batch["gene_ids"],
                    batch["expr"],
                    src_key_padding_mask=batch.get("attention_key_padding_mask"),
                )
                embeddings.append(
                    hidden[:, 1:, :][:, valid_mask, :]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                targets.append(batch_targets[:, self.fold_valid_gene_mask].numpy())
        return np.concatenate(embeddings, axis=0), np.concatenate(targets, axis=0)

    @staticmethod
    def _metrics_from_matrices(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, object]:
        pccs: list[float] = []
        sccs: list[float] = []
        for prediction, target in zip(predictions, targets):
            finite = np.isfinite(prediction) & np.isfinite(target)
            if finite.sum() < 2:
                pccs.append(float("nan"))
                sccs.append(float("nan"))
                continue
            pcc = pearsonr(prediction[finite], target[finite])[0]
            scc = spearmanr(prediction[finite], target[finite])[0]
            pccs.append(float(pcc) if np.isfinite(pcc) else float("nan"))
            sccs.append(float(scc) if np.isfinite(scc) else float("nan"))
        finite = np.isfinite(predictions) & np.isfinite(targets)
        mse = float(np.mean((predictions[finite] - targets[finite]) ** 2))
        return {
            "loss": mse,
            "pcc": float(np.nanmean(pccs)),
            "scc": float(np.nanmean(sccs)),
            "pcc_per_cell_line": pccs,
            "scc_per_cell_line": sccs,
            "n_test_samples": int(targets.shape[0]),
        }

    def _fit_predict_gene_features(
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
        train_finite = np.isfinite(train_y)
        if int(train_finite.sum()) < 2:
            raise ValueError("Fewer than two observed training gene-effect scores.")
        flat_train_x = np.asarray(train_x[train_finite], dtype=np.float32)
        flat_train_y = np.asarray(train_y[train_finite], dtype=np.float32)
        flat_test_x = np.asarray(
            test_x.reshape(-1, test_x.shape[-1]),
            dtype=np.float32,
        )

        if bool(getattr(self.task_cfg, f"{prefix}_standardize", True)):
            scaler = StandardScaler()
            flat_train_x = scaler.fit_transform(flat_train_x)
            flat_test_x = scaler.transform(flat_test_x)

        requested_components = int(
            getattr(self.task_cfg, f"{prefix}_components", 256)
        )
        n_components = min(
            requested_components,
            flat_train_x.shape[0] - 1,
            flat_train_x.shape[1],
        )
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than one component.")
        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, f"{prefix}_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(flat_train_x)
        test_z = pca.transform(flat_test_x)

        rf = RandomForestRegressor(
            n_estimators=int(
                getattr(self.task_cfg, f"{rf_prefix}_n_estimators", 500)
            ),
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
            bootstrap=bool(
                getattr(self.task_cfg, f"{rf_prefix}_bootstrap", True)
            ),
            n_jobs=int(getattr(self.task_cfg, f"{rf_prefix}_n_jobs", -1)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        rf.fit(train_z, flat_train_y)
        train_prediction = rf.predict(train_z)
        test_prediction = rf.predict(test_z).reshape(test_y.shape)
        train_loss = float(np.mean((train_prediction - flat_train_y) ** 2))
        test_metrics = self._metrics_from_matrices(test_prediction, test_y)
        test_metrics.update(
            {
                "pca_components": int(n_components),
                "input_feature_dim": int(flat_train_x.shape[1]),
                "n_training_gene_pairs": int(flat_train_y.size),
            }
        )
        return {"loss": train_loss}, test_metrics

    def _finish_distributed_embedding_phase(self) -> None:
        """Close the GPU process group before rank-zero-only CPU fitting."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("Gene-essentiality scbFM PCA+RF must use one process.")

            X_expression, Y_targets, cell_ids = self._load_depmap_data()
            splits = self._build_or_load_cv_splits(cell_ids)
            fold_gene_indices = [
                self._select_training_hvg_indices(X_expression, train_idx)
                for train_idx, _ in splits
            ]
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            aggregate_rows = []
            for model_idx, (model_key, checkpoint_path) in enumerate(
                checkpoint_paths.items()
            ):
                fold_rows = []
                cell_line_rows = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                    seed_all(
                        int(getattr(self.task_cfg, "random_seed", 42))
                        + model_idx * 10000
                        + fold_idx
                    )
                    self.fold_gene_indices = fold_gene_indices[fold_idx - 1]
                    self.fold_valid_gene_mask = self.valid_gene_mask[
                        self.fold_gene_indices
                    ]
                    self.fold_n_valid_genes = int(self.fold_valid_gene_mask.sum())
                    self._build_loaders(
                        X_expression[train_idx],
                        Y_targets[train_idx],
                        X_expression[test_idx],
                        Y_targets[test_idx],
                    )
                    backbone = self._build_backbone(checkpoint_path)
                    train_x, train_y = self._extract_gene_embeddings(
                        backbone,
                        self.train_loader,
                    )
                    test_x, test_y = self._extract_gene_embeddings(
                        backbone,
                        self.test_loader,
                    )
                    train_metrics, test_metrics = self._fit_predict_gene_features(
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
                    cell_line_rows.extend(
                        self._per_cell_line_rows(
                            model_key,
                            fold_idx,
                            checkpoint_path,
                            [cell_ids[index] for index in test_idx],
                            test_metrics,
                        )
                    )
                    del backbone, train_x, train_y, test_x, test_y
                    self._cleanup_fold_state()
                aggregate_rows.append(
                    self._write_model_results(
                        checkpoint_path,
                        fold_rows,
                        cell_line_rows,
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
