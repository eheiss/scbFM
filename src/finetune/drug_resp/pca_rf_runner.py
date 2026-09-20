from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from cancerfoundation_backbone import CancerFoundationBackbone
from finetune.drug_resp.runner import DrugRespRunner
from paths import output_root
from finetune.training_correctness import validate_backbone_checkpoint
from utils import seed_all

log = logging.getLogger(__name__)


class DrugRespPCARFRunner(DrugRespRunner):
    """Frozen scbFM CLS embeddings + KPGT drug features -> PCA -> RF regression."""

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "pca_rf_variant", "") or "").strip()
        return variant or "pca_rf"

    def _task_output_dir(self) -> Path:
        return output_root(self.cfg) / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _pca_prefix(self) -> str:
        return "pca_rf"

    def _rf_prefix(self) -> str:
        return "pca_rf"

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
            resolved = hydra.utils.to_absolute_path(str(checkpoint_path))
            checkpoint = torch.load(resolved, map_location="cpu")
            validate_backbone_checkpoint(
                checkpoint,
                resolved,
                gene_num=int(self.model_cfg.gene_num),
                selected_gene_count=self.selected_gene_count,
                max_seq_len=self.max_seq_len,
                bin_num=int(self.model_cfg.bin_num),
            )
            backbone.load_state_dict(self._strip_module_prefix(checkpoint["model_state_dict"]))
            log.info("Loaded pretrained checkpoint from %s", resolved)
        else:
            log.info("Using randomly initialized backbone for drug-response PCA+RF")
        backbone = backbone.to(self.device)
        backbone.eval()
        return backbone

    def _extract_cell_features(
        self,
        X_cell,
        checkpoint_path: str,
    ) -> np.ndarray:
        backbone = self._build_backbone(checkpoint_path)
        batch_size = max(1, int(getattr(self.task_cfg, "batch_size", 4)) * 4)
        parts = []
        with torch.no_grad():
            for start in range(0, X_cell.shape[0], batch_size):
                indices = np.arange(start, min(start + batch_size, X_cell.shape[0]))
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in self._cell_backbone_batch(X_cell, indices).items()
                }
                hidden = backbone(batch["gene_ids"], batch["expr"])
                parts.append(hidden[:, 0, :].detach().cpu().numpy())
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.vstack(parts).astype(np.float32, copy=False)

    def _regression_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        pair_cell_ids: np.ndarray,
    ) -> dict[str, object]:
        predictions = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        finite = np.isfinite(predictions) & np.isfinite(targets)
        if finite.sum() >= 2:
            global_pcc = float(pearsonr(predictions[finite], targets[finite])[0])
            global_scc = float(spearmanr(predictions[finite], targets[finite])[0])
        else:
            global_pcc = global_scc = float("nan")

        cell_pccs: dict[str, float] = {}
        cell_sccs: dict[str, float] = {}
        for cell_id in np.unique(pair_cell_ids.astype(str)):
            cell_mask = pair_cell_ids.astype(str) == cell_id
            cell_finite = finite & cell_mask
            if cell_finite.sum() < 2:
                continue
            pcc = float(pearsonr(predictions[cell_finite], targets[cell_finite])[0])
            scc = float(spearmanr(predictions[cell_finite], targets[cell_finite])[0])
            cell_pccs[cell_id] = pcc if np.isfinite(pcc) else float("nan")
            cell_sccs[cell_id] = scc if np.isfinite(scc) else float("nan")

        return {
            "loss": float(mean_squared_error(targets[finite], predictions[finite])),
            "global_pcc": global_pcc,
            "global_scc": global_scc,
            "mean_pcc_per_cell": (
                float(np.nanmean(list(cell_pccs.values())))
                if cell_pccs
                else float("nan")
            ),
            "mean_scc_per_cell": (
                float(np.nanmean(list(cell_sccs.values())))
                if cell_sccs
                else float("nan")
            ),
            "cell_pccs": cell_pccs,
            "cell_sccs": cell_sccs,
            "n_test_pairs": int(targets.size),
            "n_test_cell_lines": len(cell_pccs),
        }

    def _fit_predict_features(
        self,
        cell_features: np.ndarray,
        drug_emb_matrix: np.ndarray,
        cell_idxs: np.ndarray,
        drug_idxs: np.ndarray,
        targets: np.ndarray,
        pair_cell_ids: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, object]]:
        pca_prefix = self._pca_prefix()
        rf_prefix = self._rf_prefix()
        unique_train_cells = np.unique(cell_idxs[train_idx]).astype(np.int64)
        fit_features = cell_features[unique_train_cells]
        transformed_features = cell_features

        if bool(getattr(self.task_cfg, f"{pca_prefix}_standardize", True)):
            scaler = StandardScaler()
            scaler.fit(fit_features)
            fit_features = scaler.transform(fit_features)
            transformed_features = scaler.transform(cell_features)

        requested_components = int(
            getattr(self.task_cfg, f"{pca_prefix}_components", 256)
        )
        n_components = min(
            requested_components,
            fit_features.shape[0] - 1,
            fit_features.shape[1],
        )
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than one component.")
        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, f"{pca_prefix}_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        pca.fit(fit_features)
        cell_components = pca.transform(transformed_features).astype(np.float32)

        pair_features = np.concatenate(
            (cell_components[cell_idxs], drug_emb_matrix[drug_idxs]),
            axis=1,
        )
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
            bootstrap=bool(getattr(self.task_cfg, f"{rf_prefix}_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, f"{rf_prefix}_n_jobs", -1)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        rf.fit(pair_features[train_idx], targets[train_idx])
        train_predictions = rf.predict(pair_features[train_idx])
        test_predictions = rf.predict(pair_features[test_idx])
        train_metrics = {
            "loss": float(mean_squared_error(targets[train_idx], train_predictions)),
        }
        test_metrics = self._regression_metrics(
            test_predictions,
            targets[test_idx],
            pair_cell_ids[test_idx],
        )
        test_metrics["pca_components"] = int(n_components)
        test_metrics["cell_embedding_dim"] = int(cell_features.shape[1])
        test_metrics["pair_feature_dim"] = int(pair_features.shape[1])
        return train_metrics, test_metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("scbFM drug-response PCA+RF must use one process.")

            X_cell, drug_emb_matrix, cell_idxs, drug_idxs, targets, pair_cell_ids = (
                self._load_gdsc_data()
            )
            splits = self._build_or_load_cv_splits(
                pair_cell_ids,
                self._pair_drug_ids,
                targets,
            )
            fold_gene_indices = [
                self._select_training_hvg_indices(X_cell, cell_idxs[train_idx])
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
                    cell_features = self._extract_cell_features(X_cell, checkpoint_path)
                    train_metrics, test_metrics = self._fit_predict_features(
                        cell_features,
                        drug_emb_matrix,
                        cell_idxs,
                        drug_idxs,
                        targets,
                        pair_cell_ids,
                        train_idx,
                        test_idx,
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
                            test_metrics,
                        )
                    )
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
            from run_provenance import complete_run_metadata

            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": aggregate_rows}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
