from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version.
    StratifiedGroupKFold = None

from finetune.surv_pred_survboard.runner import (
    CHECKPOINT_MODEL_KEYS,
    RANDOM_INIT_MODEL_KEY,
    SurvPredSurvBoardRunner,
    cox_partial_log_likelihood,
    harrell_c_index,
    ipcw_weighted_c_index,
)
from preprocess import reindex_adata_genes, validate_token_matrix
from run_provenance import complete_run_metadata

log = logging.getLogger(__name__)

TASK_NAME = "surv_pred"


class SurvPredRunner(SurvPredSurvBoardRunner):
    """BulkRNABert-style TCGA Cox survival prediction.

    This task uses the TCGA AnnData source also used by cancer-type
    classification and reports Harrell C-index metrics:

    - ``test_c_index``: Harrell C-index over the whole held-out fold.
    - ``test_cohort_weighted_c_index``: sample-count-weighted mean of
      per-cohort Harrell C-indices.
    - ``test_macro_c_index``: unweighted mean of per-cohort Harrell C-indices.
    - ``test_ipcw_c_index``: additional censoring-weighted C-index, kept for
      diagnostics but not used as the BulkRNABert weighted metric.
    """

    task_name = TASK_NAME

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "surv_pred" in cfg.finetune:
            return cfg.finetune.surv_pred
        raise ValueError(
            "Could not find survival prediction config. Expected cfg.finetune.surv_pred."
        )

    def _task_output_dir(self) -> Path:
        return Path(__file__).resolve().parents[4] / "output" / TASK_NAME / self._output_variant()

    def _output_prefix(self) -> str:
        return f"{TASK_NAME}_{self._output_variant()}"

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                "finetune.surv_pred.pretrained_model_paths must define "
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
                f"Missing checkpoint paths in finetune.surv_pred.pretrained_model_paths: {missing}"
            )
        checkpoint_paths[RANDOM_INIT_MODEL_KEY] = ""
        return checkpoint_paths

    @staticmethod
    def _numeric_obs(adata: ad.AnnData, column: str) -> np.ndarray:
        if column not in adata.obs:
            raise ValueError(
                f"TCGA AnnData is missing obs['{column}']. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return np.asarray(
            adata.obs[column]
            .astype(str)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            .astype(float),
            dtype=float,
        )

    def _load_tcga_survival_data(self) -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        configured_path = getattr(self.task_cfg, "tcga_data_dir", None)
        if not configured_path:
            raise ValueError("finetune.surv_pred.tcga_data_dir must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"TCGA h5ad file not found: {data_path}")

        adata = ad.read_h5ad(data_path)
        project_col = str(getattr(self.task_cfg, "project_col", "project"))
        patient_col = str(getattr(self.task_cfg, "patient_col", "patient_id"))
        required_obs = {"sample_id", project_col, patient_col}
        missing_obs = sorted(required_obs.difference(adata.obs.columns))
        if missing_obs:
            raise ValueError(f"TCGA AnnData is missing required obs columns: {missing_obs}.")

        projects = adata.obs[project_col].astype(str).str.strip().str.upper().to_numpy()
        cohorts = [str(cohort).upper() for cohort in getattr(self.task_cfg, "cohorts", [])]
        if cohorts:
            keep = np.isin(projects, cohorts)
            adata = adata[keep].copy()
            projects = projects[keep]

        if bool(getattr(self.task_cfg, "merge_gbm_lgg", False)):
            projects = np.asarray(["GBMLGG" if p in {"GBM", "LGG"} else p for p in projects])

        times = self._numeric_obs(adata, str(getattr(self.task_cfg, "survival_time_col", "OS.time")))
        events = self._numeric_obs(adata, str(getattr(self.task_cfg, "survival_event_col", "OS")))
        valid = np.isfinite(times) & np.isfinite(events) & (times > 0)
        if valid.sum() < int(getattr(self.task_cfg, "cv_folds", 5)) * 2:
            raise ValueError(f"Only {int(valid.sum())} TCGA samples have usable survival labels.")

        adata = adata[valid].copy()
        times = times[valid].astype(np.float32)
        events = (events[valid] > 0).astype(np.float32)
        projects = projects[valid].astype(str)

        adata.obs["survival_time"] = times
        adata.obs["survival_event"] = events
        adata.obs["survival_cohort"] = projects
        adata.obs_names = adata.obs["sample_id"].astype(str)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()

        gene_list_path = self._resolve_gene_list_path()
        adata, missing_genes = reindex_adata_genes(adata, gene_list_path=gene_list_path)
        if not bool(getattr(self.task_cfg, "preprocess", True)):
            validate_token_matrix(adata.X, bin_num=int(self.model_cfg.bin_num), name="surv_pred input")

        self._missing_genes_note = (
            f"Model genes missing from TCGA GEX and filled with count 0: "
            f"{len(missing_genes)} / {int(self.model_cfg.gene_num)}"
        ) if missing_genes else ""

        expected_gene_num = int(self.model_cfg.gene_num)
        if adata.n_vars != expected_gene_num:
            raise ValueError(f"Expected {expected_gene_num} genes, got {adata.n_vars}.")

        groups = None
        patient_ids = adata.obs[patient_col].astype(str).to_numpy()
        if len(np.unique(patient_ids)) < len(patient_ids):
            groups = patient_ids
            log.info("Duplicate TCGA patient IDs detected; using patient-grouped CV.")

        if self.is_master:
            log.info(
                "Prepared BulkRNABert-style TCGA survival data: samples=%d, genes=%d, "
                "events=%d, cohorts=%d",
                adata.n_obs,
                adata.n_vars,
                int(events.sum()),
                len(np.unique(projects)),
            )
        return adata, times, events, projects, groups

    def _build_cv_splits(
        self,
        events: np.ndarray,
        cohorts: np.ndarray,
        groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = int(getattr(self.task_cfg, "cv_folds", 5))
        if n_splits < 2:
            raise ValueError("finetune.surv_pred.cv_folds must be at least 2.")
        strat_labels = np.asarray([f"{cohort}_{int(event)}" for cohort, event in zip(cohorts, events)])
        _, counts = np.unique(strat_labels, return_counts=True)
        if counts.min() < n_splits:
            strat_labels = events.astype(int)

        if groups is not None:
            if StratifiedGroupKFold is not None:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=int(getattr(self.task_cfg, "random_seed", 42)),
                )
                return list(splitter.split(np.zeros(len(events)), strat_labels, groups))
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(np.zeros(len(events)), strat_labels, groups))

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        return list(splitter.split(np.zeros(len(events)), strat_labels))

    @staticmethod
    def _cohort_c_index_metrics(
        risk: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        cohorts: np.ndarray,
    ) -> dict[str, float]:
        rows = []
        for cohort in sorted(np.unique(cohorts).tolist()):
            mask = cohorts == cohort
            cidx = harrell_c_index(risk[mask], times[mask], events[mask])
            if np.isnan(cidx):
                continue
            rows.append((cohort, int(mask.sum()), float(cidx)))
        if not rows:
            return {
                "test_cohort_weighted_c_index": float("nan"),
                "test_macro_c_index": float("nan"),
            }
        counts = np.asarray([row[1] for row in rows], dtype=float)
        cidxs = np.asarray([row[2] for row in rows], dtype=float)
        metrics = {
            "test_cohort_weighted_c_index": float(np.average(cidxs, weights=counts)),
            "test_macro_c_index": float(np.mean(cidxs)),
            "test_n_cohorts_with_c_index": int(len(rows)),
        }
        for cohort, count, cidx in rows:
            metrics[f"test_c_index_{cohort}"] = cidx
            metrics[f"test_n_{cohort}"] = count
        return metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            adata, times, events, cohorts, groups = self._load_tcga_survival_data()
            splits = self._build_cv_splits(events, cohorts, groups)
            checkpoint_paths = self._get_checkpoint_paths()
            self._save_run_metadata(checkpoint_paths)

            epochs = int(getattr(self.task_cfg, "epochs", 20))
            aggregate_rows: list[dict[str, object]] = []

            if self.is_master:
                log.info(
                    "BulkRNABert-style survival CV ready: samples=%d, folds=%d, epochs=%d",
                    adata.n_obs,
                    len(splits),
                    epochs,
                )

            for model_idx, (model_key, checkpoint_path) in enumerate(checkpoint_paths.items()):
                fold_rows: list[dict[str, object]] = []
                curves_rows: list[dict[str, object]] = []
                for split_idx, (train_ix, test_ix) in enumerate(splits, start=1):
                    self.fold_gene_indices = self._select_training_hvg_indices(adata[train_ix].copy())
                    self._build_loaders(
                        adata.X[train_ix],
                        times[train_ix],
                        events[train_ix],
                        adata.X[test_ix],
                        times[test_ix],
                        events[test_ix],
                    )
                    self._build_model(checkpoint_path)
                    self._build_optimization()

                    last_train_metrics = {"loss": float("nan")}
                    last_test_lh: np.ndarray | None = None
                    for epoch in range(1, epochs + 1):
                        last_train_metrics = self._train_one_epoch(epoch)
                        last_test_lh = self._predict_log_hazard(
                            self.test_loader,
                            len(test_ix),
                        )
                        if self.is_master:
                            validation_loss = float(
                                cox_partial_log_likelihood(
                                    torch.as_tensor(last_test_lh),
                                    torch.as_tensor(times[test_ix]),
                                    torch.as_tensor(events[test_ix]),
                                ).item()
                            )
                            validation_c_index = harrell_c_index(
                                last_test_lh,
                                times[test_ix],
                                events[test_ix],
                            )
                            log.info(
                                "Model %s | Fold %d/%d | Epoch %d/%d | "
                                "Loss: %.6f | Validation Loss: %.6f | C-index: %.4f",
                                model_key,
                                split_idx,
                                len(splits),
                                epoch,
                                epochs,
                                last_train_metrics["loss"],
                                validation_loss,
                                validation_c_index,
                            )
                            curves_rows.append(
                                {
                                    "model": model_key,
                                    "split": split_idx,
                                    "epoch": epoch,
                                    "train_loss": last_train_metrics["loss"],
                                    "validation_loss": validation_loss,
                                    "validation_c_index": validation_c_index,
                                    "learning_rates": ";".join(
                                        f"{float(group['lr']):.6g}"
                                        for group in self.optimizer.param_groups
                                    ),
                                }
                            )
                            self._write_csv(
                                self._task_output_dir()
                                / f"{self._output_prefix()}_{model_key}_curves.csv",
                                curves_rows,
                            )

                    test_lh = (
                        last_test_lh
                        if last_test_lh is not None
                        else self._predict_log_hazard(self.test_loader, len(test_ix))
                    )

                    if self.is_master:
                        risk = np.asarray(test_lh, dtype=float)
                        test_metrics = {
                            "test_c_index": harrell_c_index(risk, times[test_ix], events[test_ix]),
                            "test_ipcw_c_index": ipcw_weighted_c_index(
                                risk,
                                times[test_ix],
                                events[test_ix],
                                times[train_ix],
                                events[train_ix],
                            ),
                        }
                        test_metrics.update(
                            self._cohort_c_index_metrics(
                                risk,
                                times[test_ix],
                                events[test_ix],
                                cohorts[test_ix],
                            )
                        )
                        log.info(
                            "Model %s | Fold %d/%d | C-index: %.4f | Cohort-weighted: %.4f | Macro: %.4f",
                            model_key,
                            split_idx,
                            len(splits),
                            test_metrics["test_c_index"],
                            test_metrics["test_cohort_weighted_c_index"],
                            test_metrics["test_macro_c_index"],
                        )
                        fold_rows.append(
                            self._flatten_fold_metrics(
                                model_key=model_key,
                                split_idx=split_idx,
                                n_splits=len(splits),
                                checkpoint_path=checkpoint_path,
                                cancer="pan_cancer",
                                project="TCGA",
                                train_metrics=last_train_metrics,
                                test_metrics=test_metrics,
                            )
                        )

                    self._cleanup_fold_state()
                    if self.is_distributed:
                        dist.barrier()

                if self.is_master:
                    aggregate_rows.append(
                        self._write_model_results(checkpoint_path, fold_rows, curves_rows)
                    )

            if self.is_master:
                out_dir = self._task_output_dir()
                output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
                self._write_csv(output_path, aggregate_rows, comment=getattr(self, "_missing_genes_note", ""))
                complete_run_metadata(self._run_metadata_path, output_path)
                return {"results_path": str(output_path), "results": aggregate_rows}
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
