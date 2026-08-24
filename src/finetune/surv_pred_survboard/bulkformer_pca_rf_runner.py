from __future__ import annotations

from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from finetune.canc_type_class.bulkformer_pca_rf_runner import (
    CancTypeClassBulkFormerPCARFRunner,
)
from finetune.surv_pred_survboard.pca_rf_runner import SurvPredSurvBoardPCARFRunner
from finetune.surv_pred_survboard.runner import SurvPredSurvBoardRunner
from finetune.survival_regression import fit_pca_random_forest_regressor
from run_provenance import complete_run_metadata, update_run_metadata
from utils import seed_all


class SurvPredSurvBoardBulkFormerPCARFRunner(CancTypeClassBulkFormerPCARFRunner):
    """BulkFormer full-vocabulary embeddings with PCA+RF on SurvBoard splits."""

    task_name = "surv_pred_survboard"
    config_node = "surv_pred_survboard"

    @staticmethod
    def _validate_bulkformer_expression_scale(
        values: np.ndarray,
        absolute_limit: float,
    ) -> tuple[float, float]:
        if not np.all(np.isfinite(values)):
            raise ValueError("SurvBoard BulkFormer expression contains non-finite values.")
        expression_min = float(np.min(values))
        expression_max = float(np.max(values))
        max_observed_magnitude = max(abs(expression_min), abs(expression_max))
        if max_observed_magnitude > absolute_limit:
            raise ValueError(
                "BulkFormer expects normalized SurvBoard expression with bounded magnitude, "
                f"but values span [{expression_min:.6g}, {expression_max:.6g}] "
                f"with configured absolute limit {absolute_limit:.6g}."
            )
        return expression_min, expression_max

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        return SurvPredSurvBoardRunner._resolve_task_cfg(cfg)

    def _task_output_dir(self):
        return SurvPredSurvBoardRunner._task_output_dir(self)

    def _output_prefix(self):
        return SurvPredSurvBoardRunner._output_prefix(self)

    def _survboard_metrics(self, *args, **kwargs):
        return SurvPredSurvBoardPCARFRunner._survboard_metrics(*args, **kwargs)

    def _save_run_metadata(self) -> None:
        super()._save_run_metadata()
        if self.is_master:
            update_run_metadata(
                self._run_metadata_path,
                {
                    "baseline": "bulkformer_147m_embedding_pca_random_forest_regressor",
                    "cancer": str(getattr(self.task_cfg, "cancer", "")).upper(),
                    "project": str(getattr(self.task_cfg, "project", "TCGA")),
                    "evaluation_protocol": "survboard_repeated_five_fold_cross_validation",
                    "cv_folds": 5,
                    "cv_repetitions": 5,
                    "n_outer_splits": int(
                        getattr(self.task_cfg, "expected_outer_splits", 25)
                    ),
                    "survboard_split_fingerprint": str(
                        getattr(self, "_survboard_split_fingerprint", "")
                    ),
                },
            )

    def _bulkformer_paths(self):
        repo_dir = self._required_path("bulkformer_repo_dir")
        paths = {
            "repo_dir": repo_dir,
            "checkpoint": self._required_path("bulkformer_checkpoint_path"),
            "gene_info": self._required_path("bulkformer_gene_info_path"),
            "graph": self._required_path("bulkformer_graph_path"),
            "graph_weight": self._required_path("bulkformer_graph_weight_path"),
            "gene_emb": self._required_path("bulkformer_gene_emb_path"),
        }
        interested = getattr(self.task_cfg, "bulkformer_interested_gene_list_path", None)
        if interested:
            paths["interested_gene_list"] = Path(
                hydra.utils.to_absolute_path(str(interested))
            )
        return paths

    def _prepare_survboard_bulkformer_data(self, paths):
        base = Path(hydra.utils.to_absolute_path(str(self.task_cfg.survboard_data_dir)))
        project = str(getattr(self.task_cfg, "project", "TCGA"))
        cancer = str(getattr(self.task_cfg, "cancer", ""))
        data_path = base / project / f"{cancer}_data_complete_modalities_preprocessed.csv"
        frame = pd.read_csv(data_path, low_memory=False)
        times = frame[str(getattr(self.task_cfg, "os_days_col", "OS_days"))].to_numpy(dtype=float)
        events = frame[str(getattr(self.task_cfg, "os_col", "OS"))].to_numpy(dtype=float)
        gex_cols = [column for column in frame if column.startswith("gex_")]
        symbols = [column[4:].split("|")[0] for column in gex_cols]
        gene_info = pd.read_csv(paths["gene_info"])
        symbol_to_ensg = dict(
            zip(gene_info["gene_symbol"].astype(str), gene_info["ensg_id"].astype(str))
        )
        columns, genes, seen = [], [], set()
        for column, symbol in zip(gex_cols, symbols):
            gene = symbol_to_ensg.get(symbol)
            if gene and gene not in seen:
                columns.append(column)
                genes.append(gene)
                seen.add(gene)
        values = frame[columns].to_numpy(dtype=np.float32)
        if bool(getattr(self.task_cfg, "convert_log2_to_log1p", True)):
            values *= np.float32(np.log(2.0))
        max_expected = float(
            getattr(self.task_cfg, "bulkformer_max_expected_expression", 30.0)
        )
        expression_min, expression_max = self._validate_bulkformer_expression_scale(
            values,
            max_expected,
        )
        self._bulkformer_expression_min = expression_min
        self._bulkformer_expression_max = expression_max
        source = ad.AnnData(values)
        source.var_names = genes
        source.obs_names = pd.Index([str(index) for index in range(source.n_obs)])
        expr_frame = self._adata_to_gene_frame(source)
        vocabulary = self._strip_ensembl_version(
            gene_info["ensg_id"].astype(str)
        ).tolist()
        expected = int(getattr(self.task_cfg, "bulkformer_expected_gene_count", 20010))
        if len(vocabulary) != expected:
            raise ValueError(f"BulkFormer vocabulary has {len(vocabulary)} genes; expected {expected}.")
        expression = self._align_to_bulkformer_genes(expr_frame, vocabulary)
        matched = int(np.isin(vocabulary, expr_frame.columns.astype(str)).sum())
        self._bulkformer_input_gene_count = len(vocabulary)
        self._bulkformer_matched_gene_count = matched
        self._bulkformer_missing_gene_count = len(vocabulary) - matched
        self._bulkformer_missing_fraction = 1.0 - matched / len(vocabulary)
        return expression, times, events, self._bulkformer_missing_fraction

    def run(self) -> dict:
        try:
            self._setup_runtime()
            paths = self._bulkformer_paths()
            expression, times, events, missing_fraction = self._prepare_survboard_bulkformer_data(paths)
            train_splits, test_splits = SurvPredSurvBoardRunner._load_splits(
                self, len(times)
            )
            outer_splits = list(getattr(self.task_cfg, "outer_splits", range(len(train_splits))))
            self._save_run_metadata()
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model = self._build_bulkformer_model(paths)
            embeddings, sample_indices = self._extract_bulkformer_embeddings(
                model, expression, paths, missing_fraction
            )
            embeddings, sample_indices = self._gather_bulkformer_embeddings(
                embeddings, sample_indices, len(times)
            )
            if not np.array_equal(sample_indices, np.arange(len(times))):
                raise RuntimeError("Distributed BulkFormer extraction changed sample order.")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            output_path = self._task_output_dir() / f"{self._output_prefix()}_evaluation_metrics.csv"
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}
            cancer = str(getattr(self.task_cfg, "cancer", "")).upper()
            project = str(getattr(self.task_cfg, "project", "TCGA"))
            fold_rows = []
            for position, split_idx in enumerate(outer_splits):
                train_idx, test_idx = train_splits[split_idx], test_splits[split_idx]
                result = fit_pca_random_forest_regressor(
                    self.task_cfg, embeddings[train_idx], times[train_idx], events[train_idx],
                    embeddings[test_idx], times[test_idx], prefix="bulkformer_pca"
                )
                metrics = self._survboard_metrics(
                    result, times[train_idx], events[train_idx],
                    times[test_idx], events[test_idx],
                    checkpoint_path=str(paths["checkpoint"]),
                    model_key="bulkformer_147m",
                )
                fold_rows.append(
                    SurvPredSurvBoardRunner._flatten_fold_metrics(
                        self, "bulkformer_147m", split_idx, len(outer_splits),
                        str(paths["checkpoint"]), cancer, project,
                        {"loss": float("nan")}, metrics
                    )
                )
                SurvPredSurvBoardRunner._save_survboard_predictions(
                    self, "bulkformer_147m", split_idx,
                    result.test_survival, result.time_points
                )
            aggregate = SurvPredSurvBoardRunner._write_model_results(
                self, str(paths["checkpoint"]), fold_rows, []
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
