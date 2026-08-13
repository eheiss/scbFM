from __future__ import annotations

import logging

import numpy as np
import torch.distributed as dist

from finetune.canc_type_class.scgpt_pca_rf_runner import (
    CancTypeClassScGPTPCARFRunner,
)
from finetune.deconv.pca_rf_runner import DeconvPCARFRunner
from run_provenance import complete_run_metadata
from utils import seed_all

log = logging.getLogger(__name__)


class DeconvScGPTPCARFRunner(DeconvPCARFRunner):
    """Frozen scGPT CLS embeddings followed by PCA and multi-output RF."""

    _checkpoint_state_dict = staticmethod(
        CancTypeClassScGPTPCARFRunner._checkpoint_state_dict
    )
    _required_path = CancTypeClassScGPTPCARFRunner._required_path
    _scgpt_paths = CancTypeClassScGPTPCARFRunner._scgpt_paths
    _strip_ensembl_version = staticmethod(
        CancTypeClassScGPTPCARFRunner._strip_ensembl_version
    )
    _add_scgpt_gene_symbols = CancTypeClassScGPTPCARFRunner._add_scgpt_gene_symbols
    _build_scgpt_model = CancTypeClassScGPTPCARFRunner._build_scgpt_model
    _make_scgpt_loader = CancTypeClassScGPTPCARFRunner._make_scgpt_loader
    _extract_scgpt_embeddings = (
        CancTypeClassScGPTPCARFRunner._extract_scgpt_embeddings
    )
    _extract_scgpt_embeddings_safely = (
        CancTypeClassScGPTPCARFRunner._extract_scgpt_embeddings_safely
    )
    _gather_scgpt_embeddings = (
        CancTypeClassScGPTPCARFRunner._gather_scgpt_embeddings
    )

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "scgpt_variant", "") or "").strip()
        return variant or "scgpt_pca_rf"

    def _model_key(self) -> str:
        model_key = str(getattr(self.task_cfg, "scgpt_model_key", "") or "").strip()
        return model_key or "scgpt"

    def run(self) -> dict:
        try:
            self._setup_runtime()
            paths = self._scgpt_paths()
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model, vocab, model_configs = self._build_scgpt_model(paths)
            adata, targets, groups = self._prepare_cv_data()
            splits = self._build_or_load_cv_splits(adata, groups)
            adata = self._add_scgpt_gene_symbols(adata, paths["gene_info"], vocab)
            configured_max_length = getattr(self.task_cfg, "scgpt_max_seq_len", None)
            self._scgpt_effective_max_seq_len = (
                self.selected_gene_count + 1
                if configured_max_length is None
                else int(configured_max_length)
            )
            model_key = self._model_key()
            checkpoint_path = str(paths["checkpoint"])
            self._save_run_metadata({model_key: checkpoint_path})

            fold_rows = []
            prediction_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                selected_idx = self._select_distributed_training_hvg_indices(
                    adata[train_idx]
                )
                fold_adata = adata[:, selected_idx].copy()
                gene_ids = np.asarray(
                    vocab(fold_adata.var["scgpt_gene_symbol"].astype(str).tolist()),
                    dtype=np.int64,
                )
                train_order = np.arange(len(train_idx), dtype=np.int64)
                test_order = np.arange(len(test_idx), dtype=np.int64)
                train_loader = self._make_scgpt_loader(
                    fold_adata[train_idx].copy(),
                    train_order,
                    gene_ids,
                    vocab,
                    model_configs,
                    stage=f"fold {fold_idx} training split",
                )
                test_loader = self._make_scgpt_loader(
                    fold_adata[test_idx].copy(),
                    test_order,
                    gene_ids,
                    vocab,
                    model_configs,
                    stage=f"fold {fold_idx} validation split",
                )
                train_x, observed_train_order = self._extract_scgpt_embeddings_safely(
                    model,
                    train_loader,
                    stage=f"fold {fold_idx} training split",
                )
                test_x, observed_test_order = self._extract_scgpt_embeddings_safely(
                    model,
                    test_loader,
                    stage=f"fold {fold_idx} validation split",
                )
                train_x, observed_train_order = self._gather_scgpt_embeddings(
                    train_x,
                    observed_train_order,
                    len(train_idx),
                )
                test_x, observed_test_order = self._gather_scgpt_embeddings(
                    test_x,
                    observed_test_order,
                    len(test_idx),
                )
                if not np.array_equal(observed_train_order, train_order):
                    raise RuntimeError("Distributed scGPT extraction changed training order.")
                if not np.array_equal(observed_test_order, test_order):
                    raise RuntimeError("Distributed scGPT extraction changed validation order.")

                if not self.is_master:
                    if self.is_distributed:
                        dist.barrier()
                    continue
                train_y = targets[train_idx]
                test_y = targets[test_idx]
                train_y = train_y / train_y.sum(axis=1, keepdims=True)
                test_y = test_y / test_y.sum(axis=1, keepdims=True)
                self.fold_train_target_mean = train_y.mean(axis=0)
                train_metrics, test_metrics = self._fit_predict_features(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    prefix="scgpt_pca",
                    rf_prefix="scgpt_rf",
                )
                test_adata = adata[test_idx].copy()
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
                self.fold_train_target_mean = None
                if self.is_distributed:
                    dist.barrier()

            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            if self.is_master:
                aggregate = self._write_model_results(
                    checkpoint_path,
                    fold_rows,
                    prediction_rows,
                )
                self._write_csv(output_path, [aggregate])
                complete_run_metadata(self._run_metadata_path, output_path)
                return {"results_path": str(output_path), "results": [aggregate]}
            return {"results_path": str(output_path), "results": []}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
