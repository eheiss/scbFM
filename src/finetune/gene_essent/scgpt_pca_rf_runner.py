from __future__ import annotations

import json
import logging
from contextlib import nullcontext

import anndata as ad
import numpy as np
import torch
import torch.distributed as dist
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

from finetune.canc_type_class.scgpt_pca_rf_runner import (
    CancTypeClassScGPTPCARFRunner,
    _bin_scgpt_examples_safely,
)
from finetune.gene_essent.pca_rf_runner import GeneEssentPCARFRunner
from run_provenance import complete_run_metadata
from utils import SequentialDistributedSampler, seed_all

log = logging.getLogger(__name__)


class _ScGPTGeneExpressionDataset(Dataset):
    def __init__(
        self,
        matrix,
        sample_indices: np.ndarray,
        gene_ids: np.ndarray,
        cls_token_id: int,
        cls_value: float,
    ) -> None:
        self.matrix = matrix
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.cls_token_id = int(cls_token_id)
        self.cls_value = float(cls_value)

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int):
        row = self.matrix[index]
        row = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        genes = np.concatenate(([self.cls_token_id], self.gene_ids))
        values = np.concatenate(
            ([self.cls_value], row.astype(np.float32, copy=False))
        )
        return {
            "genes": torch.as_tensor(genes, dtype=torch.long),
            "expressions": torch.as_tensor(values, dtype=torch.float32),
        }, torch.tensor(self.sample_indices[index], dtype=torch.long)


class GeneEssentScGPTPCARFRunner(GeneEssentPCARFRunner):
    """Frozen scGPT gene states followed by PCA and a shared RF."""

    config_node = "gene_essent"
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
    _gather_scgpt_embeddings = (
        CancTypeClassScGPTPCARFRunner._gather_scgpt_embeddings
    )

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "scgpt_variant", "") or "").strip()
        return variant or "scgpt_pca_rf"

    def _model_key(self) -> str:
        model_key = str(getattr(self.task_cfg, "scgpt_model_key", "") or "").strip()
        return model_key or "scgpt"

    def _make_scgpt_gene_loader(
        self,
        adata: ad.AnnData,
        sample_indices: np.ndarray,
        gene_ids: np.ndarray,
        vocab,
        model_configs: dict,
    ) -> DataLoader:
        from scgpt.data_collator import DataCollator
        from scgpt.preprocess import binning

        pad_token = str(model_configs["pad_token"])
        pad_value = int(model_configs["pad_value"])
        dataset = _ScGPTGeneExpressionDataset(
            adata.X,
            sample_indices,
            gene_ids,
            int(vocab["<cls>"]),
            float(pad_value),
        )
        required_length = int(gene_ids.size + 1)
        configured_length = getattr(self.task_cfg, "scgpt_max_seq_len", None)
        max_length = required_length if configured_length is None else int(configured_length)
        if max_length < required_length:
            raise ValueError(
                f"scgpt_max_seq_len={max_length} would truncate the selected gene tokens."
            )
        self._scgpt_effective_max_seq_len = max_length
        collator = DataCollator(
            do_padding=True,
            pad_token_id=int(vocab[pad_token]),
            pad_value=pad_value,
            do_mlm=False,
            do_binning=False,
            mlm_probability=0.15,
            mask_value=int(model_configs.get("mask_value", -1)),
            max_length=max_length,
            sampling=False,
            keep_first_n_tokens=1,
        )

        def collate_fn(batch):
            examples, indices = zip(*batch)
            examples = _bin_scgpt_examples_safely(list(examples), binning)
            return collator(examples), torch.stack(list(indices))

        batch_size = int(getattr(self.task_cfg, "scgpt_batch_size", 4))
        workers = int(getattr(self.task_cfg, "num_workers", 2))
        sampler = (
            SequentialDistributedSampler(
                dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            if self.is_distributed
            else None
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
            collate_fn=collate_fn,
        )

    def _extract_scgpt_gene_embeddings(
        self,
        model,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = []
        sample_indices = []
        autocast_context = (
            torch.cuda.amp.autocast(enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            for batch, batch_indices in loader:
                gene = batch["gene"].to(self.device, non_blocking=True)
                expr = batch["expr"].to(self.device, non_blocking=True)
                padding_mask = gene.eq(int(model.encoder.embedding.padding_idx))
                hidden = model._encode(gene, expr, padding_mask)
                embeddings.append(
                    hidden[:, 1 : self.selected_gene_count + 1, :]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                sample_indices.append(batch_indices.numpy())
        return np.concatenate(embeddings), np.concatenate(sample_indices)

    def _save_external_metadata(self, checkpoint_path: str) -> None:
        self._save_run_metadata({self._model_key(): checkpoint_path})
        if not self.is_master:
            return
        with self._run_metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(
            {
                "baseline": "scgpt_frozen_gene_embedding_pca_random_forest",
                "scgpt_source_gene_count": int(self._scgpt_source_gene_count),
                "scgpt_vocab_matched_gene_count": int(
                    self._scgpt_vocab_matched_gene_count
                ),
                "scgpt_gene_selection": "training_fold_mad_after_vocab_matching",
                "scgpt_checkpoint_loading": self._scgpt_load_report,
                "distributed_execution": (
                    "all_ranks_extract_all_folds_then_rank0_fits_after_shutdown"
                ),
            }
        )
        with self._run_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def run(self) -> dict:
        try:
            self._setup_runtime()
            X_expression, Y_targets, cell_ids = self._load_depmap_data()
            splits = self._build_or_load_cv_splits(cell_ids)
            gene_names = [
                line.strip()
                for line in self._resolve_gene_list_path().read_text().splitlines()
                if line.strip()
            ]
            canonical_index = {gene: index for index, gene in enumerate(gene_names)}
            adata = ad.AnnData(X=X_expression)
            adata.obs_names = cell_ids
            adata.var_names = gene_names

            paths = self._scgpt_paths()
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model, vocab, model_configs = self._build_scgpt_model(paths)
            adata = self._add_scgpt_gene_symbols(adata, paths["gene_info"], vocab)
            checkpoint_path = str(paths["checkpoint"])
            self._save_external_metadata(checkpoint_path)

            extracted_folds = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                selected_idx = self._select_training_hvg_indices(adata.X, train_idx)
                selected_adata = adata[:, selected_idx].copy()
                canonical_target_idx = np.asarray(
                    [canonical_index[str(gene)] for gene in selected_adata.var_names],
                    dtype=np.int64,
                )
                valid = self.valid_gene_mask[canonical_target_idx]
                if int(valid.sum()) < 2:
                    raise ValueError("Fewer than two scGPT-selected genes have targets.")
                selected_adata = selected_adata[:, valid].copy()
                canonical_target_idx = canonical_target_idx[valid]
                self.fold_n_valid_genes = int(valid.sum())
                self.selected_gene_count = self.fold_n_valid_genes

                gene_ids = np.asarray(
                    vocab(selected_adata.var["scgpt_gene_symbol"].astype(str).tolist()),
                    dtype=np.int64,
                )
                train_order = np.arange(len(train_idx), dtype=np.int64)
                test_order = np.arange(len(test_idx), dtype=np.int64)
                train_loader = self._make_scgpt_gene_loader(
                    selected_adata[train_idx].copy(),
                    train_order,
                    gene_ids,
                    vocab,
                    model_configs,
                )
                test_loader = self._make_scgpt_gene_loader(
                    selected_adata[test_idx].copy(),
                    test_order,
                    gene_ids,
                    vocab,
                    model_configs,
                )
                train_x, observed_train = self._extract_scgpt_gene_embeddings(
                    model,
                    train_loader,
                )
                test_x, observed_test = self._extract_scgpt_gene_embeddings(
                    model,
                    test_loader,
                )
                train_x, observed_train = self._gather_scgpt_embeddings(
                    train_x,
                    observed_train,
                    len(train_idx),
                )
                test_x, observed_test = self._gather_scgpt_embeddings(
                    test_x,
                    observed_test,
                    len(test_idx),
                )
                train_order_idx = np.argsort(observed_train)
                test_order_idx = np.argsort(observed_test)
                train_x = train_x[train_order_idx]
                test_x = test_x[test_order_idx]
                if not np.array_equal(observed_train[train_order_idx], train_order):
                    raise RuntimeError("Distributed scGPT extraction changed training order.")
                if not np.array_equal(observed_test[test_order_idx], test_order):
                    raise RuntimeError("Distributed scGPT extraction changed test order.")

                if self.is_master:
                    extracted_folds.append(
                        (
                            fold_idx,
                            train_idx.copy(),
                            test_idx.copy(),
                            canonical_target_idx.copy(),
                            train_x,
                            test_x,
                            self.fold_n_valid_genes,
                        )
                    )
                else:
                    del train_x, test_x
                self.selected_gene_count = int(self.model_cfg.selected_gene_count)

            del model
            self._finish_distributed_embedding_phase()

            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}

            log.info(
                "Distributed scGPT extraction complete; fitting PCA/RF for %d folds ",
                len(extracted_folds),
            )
            fold_rows = []
            cell_line_rows = []
            while extracted_folds:
                (
                    fold_idx,
                    train_idx,
                    test_idx,
                    canonical_target_idx,
                    train_x,
                    test_x,
                    fold_n_valid_genes,
                ) = extracted_folds.pop(0)
                self.fold_n_valid_genes = int(fold_n_valid_genes)
                try:
                    train_metrics, test_metrics = self._fit_predict_gene_features(
                        train_x,
                        Y_targets[train_idx][:, canonical_target_idx],
                        test_x,
                        Y_targets[test_idx][:, canonical_target_idx],
                        prefix="scgpt_pca",
                        rf_prefix="scgpt_rf",
                    )
                except Exception:
                    log.exception("scGPT PCA/RF failed in fold %d", fold_idx)
                    raise
                fold_rows.append(
                    self._flatten_fold_metrics(
                        self._model_key(),
                        fold_idx,
                        len(splits),
                        checkpoint_path,
                        train_metrics,
                        test_metrics,
                    )
                )
                cell_line_rows.extend(
                    self._per_cell_line_rows(
                        self._model_key(),
                        fold_idx,
                        checkpoint_path,
                        [cell_ids[index] for index in test_idx],
                        test_metrics,
                    )
                )
                del train_x, test_x

            aggregate = self._write_model_results(
                checkpoint_path,
                fold_rows,
                cell_line_rows,
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
