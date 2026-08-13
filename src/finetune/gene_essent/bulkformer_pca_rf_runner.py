from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

from finetune.canc_type_class.bulkformer_pca_rf_runner import (
    CancTypeClassBulkFormerPCARFRunner,
)
from finetune.gene_essent.pca_rf_runner import GeneEssentPCARFRunner
from run_provenance import complete_run_metadata
from utils import SequentialDistributedSampler, seed_all

log = logging.getLogger(__name__)


class GeneEssentBulkFormerPCARFRunner(GeneEssentPCARFRunner):
    """Frozen BulkFormer gene states followed by PCA and a shared RF."""

    config_node = "gene_essent"
    _required_path = CancTypeClassBulkFormerPCARFRunner._required_path
    _strip_ensembl_version = staticmethod(
        CancTypeClassBulkFormerPCARFRunner._strip_ensembl_version
    )
    _align_to_bulkformer_genes = staticmethod(
        CancTypeClassBulkFormerPCARFRunner._align_to_bulkformer_genes
    )
    _build_bulkformer_model = (
        CancTypeClassBulkFormerPCARFRunner._build_bulkformer_model
    )
    _gather_bulkformer_embeddings = (
        CancTypeClassBulkFormerPCARFRunner._gather_bulkformer_embeddings
    )

    @staticmethod
    def _validate_bulkformer_expression_scale(
        matrix: np.ndarray,
        absolute_limit: float,
    ) -> tuple[float, float]:
        if not np.all(np.isfinite(matrix)):
            raise ValueError("BulkFormer DepMap expression contains non-finite values.")
        expression_min = float(np.min(matrix))
        expression_max = float(np.max(matrix))
        max_observed_magnitude = max(abs(expression_min), abs(expression_max))
        if max_observed_magnitude > absolute_limit:
            raise ValueError(
                "BulkFormer expects normalized DepMap expression with bounded magnitude, "
                f"but observed range is [{expression_min:.6g}, {expression_max:.6g}] "
                f"with configured absolute limit {absolute_limit:.6g}."
            )
        return expression_min, expression_max

    def _finetune_mode(self) -> str:
        variant = str(
            getattr(self.task_cfg, "bulkformer_variant", "") or ""
        ).strip()
        return variant or "bulkformer_pca_rf"

    def _bulkformer_paths(self) -> dict[str, Path]:
        return {
            "repo_dir": self._required_path("bulkformer_repo_dir"),
            "checkpoint": self._required_path("bulkformer_checkpoint_path"),
            "gene_info": self._required_path("bulkformer_gene_info_path"),
            "graph": self._required_path("bulkformer_graph_path"),
            "graph_weight": self._required_path("bulkformer_graph_weight_path"),
            "gene_emb": self._required_path("bulkformer_gene_emb_path"),
        }

    def _load_bulkformer_expression_source(
        self,
        canonical_cell_ids: list[str],
        gene_info: pd.DataFrame,
    ) -> ad.AnnData:
        configured = getattr(self.task_cfg, "bulkformer_depmap_data_path", None)
        source_path = (
            Path(hydra.utils.to_absolute_path(str(configured)))
            if configured
            else Path(hydra.utils.to_absolute_path(str(self.task_cfg.depmap_data_path)))
        )
        if not source_path.exists():
            raise FileNotFoundError(f"Missing BulkFormer DepMap source: {source_path}")
        source = ad.read_h5ad(source_path)
        if "expr_array" in source.layers:
            matrix = source.layers["expr_array"]
        else:
            matrix = source.X
        expression = ad.AnnData(X=matrix, obs=source.obs.copy(), var=source.var.copy())
        expression.obs_names = source.obs_names.astype(str)
        if len(set(expression.obs_names)) != expression.n_obs:
            raise ValueError("BulkFormer DepMap source contains duplicate cell-line IDs.")
        missing_cells = [
            cell_id for cell_id in canonical_cell_ids
            if cell_id not in expression.obs_names
        ]
        if missing_cells:
            raise ValueError(
                f"BulkFormer DepMap source is missing {len(missing_cells)} canonical "
                f"cell lines. First missing IDs: {missing_cells[:10]}"
            )
        expression = expression[canonical_cell_ids].copy()

        if "ensg_id" in expression.var:
            ensembl = self._strip_ensembl_version(
                expression.var["ensg_id"].astype(str)
            ).astype(str)
        else:
            source_names = expression.var_names.astype(str)
            bulk_ensg = set(
                self._strip_ensembl_version(gene_info["ensg_id"].astype(str))
                .astype(str)
                .tolist()
            )
            stripped_source = self._strip_ensembl_version(source_names).astype(str)
            if sum(value in bulk_ensg for value in stripped_source) >= expression.n_vars // 2:
                ensembl = stripped_source
            else:
                if "gene_symbol" not in gene_info:
                    raise ValueError(
                        "BulkFormer gene info needs gene_symbol to map the DepMap source."
                    )
                symbol_to_ensg = dict(
                    zip(
                        gene_info["gene_symbol"].astype(str),
                        self._strip_ensembl_version(
                            gene_info["ensg_id"].astype(str)
                        ).astype(str),
                    )
                )
                ensembl = pd.Index(
                    [symbol_to_ensg.get(symbol, "") for symbol in source_names]
                )
        mapped = np.asarray([bool(value) for value in ensembl], dtype=bool)
        expression = expression[:, mapped].copy()
        expression.var_names = pd.Index(np.asarray(ensembl)[mapped].astype(str))
        expression.var_names_make_unique()
        self._bulkformer_source_path = str(source_path)
        self._bulkformer_source_gene_count = int(source.n_vars)
        self._bulkformer_mapped_source_gene_count = int(expression.n_vars)
        return expression

    def _prepare_bulkformer_expression(
        self,
        canonical_cell_ids: list[str],
        paths: dict[str, Path],
    ) -> tuple[np.ndarray, list[str], float]:
        gene_info = pd.read_csv(paths["gene_info"])
        if "ensg_id" not in gene_info:
            raise ValueError("BulkFormer gene info must contain ensg_id.")
        bulk_genes = self._strip_ensembl_version(
            gene_info["ensg_id"].astype(str)
        ).astype(str).tolist()
        expected = int(getattr(self.task_cfg, "bulkformer_expected_gene_count", 20010))
        if len(bulk_genes) != expected or len(set(bulk_genes)) != expected:
            raise ValueError(
                f"BulkFormer vocabulary must contain {expected} unique Ensembl IDs."
            )
        expression = self._load_bulkformer_expression_source(
            canonical_cell_ids,
            gene_info,
        )
        matrix = expression.X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        configured_maximum = float(
            getattr(self.task_cfg, "bulkformer_max_expected_expression", 30.0)
        )
        expression_min, expression_max = self._validate_bulkformer_expression_scale(
            matrix,
            configured_maximum,
        )
        frame = pd.DataFrame(
            matrix,
            index=expression.obs_names.astype(str),
            columns=expression.var_names.astype(str),
        )
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        missing = ~np.isin(bulk_genes, frame.columns.astype(str))
        missing_fraction = float(np.mean(missing))
        self._bulkformer_input_gene_count = int(len(bulk_genes))
        self._bulkformer_matched_gene_count = int((~missing).sum())
        self._bulkformer_missing_gene_count = int(missing.sum())
        self._bulkformer_missing_fraction = missing_fraction
        self._bulkformer_expression_min = expression_min
        self._bulkformer_expression_max = expression_max
        log.info(
            "BulkFormer DepMap mapping: source=%d, mapped_source=%d, vocab_matched=%d, "
            "vocab_missing=%d, missing_fraction=%.4f, expression_range=[%.6g, %.6g]",
            self._bulkformer_source_gene_count,
            self._bulkformer_mapped_source_gene_count,
            self._bulkformer_matched_gene_count,
            self._bulkformer_missing_gene_count,
            missing_fraction,
            expression_min,
            expression_max,
        )
        return (
            self._align_to_bulkformer_genes(frame, bulk_genes),
            bulk_genes,
            missing_fraction,
        )

    def _extract_bulkformer_gene_embeddings(
        self,
        model,
        expression: np.ndarray,
        selected_gene_indices: np.ndarray,
        missing_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dataset = TensorDataset(
            torch.as_tensor(expression, dtype=torch.float32),
            torch.arange(expression.shape[0], dtype=torch.long),
        )
        batch_size = int(getattr(self.task_cfg, "bulkformer_batch_size", 4))
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
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        embeddings = []
        sample_indices = []
        selected_index_tensor = torch.as_tensor(
            selected_gene_indices,
            dtype=torch.long,
            device=self.device,
        )
        autocast_context = (
            torch.amp.autocast("cuda", enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            for batch, batch_indices in loader:
                hidden = model(
                    batch.to(self.device, non_blocking=True),
                    mask_prob=float(missing_fraction),
                    output_expr=False,
                )
                embeddings.append(
                    hidden.index_select(1, selected_index_tensor)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                sample_indices.append(batch_indices.numpy())
        return np.concatenate(embeddings), np.concatenate(sample_indices)

    def _save_external_metadata(self, checkpoint_path: str) -> None:
        self._save_run_metadata({"bulkformer_147m": checkpoint_path})
        if not self.is_master:
            return
        with self._run_metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(
            {
                "baseline": "bulkformer_frozen_gene_embedding_pca_random_forest",
                "bulkformer_source_data_path": self._bulkformer_source_path,
                "bulkformer_source_gene_count": self._bulkformer_source_gene_count,
                "bulkformer_mapped_source_gene_count": (
                    self._bulkformer_mapped_source_gene_count
                ),
                "bulkformer_expected_gene_count": self._bulkformer_input_gene_count,
                "bulkformer_matched_gene_count": self._bulkformer_matched_gene_count,
                "bulkformer_missing_gene_count": self._bulkformer_missing_gene_count,
                "bulkformer_missing_vocab_fraction": self._bulkformer_missing_fraction,
                "bulkformer_gene_selection": "canonical_training_fold_mad",
                "bulkformer_expression_input": (
                    "processed_benchmark_expression_passed_through_unchanged"
                ),
                "bulkformer_expression_min": self._bulkformer_expression_min,
                "bulkformer_expression_max": self._bulkformer_expression_max,
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
            fold_gene_indices = [
                self._select_training_hvg_indices(X_expression, train_idx)
                for train_idx, _ in splits
            ]
            canonical_genes = [
                line.strip()
                for line in self._resolve_gene_list_path().read_text().splitlines()
                if line.strip()
            ]
            paths = self._bulkformer_paths()
            expression, bulk_genes, missing_fraction = (
                self._prepare_bulkformer_expression(cell_ids, paths)
            )
            bulk_index = {gene: index for index, gene in enumerate(bulk_genes)}
            checkpoint_path = str(paths["checkpoint"])
            self._save_external_metadata(checkpoint_path)
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model = self._build_bulkformer_model(paths)

            extracted_folds = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                canonical_idx = fold_gene_indices[fold_idx - 1]
                valid = self.valid_gene_mask[canonical_idx]
                canonical_idx = canonical_idx[valid]
                selected_genes = [canonical_genes[index] for index in canonical_idx]
                mapped = np.asarray(
                    [gene in bulk_index for gene in selected_genes],
                    dtype=bool,
                )
                if int(mapped.sum()) < 2:
                    raise ValueError("Fewer than two selected genes map to BulkFormer.")
                if not np.all(mapped):
                    log.warning(
                        "Dropped %d fold-selected target genes absent from BulkFormer.",
                        int((~mapped).sum()),
                    )
                canonical_idx = canonical_idx[mapped]
                selected_genes = [
                    gene for gene, is_mapped in zip(selected_genes, mapped) if is_mapped
                ]
                bulk_idx = np.asarray(
                    [bulk_index[gene] for gene in selected_genes],
                    dtype=np.int64,
                )
                self.fold_n_valid_genes = int(len(selected_genes))
                embeddings, observed_order = self._extract_bulkformer_gene_embeddings(
                    model,
                    expression,
                    bulk_idx,
                    missing_fraction,
                )
                embeddings, observed_order = self._gather_bulkformer_embeddings(
                    embeddings,
                    observed_order,
                    expression.shape[0],
                )
                order = np.argsort(observed_order)
                embeddings = embeddings[order]
                if not np.array_equal(
                    observed_order[order],
                    np.arange(expression.shape[0]),
                ):
                    raise RuntimeError("Distributed BulkFormer extraction changed sample order.")

                if self.is_master:
                    extracted_folds.append(
                        (
                            fold_idx,
                            train_idx.copy(),
                            test_idx.copy(),
                            canonical_idx.copy(),
                            embeddings,
                            self.fold_n_valid_genes,
                        )
                    )
                else:
                    del embeddings

            del model
            self._finish_distributed_embedding_phase()

            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}

            log.info(
                "Distributed BulkFormer extraction complete; fitting PCA/RF for %d folds",
                len(extracted_folds),
            )
            fold_rows = []
            cell_line_rows = []
            while extracted_folds:
                (
                    fold_idx,
                    train_idx,
                    test_idx,
                    canonical_idx,
                    embeddings,
                    fold_n_valid_genes,
                ) = extracted_folds.pop(0)
                self.fold_n_valid_genes = int(fold_n_valid_genes)
                try:
                    train_metrics, test_metrics = self._fit_predict_gene_features(
                        embeddings[train_idx],
                        Y_targets[train_idx][:, canonical_idx],
                        embeddings[test_idx],
                        Y_targets[test_idx][:, canonical_idx],
                        prefix="bulkformer_pca",
                        rf_prefix="bulkformer_rf",
                    )
                except Exception:
                    log.exception("BulkFormer PCA/RF failed in fold %d", fold_idx)
                    raise
                fold_rows.append(
                    self._flatten_fold_metrics(
                        "bulkformer_147m",
                        fold_idx,
                        len(splits),
                        checkpoint_path,
                        train_metrics,
                        test_metrics,
                    )
                )
                cell_line_rows.extend(
                    self._per_cell_line_rows(
                        "bulkformer_147m",
                        fold_idx,
                        checkpoint_path,
                        [cell_ids[index] for index in test_idx],
                        test_metrics,
                    )
                )
                del embeddings

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
