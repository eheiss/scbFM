from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch.distributed as dist
from omegaconf import OmegaConf

from finetune.canc_type_class.bulkformer_pca_rf_runner import (
    CancTypeClassBulkFormerPCARFRunner,
)
from finetune.drug_resp.pca_rf_runner import DrugRespPCARFRunner
from paths import REPO_ROOT, output_root
from run_provenance import complete_run_metadata, start_run_metadata
from utils import seed_all

log = logging.getLogger(__name__)


class DrugRespBulkFormerPCARFRunner(DrugRespPCARFRunner):
    """Frozen BulkFormer cell embeddings + KPGT drug features -> PCA -> RF."""

    _build_bulkformer_model = CancTypeClassBulkFormerPCARFRunner._build_bulkformer_model
    _extract_bulkformer_embeddings = (
        CancTypeClassBulkFormerPCARFRunner._extract_bulkformer_embeddings
    )
    _gather_bulkformer_embeddings = (
        CancTypeClassBulkFormerPCARFRunner._gather_bulkformer_embeddings
    )
    _strip_ensembl_version = staticmethod(
        CancTypeClassBulkFormerPCARFRunner._strip_ensembl_version
    )
    _align_to_bulkformer_genes = staticmethod(
        CancTypeClassBulkFormerPCARFRunner._align_to_bulkformer_genes
    )

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "bulkformer_variant", "") or "").strip()
        return variant or "bulkformer_pca_rf"

    def _task_output_dir(self) -> Path:
        return output_root(self.cfg) / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _pca_prefix(self) -> str:
        return "bulkformer_pca"

    def _rf_prefix(self) -> str:
        return "bulkformer_rf"

    def _required_path(self, attr: str) -> Path:
        value = getattr(self.task_cfg, attr, None)
        if not value:
            raise ValueError(f"finetune.drug_resp.{attr} must be set.")
        path = Path(hydra.utils.to_absolute_path(str(value)))
        if not path.exists():
            raise FileNotFoundError(f"Missing BulkFormer file for {attr}: {path}")
        return path

    def _bulkformer_paths(self) -> dict[str, Path]:
        paths = {
            "repo_dir": self._required_path("bulkformer_repo_dir"),
            "checkpoint": self._required_path("bulkformer_checkpoint_path"),
            "expression": self._required_path("bulkformer_expression_data_path"),
            "gene_info": self._required_path("bulkformer_gene_info_path"),
            "graph": self._required_path("bulkformer_graph_path"),
            "graph_weight": self._required_path("bulkformer_graph_weight_path"),
            "gene_emb": self._required_path("bulkformer_gene_emb_path"),
        }
        interested = getattr(
            self.task_cfg,
            "bulkformer_interested_gene_list_path",
            None,
        )
        if interested:
            paths["interested_gene_list"] = self._required_path(
                "bulkformer_interested_gene_list_path"
            )
        return paths

    def _prepare_bulkformer_expression(
        self,
        paths: dict[str, Path],
    ) -> tuple[np.ndarray, float]:
        expression_path = paths["expression"]
        if expression_path.suffix.lower() == ".h5ad":
            import anndata as ad
            from scipy import sparse

            adata = ad.read_h5ad(expression_path)
            values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
            columns = (
                adata.var["ensg_id"].astype(str).to_numpy()
                if "ensg_id" in adata.var
                else adata.var_names.astype(str).to_numpy()
            )
            expression = pd.DataFrame(
                values,
                index=adata.obs_names.astype(str),
                columns=self._strip_ensembl_version(columns),
            )
        else:
            expression = pd.read_csv(expression_path, index_col=0, low_memory=False)
            expression.index = expression.index.astype(str)
            gene_info = pd.read_csv(paths["gene_info"])
            if not {"gene_symbol", "ensg_id"}.issubset(gene_info.columns):
                raise ValueError(
                    "BulkFormer gene info must contain gene_symbol and ensg_id columns."
                )
            symbol_to_ensg = dict(
                zip(
                    gene_info["gene_symbol"].astype(str),
                    self._strip_ensembl_version(gene_info["ensg_id"]).astype(str),
                )
            )
            mapped_columns = []
            keep_columns = []
            for column in expression.columns.astype(str):
                clean = str(self._strip_ensembl_version([column])[0])
                mapped = clean if clean.startswith("ENSG") else symbol_to_ensg.get(column)
                if mapped:
                    keep_columns.append(column)
                    mapped_columns.append(str(mapped))
            expression = expression.loc[:, keep_columns].copy()
            expression.columns = mapped_columns

        if expression.index.duplicated().any():
            raise ValueError("BulkFormer GDSC expression contains duplicate cell-line IDs.")
        missing_cells = [
            cell_id
            for cell_id in self._canonical_cell_ids
            if cell_id not in expression.index
        ]
        if missing_cells:
            raise ValueError(
                f"BulkFormer GDSC expression is missing {len(missing_cells)} canonical "
                f"cell lines. First missing IDs: {missing_cells[:10]}"
            )
        expression = expression.loc[self._canonical_cell_ids]
        expression = expression.loc[:, ~expression.columns.duplicated(keep="first")]
        observed = expression.to_numpy(dtype=np.float32, copy=False)
        if not np.all(np.isfinite(observed)):
            raise ValueError("BulkFormer GDSC expression contains non-finite values.")
        expression_min = float(observed.min())
        expression_max = float(observed.max())
        max_expected = float(
            getattr(self.task_cfg, "bulkformer_max_expected_expression", 30.0)
        )
        max_observed_magnitude = max(abs(expression_min), abs(expression_max))
        if max_observed_magnitude > max_expected:
            raise ValueError(
                "BulkFormer expects normalized GDSC expression with bounded magnitude, but "
                "observed "
                f"range is [{expression_min:.6g}, {expression_max:.6g}] with configured "
                f"absolute limit {max_expected:.6g}."
            )

        gene_info = pd.read_csv(paths["gene_info"])
        gene_list = self._strip_ensembl_version(
            gene_info["ensg_id"].astype(str)
        ).astype(str).tolist()
        expected_gene_count = int(
            getattr(self.task_cfg, "bulkformer_expected_gene_count", 20010)
        )
        if len(gene_list) != expected_gene_count or len(set(gene_list)) != len(gene_list):
            raise ValueError(
                f"BulkFormer gene vocabulary must contain {expected_gene_count} unique genes."
            )
        aligned = self._align_to_bulkformer_genes(expression, gene_list)
        present = np.isin(gene_list, expression.columns.astype(str))
        missing_fraction = float(np.mean(~present))
        self._bulkformer_matched_gene_count = int(present.sum())
        self._bulkformer_missing_gene_count = int((~present).sum())
        self._bulkformer_missing_fraction = missing_fraction
        self._bulkformer_expression_min = expression_min
        self._bulkformer_expression_max = expression_max
        log.info(
            "Prepared BulkFormer GDSC data: cells=%d genes=%d matched=%d missing=%d",
            aligned.shape[0],
            aligned.shape[1],
            self._bulkformer_matched_gene_count,
            self._bulkformer_missing_gene_count,
        )
        return aligned, missing_fraction

    def _save_external_metadata(self, paths: dict[str, Path]) -> None:
        if not self.is_master:
            return
        out_dir = self._task_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{self._output_prefix()}_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        self._run_metadata_path = out_dir / f"{self._output_prefix()}_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": "finetune.drug_resp_bulkformer_pca_rf",
                "model_key": "bulkformer_147m",
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "cv_fold_manifest_path": str(self._cv_fold_manifest_path),
                "cv_fold_fingerprint": str(self._cv_fold_fingerprint),
                "matched_gene_count": self._bulkformer_matched_gene_count,
                "missing_gene_count": self._bulkformer_missing_gene_count,
                "missing_vocab_fraction": self._bulkformer_missing_fraction,
                "aggregate_type": str(
                    getattr(self.task_cfg, "bulkformer_aggregate_type", "max")
                ),
                "world_size": int(self.world_size),
            },
            checkpoint_paths={"bulkformer_147m": str(paths["checkpoint"])},
            repo_dir=REPO_ROOT,
        )

    def run(self) -> dict:
        try:
            self._setup_runtime()
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            X_cell, drug_emb_matrix, cell_idxs, drug_idxs, targets, pair_cell_ids = (
                self._load_gdsc_data()
            )
            splits = self._build_or_load_cv_splits(
                pair_cell_ids,
                self._pair_drug_ids,
                targets,
            )
            paths = self._bulkformer_paths()
            expression, missing_fraction = self._prepare_bulkformer_expression(paths)
            model = self._build_bulkformer_model(paths)
            cell_features, cell_order = self._extract_bulkformer_embeddings(
                model,
                expression,
                paths,
                missing_fraction,
            )
            cell_features, cell_order = self._gather_bulkformer_embeddings(
                cell_features,
                cell_order,
                expression.shape[0],
            )
            if not np.array_equal(cell_order, np.arange(expression.shape[0])):
                raise RuntimeError("Distributed BulkFormer extraction changed cell-line order.")
            self._save_external_metadata(paths)
            if not self.is_master:
                return {}

            fold_rows = []
            cell_line_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
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
                        "bulkformer_147m",
                        fold_idx,
                        len(splits),
                        str(paths["checkpoint"]),
                        train_metrics,
                        test_metrics,
                    )
                )
                cell_line_rows.extend(
                    self._per_cell_line_rows(
                        "bulkformer_147m",
                        fold_idx,
                        str(paths["checkpoint"]),
                        test_metrics,
                    )
                )
            aggregate = self._write_model_results(
                str(paths["checkpoint"]),
                fold_rows,
                cell_line_rows,
            )
            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            self._write_csv(output_path, [aggregate])
            complete_run_metadata(self._run_metadata_path, output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
