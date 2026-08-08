from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch.distributed as dist
from omegaconf import OmegaConf

from finetune.canc_type_class.scgpt_pca_rf_runner import (
    CancTypeClassScGPTPCARFRunner,
)
from finetune.drug_resp.pca_rf_runner import DrugRespPCARFRunner
from finetune.drug_resp.runner import ROOT
from run_provenance import complete_run_metadata, start_run_metadata
from utils import seed_all

log = logging.getLogger(__name__)


class DrugRespScGPTPCARFRunner(DrugRespPCARFRunner):
    """Frozen scGPT cell embeddings + KPGT drug features -> PCA -> RF."""

    _checkpoint_state_dict = staticmethod(
        CancTypeClassScGPTPCARFRunner._checkpoint_state_dict
    )
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

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _pca_prefix(self) -> str:
        return "scgpt_pca"

    def _rf_prefix(self) -> str:
        return "scgpt_rf"

    def _required_path(self, attr: str) -> Path:
        value = getattr(self.task_cfg, attr, None)
        if not value:
            raise ValueError(f"finetune.drug_resp.{attr} must be set.")
        path = Path(hydra.utils.to_absolute_path(str(value)))
        if not path.exists():
            raise FileNotFoundError(f"Missing scGPT path for {attr}: {path}")
        return path

    def _scgpt_paths(self) -> dict[str, Path]:
        model_dir = self._required_path("scgpt_model_dir")
        paths = {
            "repo_dir": self._required_path("scgpt_repo_dir"),
            "model_dir": model_dir,
            "args": model_dir
            / str(getattr(self.task_cfg, "scgpt_args_filename", "args.json")),
            "vocab": model_dir
            / str(getattr(self.task_cfg, "scgpt_vocab_filename", "vocab.json")),
            "checkpoint": model_dir
            / str(
                getattr(
                    self.task_cfg,
                    "scgpt_checkpoint_filename",
                    "last_model.pt",
                )
            ),
            "gene_info": self._required_path("scgpt_gene_info_path"),
        }
        missing = [
            str(path)
            for key, path in paths.items()
            if key not in {"repo_dir", "model_dir"} and not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required scGPT model/resource files: " + ", ".join(missing)
            )
        return paths

    def _save_external_metadata(self, paths: dict[str, Path]) -> None:
        if not self.is_master:
            return
        out_dir = self._task_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_prefix()
        (out_dir / f"{prefix}_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        self._run_metadata_path = out_dir / f"{prefix}_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": "finetune.drug_resp_scgpt_pca_rf",
                "model_key": self._model_key(),
                "variant": self._finetune_mode(),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "cv_fold_manifest_path": str(self._cv_fold_manifest_path),
                "cv_fold_fingerprint": str(self._cv_fold_fingerprint),
                "source_gene_count": int(
                    getattr(self, "_scgpt_source_gene_count", 0)
                ),
                "vocab_matched_gene_count": int(
                    getattr(self, "_scgpt_vocab_matched_gene_count", 0)
                ),
                "selected_gene_count": int(self.selected_gene_count),
                "max_sequence_length": int(
                    getattr(self, "_scgpt_effective_max_seq_len", 0)
                ),
                "checkpoint_loading": getattr(self, "_scgpt_load_report", {}),
                "world_size": int(self.world_size),
                "per_device_batch_size": int(
                    getattr(self.task_cfg, "scgpt_batch_size", 4)
                ),
            },
            checkpoint_paths={self._model_key(): str(paths["checkpoint"])},
            repo_dir=ROOT / "scbFM",
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
            paths = self._scgpt_paths()
            model, vocab, model_configs = self._build_scgpt_model(paths)
            adata = self._add_scgpt_gene_symbols(
                self._canonical_adata,
                paths["gene_info"],
                vocab,
            )
            configured_max_length = getattr(self.task_cfg, "scgpt_max_seq_len", None)
            self._scgpt_effective_max_seq_len = (
                self.selected_gene_count + 1
                if configured_max_length is None
                else int(configured_max_length)
            )
            self._save_external_metadata(paths)

            fold_rows: list[dict[str, object]] = []
            cell_line_rows: list[dict[str, object]] = []
            checkpoint_path = str(paths["checkpoint"])
            model_key = self._model_key()

            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                unique_train_cells = np.unique(cell_idxs[train_idx]).astype(np.int64)
                hvg_idx = self._select_training_hvg_indices(
                    adata.X,
                    unique_train_cells,
                )
                fold_adata = adata[:, hvg_idx].copy()
                gene_symbols = (
                    fold_adata.var["scgpt_gene_symbol"].astype(str).tolist()
                )
                gene_ids = np.asarray(vocab(gene_symbols), dtype=np.int64)
                cell_order = np.arange(fold_adata.n_obs, dtype=np.int64)
                loader = self._make_scgpt_loader(
                    fold_adata,
                    cell_order,
                    gene_ids,
                    vocab,
                    model_configs,
                    stage=f"fold {fold_idx} all cell lines",
                )
                cell_features, observed_order = self._extract_scgpt_embeddings_safely(
                    model,
                    loader,
                    stage=f"fold {fold_idx} all cell lines",
                )
                cell_features, observed_order = self._gather_scgpt_embeddings(
                    cell_features,
                    observed_order,
                    fold_adata.n_obs,
                )
                if not np.array_equal(observed_order, cell_order):
                    raise RuntimeError(
                        "Distributed scGPT extraction changed cell-line order."
                    )

                if self.is_master:
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
                if self.is_distributed:
                    dist.barrier()

            output_path = self._task_output_dir() / (
                f"{self._output_prefix()}_evaluation_metrics.csv"
            )
            if not self.is_master:
                return {"results_path": str(output_path), "results": []}

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
