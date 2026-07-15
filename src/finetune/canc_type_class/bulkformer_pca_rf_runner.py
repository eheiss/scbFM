from __future__ import annotations

import logging
import sys
import types
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from finetune.canc_type_class.runner import DEFAULT_COHORTS, ROOT, CancTypeClassRunner
from utils import seed_all

log = logging.getLogger(__name__)


class CancTypeClassBulkFormerPCARFRunner(CancTypeClassRunner):
    """BulkFormer-147M frozen embeddings -> PCA -> random forest for TCGA 5-type classification."""

    task_name = "canc_type_class"
    config_node = "canc_type_class"

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "canc_type_class" in cfg.finetune:
            return cfg.finetune.canc_type_class
        raise ValueError(
            "Could not find cancer type classification config. "
            "Expected cfg.finetune.canc_type_class."
        )

    def _finetune_mode(self) -> str:
        variant = str(getattr(self.task_cfg, "bulkformer_variant", "") or "").strip()
        return variant or "bulkformer_pca_rf"

    def _task_output_dir(self) -> Path:
        return ROOT / "output" / self.task_name / self._finetune_mode()

    def _output_prefix(self) -> str:
        return f"{self.task_name}_{self._finetune_mode()}"

    def _save_run_metadata(self) -> None:
        if not self.is_master:
            return
        out_dir = self._task_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_prefix()
        (out_dir / f"{prefix}_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        self._write_json(
            out_dir / f"{prefix}_run_metadata.json",
            {
                "task": "finetune.canc_type_class_bulkformer_pca_rf",
                "baseline": "bulkformer_147m_embedding_pca_random_forest",
                "variant": self._finetune_mode(),
                "bulkformer_checkpoint_path": str(getattr(self.task_cfg, "bulkformer_checkpoint_path", "")),
                "bulkformer_repo_dir": str(getattr(self.task_cfg, "bulkformer_repo_dir", "")),
                "bulkformer_tcga_data_path": str(getattr(self.task_cfg, "bulkformer_tcga_data_path", "")),
                "pca_components": int(getattr(self.task_cfg, "bulkformer_pca_components", 256)),
                "rf_n_estimators": int(getattr(self.task_cfg, "bulkformer_rf_n_estimators", 500)),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "git_commit": self._get_git_commit(),
            },
        )

    def _required_path(self, attr: str) -> Path:
        value = getattr(self.task_cfg, attr, None)
        if not value:
            raise ValueError(f"finetune.{self.config_node}.{attr} must be set.")
        path = Path(hydra.utils.to_absolute_path(str(value)))
        if not path.exists():
            raise FileNotFoundError(f"Missing BulkFormer file for {attr}: {path}")
        return path

    def _bulkformer_paths(self) -> dict[str, Path]:
        repo_dir = self._required_path("bulkformer_repo_dir")
        paths = {
            "repo_dir": repo_dir,
            "checkpoint": self._required_path("bulkformer_checkpoint_path"),
            "tcga_data": self._required_path("bulkformer_tcga_data_path"),
            "gene_info": self._required_path("bulkformer_gene_info_path"),
            "graph": self._required_path("bulkformer_graph_path"),
            "graph_weight": self._required_path("bulkformer_graph_weight_path"),
            "gene_emb": self._required_path("bulkformer_gene_emb_path"),
        }
        interested_gene_list_path = getattr(self.task_cfg, "bulkformer_interested_gene_list_path", None)
        if interested_gene_list_path:
            path = Path(hydra.utils.to_absolute_path(str(interested_gene_list_path)))
            if not path.exists():
                raise FileNotFoundError(
                    "Missing BulkFormer file for bulkformer_interested_gene_list_path: "
                    f"{path}"
                )
            paths["interested_gene_list"] = path
        return paths

    def _load_bulkformer_tcga(self, paths: dict[str, Path]) -> ad.AnnData:
        adata = ad.read_h5ad(paths["tcga_data"])
        required_obs = {"sample_id", "patient_id", "project"}
        missing_obs = sorted(required_obs.difference(adata.obs.columns))
        if missing_obs:
            raise ValueError(
                f"BulkFormer TCGA AnnData is missing required obs columns: {missing_obs}."
            )

        adata.obs["project"] = adata.obs["project"].astype(str).str.strip().str.upper()
        cohorts = list(getattr(self.task_cfg, "cohorts", DEFAULT_COHORTS))
        selected_cancer_types = {str(cohort).upper() for cohort in cohorts}
        keep_mask = adata.obs["project"].isin(selected_cancer_types).to_numpy()
        adata = adata[keep_mask].copy()
        if adata.n_obs == 0:
            raise ValueError(
                f"No BulkFormer TCGA samples matched cohorts {cohorts} in obs['project']."
            )

        adata.obs["cancer_type"] = adata.obs["project"].astype(str)
        if bool(getattr(self.task_cfg, "merge_gbm_lgg", True)):
            adata.obs["cancer_type"] = adata.obs["cancer_type"].replace(
                {"GBM": "GBMLGG", "LGG": "GBMLGG"}
            )
        adata.obs_names = adata.obs["sample_id"].astype(str)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        return adata

    @staticmethod
    def _adata_to_gene_frame(adata: ad.AnnData) -> pd.DataFrame:
        var_names = adata.var_names.astype(str)
        if "ensg_id" in adata.var:
            var_names = adata.var["ensg_id"].astype(str).to_numpy()
        x = adata.X
        if sparse.issparse(x):
            x = x.toarray()
        return pd.DataFrame(
            np.asarray(x, dtype=np.float32),
            index=adata.obs_names.astype(str),
            columns=var_names,
        )

    @staticmethod
    def _align_to_bulkformer_genes(expr_df: pd.DataFrame, gene_list: list[str]) -> np.ndarray:
        expr_df = expr_df.loc[:, ~expr_df.columns.duplicated()].copy()
        present = [gene for gene in gene_list if gene in expr_df.columns]
        aligned = np.full((expr_df.shape[0], len(gene_list)), -10.0, dtype=np.float32)
        if present:
            gene_to_pos = {gene: idx for idx, gene in enumerate(gene_list)}
            target_cols = [gene_to_pos[gene] for gene in present]
            aligned[:, target_cols] = expr_df[present].to_numpy(dtype=np.float32, copy=False)
        return aligned

    def _prepare_bulkformer_data(
        self,
        paths: dict[str, Path],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, ad.AnnData]:
        adata = self._load_bulkformer_tcga(paths)
        labels_str = np.asarray(adata.obs["cancer_type"]).astype(str)
        self.label_dict = np.unique(labels_str)
        patient_ids = adata.obs["patient_id"].astype(str).to_numpy()
        groups = patient_ids if len(np.unique(patient_ids)) < len(patient_ids) else None

        gene_info = pd.read_csv(paths["gene_info"])
        if "ensg_id" not in gene_info.columns:
            raise ValueError(f"BulkFormer gene info must contain an 'ensg_id' column: {paths['gene_info']}")
        gene_list = gene_info["ensg_id"].astype(str).tolist()
        expr_df = self._adata_to_gene_frame(adata)
        expr_array = self._align_to_bulkformer_genes(expr_df, gene_list)

        label_to_idx = {label: idx for idx, label in enumerate(self.label_dict.tolist())}
        labels = np.asarray([label_to_idx[label] for label in labels_str], dtype=np.int64)

        missing_fraction = float(np.mean(~np.isin(gene_list, expr_df.columns.astype(str))))
        log.info(
            "Prepared BulkFormer TCGA data: samples=%d, genes=%d, missing_vocab_fraction=%.4f",
            expr_array.shape[0],
            expr_array.shape[1],
            missing_fraction,
        )
        return expr_array, labels, groups, adata

    def _build_bulkformer_model(self, paths: dict[str, Path]):
        repo_dir = paths["repo_dir"]
        repo_dir_str = str(repo_dir)
        if repo_dir_str in sys.path:
            sys.path.remove(repo_dir_str)
        sys.path.insert(0, repo_dir_str)

        # scbFM has src/utils.py, while BulkFormer imports from utils/BulkFormer.py.
        # BulkFormer/utils has no __init__.py, so it is only a namespace package.
        # If scbFM/src is also on sys.path, Python prefers the regular scbFM
        # utils.py module over the BulkFormer namespace package. Force package
        # aliases to the BulkFormer repo before importing its code.
        for module_name in list(sys.modules):
            if module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]
            if module_name == "model" or module_name.startswith("model."):
                del sys.modules[module_name]

        bulkformer_utils = types.ModuleType("utils")
        bulkformer_utils.__path__ = [str(repo_dir / "utils")]
        sys.modules["utils"] = bulkformer_utils

        bulkformer_model = types.ModuleType("model")
        bulkformer_model.__path__ = [str(repo_dir / "model")]
        sys.modules["model"] = bulkformer_model

        try:
            from model.config import model_params
            from torch_geometric.typing import SparseTensor
            from utils.BulkFormer import BulkFormer
        except ImportError as exc:
            raise ImportError(
                "BulkFormer requires its repo on PYTHONPATH plus torch-geometric/torch-cluster "
                "installed in the container."
            ) from exc

        log.info("Loading BulkFormer graph and gene resources")
        graph = torch.load(paths["graph"], map_location="cpu")
        weights = torch.load(paths["graph_weight"], map_location="cpu")
        graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(self.device)
        gene_emb = torch.load(paths["gene_emb"], map_location="cpu")

        params = dict(model_params)
        params["graph"] = graph
        params["gene_emb"] = gene_emb
        model = BulkFormer(**params).to(self.device)

        checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        state_dict = OrderedDict()
        for key, value in checkpoint.items():
            state_dict[key.removeprefix("module.")] = value
        model.load_state_dict(state_dict)
        model.eval()
        log.info(
            "Loaded BulkFormer checkpoint: %s | params=%d",
            paths["checkpoint"],
            sum(param.numel() for param in model.parameters()),
        )
        return model

    def _extract_bulkformer_embeddings(
        self,
        model,
        expr_array: np.ndarray,
        paths: dict[str, Path],
    ) -> np.ndarray:
        batch_size = int(getattr(self.task_cfg, "bulkformer_batch_size", 4))
        mask_prob = float(getattr(self.task_cfg, "bulkformer_mask_prob", 0.1))
        aggregate_type = str(getattr(self.task_cfg, "bulkformer_aggregate_type", "mean"))

        interested_gene_idx = None
        if "interested_gene_list" in paths:
            interested_gene_idx = torch.load(paths["interested_gene_list"], map_location="cpu")
            if isinstance(interested_gene_idx, torch.Tensor):
                interested_gene_idx = interested_gene_idx.cpu().numpy().tolist()

        loader = DataLoader(
            TensorDataset(torch.as_tensor(expr_array, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        log.info(
            "Extracting BulkFormer embeddings: samples=%d | genes=%d | batch_size=%d | "
            "batches=%d | aggregate=%s | interested_genes=%s",
            expr_array.shape[0],
            expr_array.shape[1],
            batch_size,
            len(loader),
            aggregate_type,
            "all" if interested_gene_idx is None else len(interested_gene_idx),
        )
        embeddings = []
        with torch.no_grad():
            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=True)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                for (batch,) in tqdm(loader, total=len(loader), disable=True):
                    batch = batch.to(self.device, non_blocking=True)
                    gene_emb = model(batch, mask_prob=mask_prob, output_expr=False)
                    gene_emb = gene_emb.detach().cpu().numpy()
                    if interested_gene_idx is not None:
                        gene_emb = gene_emb[:, interested_gene_idx, :]

                    if aggregate_type == "mean":
                        sample_emb = np.mean(gene_emb, axis=1)
                    elif aggregate_type == "max":
                        sample_emb = np.max(gene_emb, axis=1)
                    elif aggregate_type == "median":
                        sample_emb = np.median(gene_emb, axis=1)
                    elif aggregate_type == "all":
                        sample_emb = (
                            np.max(gene_emb, axis=1)
                            + np.mean(gene_emb, axis=1)
                            + np.median(gene_emb, axis=1)
                        )
                    else:
                        raise ValueError(
                            "bulkformer_aggregate_type must be one of: mean, max, median, all."
                        )
                    embeddings.append(sample_emb.astype(np.float32, copy=False))
        embeddings_array = np.vstack(embeddings)
        log.info(
            "Extracted BulkFormer embeddings: shape=%s | dtype=%s",
            embeddings_array.shape,
            embeddings_array.dtype,
        )
        return embeddings_array

    def _fit_predict_embeddings(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, object]]:
        if bool(getattr(self.task_cfg, "bulkformer_pca_standardize", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        requested_components = int(getattr(self.task_cfg, "bulkformer_pca_components", 256))
        n_components = min(requested_components, train_x.shape[0] - 1, train_x.shape[1])
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than 1 component.")
        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, "bulkformer_pca_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(train_x)
        test_z = pca.transform(test_x)

        rf = RandomForestClassifier(
            n_estimators=int(getattr(self.task_cfg, "bulkformer_rf_n_estimators", 500)),
            max_depth=getattr(self.task_cfg, "bulkformer_rf_max_depth", None),
            min_samples_leaf=int(getattr(self.task_cfg, "bulkformer_rf_min_samples_leaf", 1)),
            min_samples_split=int(getattr(self.task_cfg, "bulkformer_rf_min_samples_split", 2)),
            max_features=str(getattr(self.task_cfg, "bulkformer_rf_max_features", "sqrt")),
            class_weight=str(getattr(self.task_cfg, "bulkformer_rf_class_weight", "balanced")),
            bootstrap=bool(getattr(self.task_cfg, "bulkformer_rf_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, "bulkformer_rf_n_jobs", -1)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        rf.fit(train_z, train_y)
        train_pred = rf.predict(train_z)
        pred = rf.predict(test_z)
        precision, recall, fscore, support = precision_recall_fscore_support(
            test_y,
            pred,
            labels=np.arange(len(self.label_dict)),
            zero_division=0,
        )
        train_metrics = {
            "loss": float("nan"),
            "accuracy": 100.0 * float(accuracy_score(train_y, train_pred)),
        }
        test_metrics = {
            "loss": float("nan"),
            "accuracy": float(accuracy_score(test_y, pred)),
            "f1_macro": float(
                f1_score(
                    test_y,
                    pred,
                    labels=np.arange(len(self.label_dict)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    test_y,
                    pred,
                    labels=np.arange(len(self.label_dict)),
                    average="weighted",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                test_y,
                pred,
                labels=np.arange(len(self.label_dict)),
            ),
            "classification_report": classification_report(
                test_y,
                pred,
                labels=np.arange(len(self.label_dict)),
                target_names=self.label_dict.tolist(),
                digits=4,
                zero_division=0,
            ),
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": fscore,
            "support_per_class": support,
            "label_dict": self.label_dict.tolist(),
            "n_test_samples": int(test_y.size),
            "truth_indices": test_y,
            "prediction_indices": pred,
            "pca_components": int(n_components),
            "embedding_dim": int(train_x.shape[1]),
        }
        return train_metrics, test_metrics

    def run(self) -> dict:
        try:
            self._setup_runtime()
            if self.is_distributed and self.world_size != 1:
                raise ValueError("BulkFormer PCA+RF baseline should be launched with one process.")
            paths = self._bulkformer_paths()
            expr_array, labels, groups, adata = self._prepare_bulkformer_data(paths)
            splits = self._build_cv_splits(np.asarray(adata.obs["cancer_type"]).astype(str), groups)
            self._save_run_metadata()
            log.info(
                "Prepared BulkFormer PCA+RF CV data: samples=%d, genes=%d, folds=%d",
                expr_array.shape[0],
                expr_array.shape[1],
                len(splits),
            )

            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model = self._build_bulkformer_model(paths)
            embeddings = self._extract_bulkformer_embeddings(model, expr_array, paths)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_key = "bulkformer_147m"
            checkpoint_path = str(paths["checkpoint"])
            fold_rows: list[dict[str, object]] = []
            prediction_rows: list[dict[str, object]] = []
            confusion_matrices: list[np.ndarray] = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                log.info(
                    "BulkFormer PCA+RF | Fold %d/%d | train=%d, test=%d",
                    fold_idx,
                    len(splits),
                    len(train_idx),
                    len(test_idx),
                )
                train_metrics, test_metrics = self._fit_predict_embeddings(
                    embeddings[train_idx],
                    labels[train_idx],
                    embeddings[test_idx],
                    labels[test_idx],
                )
                log.info(
                    "BulkFormer PCA+RF | Fold %d/%d | PCA components=%d | "
                    "Train Accuracy: %.4f%% | Test Accuracy: %.4f | Weighted F1: %.4f",
                    fold_idx,
                    len(splits),
                    int(test_metrics["pca_components"]),
                    train_metrics["accuracy"],
                    float(test_metrics["accuracy"]),
                    float(test_metrics["f1_weighted"]),
                )
                test_adata = adata[test_idx].copy()
                fold_rows.append(
                    self._flatten_fold_metrics(
                        model_key=model_key,
                        fold=fold_idx,
                        n_folds=len(splits),
                        checkpoint_path=checkpoint_path,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                    )
                )
                prediction_rows.extend(
                    self._prediction_rows(
                        model_key=model_key,
                        fold=fold_idx,
                        checkpoint_path=checkpoint_path,
                        test_adata=test_adata,
                        test_metrics=test_metrics,
                    )
                )
                confusion_matrices.append(np.asarray(test_metrics["confusion_matrix"]))

            aggregate = self._write_model_results(
                checkpoint_path,
                fold_rows,
                prediction_rows,
                confusion_matrices,
            )
            out_dir = self._task_output_dir()
            output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
            self._write_csv(output_path, [aggregate])
            log.info("BulkFormer PCA+RF results written to %s", output_path)
            return {"results_path": str(output_path), "results": [aggregate]}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
