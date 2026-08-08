from __future__ import annotations

import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

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
from torch.utils.data import DataLoader, Dataset

from finetune.canc_type_class.runner import ROOT, CancTypeClassRunner
from run_provenance import complete_run_metadata, start_run_metadata
from utils import SequentialDistributedSampler, distributed_concat, seed_all

log = logging.getLogger(__name__)


class _ScGPTExpressionDataset(Dataset):
    def __init__(
        self,
        matrix,
        labels: np.ndarray,
        gene_ids: np.ndarray,
        cls_token_id: int,
        cls_value: float,
    ) -> None:
        self.matrix = matrix
        self.labels = np.asarray(labels, dtype=np.int64)
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.cls_token_id = int(cls_token_id)
        self.cls_value = float(cls_value)

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.matrix[index]
        if sparse.issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row).ravel()

        nonzero_idx = np.flatnonzero(row != 0)
        if nonzero_idx.size == 0:
            nonzero_idx = np.arange(row.size, dtype=np.int64)
        genes = np.concatenate(([self.cls_token_id], self.gene_ids[nonzero_idx]))
        values = np.concatenate(([self.cls_value], row[nonzero_idx].astype(np.float32, copy=False)))
        example = {
            "genes": torch.as_tensor(genes, dtype=torch.long),
            "expressions": torch.as_tensor(values, dtype=torch.float32),
        }
        return example, torch.tensor(self.labels[index], dtype=torch.long)


def _bin_scgpt_examples_safely(
    examples: list[dict[str, torch.Tensor]],
    binning_fn: Callable[..., torch.Tensor],
    *,
    keep_first_n_tokens: int = 1,
    n_bins: int = 51,
) -> list[dict[str, torch.Tensor]]:
    binned_examples = []
    for example in examples:
        expressions = example["expressions"].clone()
        values = expressions[keep_first_n_tokens:]
        if torch.count_nonzero(values).item() > 0:
            expressions[keep_first_n_tokens:] = binning_fn(row=values, n_bins=n_bins)
        binned_examples.append(
            {
                **example,
                "expressions": expressions,
            }
        )
    return binned_examples


class CancTypeClassScGPTPCARFRunner(CancTypeClassRunner):
    """Frozen scGPT CLS embeddings -> PCA -> random forest for TCGA 5-type classification."""

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
        variant = str(getattr(self.task_cfg, "scgpt_variant", "") or "").strip()
        return variant or "scgpt_pca_rf"

    def _model_key(self) -> str:
        model_key = str(getattr(self.task_cfg, "scgpt_model_key", "") or "").strip()
        return model_key or "scgpt"

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
        self._run_metadata_path = out_dir / f"{prefix}_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": f"finetune.{self.task_name}_scgpt_pca_rf",
                "baseline": "scgpt_frozen_cls_embedding_pca_random_forest",
                "variant": self._finetune_mode(),
                "model_key": self._model_key(),
                "scgpt_repo_dir": str(getattr(self.task_cfg, "scgpt_repo_dir", "")),
                "scgpt_model_dir": str(getattr(self.task_cfg, "scgpt_model_dir", "")),
                "scgpt_gene_info_path": str(getattr(self.task_cfg, "scgpt_gene_info_path", "")),
                "source_gene_count": int(getattr(self, "_scgpt_source_gene_count", 0)),
                "vocab_matched_gene_count": int(
                    getattr(self, "_scgpt_vocab_matched_gene_count", 0)
                ),
                "max_sequence_length": int(
                    getattr(self, "_scgpt_effective_max_seq_len", 0)
                ),
                "world_size": int(self.world_size),
                "per_device_batch_size": int(
                    getattr(self.task_cfg, "scgpt_batch_size", 4)
                ),
                "global_inference_batch_size": int(
                    getattr(self.task_cfg, "scgpt_batch_size", 4)
                )
                * int(self.world_size),
                "pca_components": int(getattr(self.task_cfg, "scgpt_pca_components", 256)),
                "rf_n_estimators": int(getattr(self.task_cfg, "scgpt_rf_n_estimators", 500)),
                "selected_gene_count": int(self.selected_gene_count),
                "checkpoint_loading": getattr(self, "_scgpt_load_report", {}),
                "cv_folds": int(getattr(self.task_cfg, "cv_folds", 5)),
                "cv_fold_manifest_path": str(
                    getattr(self, "_cv_fold_manifest_path", "")
                ),
                "cv_fold_fingerprint": str(
                    getattr(self, "_cv_fold_fingerprint", "")
                ),
            },
            checkpoint_paths={self._model_key(): str(self._scgpt_paths()["checkpoint"])},
            repo_dir=ROOT / "scbFM",
        )

    @staticmethod
    def _checkpoint_state_dict(checkpoint) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("The scGPT checkpoint must contain a state dictionary.")
        state_dict: dict[str, torch.Tensor] = {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            clean_key = str(key)
            for prefix in ("module.", "_orig_mod."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            state_dict[clean_key] = value
        return state_dict

    def _required_path(self, attr: str) -> Path:
        value = getattr(self.task_cfg, attr, None)
        if not value:
            raise ValueError(f"finetune.{self.config_node}.{attr} must be set.")
        path = Path(hydra.utils.to_absolute_path(str(value)))
        if not path.exists():
            raise FileNotFoundError(f"Missing scGPT path for {attr}: {path}")
        return path

    def _scgpt_paths(self) -> dict[str, Path]:
        repo_dir = self._required_path("scgpt_repo_dir")
        model_dir = self._required_path("scgpt_model_dir")
        args_filename = str(getattr(self.task_cfg, "scgpt_args_filename", "args.json"))
        vocab_filename = str(getattr(self.task_cfg, "scgpt_vocab_filename", "vocab.json"))
        checkpoint_filename = str(
            getattr(self.task_cfg, "scgpt_checkpoint_filename", "last_model.pt")
        )
        paths = {
            "repo_dir": repo_dir,
            "model_dir": model_dir,
            "args": model_dir / args_filename,
            "vocab": model_dir / vocab_filename,
            "checkpoint": model_dir / checkpoint_filename,
            "gene_info": self._required_path("scgpt_gene_info_path"),
        }
        missing = [str(path) for key, path in paths.items() if key != "repo_dir" and not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required scGPT model/resource files: " + ", ".join(missing))
        return paths

    def _prepare_raw_tcga_data(self) -> tuple[ad.AnnData, np.ndarray, np.ndarray | None]:
        adata = self._load_input_adata()
        adata.obs["cancer_type"] = adata.obs["cancer_type"].astype(str)
        if bool(getattr(self.task_cfg, "merge_gbm_lgg", True)):
            adata.obs["cancer_type"] = adata.obs["cancer_type"].replace(
                {"GBM": "GBMLGG", "LGG": "GBMLGG"}
            )

        self.label_dict = np.unique(np.asarray(adata.obs["cancer_type"]).astype(str))
        labels = np.asarray(adata.obs["cancer_type"]).astype(str)
        patient_ids = adata.obs["patient_id"].astype(str).to_numpy()
        groups = None
        if len(np.unique(patient_ids)) < len(patient_ids):
            groups = patient_ids
            log.info("Duplicate TCGA patient_id values detected; using patient-grouped CV.")
        return adata, labels, groups

    @staticmethod
    def _strip_ensembl_version(values: pd.Series | np.ndarray | list[str]) -> pd.Series:
        return pd.Series(values, dtype="string").str.replace(r"\.\d+$", "", regex=True)

    def _add_scgpt_gene_symbols(
        self,
        adata: ad.AnnData,
        gene_info_path: Path,
        vocab,
    ) -> ad.AnnData:
        expected_source_gene_count = int(
            getattr(self.task_cfg, "scgpt_expected_source_gene_count", 13004)
        )
        if adata.n_vars != expected_source_gene_count:
            raise ValueError(
                f"scGPT source expression input contains {adata.n_vars} genes; "
                f"expected {expected_source_gene_count}."
            )
        gene_info = pd.read_csv(gene_info_path)
        required_columns = {"ensg_id", "gene_symbol"}
        missing = sorted(required_columns.difference(gene_info.columns))
        if missing:
            raise ValueError(f"scGPT gene info file is missing required columns: {missing}")

        gene_info = gene_info.dropna(subset=["ensg_id", "gene_symbol"]).copy()
        gene_info["ensg_id_clean"] = self._strip_ensembl_version(gene_info["ensg_id"]).to_numpy()
        gene_info = gene_info.drop_duplicates("ensg_id_clean", keep="first")
        ensg_to_symbol = dict(zip(gene_info["ensg_id_clean"], gene_info["gene_symbol"].astype(str)))

        if "gene_symbol" in adata.var:
            symbols = adata.var["gene_symbol"].astype(str).to_numpy()
        else:
            if "ensg_id" in adata.var:
                ensg = self._strip_ensembl_version(adata.var["ensg_id"]).to_numpy()
            else:
                ensg = self._strip_ensembl_version(adata.var_names).to_numpy()
            symbols = np.asarray([ensg_to_symbol.get(str(gene), "") for gene in ensg], dtype=object)

        in_vocab = np.zeros(len(symbols), dtype=bool)
        seen_token_ids: set[int] = set()
        for index, symbol in enumerate(symbols):
            if not symbol or symbol not in vocab:
                continue
            token_id = int(vocab[str(symbol)])
            if token_id in seen_token_ids:
                continue
            seen_token_ids.add(token_id)
            in_vocab[index] = True
        if int(in_vocab.sum()) < self.selected_gene_count:
            raise ValueError(
                "Only "
                f"{int(in_vocab.sum())} genes map to unique scGPT vocabulary tokens, fewer "
                f"than selected_gene_count={self.selected_gene_count}."
            )

        self._scgpt_source_gene_count = int(adata.n_vars)
        adata = adata[:, in_vocab].copy()
        adata.var["scgpt_gene_symbol"] = np.asarray(symbols, dtype=object)[in_vocab]
        adata.var_names_make_unique()
        self._scgpt_vocab_matched_gene_count = int(adata.n_vars)
        log.info(
            "Mapped source genes to unique scGPT vocabulary tokens: %d/%d retained",
            int(in_vocab.sum()),
            len(in_vocab),
        )
        return adata

    def _build_scgpt_model(self, paths: dict[str, Path]):
        repo_dir = str(paths["repo_dir"])
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        try:
            from scgpt.model import TransformerModel
            from scgpt.tokenizer import GeneVocab
        except ImportError as exc:
            raise ImportError(
                "scGPT requires its repository on PYTHONPATH plus dependencies such as "
                "torchtext installed in the container."
            ) from exc

        vocab = GeneVocab.from_file(paths["vocab"])
        for token in ("<pad>", "<cls>", "<eoc>"):
            if token not in vocab:
                vocab.append_token(token)
        vocab.set_default_index(vocab["<pad>"])

        with paths["args"].open("r", encoding="utf-8") as handle:
            model_configs = json.load(handle)

        model = TransformerModel(
            ntoken=len(vocab),
            d_model=int(model_configs["embsize"]),
            nhead=int(model_configs["nheads"]),
            d_hid=int(model_configs["d_hid"]),
            nlayers=int(model_configs["nlayers"]),
            nlayers_cls=int(model_configs.get("n_layers_cls", 3)),
            n_cls=1,
            vocab=vocab,
            dropout=float(model_configs["dropout"]),
            pad_token=str(model_configs["pad_token"]),
            pad_value=int(model_configs["pad_value"]),
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            input_emb_style=str(model_configs.get("input_emb_style", "continuous")),
            n_input_bins=int(model_configs.get("n_bins", 51)),
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            use_fast_transformer=bool(getattr(self.task_cfg, "scgpt_use_fast_transformer", False)),
            fast_transformer_backend="flash",
            pre_norm=bool(model_configs.get("pre_norm", False)),
        )
        log.info("Loading scGPT checkpoint from %s", paths["checkpoint"])
        source_state = self._checkpoint_state_dict(
            torch.load(paths["checkpoint"], map_location="cpu")
        )
        if not bool(getattr(model, "use_fast_transformer", False)):
            source_state = {
                key.replace("Wqkv.", "in_proj_"): value
                for key, value in source_state.items()
            }
        target_state = model.state_dict()
        merged_state = dict(target_state)
        matched: list[str] = []
        missing: list[str] = []
        shape_mismatches: list[dict[str, object]] = []
        for key, target_value in target_state.items():
            source_value = source_state.get(key)
            if source_value is None:
                missing.append(key)
            elif tuple(source_value.shape) != tuple(target_value.shape):
                shape_mismatches.append(
                    {
                        "key": key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                    }
                )
            else:
                merged_state[key] = source_value
                matched.append(key)

        allowed_head_prefixes = ("mvc_decoder.", "cls_decoder.")
        invalid_missing = [
            key for key in missing if not key.startswith(allowed_head_prefixes)
        ]
        invalid_shapes = [
            item
            for item in shape_mismatches
            if not str(item["key"]).startswith(allowed_head_prefixes)
        ]
        if invalid_missing or invalid_shapes:
            raise ValueError(
                "The pretrained scGPT checkpoint is incompatible with args.json/vocab.json. "
                f"Core missing keys={invalid_missing[:20]}, "
                f"core shape mismatches={invalid_shapes[:20]}"
            )
        model.load_state_dict(merged_state, strict=True)
        unexpected = sorted(set(source_state).difference(target_state))
        self._scgpt_load_report = {
            "strict_core_loading": True,
            "checkpoint_tensor_count": len(source_state),
            "model_tensor_count": len(target_state),
            "matched_tensor_count": len(matched),
            "matched_parameter_fraction": (
                sum(target_state[key].numel() for key in matched)
                / sum(value.numel() for value in target_state.values())
            ),
            "newly_initialized_keys": sorted(
                set(missing).union(str(item["key"]) for item in shape_mismatches)
            ),
            "unexpected_checkpoint_keys": unexpected,
        }
        model = model.to(self.device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Loaded scGPT model: params=%d | embsize=%d | layers=%d | heads=%d | vocab=%d",
            n_params,
            int(model_configs["embsize"]),
            int(model_configs["nlayers"]),
            int(model_configs["nheads"]),
            len(vocab),
        )
        return model, vocab, model_configs

    def _make_scgpt_loader(
        self,
        adata: ad.AnnData,
        labels: np.ndarray,
        gene_ids: np.ndarray,
        vocab,
        model_configs: dict,
        *,
        stage: str,
    ) -> DataLoader:
        from scgpt.data_collator import DataCollator
        from scgpt.preprocess import binning

        pad_token = str(model_configs["pad_token"])
        pad_value = int(model_configs["pad_value"])
        if sparse.issparse(adata.X):
            empty_profile_indices = np.flatnonzero(
                np.asarray((adata.X.tocsr() != 0).getnnz(axis=1)).ravel() == 0
            )
        else:
            empty_profile_indices = np.flatnonzero(
                np.count_nonzero(np.asarray(adata.X), axis=1) == 0
            )
        if empty_profile_indices.size and self.is_master:
            sample_ids = (
                adata.obs["sample_id"].astype(str).to_numpy()
                if "sample_id" in adata.obs
                else adata.obs_names.astype(str).to_numpy()
            )
            affected_sample_ids = sample_ids[empty_profile_indices].tolist()
            log.warning(
                "scGPT all-zero selected-gene fallback | stage=%s | samples=%d | "
                "sample_ids=%s",
                stage,
                empty_profile_indices.size,
                affected_sample_ids,
            )
        dataset = _ScGPTExpressionDataset(
            matrix=adata.X,
            labels=labels,
            gene_ids=gene_ids,
            cls_token_id=int(vocab["<cls>"]),
            cls_value=float(pad_value),
        )
        required_length = int(gene_ids.size + 1)
        configured_max_length = getattr(self.task_cfg, "scgpt_max_seq_len", None)
        if configured_max_length is None:
            max_length = required_length
        else:
            max_length = int(configured_max_length)
            if max_length < required_length:
                raise ValueError(
                    f"scgpt_max_seq_len={max_length} would truncate the configured "
                    f"{gene_ids.size} selected genes plus CLS. Set it to at least "
                    f"{required_length}, or leave it null for automatic sizing."
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
            examples, batch_labels = zip(*batch)
            binned_examples = _bin_scgpt_examples_safely(list(examples), binning)
            return collator(binned_examples), torch.stack(list(batch_labels))

        workers = int(getattr(self.task_cfg, "num_workers", 2))
        batch_size = int(getattr(self.task_cfg, "scgpt_batch_size", 4))
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
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            collate_fn=collate_fn,
        )

    def _extract_scgpt_embeddings(self, model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        embeddings = []
        labels = []
        autocast_context = (
            torch.cuda.amp.autocast(enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            for batch, batch_labels in loader:
                gene = batch["gene"].to(self.device, non_blocking=True)
                expr = batch["expr"].to(self.device, non_blocking=True)
                padding_mask = gene.eq(int(model.encoder.embedding.padding_idx))
                hidden = model._encode(gene, expr, padding_mask)
                cell_emb = hidden[:, 0, :]
                cell_emb = torch.nn.functional.normalize(cell_emb, p=2, dim=1)
                embeddings.append(
                    cell_emb.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                labels.append(batch_labels.numpy())
        return np.vstack(embeddings), np.concatenate(labels)

    def _extract_scgpt_embeddings_safely(
        self,
        model,
        loader: DataLoader,
        *,
        stage: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_error: Exception | None = None
        result: tuple[np.ndarray, np.ndarray] | None = None
        try:
            result = self._extract_scgpt_embeddings(model, loader)
        except Exception as exc:
            local_error = exc
            log.exception(
                "scGPT embedding extraction failed | rank=%d | stage=%s | samples=%d",
                self.rank,
                stage,
                len(loader.dataset),
            )

        if self.is_distributed:
            failure_flag = torch.tensor(
                [int(local_error is not None)],
                dtype=torch.int32,
                device=self.device,
            )
            dist.all_reduce(failure_flag, op=dist.ReduceOp.MAX)
            if int(failure_flag.item()) != 0:
                if local_error is not None:
                    raise local_error
                raise RuntimeError(
                    "scGPT embedding extraction failed on another distributed rank "
                    f"during {stage}; inspect that rank's traceback in the Slurm error log."
                )

        if local_error is not None:
            raise local_error
        if result is None:
            raise RuntimeError(
                f"scGPT embedding extraction produced no result during {stage}."
            )
        return result

    def _gather_scgpt_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        total_examples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_distributed:
            return embeddings, labels

        embedding_tensor = torch.as_tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device,
        )
        label_tensor = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        gathered_embeddings = distributed_concat(
            embedding_tensor,
            total_examples,
            self.world_size,
        )
        gathered_labels = distributed_concat(
            label_tensor,
            total_examples,
            self.world_size,
        )
        return (
            gathered_embeddings.cpu().numpy().astype(np.float32, copy=False),
            gathered_labels.cpu().numpy().astype(np.int64, copy=False),
        )

    def _fit_predict_embeddings(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, object]]:
        if bool(getattr(self.task_cfg, "scgpt_pca_standardize", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        requested_components = int(getattr(self.task_cfg, "scgpt_pca_components", 256))
        n_components = min(requested_components, train_x.shape[0] - 1, train_x.shape[1])
        if n_components < 1:
            raise ValueError("Cannot fit PCA with fewer than 1 component.")

        pca = PCA(
            n_components=n_components,
            whiten=bool(getattr(self.task_cfg, "scgpt_pca_whiten", False)),
            random_state=int(getattr(self.task_cfg, "random_seed", 42)),
        )
        train_z = pca.fit_transform(train_x)
        test_z = pca.transform(test_x)

        rf = RandomForestClassifier(
            n_estimators=int(getattr(self.task_cfg, "scgpt_rf_n_estimators", 500)),
            max_depth=getattr(self.task_cfg, "scgpt_rf_max_depth", None),
            min_samples_leaf=int(getattr(self.task_cfg, "scgpt_rf_min_samples_leaf", 1)),
            min_samples_split=int(getattr(self.task_cfg, "scgpt_rf_min_samples_split", 2)),
            max_features=str(getattr(self.task_cfg, "scgpt_rf_max_features", "sqrt")),
            class_weight=str(getattr(self.task_cfg, "scgpt_rf_class_weight", "balanced")),
            bootstrap=bool(getattr(self.task_cfg, "scgpt_rf_bootstrap", True)),
            n_jobs=int(getattr(self.task_cfg, "scgpt_rf_n_jobs", -1)),
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

            paths = self._scgpt_paths()
            seed_all(int(getattr(self.task_cfg, "random_seed", 42)))
            model, vocab, model_configs = self._build_scgpt_model(paths)
            adata, labels_str, groups = self._prepare_raw_tcga_data()
            adata = self._add_scgpt_gene_symbols(adata, paths["gene_info"], vocab)
            splits = self._build_or_load_cv_splits(adata, labels_str, groups)
            label_to_idx = {label: idx for idx, label in enumerate(self.label_dict.tolist())}
            labels = np.asarray([label_to_idx[label] for label in labels_str], dtype=np.int64)
            configured_max_length = getattr(self.task_cfg, "scgpt_max_seq_len", None)
            self._scgpt_effective_max_seq_len = (
                self.selected_gene_count + 1
                if configured_max_length is None
                else int(configured_max_length)
            )
            self._save_run_metadata()
            log.info(
                "Prepared scGPT PCA+RF CV data: samples=%d, vocab-matched source "
                "genes=%d, selected genes per fold=%d, folds=%d",
                adata.n_obs,
                adata.n_vars,
                self.selected_gene_count,
                len(splits),
            )

            model_key = self._model_key()
            checkpoint_path = str(paths["checkpoint"])
            fold_rows: list[dict[str, object]] = []
            prediction_rows: list[dict[str, object]] = []
            confusion_matrices: list[np.ndarray] = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                log.info(
                    "scGPT PCA+RF | Fold %d/%d | train=%d, test=%d",
                    fold_idx,
                    len(splits),
                    len(train_idx),
                    len(test_idx),
                )
                hvg_idx = self._select_training_hvg_indices(adata[train_idx].copy())
                fold_adata = adata[:, hvg_idx].copy()
                gene_symbols = fold_adata.var["scgpt_gene_symbol"].astype(str).tolist()
                gene_ids = np.asarray(vocab(gene_symbols), dtype=np.int64)

                train_loader = self._make_scgpt_loader(
                    fold_adata[train_idx].copy(),
                    labels[train_idx],
                    gene_ids,
                    vocab,
                    model_configs,
                    stage=f"fold {fold_idx} training split",
                )
                test_loader = self._make_scgpt_loader(
                    fold_adata[test_idx].copy(),
                    labels[test_idx],
                    gene_ids,
                    vocab,
                    model_configs,
                    stage=f"fold {fold_idx} validation split",
                )
                train_emb, train_y = self._extract_scgpt_embeddings_safely(
                    model,
                    train_loader,
                    stage=f"fold {fold_idx} training split",
                )
                test_emb, test_y = self._extract_scgpt_embeddings_safely(
                    model,
                    test_loader,
                    stage=f"fold {fold_idx} validation split",
                )
                train_emb, train_y = self._gather_scgpt_embeddings(
                    train_emb,
                    train_y,
                    len(train_idx),
                )
                test_emb, test_y = self._gather_scgpt_embeddings(
                    test_emb,
                    test_y,
                    len(test_idx),
                )
                if not np.array_equal(train_y, labels[train_idx]):
                    raise RuntimeError(
                        "Distributed scGPT extraction changed training-sample order."
                    )
                if not np.array_equal(test_y, labels[test_idx]):
                    raise RuntimeError(
                        "Distributed scGPT extraction changed validation-sample order."
                    )
                log.info(
                    "scGPT PCA+RF | Fold %d/%d | selected genes=%d | "
                    "embeddings train=%s test=%s",
                    fold_idx,
                    len(splits),
                    fold_adata.n_vars,
                    train_emb.shape,
                    test_emb.shape,
                )
                if not self.is_master:
                    if self.is_distributed:
                        dist.barrier()
                    continue
                train_metrics, test_metrics = self._fit_predict_embeddings(
                    train_emb,
                    train_y,
                    test_emb,
                    test_y,
                )
                log.info(
                    "scGPT PCA+RF | Fold %d/%d | PCA components=%d | "
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
                if self.is_distributed:
                    dist.barrier()

            out_dir = self._task_output_dir()
            output_path = out_dir / f"{self._output_prefix()}_evaluation_metrics.csv"
            if self.is_master:
                aggregate = self._write_model_results(
                    checkpoint_path,
                    fold_rows,
                    prediction_rows,
                    confusion_matrices,
                )
                self._write_csv(output_path, [aggregate])
                complete_run_metadata(self._run_metadata_path, output_path)
                log.info("scGPT PCA+RF results written to %s", output_path)
                return {"results_path": str(output_path), "results": [aggregate]}
            return {"results_path": str(output_path), "results": []}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
