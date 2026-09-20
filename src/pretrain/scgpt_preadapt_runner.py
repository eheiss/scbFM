from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from paths import output_root
from scipy import sparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from utils import seed_all


log = logging.getLogger(__name__)


def _strip_ensembl_version(value: str) -> str:
    return str(value).split(".", maxsplit=1)[0]


def _sha256_strings(values: Iterator[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _random_train_validation_split(
    n_samples: int,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    """Return a deterministic sample-level split matching the bulk data builder."""
    if n_samples < 2:
        raise ValueError("scgpt_preadapt requires at least two bulk samples.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_split must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    n_validation = int(round(n_samples * validation_fraction))
    n_validation = min(max(n_validation, 1), n_samples - 1)
    split = np.full(n_samples, "train", dtype=object)
    split[shuffled[:n_validation]] = "validation"
    return split


class _RankSliceSampler(Sampler[int]):
    """Non-padding distributed sampler for evaluation on an unwrapped model."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.length = len(dataset)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.length:
            return 0
        return (self.length - 1 - self.rank) // self.world_size + 1


class ScGPTBulkDataset(Dataset):
    """Worker-safe backed reader over multiple aligned bulk h5ad files."""

    def __init__(
        self,
        *,
        data_paths: list[Path],
        file_indices: np.ndarray,
        row_indices: np.ndarray,
        record_indices: np.ndarray,
        source_ids: np.ndarray,
        source_gene_indices: list[np.ndarray],
        vocab_gene_ids: list[np.ndarray],
        cls_token_id: int,
        cls_value: float,
    ) -> None:
        self.data_paths = [str(path) for path in data_paths]
        self.file_indices = np.asarray(file_indices, dtype=np.int16)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.record_indices = np.asarray(record_indices, dtype=np.int64)
        self.source_ids = np.asarray(source_ids, dtype=np.int16)
        self.source_gene_indices = [
            np.asarray(indices, dtype=np.int64) for indices in source_gene_indices
        ]
        self.vocab_gene_ids = [
            np.asarray(indices, dtype=np.int64) for indices in vocab_gene_ids
        ]
        self.cls_token_id = int(cls_token_id)
        self.cls_value = float(cls_value)
        self._adatas: dict[int, ad.AnnData] = {}

        lengths = {
            self.file_indices.size,
            self.row_indices.size,
            self.record_indices.size,
            self.source_ids.size,
        }
        if len(lengths) != 1:
            raise ValueError("All scGPT bulk dataset index arrays must have equal length.")

    def __len__(self) -> int:
        return int(self.row_indices.size)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_adatas"] = {}
        return state

    def __del__(self):
        for adata in getattr(self, "_adatas", {}).values():
            file_obj = getattr(adata, "file", None)
            if file_obj is not None:
                file_obj.close()

    def _ensure_open(self, file_index: int) -> ad.AnnData:
        if file_index not in self._adatas:
            self._adatas[file_index] = ad.read_h5ad(
                self.data_paths[file_index],
                backed="r",
            )
        return self._adatas[file_index]

    @staticmethod
    def _dense_row(row) -> np.ndarray:
        if sparse.issparse(row):
            return np.asarray(row.toarray()).ravel()
        return np.asarray(row).ravel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_index = int(self.file_indices[index])
        row_index = int(self.row_indices[index])
        adata = self._ensure_open(file_index)
        source_columns = self.source_gene_indices[file_index]
        values = self._dense_row(adata.X[row_index])[source_columns].astype(
            np.float32,
            copy=False,
        )
        expressed = np.flatnonzero(values)
        if expressed.size == 0:
            raise ValueError(
                "scGPT pre-adaptation encountered a profile with no non-zero "
                "vocabulary-matched genes: "
                f"record_index={int(self.record_indices[index])}, "
                f"source_id={int(self.source_ids[index])}, row_index={row_index}."
            )
        values = values[expressed]
        vocab_gene_ids = self.vocab_gene_ids[file_index][expressed]
        genes = np.concatenate(
            (
                np.asarray([self.cls_token_id], dtype=np.int64),
                vocab_gene_ids,
            )
        )
        expressions = np.concatenate(
            (
                np.asarray([self.cls_value], dtype=np.float32),
                values,
            )
        )
        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(expressions).float(),
            "record_index": torch.tensor(
                int(self.record_indices[index]),
                dtype=torch.long,
            ),
            "source_id": torch.tensor(int(self.source_ids[index]), dtype=torch.long),
        }


class SeededScGPTCollator:
    """Deterministically seed scGPT's native collator per epoch and batch."""

    def __init__(self, collator, seed: int, fixed: bool) -> None:
        self.collator = collator
        self.seed = int(seed)
        self.fixed = bool(fixed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = 0 if self.fixed else int(epoch)

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        record_indices = torch.stack([example["record_index"] for example in examples])
        source_ids = torch.stack([example["source_id"] for example in examples])
        batch_fingerprint = sum(
            (position + 1) * int(record_index)
            for position, record_index in enumerate(record_indices.tolist())
        )
        batch_seed = (
            self.seed + self.epoch * 1_000_003 + batch_fingerprint * 97
        ) % (2**63 - 1)
        native_examples = [
            {
                "genes": example["genes"],
                "expressions": example["expressions"],
            }
            for example in examples
        ]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(batch_seed)
            batch = self.collator(native_examples)
        batch["record_index"] = record_indices
        batch["source_id"] = source_ids
        return batch


class ScGPTPreadaptRunner:
    """Continue a pretrained scGPT model on all filtered bulk expression data."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.pretrain_cfg = cfg.pretrain
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_dataset = None
        self.validation_dataset = None
        self.train_loader = None
        self.validation_loader = None
        self.train_collator = None
        self.validation_collator = None

        self.data_paths: list[Path] = []
        self.source_names: list[str] = []
        self.source_gene_indices: list[np.ndarray] = []
        self.vocab_gene_ids: list[np.ndarray] = []
        self.data_fingerprint = ""
        self.gene_mapping_fingerprint = ""
        self.resource_fingerprints: dict[str, str] = {}
        self.load_report: dict[str, object] = {}
        self.model_configs: dict[str, object] = {}
        self.vocab = None
        self.paths: dict[str, Path] = {}
        self.amp_dtype: torch.dtype | None = None
        self.backward_scale = 1.0

    def _setup_runtime(self) -> None:
        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        seed_all(int(self.pretrain_cfg.seed) + self.rank)

    def _required_path(self, name: str) -> Path:
        configured = getattr(self.pretrain_cfg, name, None)
        if not configured:
            raise ValueError(f"pretrain.{name} must be set.")
        path = Path(hydra.utils.to_absolute_path(str(configured)))
        if not path.exists():
            raise FileNotFoundError(f"Missing pretrain.{name}: {path}")
        return path

    def _resolve_paths(self) -> None:
        configured_data_paths = list(getattr(self.pretrain_cfg, "data_paths", []) or [])
        if len(configured_data_paths) != 2:
            raise ValueError(
                "pretrain.data_paths must contain pretraining_bulk_RAW.h5ad and "
                "preadapt_bulk_RAW.h5ad."
            )
        self.data_paths = [
            Path(hydra.utils.to_absolute_path(str(path)))
            for path in configured_data_paths
        ]
        missing_data = [str(path) for path in self.data_paths if not path.is_file()]
        if missing_data:
            raise FileNotFoundError("Missing bulk data files: " + ", ".join(missing_data))
        expected_names = ["pretraining_bulk_RAW.h5ad", "preadapt_bulk_RAW.h5ad"]
        observed_names = [path.name for path in self.data_paths]
        if observed_names != expected_names:
            raise ValueError(
                "pretrain.data_paths must list pretraining_bulk_RAW.h5ad followed by "
                f"preadapt_bulk_RAW.h5ad; observed={observed_names}."
            )

        repo_dir = self._required_path("scgpt_repo_dir")
        model_dir = self._required_path("scgpt_model_dir")
        self.paths = {
            "repo_dir": repo_dir,
            "model_dir": model_dir,
            "args": model_dir / str(self.pretrain_cfg.scgpt_args_filename),
            "vocab": model_dir / str(self.pretrain_cfg.scgpt_vocab_filename),
            "checkpoint": model_dir / str(self.pretrain_cfg.scgpt_checkpoint_filename),
            "gene_info": self._required_path("gene_info_path"),
        }
        missing = [
            str(path)
            for name, path in self.paths.items()
            if name not in {"repo_dir", "model_dir"} and not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing scGPT resources: " + ", ".join(missing))

    def _output_dir(self) -> Path:
        return output_root(self.cfg) / str(self.pretrain_cfg.model_name)

    def _output_path(self, suffix: str) -> Path:
        return self._output_dir() / f"{self.pretrain_cfg.model_name}_{suffix}"

    def _import_scgpt(self):
        repo_dir = str(self.paths["repo_dir"])
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from scgpt.data_collator import DataCollator
            from scgpt.model import TransformerModel
            from scgpt.tokenizer import GeneVocab
        except ImportError as exc:
            raise ImportError(
                "scgpt_preadapt requires the pinned scGPT repository and its "
                "torchtext-compatible environment."
            ) from exc
        return DataCollator, TransformerModel, GeneVocab

    def _load_resources(self) -> None:
        _, _, GeneVocab = self._import_scgpt()
        self.vocab = GeneVocab.from_file(self.paths["vocab"])
        required_tokens = ("<pad>", "<cls>", "<eoc>")
        missing_tokens = [token for token in required_tokens if token not in self.vocab]
        if missing_tokens:
            raise ValueError(
                "The frozen scGPT vocabulary is missing required tokens: "
                + ", ".join(missing_tokens)
            )
        self.vocab.set_default_index(self.vocab["<pad>"])
        with self.paths["args"].open("r", encoding="utf-8") as handle:
            self.model_configs = json.load(handle)
        self.resource_fingerprints = {
            name: _sha256_file(self.paths[name])
            for name in ("args", "vocab", "gene_info")
        }

        configured_bins = int(self.pretrain_cfg.n_bins)
        checkpoint_bins = int(self.model_configs.get("n_bins", configured_bins))
        if configured_bins != 51 or checkpoint_bins != configured_bins:
            raise ValueError(
                "Native scGPT DataCollator binning requires n_bins=51 and it must "
                f"match the checkpoint; configured={configured_bins}, checkpoint={checkpoint_bins}."
            )

    def _validate_config(self) -> None:
        if int(self.pretrain_cfg.max_seq_len) != 1200:
            raise ValueError(
                "scgpt_preadapt must use max_seq_len=1200 (1,199 genes plus CLS)."
            )
        if str(self.pretrain_cfg.input_gene_filter) != "nonzero_per_profile":
            raise ValueError(
                "scgpt_preadapt must use input_gene_filter=nonzero_per_profile "
                "to match native scGPT foundation pretraining."
            )
        if not bool(self.pretrain_cfg.do_mvc):
            raise ValueError("scgpt_preadapt requires both MLM and MVC objectives.")
        if float(self.pretrain_cfg.mvc_loss_weight) <= 0:
            raise ValueError("mvc_loss_weight must be positive.")
        if not 0.0 < float(self.pretrain_cfg.mask_ratio) < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1.")
        positive_integer_fields = (
            "epochs",
            "batch_size",
            "grad_acc",
            "prefetch_factor",
            "valid_every",
        )
        invalid = [
            name
            for name in positive_integer_fields
            if int(getattr(self.pretrain_cfg, name)) <= 0
        ]
        if invalid:
            raise ValueError(
                "These scgpt_preadapt settings must be positive integers: "
                + ", ".join(invalid)
            )
        if int(self.pretrain_cfg.num_workers) < 0:
            raise ValueError("num_workers must be non-negative.")
        if int(self.pretrain_cfg.early_stopping_patience) < 0:
            raise ValueError("early_stopping_patience must be non-negative.")
        if float(self.pretrain_cfg.max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be positive.")
        positive_float_fields = (
            "backbone_learning_rate",
            "new_module_learning_rate",
            "adam_eps",
        )
        invalid = [
            name
            for name in positive_float_fields
            if float(getattr(self.pretrain_cfg, name)) <= 0
        ]
        if invalid:
            raise ValueError(
                "These scgpt_preadapt settings must be positive: "
                + ", ".join(invalid)
            )
        if float(self.pretrain_cfg.weight_decay) < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 <= float(self.pretrain_cfg.warmup_fraction) < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if not 0.0 <= float(self.pretrain_cfg.min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1].")
        if float(self.pretrain_cfg.early_stopping_min_delta) < 0:
            raise ValueError("early_stopping_min_delta must be non-negative.")

    def _build_manifest(self) -> pd.DataFrame:
        frames = []
        fingerprint_parts = []
        record_offset = 0
        for file_index, path in enumerate(self.data_paths):
            backed = ad.read_h5ad(path, backed="r")
            try:
                n_obs, n_vars = map(int, backed.shape)
                expected_n_vars = int(
                    self.pretrain_cfg.expected_source_gene_count
                )
                if n_vars != expected_n_vars:
                    raise ValueError(
                        f"{path} contains {n_vars} genes; expected {expected_n_vars}."
                    )
                obs_names = backed.obs_names.astype(str).to_numpy()
                if "bulk_source" in backed.obs:
                    sources = backed.obs["bulk_source"].astype(str).to_numpy()
                else:
                    sources = np.full(n_obs, path.stem, dtype=object)
                var_names = backed.var_names.astype(str).tolist()
            finally:
                backed.file.close()

            records = pd.DataFrame(
                {
                    "record_index": np.arange(
                        record_offset,
                        record_offset + n_obs,
                        dtype=np.int64,
                    ),
                    "file_index": file_index,
                    "row_index": np.arange(n_obs, dtype=np.int64),
                    "sample_id": obs_names,
                    "bulk_source": sources,
                }
            )
            frames.append(records)
            record_offset += n_obs
            fingerprint_parts.extend(
                [
                    str(path.resolve()),
                    str(path.stat().st_size),
                    str(path.stat().st_mtime_ns),
                    str(n_obs),
                    str(n_vars),
                    _sha256_strings(iter(var_names)),
                    _sha256_strings(iter(obs_names.tolist())),
                ]
            )

        manifest = pd.concat(frames, ignore_index=True)
        manifest["split"] = _random_train_validation_split(
            len(manifest),
            float(self.pretrain_cfg.validation_split),
            int(self.pretrain_cfg.seed),
        )
        self.data_fingerprint = _sha256_strings(iter(fingerprint_parts))
        return manifest

    def _prepare_manifest(self) -> pd.DataFrame:
        manifest_path = self._output_path("split_manifest.csv")
        metadata_path = self._output_path("split_manifest.json")
        if self.is_master:
            manifest = self._build_manifest()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = manifest_path.with_suffix(".csv.tmp")
            manifest.to_csv(temporary_path, index=False)
            os.replace(temporary_path, manifest_path)
            metadata = {
                "data_fingerprint": self.data_fingerprint,
                "seed": int(self.pretrain_cfg.seed),
                "validation_split": float(self.pretrain_cfg.validation_split),
                "sample_count": int(len(manifest)),
                "train_count": int((manifest["split"] == "train").sum()),
                "validation_count": int(
                    (manifest["split"] == "validation").sum()
                ),
                "split_strategy": "deterministic_sample_level_random",
                "data_paths": [str(path) for path in self.data_paths],
            }
            self._atomic_json(metadata_path, metadata)
        if self.is_distributed:
            dist.barrier()
        manifest = pd.read_csv(manifest_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.data_fingerprint = str(metadata["data_fingerprint"])
        if int(metadata["sample_count"]) != len(manifest):
            raise ValueError("Split manifest row count does not match its metadata.")
        return manifest

    def _prepare_gene_mapping(self) -> dict[str, int]:
        gene_info = pd.read_csv(self.paths["gene_info"])
        required = {"ensg_id", "gene_symbol"}
        missing_columns = sorted(required.difference(gene_info.columns))
        if missing_columns:
            raise ValueError(
                f"gene_info_path is missing columns: {missing_columns}"
            )
        gene_info = gene_info.dropna(subset=["ensg_id", "gene_symbol"]).copy()
        gene_info["ensg_clean"] = gene_info["ensg_id"].map(_strip_ensembl_version)
        gene_info = gene_info.drop_duplicates("ensg_clean", keep="first")
        symbol_by_ensembl = dict(
            zip(gene_info["ensg_clean"], gene_info["gene_symbol"].astype(str))
        )

        mapping_rows: list[dict[str, object]] = []
        reference_var_names: list[str] | None = None
        self.source_gene_indices = []
        self.vocab_gene_ids = []
        retained_counts = []
        for file_index, path in enumerate(self.data_paths):
            backed = ad.read_h5ad(path, backed="r")
            try:
                var_names = backed.var_names.astype(str).tolist()
            finally:
                backed.file.close()
            if reference_var_names is None:
                reference_var_names = var_names
            elif var_names != reference_var_names:
                raise ValueError(
                    "The two bulk files must have identical gene order before scGPT mapping."
                )

            seen_vocab_ids: set[int] = set()
            source_indices = []
            vocab_ids = []
            for source_index, ensembl_id in enumerate(var_names):
                clean_id = _strip_ensembl_version(ensembl_id)
                symbol = symbol_by_ensembl.get(clean_id)
                status = "retained"
                vocab_id: int | None = None
                if symbol is None:
                    status = "missing_gene_info"
                elif symbol not in self.vocab:
                    status = "missing_from_vocab"
                else:
                    vocab_id = int(self.vocab[symbol])
                    if vocab_id in seen_vocab_ids:
                        status = "duplicate_vocab_token"
                    else:
                        seen_vocab_ids.add(vocab_id)
                        source_indices.append(source_index)
                        vocab_ids.append(vocab_id)
                if file_index == 0:
                    mapping_rows.append(
                        {
                            "source_index": source_index,
                            "ensembl_id": ensembl_id,
                            "ensembl_id_clean": clean_id,
                            "gene_symbol": symbol or "",
                            "vocab_id": "" if vocab_id is None else vocab_id,
                            "status": status,
                        }
                    )
            self.source_gene_indices.append(np.asarray(source_indices, dtype=np.int64))
            self.vocab_gene_ids.append(np.asarray(vocab_ids, dtype=np.int64))
            retained_counts.append(len(source_indices))

        if len(set(retained_counts)) != 1:
            raise ValueError(f"Mapped gene counts differ between bulk files: {retained_counts}")
        retained = retained_counts[0]
        minimum = max(
            int(self.pretrain_cfg.min_mapped_genes),
            int(self.pretrain_cfg.max_seq_len) - 1,
        )
        if retained < minimum:
            raise ValueError(
                f"Only {retained} genes map uniquely to scGPT; at least {minimum} are required."
            )

        self.gene_mapping_fingerprint = _sha256_strings(
            iter(
                "\t".join(str(value) for value in row.values())
                for row in mapping_rows
            )
        )
        mapping = pd.DataFrame(mapping_rows)
        if self.is_master:
            path = self._output_path("gene_mapping.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            mapping.to_csv(path, index=False)
        counts = {
            str(status): int(count)
            for status, count in mapping["status"].value_counts().items()
        }
        counts["source_gene_count"] = int(len(mapping))
        counts["retained_gene_count"] = int(retained)
        return counts

    @staticmethod
    def _checkpoint_state_dict(checkpoint) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("The scGPT checkpoint must contain a PyTorch state dictionary.")
        state_dict = {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            clean_key = str(key)
            for prefix in ("module.", "_orig_mod."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            state_dict[clean_key] = value
        return state_dict

    def _build_model(self) -> None:
        _, TransformerModel, _ = self._import_scgpt()
        configs = self.model_configs
        model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=int(configs["embsize"]),
            nhead=int(configs["nheads"]),
            d_hid=int(configs["d_hid"]),
            nlayers=int(configs["nlayers"]),
            nlayers_cls=int(configs.get("n_layers_cls", 3)),
            n_cls=1,
            vocab=self.vocab,
            dropout=float(configs["dropout"]),
            pad_token=str(configs["pad_token"]),
            pad_value=int(configs["pad_value"]),
            do_mvc=bool(self.pretrain_cfg.do_mvc),
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style=str(configs.get("input_emb_style", "continuous")),
            n_input_bins=int(configs.get("n_bins", 51)),
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            explicit_zero_prob=False,
            use_fast_transformer=bool(self.pretrain_cfg.use_fast_transformer),
            fast_transformer_backend="flash",
            pre_norm=bool(configs.get("pre_norm", False)),
        )

        source_checkpoint = torch.load(self.paths["checkpoint"], map_location="cpu")
        source_state = self._checkpoint_state_dict(source_checkpoint)
        if not bool(getattr(model, "use_fast_transformer", False)):
            source_state = {
                key.replace("Wqkv.", "in_proj_"): value
                for key, value in source_state.items()
            }
        target_state = model.state_dict()
        merged_state = dict(target_state)
        matched = []
        missing = []
        shape_mismatches = []
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

        allowed_new_prefixes = ("mvc_decoder.", "cls_decoder.")
        invalid_missing = [
            key for key in missing if not key.startswith(allowed_new_prefixes)
        ]
        invalid_shapes = [
            item
            for item in shape_mismatches
            if not str(item["key"]).startswith(allowed_new_prefixes)
        ]
        if invalid_missing or invalid_shapes:
            raise ValueError(
                "The pretrained scGPT checkpoint is incompatible with its args/vocab. "
                f"Core missing keys={invalid_missing[:20]}, core shape mismatches={invalid_shapes[:20]}"
            )
        model.load_state_dict(merged_state, strict=True)

        newly_initialized = set(missing)
        newly_initialized.update(str(item["key"]) for item in shape_mismatches)
        for name, parameter in model.named_parameters():
            if name.startswith("cls_decoder."):
                parameter.requires_grad = False

        unexpected = sorted(set(source_state).difference(target_state))
        self.load_report = {
            "checkpoint_tensor_count": len(source_state),
            "model_tensor_count": len(target_state),
            "matched_tensor_count": len(matched),
            "matched_parameter_fraction": (
                sum(target_state[key].numel() for key in matched)
                / sum(value.numel() for value in target_state.values())
            ),
            "newly_initialized_keys": sorted(newly_initialized),
            "shape_mismatches": shape_mismatches,
            "unexpected_checkpoint_keys": unexpected,
            "mvc_pretrained": not any(
                key.startswith("mvc_decoder.") for key in newly_initialized
            ),
        }
        self.model = model.to(self.device)

    def _native_collator(self, fixed: bool):
        DataCollator, _, _ = self._import_scgpt()
        pad_token = str(self.model_configs["pad_token"])
        native = DataCollator(
            do_padding=True,
            pad_token_id=int(self.vocab[pad_token]),
            pad_value=int(self.model_configs["pad_value"]),
            do_mlm=True,
            do_binning=True,
            mlm_probability=float(self.pretrain_cfg.mask_ratio),
            mask_value=int(self.model_configs.get("mask_value", -1)),
            max_length=int(self.pretrain_cfg.max_seq_len),
            sampling=True,
            keep_first_n_tokens=1,
        )
        return SeededScGPTCollator(native, int(self.pretrain_cfg.seed), fixed=fixed)

    def _new_dataset(self, frame: pd.DataFrame) -> ScGPTBulkDataset:
        return ScGPTBulkDataset(
            data_paths=self.data_paths,
            file_indices=frame["file_index"].to_numpy(dtype=np.int16),
            row_indices=frame["row_index"].to_numpy(dtype=np.int64),
            record_indices=frame["record_index"].to_numpy(dtype=np.int64),
            source_ids=frame["source_id"].to_numpy(dtype=np.int16),
            source_gene_indices=self.source_gene_indices,
            vocab_gene_ids=self.vocab_gene_ids,
            cls_token_id=int(self.vocab["<cls>"]),
            cls_value=float(self.model_configs["pad_value"]),
        )

    def _build_loaders(self, manifest: pd.DataFrame) -> None:
        self.source_names = sorted(manifest["bulk_source"].astype(str).unique())
        source_to_id = {source: index for index, source in enumerate(self.source_names)}
        manifest = manifest.copy()
        manifest["source_id"] = manifest["bulk_source"].astype(str).map(source_to_id)
        train_frame = manifest[manifest["split"] == "train"]
        validation_frame = manifest[manifest["split"] == "validation"]
        self.train_dataset = self._new_dataset(train_frame)
        self.validation_dataset = self._new_dataset(validation_frame)
        self.train_collator = self._native_collator(fixed=False)
        self.validation_collator = self._native_collator(fixed=True)

        workers = int(self.pretrain_cfg.num_workers)
        loader_kwargs: dict[str, object] = {
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": False,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = int(self.pretrain_cfg.prefetch_factor)

        train_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=int(self.pretrain_cfg.seed),
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(self.pretrain_cfg.batch_size),
            sampler=train_sampler,
            shuffle=train_sampler is None,
            collate_fn=self.train_collator,
            drop_last=False,
            **loader_kwargs,
        )

        validation_sampler = None
        if self.is_distributed:
            validation_sampler = _RankSliceSampler(
                self.validation_dataset,
                self.rank,
                self.world_size,
            )
        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=int(self.pretrain_cfg.batch_size),
            sampler=validation_sampler,
            shuffle=False,
            collate_fn=self.validation_collator,
            drop_last=False,
            **loader_kwargs,
        )

    def _raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _configure_amp(self) -> None:
        if not bool(self.pretrain_cfg.amp) or self.device.type != "cuda":
            self.amp_dtype = None
        else:
            configured = str(self.pretrain_cfg.amp_dtype).lower()
            if configured == "auto":
                self.amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            elif configured in {"bfloat16", "bf16"}:
                self.amp_dtype = torch.bfloat16
            elif configured in {"float16", "fp16"}:
                self.amp_dtype = torch.float16
            else:
                raise ValueError(f"Unsupported amp_dtype: {configured}")
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_dtype == torch.float16
        )
        self.backward_scale = float(
            int(self.pretrain_cfg.batch_size)
            * (int(self.pretrain_cfg.max_seq_len) - 1)
        )

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _build_optimization(self) -> None:
        new_keys = set(self.load_report["newly_initialized_keys"])
        backbone_parameters = []
        new_parameters = []
        for name, parameter in self._raw_model().named_parameters():
            if not parameter.requires_grad:
                continue
            if name in new_keys:
                new_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        parameter_groups = [
            {
                "params": backbone_parameters,
                "lr": float(self.pretrain_cfg.backbone_learning_rate),
                "group_name": "backbone",
            }
        ]
        if new_parameters:
            parameter_groups.append(
                {
                    "params": new_parameters,
                    "lr": float(self.pretrain_cfg.new_module_learning_rate),
                    "group_name": "new_modules",
                }
            )
        self.optimizer = AdamW(
            parameter_groups,
            weight_decay=float(self.pretrain_cfg.weight_decay),
            eps=float(self.pretrain_cfg.adam_eps),
        )

        updates_per_epoch = math.ceil(
            len(self.train_loader) / int(self.pretrain_cfg.grad_acc)
        )
        total_updates = max(1, updates_per_epoch * int(self.pretrain_cfg.epochs))
        warmup_updates = int(round(total_updates * float(self.pretrain_cfg.warmup_fraction)))
        warmup_updates = min(max(warmup_updates, 1), max(total_updates - 1, 1))
        min_ratio = float(self.pretrain_cfg.min_lr_ratio)

        def lr_multiplier(step: int) -> float:
            if step < warmup_updates:
                return max((step + 1) / warmup_updates, min_ratio)
            progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_multiplier)
        self._configure_amp()

    def _wrap_distributed(self) -> None:
        if not self.is_distributed:
            return
        if self.device.type == "cuda":
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = DDP(
                self.model,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

    def _optimizer_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]

    @staticmethod
    def _is_accumulation_boundary(
        step: int,
        total_steps: int,
        accumulation_steps: int,
    ) -> bool:
        return step % accumulation_steps == 0 or step == total_steps

    def _normalize_accumulated_gradients(self, local_target_count: float) -> bool:
        normalizer = torch.tensor(
            local_target_count,
            dtype=torch.float64,
            device=self.device,
        )
        if self.is_distributed:
            dist.all_reduce(normalizer, op=dist.ReduceOp.SUM)
            normalizer /= self.world_size
        divisor = float(normalizer.item())
        if divisor <= 0.0:
            return False
        if not np.isfinite(divisor):
            raise ValueError(f"Masked-target normalizer is not finite: {divisor}")
        for parameter in self._optimizer_parameters():
            if parameter.grad is not None:
                parameter.grad.div_(divisor)
        return True

    def _forward_loss_parts(
        self,
        model,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        gene_ids = batch["gene"].to(self.device, non_blocking=True)
        targets = batch["expr"].to(self.device, non_blocking=True)
        masked_values = batch["masked_expr"].to(self.device, non_blocking=True)
        source_ids = batch["source_id"].to(self.device, non_blocking=True)
        padding_mask = gene_ids.eq(int(self.vocab[self.model_configs["pad_token"]]))
        output = model(
            gene_ids,
            masked_values,
            src_key_padding_mask=padding_mask,
            MVC=bool(self.pretrain_cfg.do_mvc),
            ECS=False,
        )
        masked_positions = masked_values.eq(
            float(self.model_configs.get("mask_value", -1))
        )
        mask = masked_positions.float()
        counts = mask.sum(dim=1)
        mlm_output = output["mlm_output"].float()
        float_targets = targets.float()
        mlm_per_sample = ((mlm_output - float_targets) * mask).square().sum(dim=1)
        if bool(self.pretrain_cfg.do_mvc):
            mvc_output = output["mvc_output"].float()
            mvc_per_sample = ((mvc_output - float_targets) * mask).square().sum(dim=1)
        else:
            mvc_output = None
            mvc_per_sample = torch.zeros_like(mlm_per_sample)
        total_per_sample = (
            mlm_per_sample
            + float(self.pretrain_cfg.mvc_loss_weight) * mvc_per_sample
        )
        return (
            {
                "total_per_sample": total_per_sample,
                "mlm_per_sample": mlm_per_sample,
                "mvc_per_sample": mvc_per_sample,
                "counts": counts,
                "mlm_output": mlm_output,
                "mvc_output": mvc_output,
                "masked_positions": masked_positions,
            },
            targets,
            source_ids,
        )

    def _new_epoch_stats(self) -> dict[str, torch.Tensor]:
        return {
            "source": torch.zeros(
                (len(self.source_names), 4),
                dtype=torch.float64,
                device=self.device,
            ),
            "bins": torch.zeros(
                (int(self.pretrain_cfg.n_bins), 3),
                dtype=torch.float64,
                device=self.device,
            ),
        }

    def _update_epoch_stats(
        self,
        stats: dict[str, torch.Tensor],
        parts: dict[str, torch.Tensor],
        targets: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> None:
        source_values = torch.stack(
            (
                parts["total_per_sample"],
                parts["mlm_per_sample"],
                parts["mvc_per_sample"],
                parts["counts"],
            ),
            dim=1,
        ).double()
        stats["source"].index_add_(0, source_ids, source_values)

        mask = parts["masked_positions"]
        target_bins = targets[mask].long().clamp(0, int(self.pretrain_cfg.n_bins) - 1)
        if target_bins.numel() == 0:
            return
        mlm_errors = (parts["mlm_output"][mask] - targets[mask]).square().double()
        bin_counts = torch.bincount(
            target_bins,
            minlength=int(self.pretrain_cfg.n_bins),
        ).double()
        stats["bins"][:, 0] += bin_counts
        stats["bins"][:, 1].scatter_add_(0, target_bins, mlm_errors)
        if bool(self.pretrain_cfg.do_mvc):
            mvc_errors = (parts["mvc_output"][mask] - targets[mask]).square().double()
            stats["bins"][:, 2].scatter_add_(0, target_bins, mvc_errors)

    def _reduce_stats(self, stats: dict[str, torch.Tensor]) -> None:
        if self.is_distributed:
            dist.all_reduce(stats["source"], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats["bins"], op=dist.ReduceOp.SUM)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    def _format_stats(
        self,
        *,
        epoch: int,
        split: str,
        stats: dict[str, torch.Tensor],
    ) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
        source_stats = stats["source"].detach().cpu().numpy()
        overall = source_stats.sum(axis=0)
        total_loss = self._safe_divide(float(overall[0]), float(overall[3]))
        mlm_loss = self._safe_divide(float(overall[1]), float(overall[3]))
        mvc_loss = self._safe_divide(float(overall[2]), float(overall[3]))
        if float(overall[3]) <= 0 or not all(
            np.isfinite(value) for value in (total_loss, mlm_loss, mvc_loss)
        ):
            raise ValueError(
                f"Non-finite {split} objective at epoch {epoch}: "
                f"loss={total_loss}, mlm={mlm_loss}, mvc={mvc_loss}, "
                f"masked_targets={float(overall[3])}."
            )
        metrics = {
            f"{split}_loss": total_loss,
            f"{split}_mlm_loss": mlm_loss,
            f"{split}_mvc_loss": mvc_loss,
            f"{split}_masked_target_count": float(overall[3]),
        }
        learning_rates = {
            str(group.get("group_name", index)): float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        } if self.optimizer is not None else {}
        rows = [
            {
                "epoch": epoch,
                "split": split,
                "scope": "all",
                "loss": total_loss,
                "mlm_loss": mlm_loss,
                "mvc_loss": mvc_loss,
                "masked_target_count": int(overall[3]),
                "backbone_learning_rate": learning_rates.get("backbone", float("nan")),
                "new_module_learning_rate": learning_rates.get(
                    "new_modules", float("nan")
                ),
            }
        ]
        for source_index, source_name in enumerate(self.source_names):
            values = source_stats[source_index]
            if float(values[3]) <= 0:
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "scope": source_name,
                    "loss": self._safe_divide(float(values[0]), float(values[3])),
                    "mlm_loss": self._safe_divide(float(values[1]), float(values[3])),
                    "mvc_loss": self._safe_divide(float(values[2]), float(values[3])),
                    "masked_target_count": int(values[3]),
                    "backbone_learning_rate": learning_rates.get(
                        "backbone", float("nan")
                    ),
                    "new_module_learning_rate": learning_rates.get(
                        "new_modules", float("nan")
                    ),
                }
            )

        bin_values = stats["bins"].detach().cpu().numpy()
        bin_rows = []
        for bin_id, (count, mlm_sum, mvc_sum) in enumerate(bin_values):
            bin_rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "bin_id": bin_id,
                    "target_count": int(count),
                    "mlm_loss": self._safe_divide(float(mlm_sum), float(count)),
                    "mvc_loss": self._safe_divide(float(mvc_sum), float(count)),
                }
            )
        return metrics, rows, bin_rows

    def _train_epoch(
        self,
        epoch: int,
    ) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
        epoch_seed = (
            int(self.pretrain_cfg.seed)
            + self.rank * 1_000_003
            + int(epoch) * 10_000_019
        ) % (2**63 - 1)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        self.model.train()
        self.train_collator.set_epoch(epoch)
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        accumulation_steps = int(self.pretrain_cfg.grad_acc)
        total_steps = len(self.train_loader)
        target_normalizer = 0.0
        stats = self._new_epoch_stats()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()

        for step, batch in enumerate(self.train_loader, start=1):
            should_step = self._is_accumulation_boundary(
                step,
                total_steps,
                accumulation_steps,
            )
            sync_context = (
                self.model.no_sync()
                if isinstance(self.model, DDP) and not should_step
                else nullcontext()
            )
            with sync_context, self._autocast():
                parts, targets, source_ids = self._forward_loss_parts(self.model, batch)
                loss_sum = parts["total_per_sample"].sum() / self.backward_scale
            self.scaler.scale(loss_sum).backward()
            target_normalizer += (
                float(parts["counts"].sum().detach().item()) / self.backward_scale
            )

            if should_step:
                self.scaler.unscale_(self.optimizer)
                has_targets = self._normalize_accumulated_gradients(target_normalizer)
                if has_targets:
                    torch.nn.utils.clip_grad_norm_(
                        self._optimizer_parameters(),
                        float(self.pretrain_cfg.max_grad_norm),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                target_normalizer = 0.0

            with torch.no_grad():
                self._update_epoch_stats(stats, parts, targets, source_ids)
            if (
                self.is_master
                and int(self.pretrain_cfg.log_every_batches) > 0
                and step % int(self.pretrain_cfg.log_every_batches) == 0
            ):
                log.info(
                    "scgpt_preadapt | epoch=%d batch=%d/%d elapsed=%.1fs",
                    epoch,
                    step,
                    total_steps,
                    time.monotonic() - started,
                )

        self._reduce_stats(stats)
        return self._format_stats(epoch=epoch, split="train", stats=stats)

    def _validate(
        self,
        epoch: int,
    ) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
        model = self._raw_model()
        model.eval()
        self.validation_collator.set_epoch(0)
        stats = self._new_epoch_stats()
        with torch.no_grad():
            for batch in self.validation_loader:
                with self._autocast():
                    parts, targets, source_ids = self._forward_loss_parts(model, batch)
                self._update_epoch_stats(stats, parts, targets, source_ids)
        self._reduce_stats(stats)
        return self._format_stats(epoch=epoch, split="validation", stats=stats)

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        fieldnames = list(rows[0])
        extras = sorted({key for row in rows for key in row}.difference(fieldnames))
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extras])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_torch_save(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def _copy_model_assets(self) -> None:
        out_dir = self._output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.paths["args"], out_dir / "args.json")
        shutil.copy2(self.paths["vocab"], out_dir / "vocab.json")
        config_path = self._output_path("config.yaml")
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        os.replace(temporary, config_path)

    def _save_native_model(self, filename: str) -> Path:
        path = self._output_dir() / filename
        self._atomic_torch_save(path, self._raw_model().state_dict())
        return path

    def _resume_signature(self) -> dict[str, object]:
        return {
            "data_fingerprint": self.data_fingerprint,
            "gene_mapping_fingerprint": self.gene_mapping_fingerprint,
            "resource_fingerprints": dict(self.resource_fingerprints),
            "scgpt_git_commit": self._scgpt_git_commit(),
            "seed": int(self.pretrain_cfg.seed),
            "validation_split": float(self.pretrain_cfg.validation_split),
            "max_seq_len": int(self.pretrain_cfg.max_seq_len),
            "input_gene_filter": str(self.pretrain_cfg.input_gene_filter),
            "expected_source_gene_count": int(
                self.pretrain_cfg.expected_source_gene_count
            ),
            "n_bins": int(self.pretrain_cfg.n_bins),
            "mask_ratio": float(self.pretrain_cfg.mask_ratio),
            "do_mvc": bool(self.pretrain_cfg.do_mvc),
            "mvc_loss_weight": float(self.pretrain_cfg.mvc_loss_weight),
            "use_fast_transformer": bool(self.pretrain_cfg.use_fast_transformer),
            "epochs": int(self.pretrain_cfg.epochs),
            "batch_size": int(self.pretrain_cfg.batch_size),
            "grad_acc": int(self.pretrain_cfg.grad_acc),
            "max_grad_norm": float(self.pretrain_cfg.max_grad_norm),
            "amp": bool(self.pretrain_cfg.amp),
            "amp_dtype": str(self.pretrain_cfg.amp_dtype),
            "backbone_learning_rate": float(
                self.pretrain_cfg.backbone_learning_rate
            ),
            "new_module_learning_rate": float(
                self.pretrain_cfg.new_module_learning_rate
            ),
            "weight_decay": float(self.pretrain_cfg.weight_decay),
            "adam_eps": float(self.pretrain_cfg.adam_eps),
            "warmup_fraction": float(self.pretrain_cfg.warmup_fraction),
            "min_lr_ratio": float(self.pretrain_cfg.min_lr_ratio),
            "valid_every": int(self.pretrain_cfg.valid_every),
            "early_stopping_patience": int(
                self.pretrain_cfg.early_stopping_patience
            ),
            "early_stopping_min_delta": float(
                self.pretrain_cfg.early_stopping_min_delta
            ),
            "world_size": int(self.world_size),
        }

    def _save_training_state(
        self,
        *,
        epoch: int,
        best_validation_loss: float,
        stale_validations: int,
        epoch_rows: list[dict[str, object]],
        bin_rows: list[dict[str, object]],
    ) -> Path:
        path = self._output_dir() / "training_state.pt"
        state = {
            "task": "pretrain.scgpt_preadapt",
            "model_name": str(self.pretrain_cfg.model_name),
            "epoch": int(epoch),
            "model_state_dict": self._raw_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_validation_loss": float(best_validation_loss),
            "stale_validations": int(stale_validations),
            "epoch_metric_rows": epoch_rows,
            "bin_metric_rows": bin_rows,
            "resume_signature": self._resume_signature(),
        }
        self._atomic_torch_save(path, state)
        return path

    def _resume_training_state(self) -> tuple[int, float, int, list, list]:
        configured = getattr(self.pretrain_cfg, "resume_state_path", None)
        if not configured:
            return 1, float("inf"), 0, [], []
        path = Path(hydra.utils.to_absolute_path(str(configured)))
        if not path.is_file():
            raise FileNotFoundError(f"Missing resume_state_path: {path}")
        state = torch.load(path, map_location=self.device)
        if state.get("task") != "pretrain.scgpt_preadapt":
            raise ValueError(f"Not an scgpt_preadapt training state: {path}")
        expected = self._resume_signature()
        observed = state.get("resume_signature")
        if not isinstance(observed, dict):
            raise ValueError(
                "Resume state has no scgpt_preadapt resume_signature; start a fresh "
                "run with the new implementation."
            )
        mismatches = {
            key: {"resume": observed.get(key), "current": value}
            for key, value in expected.items()
            if observed.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Resume state training configuration mismatch: {mismatches}")
        self._raw_model().load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state.get("scaler_state_dict", {}))
        return (
            int(state["epoch"]) + 1,
            float(state["best_validation_loss"]),
            int(state.get("stale_validations", 0)),
            list(state.get("epoch_metric_rows", [])),
            list(state.get("bin_metric_rows", [])),
        )

    def _audit_data(self, manifest: pd.DataFrame) -> dict[str, object]:
        rows_per_file = max(1, int(self.pretrain_cfg.audit_sample_rows))
        file_reports = []
        for file_index, path in enumerate(self.data_paths):
            file_manifest = manifest[manifest["file_index"] == file_index]
            sample_count = min(rows_per_file, len(file_manifest))
            rng = np.random.default_rng(int(self.pretrain_cfg.seed) + file_index)
            selected = rng.choice(len(file_manifest), size=sample_count, replace=False)
            row_indices = file_manifest.iloc[selected]["row_index"].to_numpy(dtype=np.int64)
            backed = ad.read_h5ad(path, backed="r")
            minimum = float("inf")
            maximum = float("-inf")
            nonzero = 0
            values_seen = 0
            try:
                for row_index in row_indices:
                    row = ScGPTBulkDataset._dense_row(backed.X[int(row_index)])[
                        self.source_gene_indices[file_index]
                    ]
                    if row.size:
                        minimum = min(minimum, float(np.min(row)))
                        maximum = max(maximum, float(np.max(row)))
                        nonzero += int(np.count_nonzero(row))
                        values_seen += int(row.size)
            finally:
                backed.file.close()
            file_reports.append(
                {
                    "path": str(path),
                    "samples": int(len(file_manifest)),
                    "audited_rows": sample_count,
                    "minimum": minimum,
                    "maximum": maximum,
                    "nonzero_fraction": self._safe_divide(nonzero, values_seen),
                }
            )
            if not np.isfinite(minimum) or not np.isfinite(maximum):
                raise ValueError(f"Bulk matrix audit found non-finite values in {path}.")
            if minimum < 0:
                raise ValueError(
                    f"Bulk matrix audit found negative expression values in {path}: {minimum}"
                )
        return {
            "files": file_reports,
            "total_samples": int(len(manifest)),
            "train_samples": int((manifest["split"] == "train").sum()),
            "validation_samples": int((manifest["split"] == "validation").sum()),
            "source_counts": {
                str(source): int(count)
                for source, count in manifest["bulk_source"].value_counts().items()
            },
        }

    def _scgpt_git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.paths["repo_dir"]), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    def _write_audit(
        self,
        manifest: pd.DataFrame,
        gene_counts: dict[str, int],
    ) -> Path:
        report = {
            "task": "pretrain.scgpt_preadapt",
            "model_name": str(self.pretrain_cfg.model_name),
            "scgpt_git_commit": self._scgpt_git_commit(),
            "checkpoint_path": str(self.paths["checkpoint"]),
            "checkpoint_sha256": _sha256_file(self.paths["checkpoint"]),
            "vocab_size": len(self.vocab),
            "model_parameter_count": sum(
                parameter.numel() for parameter in self._raw_model().parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self._raw_model().parameters()
                if parameter.requires_grad
            ),
            "max_seq_len": int(self.pretrain_cfg.max_seq_len),
            "input_gene_filter": str(self.pretrain_cfg.input_gene_filter),
            "source_gene_count": int(
                self.pretrain_cfg.expected_source_gene_count
            ),
            "mask_ratio": float(self.pretrain_cfg.mask_ratio),
            "epochs": int(self.pretrain_cfg.epochs),
            "data_fingerprint": self.data_fingerprint,
            "gene_mapping_fingerprint": self.gene_mapping_fingerprint,
            "resource_fingerprints": dict(self.resource_fingerprints),
            "gene_mapping": gene_counts,
            "checkpoint_loading": self.load_report,
            "data": self._audit_data(manifest),
        }
        path = self._output_path("audit.json")
        self._atomic_json(path, report)
        return path

    def _training_metadata(
        self,
        manifest: pd.DataFrame,
        gene_counts: dict[str, int],
    ) -> dict[str, object]:
        return {
            "task": "pretrain.scgpt_preadapt",
            "model_name": str(self.pretrain_cfg.model_name),
            "source_checkpoint": str(self.paths["checkpoint"]),
            "scgpt_git_commit": self._scgpt_git_commit(),
            "data_paths": [str(path) for path in self.data_paths],
            "data_fingerprint": self.data_fingerprint,
            "gene_mapping_fingerprint": self.gene_mapping_fingerprint,
            "resource_fingerprints": dict(self.resource_fingerprints),
            "split_strategy": "deterministic_sample_level_random",
            "sample_count": int(len(manifest)),
            "train_sample_count": int((manifest["split"] == "train").sum()),
            "validation_sample_count": int(
                (manifest["split"] == "validation").sum()
            ),
            "gene_mapping": gene_counts,
            "checkpoint_loading": self.load_report,
            "max_seq_len": int(self.pretrain_cfg.max_seq_len),
            "input_gene_filter": str(self.pretrain_cfg.input_gene_filter),
            "maximum_selected_gene_count": int(self.pretrain_cfg.max_seq_len) - 1,
            "do_mvc": bool(self.pretrain_cfg.do_mvc),
            "mvc_loss_weight": float(self.pretrain_cfg.mvc_loss_weight),
            "epochs": int(self.pretrain_cfg.epochs),
            "effective_global_batch_size": (
                int(self.pretrain_cfg.batch_size)
                * self.world_size
                * int(self.pretrain_cfg.grad_acc)
            ),
            "world_size": self.world_size,
            "amp_dtype": str(self.amp_dtype).replace("torch.", "")
            if self.amp_dtype is not None
            else "disabled",
        }

    def run(self) -> dict[str, object]:
        try:
            self._setup_runtime()
            self._resolve_paths()
            self._load_resources()
            self._validate_config()
            manifest = self._prepare_manifest()
            gene_counts = self._prepare_gene_mapping()
            self._build_model()

            if self.is_master:
                self._copy_model_assets()
                audit_path = self._write_audit(manifest, gene_counts)
                log.info("Wrote scgpt_preadapt audit to %s", audit_path)
            if self.is_distributed:
                dist.barrier()
            if bool(self.pretrain_cfg.audit_only):
                return {
                    "audit": str(self._output_path("audit.json")),
                    "samples": int(len(manifest)),
                    "mapped_genes": int(gene_counts["retained_gene_count"]),
                }

            self._build_loaders(manifest)
            self._wrap_distributed()
            self._build_optimization()
            start_epoch, best_loss, stale_validations, epoch_rows, bin_rows = (
                self._resume_training_state()
            )

            if self.is_master:
                metadata = self._training_metadata(manifest, gene_counts)
                self._atomic_json(self._output_path("run_metadata.json"), metadata)
                log.info(
                    "scgpt_preadapt | samples=%d train=%d validation=%d mapped_genes=%d "
                    "effective_batch=%d device=%s",
                    len(manifest),
                    len(self.train_dataset),
                    len(self.validation_dataset),
                    gene_counts["retained_gene_count"],
                    metadata["effective_global_batch_size"],
                    self.device,
                )

            if start_epoch == 1 and not epoch_rows:
                validation_metrics, validation_rows, validation_bin_rows = self._validate(0)
                epoch_rows.extend(validation_rows)
                bin_rows.extend(validation_bin_rows)
                best_loss = float(validation_metrics["validation_loss"])
                if self.is_master:
                    self._save_native_model("last_model.pt")
                    self._save_training_state(
                        epoch=0,
                        best_validation_loss=best_loss,
                        stale_validations=0,
                        epoch_rows=epoch_rows,
                        bin_rows=bin_rows,
                    )
                    self._write_csv(self._output_path("epoch_metrics.csv"), epoch_rows)
                    self._write_csv(self._output_path("bin_metrics.csv"), bin_rows)
                    log.info(
                        "scgpt_preadapt | epoch=0 validation_loss=%.6f mlm=%.6f mvc=%.6f",
                        validation_metrics["validation_loss"],
                        validation_metrics["validation_mlm_loss"],
                        validation_metrics["validation_mvc_loss"],
                    )

            completed_epoch = start_epoch - 1
            early_stopping_patience = int(
                self.pretrain_cfg.early_stopping_patience
            )
            early_stopping_enabled = early_stopping_patience > 0
            stop_early = (
                early_stopping_enabled
                and stale_validations >= early_stopping_patience
            )
            for epoch in range(start_epoch, int(self.pretrain_cfg.epochs) + 1):
                if stop_early:
                    break
                train_metrics, train_rows, train_bin_rows = self._train_epoch(epoch)
                epoch_rows.extend(train_rows)
                bin_rows.extend(train_bin_rows)
                completed_epoch = epoch
                validation_metrics = None
                if epoch % int(self.pretrain_cfg.valid_every) == 0:
                    validation_metrics, validation_rows, validation_bin_rows = self._validate(epoch)
                    epoch_rows.extend(validation_rows)
                    bin_rows.extend(validation_bin_rows)
                    current_loss = float(validation_metrics["validation_loss"])
                    if current_loss < best_loss - float(
                        self.pretrain_cfg.early_stopping_min_delta
                    ):
                        best_loss = current_loss
                        stale_validations = 0
                    else:
                        stale_validations += 1
                    stop_early = (
                        early_stopping_enabled
                        and stale_validations >= early_stopping_patience
                    )

                if self.is_master:
                    self._save_native_model("last_model.pt")
                    self._save_training_state(
                        epoch=epoch,
                        best_validation_loss=best_loss,
                        stale_validations=stale_validations,
                        epoch_rows=epoch_rows,
                        bin_rows=bin_rows,
                    )
                    self._write_csv(self._output_path("epoch_metrics.csv"), epoch_rows)
                    self._write_csv(self._output_path("bin_metrics.csv"), bin_rows)
                    log.info(
                        "scgpt_preadapt | epoch=%d train_loss=%.6f validation_loss=%s "
                        "best=%.6f stale=%d",
                        epoch,
                        train_metrics["train_loss"],
                        (
                            f"{validation_metrics['validation_loss']:.6f}"
                            if validation_metrics is not None
                            else "not_run"
                        ),
                        best_loss,
                        stale_validations,
                    )
                if self.is_distributed:
                    stop_tensor = torch.tensor(
                        int(stop_early),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    dist.broadcast(stop_tensor, src=0)
                    stop_early = bool(stop_tensor.item())
                    dist.barrier()
                if stop_early:
                    break

            return {
                "last_model": str(self._output_dir() / "last_model.pt"),
                "training_state": str(self._output_dir() / "training_state.pt"),
                "completed_epoch": completed_epoch,
                "best_validation_loss": best_loss,
                "stopped_early": stop_early,
            }
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
