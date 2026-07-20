from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils import CosineAnnealingWarmupRestarts, seed_all

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


class ScGPTBulkMaskedDataset(Dataset):
    """Backed multi-h5ad dataset for scGPT masked-value preadaptation."""

    def __init__(
        self,
        *,
        data_paths: list[str],
        path_indices: np.ndarray,
        row_indices: np.ndarray,
        n_obs_by_path: list[int],
        source_gene_indices: list[np.ndarray],
        vocab_gene_ids: list[np.ndarray],
        cls_token_id: int,
        cls_value: float,
        selected_gene_count: int,
        io_block_size: int,
        io_cache_blocks: int,
        seed: int,
    ) -> None:
        self.data_paths = [str(path) for path in data_paths]
        self.path_indices = np.asarray(path_indices, dtype=np.int16)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.n_obs_by_path = [int(n_obs) for n_obs in n_obs_by_path]
        self.source_gene_indices = [
            np.asarray(indices, dtype=np.int64) for indices in source_gene_indices
        ]
        self.vocab_gene_ids = [np.asarray(ids, dtype=np.int64) for ids in vocab_gene_ids]
        self.cls_token_id = int(cls_token_id)
        self.cls_value = float(cls_value)
        self.selected_gene_count = int(selected_gene_count)
        self.io_block_size = max(1, int(io_block_size))
        self.io_cache_blocks = max(1, int(io_cache_blocks))
        self.seed = int(seed)
        self.epoch = 0
        self._adatas: dict[int, ad.AnnData] = {}
        self._block_cache: dict[tuple[int, int], np.ndarray] = {}
        self._block_cache_order: list[tuple[int, int]] = []

        if self.path_indices.shape[0] != self.row_indices.shape[0]:
            raise ValueError("path_indices and row_indices must have equal length.")

    def __len__(self) -> int:
        return int(self.row_indices.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_adatas"] = {}
        state["_block_cache"] = {}
        state["_block_cache_order"] = []
        return state

    def __del__(self):
        for adata in getattr(self, "_adatas", {}).values():
            file_obj = getattr(adata, "file", None)
            if file_obj is not None:
                file_obj.close()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._block_cache.clear()
        self._block_cache_order.clear()

    def _ensure_open(self, path_idx: int) -> ad.AnnData:
        if path_idx not in self._adatas:
            self._adatas[path_idx] = ad.read_h5ad(self.data_paths[path_idx], backed="r")
        return self._adatas[path_idx]

    @staticmethod
    def _to_dense_row(row) -> np.ndarray:
        if sparse.issparse(row):
            return np.asarray(row.toarray()).ravel()
        return np.asarray(row).ravel()

    @staticmethod
    def _to_dense_block(block) -> np.ndarray:
        if sparse.issparse(block):
            return np.asarray(block.toarray())
        return np.asarray(block)

    def _read_block(self, path_idx: int, row_idx: int) -> tuple[np.ndarray, int]:
        block_start = (row_idx // self.io_block_size) * self.io_block_size
        block_stop = min(block_start + self.io_block_size, self.n_obs_by_path[path_idx])
        cache_key = (path_idx, block_start)
        if cache_key not in self._block_cache:
            adata = self._ensure_open(path_idx)
            source_indices = self.source_gene_indices[path_idx]
            block = self._to_dense_block(adata.X[block_start:block_stop, source_indices]).astype(
                np.float32,
                copy=False,
            )
            self._block_cache[cache_key] = block
            self._block_cache_order.append(cache_key)
            while len(self._block_cache_order) > self.io_cache_blocks:
                old_key = self._block_cache_order.pop(0)
                self._block_cache.pop(old_key, None)
        return self._block_cache[cache_key], block_start

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path_idx = int(self.path_indices[index])
        row_idx = int(self.row_indices[index])
        vocab_ids = self.vocab_gene_ids[path_idx]

        n_genes = int(vocab_ids.shape[0])
        if self.selected_gene_count < n_genes:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            selected = rng.choice(n_genes, size=self.selected_gene_count, replace=False)
            selected.sort()
        else:
            selected = np.arange(n_genes, dtype=np.int64)

        # Read a row block from backed h5ad, then sample genes from memory.
        # Per-sample backed fancy indexing is prohibitively slow for HDF5.
        block, block_start = self._read_block(path_idx, row_idx)
        row = block[row_idx - block_start, selected].astype(np.float32, copy=False)

        genes = np.concatenate(
            (
                np.asarray([self.cls_token_id], dtype=np.int64),
                vocab_ids[selected].astype(np.int64, copy=False),
            )
        )
        values = np.concatenate(
            (
                np.asarray([self.cls_value], dtype=np.float32),
                row.astype(np.float32, copy=False),
            )
        )
        return {
            "genes": torch.as_tensor(genes, dtype=torch.long),
            "expressions": torch.as_tensor(values, dtype=torch.float32),
        }


class ScGPTPreTokenizedDataset(Dataset):
    """Memmap-backed pre-tokenized scGPT dataset.

    The cache stores gene ids and already-binned expression values. Runtime
    masking is still done in the collator, so each epoch still sees fresh masks.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        split: str,
        n_samples: int,
        seq_len: int,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.n_samples = int(n_samples)
        self.seq_len = int(seq_len)
        self.epoch = 0
        self._genes: np.memmap | None = None
        self._expr: np.memmap | None = None
        self._opened_epoch: int | None = None

    def __len__(self) -> int:
        return self.n_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_genes"] = None
        state["_expr"] = None
        state["_opened_epoch"] = None
        return state

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self._opened_epoch != self.epoch:
            self._genes = None
            self._expr = None
            self._opened_epoch = None

    def _prefix(self) -> str:
        return f"{self.split}_epoch{self.epoch:03d}"

    def _ensure_open(self) -> None:
        if self._opened_epoch == self.epoch and self._genes is not None and self._expr is not None:
            return
        prefix = self._prefix()
        genes_path = self.cache_dir / f"{prefix}_genes.int32.dat"
        expr_path = self.cache_dir / f"{prefix}_expr.int16.dat"
        if not genes_path.exists() or not expr_path.exists():
            raise FileNotFoundError(
                f"Missing pre-tokenized scGPT cache for {self.split} epoch {self.epoch}: "
                f"{genes_path}, {expr_path}"
            )
        self._genes = np.memmap(genes_path, mode="r", dtype=np.int32, shape=(self.n_samples, self.seq_len))
        self._expr = np.memmap(expr_path, mode="r", dtype=np.int16, shape=(self.n_samples, self.seq_len))
        self._opened_epoch = self.epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._ensure_open()
        assert self._genes is not None and self._expr is not None
        genes = np.asarray(self._genes[index], dtype=np.int64)
        expr = np.asarray(self._expr[index], dtype=np.float32)
        return {
            "genes": torch.as_tensor(genes, dtype=torch.long),
            "expressions": torch.as_tensor(expr, dtype=torch.float32),
        }


class PreBinnedMaskCollator:
    """Collator for pre-tokenized and pre-binned fixed-length scGPT examples."""

    def __init__(
        self,
        *,
        pad_value: int,
        mask_value: int,
        mlm_probability: float,
        keep_first_n_tokens: int = 1,
    ) -> None:
        self.pad_value = int(pad_value)
        self.mask_value = int(mask_value)
        self.mlm_probability = float(mlm_probability)
        self.keep_first_n_tokens = int(keep_first_n_tokens)
        if self.mlm_probability <= 0 or self.mlm_probability >= 1:
            raise ValueError("mlm_probability must be between 0 and 1.")

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        genes = torch.stack([example["genes"] for example in examples], dim=0)
        expressions = torch.stack([example["expressions"] for example in examples], dim=0)
        probability_matrix = torch.full(expressions.shape, self.mlm_probability)
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0
        mask = torch.bernoulli(probability_matrix).bool()
        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        return {
            "gene": genes,
            "expr": expressions,
            "masked_expr": masked_expressions,
        }


class ScGPTPreadaptRunner:
    """Continue scGPT masked-value training on unsupervised bulk data."""

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
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.vocab = None
        self.model_configs: dict | None = None
        self.scaler = None
        self._pretokenize_context: dict[str, object] | None = None

    def _setup_runtime(self) -> None:
        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
            dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        seed_all(int(getattr(self.pretrain_cfg, "seed", 2021)) + self.rank)

    def _required_path(self, attr: str) -> Path:
        value = getattr(self.pretrain_cfg, attr, None)
        if not value:
            raise ValueError(f"pretrain.{attr} must be set.")
        path = Path(hydra.utils.to_absolute_path(str(value)))
        if not path.exists():
            raise FileNotFoundError(f"Missing path for pretrain.{attr}: {path}")
        return path

    def _data_paths(self) -> list[Path]:
        values = list(getattr(self.pretrain_cfg, "data_paths", []) or [])
        if not values:
            raise ValueError("pretrain.data_paths must contain at least one bulk h5ad path.")
        paths = [Path(hydra.utils.to_absolute_path(str(value))) for value in values]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing bulk h5ad files: " + ", ".join(missing))
        return paths

    def _scgpt_paths(self) -> dict[str, Path]:
        repo_dir = self._required_path("scgpt_repo_dir")
        model_dir = self._required_path("scgpt_model_dir")
        paths = {
            "repo_dir": repo_dir,
            "model_dir": model_dir,
            "args": model_dir / str(getattr(self.pretrain_cfg, "scgpt_args_filename", "args.json")),
            "vocab": model_dir / str(getattr(self.pretrain_cfg, "scgpt_vocab_filename", "vocab.json")),
            "checkpoint": model_dir
            / str(getattr(self.pretrain_cfg, "scgpt_checkpoint_filename", "best_model.pt")),
            "gene_info": self._required_path("gene_info_path"),
        }
        missing = [str(path) for key, path in paths.items() if key != "repo_dir" and not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required scGPT files: " + ", ".join(missing))
        return paths

    def _import_scgpt(self, repo_dir: Path):
        repo_dir_str = str(repo_dir)
        if repo_dir_str not in sys.path:
            sys.path.insert(0, repo_dir_str)
        from scgpt.data_collator import DataCollator
        from scgpt.model import TransformerModel
        from scgpt.preprocess import binning
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        return DataCollator, TransformerModel, GeneVocab, load_pretrained, binning

    @staticmethod
    def _strip_ensembl_version(values: pd.Series | np.ndarray | list[str]) -> pd.Series:
        return pd.Series(values, dtype="string").str.replace(r"\.\d+$", "", regex=True)

    def _gene_info_mapping(self, gene_info_path: Path) -> dict[str, str]:
        gene_info = pd.read_csv(gene_info_path)
        required = {"ensg_id", "gene_symbol"}
        missing = sorted(required.difference(gene_info.columns))
        if missing:
            raise ValueError(f"gene_info_path is missing required columns: {missing}")
        gene_info = gene_info.dropna(subset=["ensg_id", "gene_symbol"]).copy()
        gene_info["ensg_id_clean"] = self._strip_ensembl_version(gene_info["ensg_id"]).to_numpy()
        gene_info = gene_info.drop_duplicates("ensg_id_clean", keep="first")
        return dict(zip(gene_info["ensg_id_clean"], gene_info["gene_symbol"].astype(str)))

    def _map_genes_for_path(
        self,
        data_path: Path,
        ensg_to_symbol: dict[str, str],
        vocab,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        backed = ad.read_h5ad(data_path, backed="r")
        try:
            var = backed.var.copy()
            n_vars = int(backed.n_vars)
            if "gene_symbol" in var:
                symbols = var["gene_symbol"].astype(str).to_numpy()
            else:
                if "ensg_id" in var:
                    ensg = self._strip_ensembl_version(var["ensg_id"]).to_numpy()
                else:
                    ensg = self._strip_ensembl_version(backed.var_names).to_numpy()
                symbols = np.asarray([ensg_to_symbol.get(str(gene), "") for gene in ensg], dtype=object)
        finally:
            backed.file.close()

        in_vocab = np.asarray([bool(symbol) and symbol in vocab for symbol in symbols], dtype=bool)
        source_indices = np.flatnonzero(in_vocab).astype(np.int64)
        vocab_ids = np.asarray([int(vocab[str(symbols[idx])]) for idx in source_indices], dtype=np.int64)
        if source_indices.size < int(getattr(self.pretrain_cfg, "min_mapped_genes", 200)):
            raise ValueError(
                f"{data_path} maps only {source_indices.size}/{n_vars} genes to scGPT vocab."
            )
        return source_indices, vocab_ids, int(source_indices.size), n_vars

    def _build_model(self, paths: dict[str, Path]) -> None:
        _, TransformerModel, GeneVocab, load_pretrained, _ = self._import_scgpt(paths["repo_dir"])
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
            nlayers_cls=int(model_configs["n_layers_cls"]),
            n_cls=1,
            vocab=vocab,
            dropout=float(model_configs["dropout"]),
            pad_token=str(model_configs["pad_token"]),
            pad_value=int(model_configs["pad_value"]),
            do_mvc=bool(getattr(self.pretrain_cfg, "do_mvc", False)),
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            input_emb_style=str(model_configs.get("input_emb_style", "continuous")),
            n_input_bins=int(model_configs.get("n_bins", 51)),
            cell_emb_style="cls",
            use_fast_transformer=bool(getattr(self.pretrain_cfg, "use_fast_transformer", False)),
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        log.info("Loading pretrained scGPT checkpoint from %s", paths["checkpoint"])
        load_pretrained(model, torch.load(paths["checkpoint"], map_location="cpu"), verbose=False)
        if bool(getattr(self.pretrain_cfg, "freeze_unused_cls_decoder", True)) and hasattr(model, "cls_decoder"):
            for param in model.cls_decoder.parameters():
                param.requires_grad_(False)
            if self.is_master:
                log.info("Froze unused scGPT cls_decoder parameters for unsupervised preadaptation.")
        model = model.to(self.device)

        if self.is_distributed:
            find_unused = bool(getattr(self.pretrain_cfg, "ddp_find_unused_parameters", False))
            if self.device.type == "cuda":
                model = DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=find_unused,
                )
            else:
                model = DDP(model, find_unused_parameters=find_unused)

        self.model = model
        self.vocab = vocab
        self.model_configs = model_configs

    def _split_rows(self, n_obs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(n_obs, dtype=np.int64)
        val_fraction = float(getattr(self.pretrain_cfg, "validation_split", 0.05))
        if n_obs < 2 or val_fraction <= 0:
            return indices, np.asarray([], dtype=np.int64)
        train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed)
        return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)

    def _pretokenized_cache_dir(self) -> Path:
        cache_dir = getattr(self.pretrain_cfg, "pretokenized_cache_dir", None)
        if cache_dir:
            return Path(hydra.utils.to_absolute_path(str(cache_dir)))
        return self._output_dir() / "pretokenized_cache"

    def _cache_prefix(self, split: str, epoch: int) -> str:
        return f"{split}_epoch{int(epoch):03d}"

    def _cache_paths(self, split: str, epoch: int) -> dict[str, Path]:
        cache_dir = self._pretokenized_cache_dir()
        prefix = self._cache_prefix(split, epoch)
        return {
            "dir": cache_dir,
            "genes": cache_dir / f"{prefix}_genes.int32.dat",
            "expr": cache_dir / f"{prefix}_expr.int16.dat",
            "metadata": cache_dir / f"{prefix}_metadata.json",
        }

    def _cache_metadata(
        self,
        *,
        split: str,
        epoch: int,
        n_samples: int,
        seq_len: int,
        data_path_strs: list[str],
        seed: int,
    ) -> dict[str, object]:
        return {
            "split": str(split),
            "epoch": int(epoch),
            "n_samples": int(n_samples),
            "seq_len": int(seq_len),
            "selected_gene_count": int(getattr(self.pretrain_cfg, "selected_gene_count", 1199)),
            "n_bins": int(self.model_configs.get("n_bins", 51)),
            "seed": int(seed),
            "mask_prob": float(getattr(self.pretrain_cfg, "mask_prob", 0.15)),
            "data_paths": [str(path) for path in data_path_strs],
        }

    @staticmethod
    def _cache_is_ready(paths: dict[str, Path], expected_metadata: dict[str, object]) -> bool:
        if not paths["genes"].exists() or not paths["expr"].exists() or not paths["metadata"].exists():
            return False
        try:
            current = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except Exception:
            return False
        return all(current.get(key) == value for key, value in expected_metadata.items())

    def _write_pretokenized_cache(
        self,
        *,
        split: str,
        epoch: int,
        path_indices: np.ndarray,
        row_indices: np.ndarray,
        data_path_strs: list[str],
        n_obs_by_path: list[int],
        source_gene_indices: list[np.ndarray],
        vocab_gene_ids: list[np.ndarray],
        cls_token_id: int,
        pad_token_id: int,
        cls_value: float,
        pad_value: int,
        selected_gene_count: int,
        io_block_size: int,
        seed: int,
        binning,
    ) -> None:
        n_samples = int(row_indices.shape[0])
        seq_len = int(selected_gene_count) + 1
        paths = self._cache_paths(split, epoch)
        metadata = self._cache_metadata(
            split=split,
            epoch=epoch,
            n_samples=n_samples,
            seq_len=seq_len,
            data_path_strs=data_path_strs,
            seed=seed,
        )
        if self._cache_is_ready(paths, metadata):
            log.info("scGPT preadapt | using existing pre-tokenized cache: split=%s epoch=%d", split, epoch)
            return

        paths["dir"].mkdir(parents=True, exist_ok=True)
        tmp_genes = paths["genes"].with_suffix(paths["genes"].suffix + ".tmp")
        tmp_expr = paths["expr"].with_suffix(paths["expr"].suffix + ".tmp")
        tmp_metadata = paths["metadata"].with_suffix(paths["metadata"].suffix + ".tmp")
        for path in (tmp_genes, tmp_expr, tmp_metadata):
            if path.exists():
                path.unlink()

        genes_mm = np.memmap(tmp_genes, mode="w+", dtype=np.int32, shape=(n_samples, seq_len))
        expr_mm = np.memmap(tmp_expr, mode="w+", dtype=np.int16, shape=(n_samples, seq_len))
        genes_mm[:, :] = int(pad_token_id)
        expr_mm[:, :] = int(pad_value)
        genes_mm[:, 0] = int(cls_token_id)
        expr_mm[:, 0] = int(cls_value)

        start_time = time.perf_counter()
        blocks_done = 0
        cache_log_every_blocks = int(getattr(self.pretrain_cfg, "cache_log_every_blocks", 100))
        log.info(
            "scGPT preadapt | cache split=%s epoch=%d starting: samples=%d seq_len=%d io_block_size=%d",
            split,
            epoch,
            n_samples,
            seq_len,
            int(io_block_size),
        )
        if n_samples == 0:
            genes_mm.flush()
            expr_mm.flush()
        else:
            total_blocks = int(
                sum(
                    np.unique((row_indices[np.flatnonzero(path_indices == path_idx)] // int(io_block_size)) * int(io_block_size)).shape[0]
                    for path_idx in np.unique(path_indices)
                )
            )
            for path_idx in np.unique(path_indices):
                path_idx_int = int(path_idx)
                source_indices = source_gene_indices[path_idx_int]
                vocab_ids = vocab_gene_ids[path_idx_int]
                n_genes = int(vocab_ids.shape[0])
                sample_positions = np.flatnonzero(path_indices == path_idx)
                rows_for_path = row_indices[sample_positions]
                block_starts = (rows_for_path // int(io_block_size)) * int(io_block_size)

                backed = ad.read_h5ad(data_path_strs[path_idx_int], backed="r")
                try:
                    for block_start in np.unique(block_starts):
                        block_start_int = int(block_start)
                        block_stop = min(block_start_int + int(io_block_size), n_obs_by_path[path_idx_int])
                        in_block = np.flatnonzero(block_starts == block_start)
                        positions = sample_positions[in_block]
                        rows = rows_for_path[in_block]
                        should_log_block = cache_log_every_blocks > 0 and (
                            blocks_done == 0
                            or (blocks_done + 1) % cache_log_every_blocks == 0
                            or blocks_done + 1 == total_blocks
                        )
                        if should_log_block:
                            log.info(
                                "scGPT preadapt | cache split=%s epoch=%d block=%d/%d reading rows=%d:%d path=%d samples_in_block=%d elapsed=%.1fs",
                                split,
                                epoch,
                                blocks_done + 1,
                                total_blocks,
                                block_start_int,
                                block_stop,
                                path_idx_int,
                                len(positions),
                                time.perf_counter() - start_time,
                            )

                        read_start = time.perf_counter()
                        raw_block = backed.X[block_start_int:block_stop, :]
                        if sparse.issparse(raw_block):
                            block = raw_block[:, source_indices].toarray().astype(np.float32, copy=False)
                        else:
                            block = np.asarray(raw_block)[:, source_indices].astype(np.float32, copy=False)
                        read_elapsed = time.perf_counter() - read_start

                        for pos, row_idx in zip(positions, rows, strict=False):
                            if selected_gene_count < n_genes:
                                rng = np.random.default_rng(seed + int(epoch) * 1_000_003 + int(pos))
                                selected = rng.choice(n_genes, size=selected_gene_count, replace=False)
                                selected.sort()
                            else:
                                selected = np.arange(n_genes, dtype=np.int64)
                            selected_len = int(selected.shape[0])
                            row = block[int(row_idx) - block_start_int, selected].astype(np.float32, copy=False)
                            binned = binning(row=row, n_bins=int(self.model_configs.get("n_bins", 51)))
                            genes_mm[int(pos), 1 : selected_len + 1] = vocab_ids[selected].astype(np.int32, copy=False)
                            expr_mm[int(pos), 1 : selected_len + 1] = np.asarray(binned, dtype=np.int16)
                        blocks_done += 1
                        if cache_log_every_blocks > 0 and (
                            blocks_done == 1
                            or blocks_done % cache_log_every_blocks == 0
                            or blocks_done == total_blocks
                        ):
                            log.info(
                                "scGPT preadapt | cache split=%s epoch=%d block=%d/%d done read=%.1fs elapsed=%.1fs",
                                split,
                                epoch,
                                blocks_done,
                                total_blocks,
                                read_elapsed,
                                time.perf_counter() - start_time,
                            )
                finally:
                    backed.file.close()

        genes_mm.flush()
        expr_mm.flush()
        del genes_mm
        del expr_mm
        tmp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tmp_genes.replace(paths["genes"])
        tmp_expr.replace(paths["expr"])
        tmp_metadata.replace(paths["metadata"])
        log.info(
            "scGPT preadapt | wrote pre-tokenized cache: split=%s epoch=%d samples=%d seq_len=%d elapsed=%.1fs",
            split,
            epoch,
            n_samples,
            seq_len,
            time.perf_counter() - start_time,
        )

    def _ensure_pretokenized_cache(self, split: str, epoch: int) -> None:
        if not bool(getattr(self.pretrain_cfg, "use_pretokenized_cache", False)):
            return
        if self._pretokenize_context is None:
            raise RuntimeError("Pretokenized cache context has not been initialized.")
        ctx = self._pretokenize_context
        path_indices = ctx[f"{split}_path_indices"]
        row_indices = ctx[f"{split}_row_indices"]
        n_samples = int(row_indices.shape[0])
        seq_len = int(ctx["selected_gene_count"]) + 1
        data_path_strs = list(ctx["data_path_strs"])
        seed = int(ctx["seed"])
        paths = self._cache_paths(split, epoch)
        metadata = self._cache_metadata(
            split=split,
            epoch=epoch,
            n_samples=n_samples,
            seq_len=seq_len,
            data_path_strs=data_path_strs,
            seed=seed,
        )
        cache_ready = self._cache_is_ready(paths, metadata)
        build_cache = bool(getattr(self.pretrain_cfg, "build_pretokenized_cache", False))
        if not cache_ready and not build_cache:
            raise FileNotFoundError(
                "Missing pre-tokenized scGPT cache for "
                f"split={split}, epoch={epoch}: {paths['metadata']}. "
                "Build it first with pretrain.pretokenize_only=true and "
                "pretrain.build_pretokenized_cache=true, or set "
                "pretrain.build_pretokenized_cache=true for single-process debugging."
            )

        if self.is_master and not cache_ready:
            _, _, _, _, binning = self._import_scgpt(Path(ctx["repo_dir"]))
            self._write_pretokenized_cache(
                split=split,
                epoch=epoch,
                path_indices=path_indices,
                row_indices=row_indices,
                data_path_strs=data_path_strs,
                n_obs_by_path=list(ctx["n_obs_by_path"]),
                source_gene_indices=list(ctx["source_gene_indices"]),
                vocab_gene_ids=list(ctx["vocab_gene_ids"]),
                cls_token_id=int(ctx["cls_token_id"]),
                pad_token_id=int(ctx["pad_token_id"]),
                cls_value=float(ctx["cls_value"]),
                pad_value=int(ctx["pad_value"]),
                selected_gene_count=int(ctx["selected_gene_count"]),
                io_block_size=int(ctx["io_block_size"]),
                seed=seed,
                binning=binning,
            )
        if self.is_distributed:
            dist.barrier()
        if not self._cache_is_ready(paths, metadata):
            raise FileNotFoundError(f"Pre-tokenized cache was not created for split={split}, epoch={epoch}.")

    def _pretokenize_all(self) -> dict[str, object]:
        if not bool(getattr(self.pretrain_cfg, "use_pretokenized_cache", False)):
            raise RuntimeError("pretokenize_only requires pretrain.use_pretokenized_cache=true.")

        epochs = int(
            getattr(
                self.pretrain_cfg,
                "pretokenize_epochs",
                int(getattr(self.pretrain_cfg, "epochs", 5)),
            )
            or int(getattr(self.pretrain_cfg, "epochs", 5))
        )
        start_time = time.perf_counter()
        if self.is_master:
            log.info(
                "scGPT preadapt | pre-tokenization-only mode: building val epoch=0 and train epochs=1..%d",
                epochs,
            )

        self._ensure_pretokenized_cache("val", 0)
        for epoch in range(1, epochs + 1):
            self._ensure_pretokenized_cache("train", epoch)

        elapsed = time.perf_counter() - start_time
        if self.is_master:
            log.info("scGPT preadapt | pre-tokenization complete in %.1fs", elapsed)
        return {
            "pretokenized_cache_dir": str(self._pretokenized_cache_dir()),
            "pretokenize_epochs": epochs,
            "elapsed_seconds": elapsed,
        }

    def _build_loaders(self, data_paths: list[Path], paths: dict[str, Path]) -> None:
        DataCollator, _, _, _, _ = self._import_scgpt(paths["repo_dir"])
        ensg_to_symbol = self._gene_info_mapping(paths["gene_info"])
        data_path_strs = [str(path) for path in data_paths]

        source_gene_indices: list[np.ndarray] = []
        vocab_gene_ids: list[np.ndarray] = []
        n_obs_by_path: list[int] = []
        train_path_ids = []
        train_row_ids = []
        val_path_ids = []
        val_row_ids = []
        mapping_rows = []
        seed = int(getattr(self.pretrain_cfg, "seed", 2021))

        for path_idx, data_path in enumerate(data_paths):
            backed = ad.read_h5ad(data_path, backed="r")
            try:
                n_obs = int(backed.n_obs)
            finally:
                backed.file.close()
            n_obs_by_path.append(n_obs)
            src_idx, vocab_ids, mapped, total = self._map_genes_for_path(
                data_path,
                ensg_to_symbol,
                self.vocab,
            )
            source_gene_indices.append(src_idx)
            vocab_gene_ids.append(vocab_ids)
            train_idx, val_idx = self._split_rows(n_obs, seed + path_idx)
            train_path_ids.append(np.full(train_idx.shape, path_idx, dtype=np.int16))
            train_row_ids.append(train_idx)
            if val_idx.size:
                val_path_ids.append(np.full(val_idx.shape, path_idx, dtype=np.int16))
                val_row_ids.append(val_idx)
            mapping_rows.append(
                {
                    "data_path": str(data_path),
                    "samples": n_obs,
                    "genes_total": total,
                    "genes_mapped_to_scgpt_vocab": mapped,
                    "genes_dropped": total - mapped,
                }
            )

        train_path_indices = np.concatenate(train_path_ids)
        train_row_indices = np.concatenate(train_row_ids)
        train_order = np.lexsort((train_row_indices, train_path_indices))
        train_path_indices = train_path_indices[train_order]
        train_row_indices = train_row_indices[train_order]
        if val_path_ids:
            val_path_indices = np.concatenate(val_path_ids)
            val_row_indices = np.concatenate(val_row_ids)
            val_order = np.lexsort((val_row_indices, val_path_indices))
            val_path_indices = val_path_indices[val_order]
            val_row_indices = val_row_indices[val_order]
        else:
            val_path_indices = np.asarray([], dtype=np.int16)
            val_row_indices = np.asarray([], dtype=np.int64)

        selected_gene_count = int(getattr(self.pretrain_cfg, "selected_gene_count", 1199))
        io_block_size = int(getattr(self.pretrain_cfg, "io_block_size", 1024))
        io_cache_blocks = int(getattr(self.pretrain_cfg, "io_cache_blocks", 2))
        pad_value = int(self.model_configs["pad_value"])
        pad_token_id = int(self.vocab[str(self.model_configs["pad_token"])])
        cls_token_id = int(self.vocab["<cls>"])
        use_pretokenized_cache = bool(getattr(self.pretrain_cfg, "use_pretokenized_cache", False))
        if use_pretokenized_cache:
            seq_len = selected_gene_count + 1
            cache_dir = self._pretokenized_cache_dir()
            self._pretokenize_context = {
                "repo_dir": str(paths["repo_dir"]),
                "data_path_strs": data_path_strs,
                "n_obs_by_path": n_obs_by_path,
                "source_gene_indices": source_gene_indices,
                "vocab_gene_ids": vocab_gene_ids,
                "train_path_indices": train_path_indices,
                "train_row_indices": train_row_indices,
                "val_path_indices": val_path_indices,
                "val_row_indices": val_row_indices,
                "cls_token_id": cls_token_id,
                "pad_token_id": pad_token_id,
                "cls_value": float(pad_value),
                "pad_value": pad_value,
                "selected_gene_count": selected_gene_count,
                "io_block_size": io_block_size,
                "seed": seed,
            }
            self.train_dataset = ScGPTPreTokenizedDataset(
                cache_dir=cache_dir,
                split="train",
                n_samples=int(train_row_indices.shape[0]),
                seq_len=seq_len,
            )
            self.val_dataset = ScGPTPreTokenizedDataset(
                cache_dir=cache_dir,
                split="val",
                n_samples=int(val_row_indices.shape[0]),
                seq_len=seq_len,
            )
            collator = PreBinnedMaskCollator(
                pad_value=pad_value,
                mask_value=int(self.model_configs.get("mask_value", -1)),
                mlm_probability=float(getattr(self.pretrain_cfg, "mask_prob", 0.15)),
                keep_first_n_tokens=1,
            )
        else:
            self.train_dataset = ScGPTBulkMaskedDataset(
                data_paths=data_path_strs,
                path_indices=train_path_indices,
                row_indices=train_row_indices,
                n_obs_by_path=n_obs_by_path,
                source_gene_indices=source_gene_indices,
                vocab_gene_ids=vocab_gene_ids,
                cls_token_id=cls_token_id,
                cls_value=float(pad_value),
                selected_gene_count=selected_gene_count,
                io_block_size=io_block_size,
                io_cache_blocks=io_cache_blocks,
                seed=seed + self.rank,
            )
            self.val_dataset = ScGPTBulkMaskedDataset(
                data_paths=data_path_strs,
                path_indices=val_path_indices,
                row_indices=val_row_indices,
                n_obs_by_path=n_obs_by_path,
                source_gene_indices=source_gene_indices,
                vocab_gene_ids=vocab_gene_ids,
                cls_token_id=cls_token_id,
                cls_value=float(pad_value),
                selected_gene_count=selected_gene_count,
                io_block_size=io_block_size,
                io_cache_blocks=io_cache_blocks,
                seed=seed,
            )

            collator = DataCollator(
                do_padding=True,
                pad_token_id=pad_token_id,
                pad_value=pad_value,
                do_mlm=True,
                do_binning=True,
                mlm_probability=float(getattr(self.pretrain_cfg, "mask_prob", 0.15)),
                mask_value=int(self.model_configs.get("mask_value", -1)),
                max_length=selected_gene_count + 1,
                sampling=False,
                keep_first_n_tokens=1,
            )

        loader_kwargs: dict[str, object] = {
            "num_workers": int(getattr(self.pretrain_cfg, "num_workers", 2)),
            "pin_memory": self.device.type == "cuda",
            "collate_fn": collator,
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["prefetch_factor"] = int(getattr(self.pretrain_cfg, "prefetch_factor", 2))
            loader_kwargs["persistent_workers"] = bool(getattr(self.pretrain_cfg, "persistent_workers", False))

        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=int(getattr(self.pretrain_cfg, "batch_size", 16)),
                sampler=train_sampler,
                shuffle=False,
                **loader_kwargs,
            )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=int(getattr(self.pretrain_cfg, "batch_size", 16)),
                sampler=val_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=int(getattr(self.pretrain_cfg, "batch_size", 16)),
                shuffle=False,
                **loader_kwargs,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=int(getattr(self.pretrain_cfg, "batch_size", 16)),
                shuffle=False,
                **loader_kwargs,
            )

        if self.is_master:
            self._write_csv(self._output_dir() / f"{self._output_prefix()}_gene_mapping.csv", mapping_rows)
            log.info(
                "Prepared scGPT bulk preadaptation data: train=%d, val=%d, files=%d, io_block_size=%d, io_cache_blocks=%d, num_workers=%d, pretokenized_cache=%s",
                len(self.train_dataset),
                len(self.val_dataset),
                len(data_paths),
                io_block_size,
                io_cache_blocks,
                int(loader_kwargs["num_workers"]),
                use_pretokenized_cache,
            )

    def _build_optimization(self) -> None:
        learning_rate = float(getattr(self.pretrain_cfg, "learning_rate", 1e-4))
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = Adam(trainable_params, lr=learning_rate)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.pretrain_cfg, "first_cycle_steps", 5)),
            cycle_mult=float(getattr(self.pretrain_cfg, "cycle_mult", 1)),
            max_lr=learning_rate,
            min_lr=float(getattr(self.pretrain_cfg, "min_lr", 1e-6)),
            warmup_steps=int(getattr(self.pretrain_cfg, "warmup_steps", 1)),
            gamma=float(getattr(self.pretrain_cfg, "gamma", 1.0)),
        )
        amp_enabled = bool(getattr(self.pretrain_cfg, "amp", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def _amp_dtype(self) -> torch.dtype:
        value = str(getattr(self.pretrain_cfg, "amp_dtype", "float16")).lower()
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float16

    def _model_name(self) -> str:
        return str(getattr(self.pretrain_cfg, "model_name", "scgpt_preadapt_bulk"))

    def _output_dir(self) -> Path:
        return ROOT / "output" / self._model_name()

    def _output_prefix(self) -> str:
        return self._model_name()

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({field for row in rows for field in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_config(self, paths: dict[str, Path], data_paths: list[Path]) -> None:
        if not self.is_master:
            return
        out_dir = self._output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_prefix()
        (out_dir / f"{prefix}_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )
        metadata = {
            "task": "pretrain.scgpt_preadapt",
            "model_name": self._model_name(),
            "scgpt_repo_dir": str(paths["repo_dir"]),
            "scgpt_model_dir": str(paths["model_dir"]),
            "scgpt_checkpoint": str(paths["checkpoint"]),
            "data_paths": [str(path) for path in data_paths],
            "selected_gene_count": int(getattr(self.pretrain_cfg, "selected_gene_count", 1199)),
            "mask_prob": float(getattr(self.pretrain_cfg, "mask_prob", 0.15)),
            "validation_split": float(getattr(self.pretrain_cfg, "validation_split", 0.05)),
        }
        (out_dir / f"{prefix}_run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        shutil.copy2(paths["args"], out_dir / str(getattr(self.pretrain_cfg, "scgpt_args_filename", "args.json")))
        shutil.copy2(paths["vocab"], out_dir / str(getattr(self.pretrain_cfg, "scgpt_vocab_filename", "vocab.json")))

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gene = batch["gene"].to(self.device, non_blocking=True)
        target_expr = batch["expr"].to(self.device, non_blocking=True)
        masked_expr = batch["masked_expr"].to(self.device, non_blocking=True)
        padding_mask = gene.eq(int(self.vocab[str(self.model_configs["pad_token"])]))
        mask_value = int(self.model_configs.get("mask_value", -1))
        masked_positions = masked_expr.eq(mask_value)

        amp_enabled = bool(getattr(self.pretrain_cfg, "amp", True)) and self.device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=self._amp_dtype()):
            output = self.model(
                gene,
                masked_expr,
                src_key_padding_mask=padding_mask,
                MVC=bool(getattr(self.pretrain_cfg, "do_mvc", False)),
            )
            mask = masked_positions.float()
            token_count = mask.sum()
            if token_count.item() == 0:
                loss_sum = output["mlm_output"].sum() * 0.0
                return loss_sum, loss_sum.detach(), token_count.detach()

            loss_sum = F.mse_loss(
                output["mlm_output"] * mask,
                target_expr * mask,
                reduction="sum",
            )
            total_loss_sum = loss_sum
            if bool(getattr(self.pretrain_cfg, "do_mvc", False)):
                mvc_loss_sum = F.mse_loss(
                    output["mvc_output"] * mask,
                    target_expr * mask,
                    reduction="sum",
                )
                total_loss_sum = total_loss_sum + mvc_loss_sum
        loss = total_loss_sum / token_count
        return loss, total_loss_sum.detach(), token_count.detach()

    def _reduce_pair(self, loss_sum: torch.Tensor, token_count: torch.Tensor) -> tuple[float, float]:
        if self.is_distributed:
            values = torch.stack([loss_sum.to(self.device), token_count.to(self.device)])
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            loss_sum = values[0]
            token_count = values[1]
        return float(loss_sum.detach().cpu().item()), float(token_count.detach().cpu().item())

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self._ensure_pretokenized_cache("train", epoch)
        self.model.train()
        self.train_dataset.set_epoch(epoch)
        if self.is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        grad_acc = int(getattr(self.pretrain_cfg, "grad_acc", 1))
        max_grad_norm = float(getattr(self.pretrain_cfg, "max_grad_norm", 1e2))
        log_every_batches = int(getattr(self.pretrain_cfg, "log_every_batches", 500))
        local_loss_sum = torch.zeros((), device=self.device)
        local_token_count = torch.zeros((), device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        last_log_time = epoch_start
        num_batches = len(self.train_loader)

        if self.is_master:
            log.info(
                "scGPT preadapt | epoch=%d | starting train loop: batches=%d | batch_size=%d | grad_acc=%d",
                epoch,
                num_batches,
                int(getattr(self.pretrain_cfg, "batch_size", 16)),
                grad_acc,
            )

        for step, batch in enumerate(self.train_loader, start=1):
            loss, loss_sum, token_count = self._compute_loss(batch)
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss / grad_acc).backward()
            else:
                (loss / grad_acc).backward()
            local_loss_sum += loss_sum
            local_token_count += token_count

            if step % grad_acc == 0 or step == len(self.train_loader):
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in self.model.parameters() if param.requires_grad],
                    max_grad_norm,
                )
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if (
                self.is_master
                and log_every_batches > 0
                and (step == 1 or step % log_every_batches == 0 or step == num_batches)
            ):
                now = time.perf_counter()
                elapsed = now - epoch_start
                recent = now - last_log_time
                last_log_time = now
                seen_tokens = float(local_token_count.detach().cpu().item())
                running_loss = (
                    float(local_loss_sum.detach().cpu().item()) / seen_tokens
                    if seen_tokens
                    else float("nan")
                )
                log.info(
                    "scGPT preadapt | epoch=%d | train batch=%d/%d | running_loss=%.6f | masked_tokens=%d | elapsed=%.1fs | recent=%.1fs",
                    epoch,
                    step,
                    num_batches,
                    running_loss,
                    int(seen_tokens),
                    elapsed,
                    recent,
                )

        loss_sum, token_count = self._reduce_pair(local_loss_sum, local_token_count)
        return {"loss": loss_sum / token_count if token_count else float("nan"), "masked_tokens": token_count}

    def _validate(self, epoch: int) -> dict[str, float]:
        self._ensure_pretokenized_cache("val", 0)
        self.model.eval()
        self.val_dataset.set_epoch(0)
        log_every_batches = int(getattr(self.pretrain_cfg, "log_every_batches", 500))
        local_loss_sum = torch.zeros((), device=self.device)
        local_token_count = torch.zeros((), device=self.device)
        num_batches = len(self.val_loader)
        val_start = time.perf_counter()

        if self.is_master:
            log.info(
                "scGPT preadapt | epoch=%d | starting validation loop: batches=%d",
                epoch,
                num_batches,
            )

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader, start=1):
                _, loss_sum, token_count = self._compute_loss(batch)
                local_loss_sum += loss_sum
                local_token_count += token_count
                if (
                    self.is_master
                    and log_every_batches > 0
                    and (step == 1 or step % log_every_batches == 0 or step == num_batches)
                ):
                    seen_tokens = float(local_token_count.detach().cpu().item())
                    running_loss = (
                        float(local_loss_sum.detach().cpu().item()) / seen_tokens
                        if seen_tokens
                        else float("nan")
                    )
                    log.info(
                        "scGPT preadapt | epoch=%d | val batch=%d/%d | running_loss=%.6f | masked_tokens=%d | elapsed=%.1fs",
                        epoch,
                        step,
                        num_batches,
                        running_loss,
                        int(seen_tokens),
                        time.perf_counter() - val_start,
                    )
        loss_sum, token_count = self._reduce_pair(local_loss_sum, local_token_count)
        return {"loss": loss_sum / token_count if token_count else float("nan"), "masked_tokens": token_count}

    def _state_dict(self) -> dict[str, torch.Tensor]:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        return model.state_dict()

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float) -> None:
        if not self.is_master:
            return
        out_dir = self._output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_prefix()
        state_dict = self._state_dict()
        torch.save(state_dict, out_dir / f"{prefix}_{name}_model.pt")
        if name == "best":
            torch.save(state_dict, out_dir / "scgpt_preadapt.pt")
        torch.save(
            {
                "epoch": int(epoch),
                "val_loss": float(val_loss),
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            },
            out_dir / f"{prefix}_{name}_training_checkpoint.pth",
        )

    def run(self) -> dict:
        try:
            self._setup_runtime()
            paths = self._scgpt_paths()
            data_paths = self._data_paths()
            self._build_model(paths)
            self._save_config(paths, data_paths)
            self._build_loaders(data_paths, paths)
            if bool(getattr(self.pretrain_cfg, "pretokenize_only", False)):
                return self._pretokenize_all()
            self._build_optimization()

            history: list[dict[str, object]] = []
            best_val_loss = float("inf")
            best_epoch = -1
            epochs = int(getattr(self.pretrain_cfg, "epochs", 5))
            valid_every = int(getattr(self.pretrain_cfg, "valid_every", 1))

            for epoch in range(1, epochs + 1):
                train_metrics = self._train_epoch(epoch)
                val_metrics = {"loss": float("nan"), "masked_tokens": 0.0}
                if valid_every > 0 and epoch % valid_every == 0 and len(self.val_dataset) > 0:
                    val_metrics = self._validate(epoch)
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = float(val_metrics["loss"])
                        best_epoch = epoch
                        self._save_checkpoint("best", epoch, best_val_loss)

                self.scheduler.step()
                row = {
                    "epoch": epoch,
                    "train_loss": float(train_metrics["loss"]),
                    "train_masked_tokens": int(train_metrics["masked_tokens"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_masked_tokens": int(val_metrics["masked_tokens"]),
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                }
                history.append(row)
                if self.is_master:
                    log.info(
                        "scGPT preadapt | epoch=%d | train_loss=%.6f | val_loss=%.6f | lr=%.6g",
                        epoch,
                        row["train_loss"],
                        row["val_loss"],
                        row["learning_rate"],
                    )
                    self._write_csv(
                        self._output_dir() / f"{self._output_prefix()}_metrics.csv",
                        history,
                    )
                self._save_checkpoint("latest", epoch, row["val_loss"])

            self._save_checkpoint("last", epochs, history[-1]["val_loss"])
            return {
                "metrics_path": str(self._output_dir() / f"{self._output_prefix()}_metrics.csv"),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
