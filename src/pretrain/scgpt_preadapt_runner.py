from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
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
        source_gene_indices: list[np.ndarray],
        vocab_gene_ids: list[np.ndarray],
        cls_token_id: int,
        cls_value: float,
        selected_gene_count: int,
        seed: int,
    ) -> None:
        self.data_paths = [str(path) for path in data_paths]
        self.path_indices = np.asarray(path_indices, dtype=np.int16)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.source_gene_indices = [
            np.asarray(indices, dtype=np.int64) for indices in source_gene_indices
        ]
        self.vocab_gene_ids = [np.asarray(ids, dtype=np.int64) for ids in vocab_gene_ids]
        self.cls_token_id = int(cls_token_id)
        self.cls_value = float(cls_value)
        self.selected_gene_count = int(selected_gene_count)
        self.seed = int(seed)
        self.epoch = 0
        self._adatas: dict[int, ad.AnnData] = {}

        if self.path_indices.shape[0] != self.row_indices.shape[0]:
            raise ValueError("path_indices and row_indices must have equal length.")

    def __len__(self) -> int:
        return int(self.row_indices.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_adatas"] = {}
        return state

    def __del__(self):
        for adata in getattr(self, "_adatas", {}).values():
            file_obj = getattr(adata, "file", None)
            if file_obj is not None:
                file_obj.close()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ensure_open(self, path_idx: int) -> ad.AnnData:
        if path_idx not in self._adatas:
            self._adatas[path_idx] = ad.read_h5ad(self.data_paths[path_idx], backed="r")
        return self._adatas[path_idx]

    @staticmethod
    def _to_dense_row(row) -> np.ndarray:
        if sparse.issparse(row):
            return np.asarray(row.toarray()).ravel()
        return np.asarray(row).ravel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path_idx = int(self.path_indices[index])
        row_idx = int(self.row_indices[index])
        adata = self._ensure_open(path_idx)

        source_indices = self.source_gene_indices[path_idx]
        vocab_ids = self.vocab_gene_ids[path_idx]

        n_genes = int(vocab_ids.shape[0])
        if self.selected_gene_count < n_genes:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            selected = rng.choice(n_genes, size=self.selected_gene_count, replace=False)
            selected.sort()
        else:
            selected = np.arange(n_genes, dtype=np.int64)

        # Read only the sampled genes from backed h5ad. Reading all mapped genes
        # first is prohibitive for ARCHS4-scale bulk matrices and can exceed
        # job memory before the first training epoch starts.
        row = self._to_dense_row(adata.X[row_idx, source_indices[selected]]).astype(
            np.float32,
            copy=False,
        )

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
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        return DataCollator, TransformerModel, GeneVocab, load_pretrained

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
        _, TransformerModel, GeneVocab, load_pretrained = self._import_scgpt(paths["repo_dir"])
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
        model = model.to(self.device)

        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                model = DDP(model)

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

    def _build_loaders(self, data_paths: list[Path], paths: dict[str, Path]) -> None:
        DataCollator, _, _, _ = self._import_scgpt(paths["repo_dir"])
        ensg_to_symbol = self._gene_info_mapping(paths["gene_info"])
        data_path_strs = [str(path) for path in data_paths]

        source_gene_indices: list[np.ndarray] = []
        vocab_gene_ids: list[np.ndarray] = []
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
        if val_path_ids:
            val_path_indices = np.concatenate(val_path_ids)
            val_row_indices = np.concatenate(val_row_ids)
        else:
            val_path_indices = np.asarray([], dtype=np.int16)
            val_row_indices = np.asarray([], dtype=np.int64)

        selected_gene_count = int(getattr(self.pretrain_cfg, "selected_gene_count", 1199))
        pad_value = int(self.model_configs["pad_value"])
        cls_token_id = int(self.vocab["<cls>"])
        self.train_dataset = ScGPTBulkMaskedDataset(
            data_paths=data_path_strs,
            path_indices=train_path_indices,
            row_indices=train_row_indices,
            source_gene_indices=source_gene_indices,
            vocab_gene_ids=vocab_gene_ids,
            cls_token_id=cls_token_id,
            cls_value=float(pad_value),
            selected_gene_count=selected_gene_count,
            seed=seed + self.rank,
        )
        self.val_dataset = ScGPTBulkMaskedDataset(
            data_paths=data_path_strs,
            path_indices=val_path_indices,
            row_indices=val_row_indices,
            source_gene_indices=source_gene_indices,
            vocab_gene_ids=vocab_gene_ids,
            cls_token_id=cls_token_id,
            cls_value=float(pad_value),
            selected_gene_count=selected_gene_count,
            seed=seed,
        )

        collator = DataCollator(
            do_padding=True,
            pad_token_id=int(self.vocab[str(self.model_configs["pad_token"])]),
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
            loader_kwargs["persistent_workers"] = False

        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
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
                shuffle=True,
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
                "Prepared scGPT bulk preadaptation data: train=%d, val=%d, files=%d",
                len(self.train_dataset),
                len(self.val_dataset),
                len(data_paths),
            )

    def _build_optimization(self) -> None:
        learning_rate = float(getattr(self.pretrain_cfg, "learning_rate", 1e-4))
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.pretrain_cfg, "first_cycle_steps", 5)),
            cycle_mult=float(getattr(self.pretrain_cfg, "cycle_mult", 1)),
            max_lr=learning_rate,
            min_lr=float(getattr(self.pretrain_cfg, "min_lr", 1e-6)),
            warmup_steps=int(getattr(self.pretrain_cfg, "warmup_steps", 1)),
            gamma=float(getattr(self.pretrain_cfg, "gamma", 1.0)),
        )

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
        self.model.train()
        self.train_dataset.set_epoch(epoch)
        if self.is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        grad_acc = int(getattr(self.pretrain_cfg, "grad_acc", 1))
        max_grad_norm = float(getattr(self.pretrain_cfg, "max_grad_norm", 1e2))
        local_loss_sum = torch.zeros((), device=self.device)
        local_token_count = torch.zeros((), device=self.device)
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader, start=1):
            loss, loss_sum, token_count = self._compute_loss(batch)
            (loss / grad_acc).backward()
            local_loss_sum += loss_sum
            local_token_count += token_count

            if step % grad_acc == 0 or step == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        loss_sum, token_count = self._reduce_pair(local_loss_sum, local_token_count)
        return {"loss": loss_sum / token_count if token_count else float("nan"), "masked_tokens": token_count}

    def _validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        self.val_dataset.set_epoch(0)
        local_loss_sum = torch.zeros((), device=self.device)
        local_token_count = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch in self.val_loader:
                _, loss_sum, token_count = self._compute_loss(batch)
                local_loss_sum += loss_sum
                local_token_count += token_count
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

            self._save_checkpoint("last", epochs, history[-1]["val_loss"])
            return {
                "metrics_path": str(self._output_dir() / f"{self._output_prefix()}_metrics.csv"),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
