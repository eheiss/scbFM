#!/usr/bin/env python3
"""Train fold-1 models and generate held-out PCA and UMAP coordinates."""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


ROOT = Path(os.environ.get("SCBFM_CLUSTER_ROOT", "/cluster/work/boeva/eheiss"))
SCBFM_ROOT = ROOT / "scbFM"
SRC = SCBFM_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.umap_pca.common import (
    MODEL_KEYS,
    REPRESENTATION_KEYS,
    dense_rows,
    fit_reductions,
    save_coordinate_archive,
    select_plot_indices,
    validate_coordinate_archive,
)
from finetune.canc_type_class.runner import CancTypeClassDataset, CancTypeClassRunner
from finetune.canc_type_class_33.runner import CancTypeClass33Runner
from finetune.disease_class.runner import DiseaseClassRunner
from utils import seed_all


log = logging.getLogger("umap_pca")
INITIAL_CHECKPOINT_PATHS = {
    "random_init": "",
    "pretrain_sc": ROOT / "output" / "pretrain_sc" / "pretrain_sc.pth",
    "preadapt_sc": ROOT / "output" / "preadapt_sc" / "preadapt_sc.pth",
    "pretrain_bulk": ROOT / "output" / "pretrain_bulk" / "pretrain_bulk.pth",
    "preadapt_bulk": ROOT / "output" / "preadapt_bulk" / "preadapt_bulk.pth",
}
TASK_RUNNERS = {
    "canc_type_class": CancTypeClassRunner,
    "canc_type_class_33": CancTypeClass33Runner,
    "disease_class": DiseaseClassRunner,
}
TASK_DATA_PATHS = {
    "canc_type_class": ("tcga_data_dir", ROOT / "datasets" / "TCGA" / "tcga.h5ad"),
    "canc_type_class_33": (
        "tcga_data_dir",
        ROOT / "datasets" / "TCGA" / "tcga.h5ad",
    ),
    "disease_class": (
        "disignatlas_data_path",
        ROOT / "datasets" / "DiSignAtlas" / "disignatlas.h5ad",
    ),
}
CURVE_FIELDS = (
    "task",
    "model",
    "fold",
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "validation_f1_macro",
    "validation_f1_weighted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASK_RUNNERS))
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--fit-max", type=int, default=10_000)
    parser.add_argument("--plot-max", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return ROOT / "output" / "umap_pca" / args.task / f"{args.task}_coordinates.npz"


def checkpoint_path(task: str, model_key: str, fold: int) -> Path:
    return (
        ROOT
        / "output"
        / "umap_pca_backbones"
        / task
        / f"{task}_full_ft_{model_key}_fold_{fold}_backbone.pth"
    )


def curve_path(args: argparse.Namespace) -> Path:
    path = output_path(args)
    return path.with_name(f"{args.task}_full_ft_fold_{args.fold_index + 1}_curves.csv")


def load_config(task: str):
    model_cfg = OmegaConf.load(SRC / "configs" / "pretrain" / "pretrain.yaml")
    task_cfg = OmegaConf.load(SRC / "configs" / "finetune" / f"{task}.yaml")
    cfg = OmegaConf.create(
        {
            "pretrain": OmegaConf.to_container(model_cfg, resolve=True),
            "finetune": {task: OmegaConf.to_container(task_cfg, resolve=True)},
        }
    )
    node = cfg.finetune[task]
    node.gene_list_path = str(SCBFM_ROOT / "data" / "gene_list.txt")
    node.finetune_mode = "full_ft"
    node.pretrained_model_paths = {
        key: str(path)
        for key, path in INITIAL_CHECKPOINT_PATHS.items()
        if key != "random_init"
    }
    return cfg


def ensure_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing analysis inputs: {missing}")


def reduction_kwargs(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "seed": args.seed,
        "pca_components": args.pca_components,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
    }


def reduce_all(
    representations: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, dict[str, np.ndarray | int | float]]:
    if tuple(representations) != REPRESENTATION_KEYS:
        raise ValueError(
            f"Representations must use the canonical order {REPRESENTATION_KEYS}; "
            f"got {tuple(representations)}."
        )
    coordinates = {}
    for key, (train_values, test_values) in representations.items():
        log.info(
            "Reducing %s: train=%s held-out=%s",
            key,
            train_values.shape,
            test_values.shape,
        )
        coordinates[key] = fit_reductions(
            train_values,
            test_values,
            **reduction_kwargs(args),
        )
        representations[key] = (np.empty((0, 0)), np.empty((0, 0)))
        gc.collect()
    return coordinates


def make_expression_loader(
    runner,
    matrix,
    labels: np.ndarray,
    gene_indices: np.ndarray,
    *,
    batch_size: int = 4,
) -> DataLoader:
    dataset = CancTypeClassDataset(
        matrix,
        labels,
        bin_num=int(runner.model_cfg.bin_num),
        cls_gene_id=runner.cls_gene_id,
        gene_token_offset=runner.gene_token_offset,
        cls_value=runner.cls_value,
        selected_gene_count=runner.selected_gene_count,
        seed=int(getattr(runner.task_cfg, "random_seed", 42)),
        do_binning=runner._should_preprocess_input(),
        fixed_gene_indices=gene_indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=runner.device.type == "cuda",
        persistent_workers=True,
    )


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def expected_checkpoint_metadata(
    runner,
    *,
    model_key: str,
    fold: int,
    source_checkpoint_path: str,
) -> dict[str, object]:
    return {
        "task": runner.task_name,
        "finetune_mode": "full_ft",
        "model_key": model_key,
        "fold": int(fold),
        "epoch": int(runner.task_cfg.epochs),
        "source_checkpoint_path": str(source_checkpoint_path),
        "cv_fold_fingerprint": str(getattr(runner, "_cv_fold_fingerprint", "")),
        "world_size": int(runner.world_size),
        "global_effective_batch_size": int(runner.task_cfg.batch_size)
        * int(runner.task_cfg.grad_accumulation_steps)
        * int(runner.world_size),
    }


def load_reusable_checkpoint(
    path: Path,
    *,
    runner,
    model_key: str,
    fold: int,
    source_checkpoint_path: str,
    selected_genes: np.ndarray,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    runner._validate_backbone_checkpoint(checkpoint, str(path))
    expected = expected_checkpoint_metadata(
        runner,
        model_key=model_key,
        fold=fold,
        source_checkpoint_path=source_checkpoint_path,
    )
    mismatches = {
        key: {"expected": value, "observed": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    observed_genes = np.asarray(
        checkpoint.get("selected_gene_indices", []), dtype=np.int64
    )
    if not np.array_equal(observed_genes, np.asarray(selected_genes, dtype=np.int64)):
        mismatches["selected_gene_indices"] = {
            "expected_count": int(len(selected_genes)),
            "observed_count": int(len(observed_genes)),
        }
    if mismatches:
        raise ValueError(f"Cannot reuse incompatible checkpoint {path}: {mismatches}")
    log.info("Reusing completed full fine-tuned backbone: %s", path)
    return checkpoint


def save_backbone_checkpoint(
    path: Path,
    *,
    runner,
    model_key: str,
    fold: int,
    source_checkpoint_path: str,
    selected_genes: np.ndarray,
) -> None:
    if runner.model is None:
        raise RuntimeError("Cannot save a checkpoint before building the model.")
    raw_model = (
        runner.model.module if isinstance(runner.model, DDP) else runner.model
    )
    state_dict = {
        key: value.detach().cpu()
        for key, value in raw_model.backbone.state_dict().items()
    }
    checkpoint = {
        "backbone": "cancerfoundation",
        "gene_num": int(runner.model_cfg.gene_num),
        "selected_gene_count": int(runner.selected_gene_count),
        "max_seq_len": int(runner.max_seq_len),
        "bin_num": int(runner.model_cfg.bin_num),
        "model_state_dict": state_dict,
        "cv_fold_manifest_path": str(getattr(runner, "_cv_fold_manifest_path", "")),
        "selected_gene_indices": np.asarray(selected_genes, dtype=np.int64).tolist(),
        **expected_checkpoint_metadata(
            runner,
            model_key=model_key,
            fold=fold,
            source_checkpoint_path=source_checkpoint_path,
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    log.info("Saved final full fine-tuned backbone: %s", path)


def load_curve_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_curve_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def extract_backbone_embeddings(runner, fit_loader, plot_loader):
    if runner.model is None:
        raise RuntimeError("Cannot extract embeddings before building the model.")
    raw_model = (
        runner.model.module if isinstance(runner.model, DDP) else runner.model
    )
    raw_model.backbone.eval()

    def extract(loader) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for batch, _ in loader:
                model_batch = {
                    key: value.to(runner.device, non_blocking=True)
                    for key, value in batch.items()
                }
                hidden = raw_model.backbone(
                    model_batch["gene_ids"],
                    model_batch["expr"],
                    src_key_padding_mask=model_batch.get(
                        "attention_key_padding_mask"
                    ),
                )
                embeddings.append(hidden[:, 0, :].detach().cpu().numpy())
        return np.vstack(embeddings)

    return extract(fit_loader), extract(plot_loader)


def generate_classification(args: argparse.Namespace) -> None:
    task = args.task
    cfg = load_config(task)
    path_key, data_path = TASK_DATA_PATHS[task]
    cfg.finetune[task][path_key] = str(data_path)
    ensure_inputs(
        [
            data_path,
            *[Path(path) for path in INITIAL_CHECKPOINT_PATHS.values() if path],
        ]
    )

    runner = TASK_RUNNERS[task](cfg)
    runner._setup_runtime()
    effective_batch_size = (
        int(runner.task_cfg.batch_size)
        * int(runner.task_cfg.grad_accumulation_steps)
        * runner.world_size
    )
    if runner.world_size != 4 or effective_batch_size != 64:
        raise ValueError(
            "Fold-1 full fine-tuning must run with four processes to match the "
            "benchmark's global effective batch size of 64."
        )
    adata, labels, groups = runner._prepare_cv_data()
    splits = runner._build_or_load_cv_splits(adata, labels, groups)
    if not 0 <= args.fold_index < len(splits):
        raise ValueError(f"fold-index must be in [0, {len(splits) - 1}].")
    fold_number = args.fold_index + 1
    train_indices, test_indices = splits[args.fold_index]
    selected_genes = runner._select_training_hvg_indices(adata[train_indices])
    train_adata = adata[train_indices].copy()
    test_adata = adata[test_indices].copy()

    representations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    curve_file = curve_path(args)
    curve_rows = load_curve_rows(curve_file) if runner.is_master else []
    checkpoint_paths = {
        model_key: checkpoint_path(task, model_key, fold_number)
        for model_key in MODEL_KEYS
    }

    if runner.is_master:
        train_local = select_plot_indices(
            labels[train_indices], args.fit_max, seed=args.seed
        )
        test_local = select_plot_indices(
            labels[test_indices], args.plot_max, seed=args.seed + 1
        )
        fit_indices = np.asarray(train_indices)[train_local]
        plot_indices = np.asarray(test_indices)[test_local]
        label_order = np.asarray(runner.label_dict).astype(str)
        label_to_index = {label: index for index, label in enumerate(label_order)}
        fit_y = np.asarray(
            [label_to_index[str(label)] for label in labels[fit_indices]]
        )
        plot_y = np.asarray(
            [label_to_index[str(label)] for label in labels[plot_indices]]
        )
        fit_loader = make_expression_loader(
            runner, adata.X[fit_indices], fit_y, selected_genes
        )
        plot_loader = make_expression_loader(
            runner, adata.X[plot_indices], plot_y, selected_genes
        )
    else:
        fit_indices = plot_indices = label_order = fit_loader = plot_loader = None

    epochs = int(runner.task_cfg.epochs)
    for model_key in MODEL_KEYS:
        source_path = str(INITIAL_CHECKPOINT_PATHS[model_key])
        seed_all(int(runner.task_cfg.random_seed) + runner.rank + fold_number)
        runner._build_loaders(train_adata, test_adata)
        if not np.array_equal(runner.fold_gene_indices, selected_genes):
            raise ValueError("Full fine-tuning and visualization selected different genes.")
        runner._build_model(source_path)

        rng_state = capture_rng_state() if runner.is_master else None
        if runner.is_master:
            head_fit, head_plot = extract_backbone_embeddings(
                runner, fit_loader, plot_loader
            )
            representations[f"head_only__{model_key}"] = (head_fit, head_plot)
            restore_rng_state(rng_state)
        if runner.is_distributed:
            dist.barrier()

        saved_checkpoint = load_reusable_checkpoint(
            checkpoint_paths[model_key],
            runner=runner,
            model_key=model_key,
            fold=fold_number,
            source_checkpoint_path=source_path,
            selected_genes=selected_genes,
        )
        if saved_checkpoint is None:
            runner._build_optimization()
            if runner.is_master:
                curve_rows = [
                    row for row in curve_rows if str(row.get("model")) != model_key
                ]
            for epoch in range(1, epochs + 1):
                train_metrics = runner._train_one_epoch(epoch)
                validation_metrics = runner._evaluate()
                if runner.is_master:
                    curve_rows.append(
                        {
                            "task": task,
                            "model": model_key,
                            "fold": fold_number,
                            "epoch": epoch,
                            "train_loss": float(train_metrics["loss"]),
                            "train_accuracy": float(train_metrics["accuracy"]),
                            "validation_loss": float(validation_metrics["loss"]),
                            "validation_accuracy": float(validation_metrics["accuracy"]),
                            "validation_f1_macro": float(
                                validation_metrics["f1_macro"]
                            ),
                            "validation_f1_weighted": float(
                                validation_metrics["f1_weighted"]
                            ),
                        }
                    )
                    write_curve_rows(curve_file, curve_rows)
                    log.info(
                        "%s | Fold %d | Epoch %d/%d | train loss %.6f | "
                        "validation loss %.6f",
                        model_key,
                        fold_number,
                        epoch,
                        epochs,
                        train_metrics["loss"],
                        validation_metrics["loss"],
                    )
            if runner.is_master:
                save_backbone_checkpoint(
                    checkpoint_paths[model_key],
                    runner=runner,
                    model_key=model_key,
                    fold=fold_number,
                    source_checkpoint_path=source_path,
                    selected_genes=selected_genes,
                )
        else:
            raw_model = (
                runner.model.module if isinstance(runner.model, DDP) else runner.model
            )
            raw_model.backbone.load_state_dict(saved_checkpoint["model_state_dict"])
            raw_model.eval()
        if runner.is_distributed:
            dist.barrier()

        if runner.is_master:
            full_fit, full_plot = extract_backbone_embeddings(
                runner, fit_loader, plot_loader
            )
            representations[f"full_ft__{model_key}"] = (full_fit, full_plot)
        if runner.is_distributed:
            dist.barrier()
        runner._cleanup_fold_state()

    if not runner.is_master:
        return
    representations["raw_all_genes"] = (
        dense_rows(adata.X, fit_indices),
        dense_rows(adata.X, plot_indices),
    )
    representations["raw_mad1199"] = (
        dense_rows(adata.X, fit_indices, selected_genes),
        dense_rows(adata.X, plot_indices, selected_genes),
    )
    coordinates = reduce_all(representations, args)
    save_coordinate_archive(
        output_path(args),
        labels=np.asarray(labels[plot_indices]).astype(str),
        sample_ids=adata.obs_names[plot_indices].astype(str).to_numpy(),
        coordinates=coordinates,
        metadata={
            "task": task,
            "target_type": "categorical",
            "label_order": label_order.tolist(),
            "source_data": str(data_path),
            "cv_fold_index_zero_based": args.fold_index,
            "cv_fold_number": fold_number,
            "cv_fold_manifest_path": str(getattr(runner, "_cv_fold_manifest_path", "")),
            "cv_fold_fingerprint": str(getattr(runner, "_cv_fold_fingerprint", "")),
            "gene_selection": "median_absolute_deviation_on_complete_training_fold",
            "selected_gene_count": int(selected_genes.size),
            "complete_training_count": int(len(train_indices)),
            "complete_held_out_count": int(len(test_indices)),
            "reducer_fit_count": int(len(fit_indices)),
            "plotted_held_out_count": int(len(plot_indices)),
            "initial_checkpoint_paths": {
                key: str(value) for key, value in INITIAL_CHECKPOINT_PATHS.items()
            },
            "full_ft_checkpoint_paths": {
                key: str(value) for key, value in checkpoint_paths.items()
            },
            "full_ft_epochs": epochs,
            "world_size": runner.world_size,
            "global_effective_batch_size": effective_batch_size,
            "reduction_seed": args.seed,
        },
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    path = output_path(args)
    if args.validate_only:
        validate_coordinate_archive(path)
        log.info("Validated coordinate output: %s", path)
        return
    try:
        from umap import UMAP  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The selected Singularity image does not provide umap-learn."
        ) from exc
    try:
        generate_classification(args)
        if int(os.environ.get("RANK", 0)) == 0:
            log.info("Saved coordinate output: %s", path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
