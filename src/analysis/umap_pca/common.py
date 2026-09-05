from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MODEL_KEYS = (
    "random_init",
    "pretrain_sc",
    "preadapt_sc",
    "pretrain_bulk",
    "preadapt_bulk",
)
FINETUNE_MODES = ("head_only", "full_ft")
RAW_REPRESENTATION_KEYS = (
    "raw_all_genes",
    "raw_mad1199",
)
REPRESENTATION_KEYS = tuple(
    f"{mode}__{model_key}"
    for model_key in MODEL_KEYS
    for mode in FINETUNE_MODES
) + RAW_REPRESENTATION_KEYS


def select_plot_indices(
    labels: np.ndarray,
    max_points: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select a reproducible, approximately stratified plotting subset."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Subsampling labels must be one-dimensional.")
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    indices = np.arange(labels.size, dtype=np.int64)
    if indices.size <= max_points:
        return indices

    _, counts = np.unique(labels, return_counts=True)
    can_stratify = (
        counts.min() >= 2
        and max_points >= counts.size
        and labels.size - max_points >= counts.size
    )
    selected, _ = train_test_split(
        indices,
        train_size=max_points,
        random_state=seed,
        shuffle=True,
        stratify=labels if can_stratify else None,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))


def dense_rows(
    matrix,
    row_indices: np.ndarray,
    column_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Materialize only the requested rows and columns as float32."""
    selected = matrix[np.asarray(row_indices, dtype=np.int64)]
    if column_indices is not None:
        selected = selected[:, np.asarray(column_indices, dtype=np.int64)]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float32)


def _pad_two_dimensions(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 2:
        return values
    if values.shape[1] == 1:
        return np.column_stack((values[:, 0], np.zeros(values.shape[0])))
    raise ValueError(f"Expected one or two coordinates, got shape {values.shape}.")


def fit_reductions(
    train_values: np.ndarray,
    test_values: np.ndarray,
    *,
    seed: int,
    pca_components: int = 50,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.3,
) -> dict[str, np.ndarray | int | float]:
    """Fit scaling, PCA, and UMAP on training data and transform held-out data."""
    train_values = np.asarray(train_values, dtype=np.float32)
    test_values = np.asarray(test_values, dtype=np.float32)
    if train_values.ndim != 2 or test_values.ndim != 2:
        raise ValueError("Reduction inputs must be two-dimensional matrices.")
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError("Training and held-out matrices must have the same feature count.")
    if train_values.shape[0] < 3 or test_values.shape[0] < 1:
        raise ValueError("At least three training samples and one held-out sample are required.")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError("Reduction inputs contain non-finite values.")

    scaler = StandardScaler(copy=False)
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)
    n_components = min(
        int(pca_components),
        train_scaled.shape[0] - 1,
        train_scaled.shape[1],
    )
    if n_components < 1:
        raise ValueError("PCA requires at least one component.")
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized" if n_components < min(train_scaled.shape) else "auto",
        random_state=seed,
    )
    train_pca = pca.fit_transform(train_scaled).astype(np.float32, copy=False)
    test_pca = pca.transform(test_scaled).astype(np.float32, copy=False)

    try:
        from umap import UMAP
    except ImportError as exc:  # pragma: no cover - exercised in the cluster image
        raise RuntimeError(
            "umap-learn is required in the cluster environment for this analysis."
        ) from exc

    n_neighbors = min(int(umap_neighbors), train_pca.shape[0] - 1)
    if n_neighbors < 2:
        raise ValueError("UMAP requires at least two neighbors.")
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(umap_min_dist),
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
        low_memory=True,
    )
    reducer.fit(train_pca)
    test_umap = reducer.transform(test_pca).astype(np.float32, copy=False)
    pca_two = _pad_two_dimensions(test_pca[:, :2]).astype(np.float32, copy=False)
    explained = np.zeros(2, dtype=np.float32)
    explained[: min(2, pca.explained_variance_ratio_.size)] = (
        pca.explained_variance_ratio_[:2]
    )
    return {
        "pca": pca_two,
        "umap": test_umap,
        "pca_explained_variance_ratio": explained,
        "pca_components": int(n_components),
        "umap_neighbors": int(n_neighbors),
    }


def save_coordinate_archive(
    path: Path,
    *,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    coordinates: dict[str, dict[str, np.ndarray | int | float]],
    metadata: dict[str, object],
    continuous_values: np.ndarray | None = None,
) -> None:
    """Write a compact, validated coordinate archive and adjacent metadata JSON."""
    path = Path(path)
    missing = set(REPRESENTATION_KEYS).difference(coordinates)
    if missing:
        raise ValueError(f"Missing representation coordinates: {sorted(missing)}")
    labels = np.asarray(labels).astype(str)
    sample_ids = np.asarray(sample_ids).astype(str)
    if labels.shape != sample_ids.shape:
        raise ValueError("labels and sample_ids must have identical one-dimensional shapes.")

    payload: dict[str, np.ndarray] = {
        "labels": labels,
        "sample_ids": sample_ids,
        "representation_keys": np.asarray(REPRESENTATION_KEYS),
    }
    if continuous_values is not None:
        continuous_values = np.asarray(continuous_values, dtype=np.float32)
        if continuous_values.shape != labels.shape:
            raise ValueError("continuous_values must match the labels shape.")
        payload["continuous_values"] = continuous_values
    reduction_metadata: dict[str, object] = {}
    for key in REPRESENTATION_KEYS:
        result = coordinates[key]
        for method in ("pca", "umap"):
            values = np.asarray(result[method], dtype=np.float32)
            if values.shape != (labels.size, 2) or not np.isfinite(values).all():
                raise ValueError(
                    f"{method} coordinates for {key} have invalid shape or values: "
                    f"{values.shape}."
                )
            payload[f"{method}__{key}"] = values
        payload[f"pca_variance__{key}"] = np.asarray(
            result["pca_explained_variance_ratio"], dtype=np.float32
        )
        reduction_metadata[key] = {
            "pca_components": int(result["pca_components"]),
            "umap_neighbors": int(result["umap_neighbors"]),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary_path, **payload)
    os.replace(temporary_path, path)
    metadata = {**metadata, "reductions": reduction_metadata}
    metadata_path = path.with_suffix(".json")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_metadata, metadata_path)
    validate_coordinate_archive(path)


def validate_coordinate_archive(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Coordinate archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = tuple(archive["representation_keys"].astype(str).tolist())
        if keys != REPRESENTATION_KEYS:
            raise ValueError(f"Unexpected representation order in {path}: {keys}")
        n_samples = archive["labels"].shape[0]
        if archive["sample_ids"].shape != (n_samples,):
            raise ValueError(f"Invalid sample_ids shape in {path}.")
        for key in REPRESENTATION_KEYS:
            for method in ("pca", "umap"):
                values = archive[f"{method}__{key}"]
                if values.shape != (n_samples, 2) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {method} coordinates for {key} in {path}.")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Coordinate metadata not found: {metadata_path}")
