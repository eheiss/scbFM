from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.umap_pca.common import (
    MODEL_KEYS,
    REPRESENTATION_KEYS,
    fit_reductions,
    save_coordinate_archive,
    select_plot_indices,
    validate_coordinate_archive,
)

REPO = Path(__file__).resolve().parents[1]


class _FakeUMAP:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, values: np.ndarray) -> "_FakeUMAP":
        self.fit_shape = values.shape
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return values[:, :2]


def test_plot_subsampling_is_deterministic_and_stratified() -> None:
    labels = np.repeat(np.asarray(["a", "b", "c"]), [60, 30, 10])
    first = select_plot_indices(labels, 30, seed=42)
    second = select_plot_indices(labels, 30, seed=42)

    assert np.array_equal(first, second)
    assert first.shape == (30,)
    assert set(labels[first]) == {"a", "b", "c"}


def test_reductions_fit_training_and_transform_held_out(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=_FakeUMAP))
    rng = np.random.default_rng(42)
    result = fit_reductions(
        rng.normal(size=(40, 8)),
        rng.normal(size=(13, 8)),
        seed=42,
        pca_components=5,
        umap_neighbors=10,
    )

    assert result["pca"].shape == (13, 2)
    assert result["umap"].shape == (13, 2)
    assert result["pca_components"] == 5
    assert result["umap_neighbors"] == 10


def test_coordinate_archive_round_trip(tmp_path: Path) -> None:
    n_samples = 11
    coordinates = {
        key: {
            "pca": np.full((n_samples, 2), index, dtype=np.float32),
            "umap": np.full((n_samples, 2), index + 0.5, dtype=np.float32),
            "pca_explained_variance_ratio": np.asarray([0.4, 0.2]),
            "pca_components": 5,
            "umap_neighbors": 4,
        }
        for index, key in enumerate(REPRESENTATION_KEYS)
    }
    path = tmp_path / "coordinates.npz"
    save_coordinate_archive(
        path,
        labels=np.asarray(["a"] * n_samples),
        sample_ids=np.asarray([f"sample-{index}" for index in range(n_samples)]),
        continuous_values=np.linspace(0.0, 1.0, n_samples),
        coordinates=coordinates,
        metadata={"task": "test"},
    )

    validate_coordinate_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        assert tuple(archive["representation_keys"].astype(str)) == REPRESENTATION_KEYS
        assert archive["umap__full_ft__preadapt_bulk"].shape == (n_samples, 2)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["task"] == "test"
    assert metadata["reductions"]["head_only__random_init"]["pca_components"] == 5


def test_representation_order_matches_six_by_two_layout() -> None:
    expected = tuple(
        f"{mode}__{model_key}"
        for model_key in MODEL_KEYS
        for mode in ("head_only", "full_ft")
    ) + ("raw_all_genes", "raw_mad1199")

    assert REPRESENTATION_KEYS == expected
    assert len(REPRESENTATION_KEYS) == 12




def test_notebook_renders_one_six_by_two_figure_per_task_and_method() -> None:
    notebook = json.loads(
        (REPO / "src" / "analysis" / "umap_pca.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "axes = np.empty((6, 2)" in source
    assert "('canc_type_class', 'Five-type cancer classification')" in source
    assert "('canc_type_class_33', '33-type cancer classification')" in source
    assert "('disease_class', 'Disease classification')" in source
    assert "surv_pred_binary" not in source
    assert "drug_resp" not in source
    assert "cell_type_annotation" not in source


def test_coordinate_archive_rejects_missing_representation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing representation"):
        save_coordinate_archive(
            tmp_path / "coordinates.npz",
            labels=np.asarray(["a"]),
            sample_ids=np.asarray(["sample"]),
            coordinates={},
            metadata={},
        )
