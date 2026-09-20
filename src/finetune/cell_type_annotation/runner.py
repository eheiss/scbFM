from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from paths import output_root
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors

from finetune.single_cell import (
    DISPLAY_NAMES,
    RAW_PCA_MODEL_KEY,
    SCBFM_ROOT,
    FrozenSingleCellBenchmark,
    SingleCellDatasetBundle,
    plot_single_cell_benchmark_summary,
)
from run_provenance import complete_run_metadata, start_run_metadata, update_run_metadata

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

TASK_NAME = "cell_type_annotation"
ANNOTATION_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


class CellTypeAnnotationRunner:
    """scGPT-style zero-shot reference mapping with frozen cell embeddings."""

    task_name = TASK_NAME
    config_node = TASK_NAME

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.task_cfg = self._resolve_task_cfg(cfg)
        self.benchmark = FrozenSingleCellBenchmark(cfg, self.task_cfg)
        self._run_metadata_path: Path | None = None

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and TASK_NAME in cfg.finetune:
            return cfg.finetune[TASK_NAME]
        raise ValueError(
            "Could not find cell type annotation config. "
            "Expected cfg.finetune.cell_type_annotation."
        )

    def _output_dir(self) -> Path:
        variant = str(getattr(self.task_cfg, "output_variant", "zero_shot_scgpt"))
        return output_root(self.cfg) / TASK_NAME / variant

    @staticmethod
    def _majority_vote(neighbor_labels: np.ndarray) -> str:
        counts = Counter(str(label) for label in neighbor_labels)
        maximum = max(counts.values())
        # scGPT uses majority voting. Sorting only defines deterministic behavior
        # for an otherwise unspecified exact tie.
        return sorted(label for label, count in counts.items() if count == maximum)[0]

    @classmethod
    def _predict_knn(
        cls,
        reference_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        reference_labels: np.ndarray,
        *,
        n_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if reference_embeddings.shape[0] < n_neighbors:
            raise ValueError(
                f"Reference has {reference_embeddings.shape[0]} cells but "
                f"n_neighbors={n_neighbors}."
            )
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        neighbors.fit(reference_embeddings)
        distances, indices = neighbors.kneighbors(query_embeddings, return_distance=True)
        predictions = np.asarray(
            [cls._majority_vote(reference_labels[row]) for row in indices], dtype=object
        )
        return predictions, distances

    @staticmethod
    def _evaluate_predictions(
        *,
        dataset_key: str,
        model_key: str,
        truth: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        labels = sorted(set(truth.astype(str)).union(predictions.astype(str)))
        precision, recall, f1, support = precision_recall_fscore_support(
            truth,
            predictions,
            labels=labels,
            average=None,
            zero_division=0,
        )
        per_class_rows = [
            {
                "dataset": dataset_key,
                "model": model_key,
                "display_name": DISPLAY_NAMES.get(model_key, model_key),
                "cell_type": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ]
        summary = {
            "dataset": dataset_key,
            "model": model_key,
            "display_name": DISPLAY_NAMES.get(model_key, model_key),
            "accuracy": float(accuracy_score(truth, predictions)),
            "macro_precision": float(
                precision_score(truth, predictions, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(truth, predictions, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(truth, predictions, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(truth, predictions, average="weighted", zero_division=0)
            ),
            "query_cell_count": int(truth.size),
            "cell_type_count": len(labels),
        }
        return summary, per_class_rows

    @staticmethod
    def _plot_confusion_matrix(
        path: Path,
        *,
        dataset_name: str,
        model_key: str,
        truth: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        labels = sorted(set(truth.astype(str)).union(predictions.astype(str)))
        matrix = confusion_matrix(truth, predictions, labels=labels, normalize="true")
        size = max(7.0, min(16.0, 0.5 * len(labels) + 4.0))
        figure, axis = plt.subplots(figsize=(size, size))
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xlabel("Predicted cell type")
        axis.set_ylabel("True cell type")
        axis.set_title(
            f"{dataset_name}: {DISPLAY_NAMES.get(model_key, model_key)}"
        )
        if len(labels) <= 18:
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if matrix[row, column] > 0.55 else "black",
                    )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
        plot_single_cell_benchmark_summary(
            path,
            rows,
            metric="macro_f1",
            ylabel="Macro F1",
            title="Zero-shot cell-type annotation: Macro F1",
        )

    @staticmethod
    def _plot_forgetting(path: Path, rows: list[dict[str, object]]) -> None:
        frame = pd.DataFrame(rows)
        frame = frame[
            (frame["comparison"] == "sc_preadaptation")
            & (frame["metric"] == "macro_f1")
        ]
        if frame.empty:
            return
        x = np.arange(len(frame))
        values = frame["delta_after_minus_before"].to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        axis.bar(x, values, color=np.where(values < 0, "#C44E52", "#55A868"))
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(x, frame["dataset"].astype(str))
        axis.set_ylabel("Preadapt sc minus pretrain sc")
        axis.set_title("Cell-annotation retention: Macro F1 change")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)

    def _start_metadata(self, checkpoint_paths: dict[str, str]) -> None:
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cell_type_annotation_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8"
        )
        self._run_metadata_path = output_dir / "cell_type_annotation_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": "finetune.cell_type_annotation",
                "protocol": "scgpt_frozen_zero_shot_reference_mapping",
                "datasets": self.benchmark.dataset_keys(),
                "label_driven_training": False,
                "cross_validation_folds": 0,
                "embedding": "l2_normalized_final_cls",
                "selected_gene_count": self.benchmark.selected_gene_count,
                "gene_selection": "reference_only_mad",
                "reference_mapping": "euclidean_10_nearest_neighbor_majority_vote",
                "primary_metric": "macro_f1",
                "world_size": self.benchmark.world_size,
            },
            checkpoint_paths=checkpoint_paths,
            repo_dir=SCBFM_ROOT,
        )

    def _evaluate_dataset(
        self,
        output_dir: Path,
        bundle: SingleCellDatasetBundle,
        embeddings: dict[str, np.ndarray],
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        dataset_dir = output_dir / bundle.key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        bundle.manifest.to_csv(
            dataset_dir / "cell_type_annotation_selected_genes.csv", index=False
        )
        n_reference = bundle.reference.n_obs
        reference_labels = bundle.reference.obs[bundle.label_key].astype(str).to_numpy()
        truth = bundle.query.obs[bundle.label_key].astype(str).to_numpy()
        query_ids = bundle.query.obs_names.astype(str).to_numpy()
        n_neighbors = int(getattr(self.task_cfg, "n_neighbors", 10))

        summary_rows: list[dict[str, object]] = []
        per_class_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        for model_key, values in embeddings.items():
            reference_embeddings = values[:n_reference]
            query_embeddings = values[n_reference:]
            predictions, distances = self._predict_knn(
                reference_embeddings,
                query_embeddings,
                reference_labels,
                n_neighbors=n_neighbors,
            )
            summary, per_class = self._evaluate_predictions(
                dataset_key=bundle.key,
                model_key=model_key,
                truth=truth,
                predictions=predictions,
            )
            summary_rows.append(summary)
            per_class_rows.extend(per_class)
            prediction_rows.extend(
                {
                    "dataset": bundle.key,
                    "model": model_key,
                    "cell_id": query_ids[index],
                    "true_cell_type": str(truth[index]),
                    "predicted_cell_type": str(predictions[index]),
                    "mean_neighbor_distance": float(distances[index].mean()),
                    "nearest_neighbor_distance": float(distances[index].min()),
                    "correct": bool(predictions[index] == truth[index]),
                }
                for index in range(truth.size)
            )
            self._plot_confusion_matrix(
                dataset_dir / f"cell_type_annotation_confusion_{model_key}.png",
                dataset_name=bundle.display_name,
                model_key=model_key,
                truth=truth,
                predictions=predictions,
            )

        for key, values in embeddings.items():
            bundle.combined.obsm[key] = values
        bundle.combined.uns = {
            "single_cell_protocol": {
                "dataset": bundle.key,
                "label_key": bundle.label_key,
                "batch_key": bundle.batch_key,
                "embedding_keys": list(embeddings),
                "reference_cell_count": int(n_reference),
                "query_cell_count": int(bundle.query.n_obs),
                "n_neighbors": n_neighbors,
                "zero_shot": True,
            }
        }
        bundle.combined.write_h5ad(
            dataset_dir / "cell_type_annotation_embeddings.h5ad", compression="gzip"
        )
        return summary_rows, per_class_rows, prediction_rows

    def run(self) -> dict[str, object]:
        self.benchmark.setup_runtime()
        try:
            checkpoint_paths = self.benchmark.checkpoint_paths()
            output_dir = self._output_dir()
            if self.benchmark.is_master:
                self._start_metadata(checkpoint_paths)

            all_summary_rows: list[dict[str, object]] = []
            all_per_class_rows: list[dict[str, object]] = []
            all_prediction_rows: list[dict[str, object]] = []
            for dataset_key in self.benchmark.dataset_keys():
                bundle = self.benchmark.load_dataset(dataset_key)
                embeddings = self.benchmark.extract_model_embeddings(
                    bundle, checkpoint_paths
                )
                if self.benchmark.is_master:
                    if bool(getattr(self.task_cfg, "include_raw_pca", True)):
                        embeddings[RAW_PCA_MODEL_KEY] = self.benchmark.raw_pca_embeddings(
                            bundle, fit_reference_only=True
                        )
                    summary, per_class, predictions = self._evaluate_dataset(
                        output_dir, bundle, embeddings
                    )
                    all_summary_rows.extend(summary)
                    all_per_class_rows.extend(per_class)
                    all_prediction_rows.extend(predictions)
                if self.benchmark.is_distributed:
                    import torch.distributed as dist

                    dist.barrier()

            if not self.benchmark.is_master:
                return {"status": "distributed_worker_complete"}

            summary_rows = self.benchmark.append_macro_average(
                all_summary_rows, metric_fields=list(ANNOTATION_METRICS)
            )
            summary_path = output_dir / "cell_type_annotation_summary.csv"
            per_class_path = output_dir / "cell_type_annotation_per_class.csv"
            predictions_path = output_dir / "cell_type_annotation_predictions.csv"
            self.benchmark.write_rows(summary_path, summary_rows)
            self.benchmark.write_rows(per_class_path, all_per_class_rows)
            self.benchmark.write_rows(predictions_path, all_prediction_rows)
            forgetting_rows = self.benchmark.forgetting_rows(
                summary_rows, metric_fields=list(ANNOTATION_METRICS)
            )
            forgetting_path = output_dir / "cell_type_annotation_forgetting.csv"
            self.benchmark.write_rows(forgetting_path, forgetting_rows)
            self._plot_summary(output_dir / "cell_type_annotation_summary.png", summary_rows)
            self._plot_forgetting(
                output_dir / "cell_type_annotation_forgetting.png", forgetting_rows
            )

            if self._run_metadata_path is None:
                raise RuntimeError("Run metadata was not initialized.")
            update_run_metadata(
                self._run_metadata_path,
                {
                    "mapping": self.benchmark.mapping_stats,
                    "external_scgpt_loading": self.benchmark.external_scgpt_load_reports,
                    "summary_path": str(summary_path.resolve()),
                    "predictions_path": str(predictions_path.resolve()),
                    "forgetting_path": str(forgetting_path.resolve()),
                },
            )
            complete_run_metadata(self._run_metadata_path, summary_path)
            return {
                "summary": summary_rows,
                "summary_path": str(summary_path),
                "per_class_path": str(per_class_path),
                "predictions_path": str(predictions_path),
                "forgetting_path": str(forgetting_path),
            }
        finally:
            self.benchmark.close_runtime()
