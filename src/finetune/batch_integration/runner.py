from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

from finetune.single_cell import (
    DISPLAY_NAMES,
    RAW_PCA_MODEL_KEY,
    SCBFM_ROOT,
    WORK_ROOT,
    FrozenSingleCellBenchmark,
    SingleCellDatasetBundle,
    plot_single_cell_benchmark_summary,
)
from run_provenance import complete_run_metadata, start_run_metadata, update_run_metadata

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

TASK_NAME = "batch_integration"
SCGPT_METRICS = (
    "nmi_cell",
    "ari_cell",
    "asw_cell",
    "asw_batch",
    "graph_connectivity",
)


class BatchIntegrationRunner:
    """Frozen scGPT-style batch integration on COVID-19 and Lung-Kim cells."""

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
            "Could not find batch integration config. "
            "Expected cfg.finetune.batch_integration."
        )

    def _output_dir(self) -> Path:
        variant = str(getattr(self.task_cfg, "output_variant", "zero_shot_scgpt"))
        return WORK_ROOT / "output" / TASK_NAME / variant

    @staticmethod
    def _orient_results(results: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
        expected = set(model_keys)
        row_overlap = len(expected.intersection(results.index.astype(str)))
        column_overlap = len(expected.intersection(results.columns.astype(str)))
        return results.T if column_overlap > row_overlap else results

    @staticmethod
    def _canonical_metric(name: str) -> str | None:
        normalized = " ".join(
            name.lower()
            .replace("_", " ")
            .replace("/", " ")
            .replace("-", " ")
            .split()
        )
        if "nmi" in normalized and (
            "label" in normalized or "cell" in normalized or "leiden" in normalized
        ):
            return "nmi_cell"
        if "ari" in normalized and (
            "label" in normalized or "cell" in normalized or "leiden" in normalized
        ):
            return "ari_cell"
        if ("silhouette" in normalized or "asw" in normalized) and "batch" in normalized:
            return "asw_batch"
        if (
            ("silhouette" in normalized or "asw" in normalized)
            and ("label" in normalized or "cell" in normalized)
            and "batch" not in normalized
        ):
            return "asw_cell"
        if "graph" in normalized and ("connect" in normalized or "conn" in normalized):
            return "graph_connectivity"
        return None

    @classmethod
    def _scgpt_result_rows(
        cls,
        results: pd.DataFrame,
        model_keys: list[str],
        dataset_key: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        results = cls._orient_results(results.copy(), model_keys)
        metric_type_row = next(
            (
                index
                for index in results.index
                if str(index).strip().lower() == "metric type"
            ),
            None,
        )
        if metric_type_row is not None:
            results = results.drop(index=metric_type_row)

        metric_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        for model_key in model_keys:
            if model_key not in results.index:
                raise ValueError(
                    f"scIB results for {dataset_key} do not contain '{model_key}'."
                )
            canonical: dict[str, float] = {}
            for source_metric in results.columns:
                metric = cls._canonical_metric(str(source_metric))
                if metric is None or metric in canonical:
                    continue
                value = pd.to_numeric(
                    pd.Series([results.loc[model_key, source_metric]]), errors="coerce"
                ).iloc[0]
                if pd.isna(value):
                    continue
                canonical[metric] = float(value)
                metric_rows.append(
                    {
                        "dataset": dataset_key,
                        "model": model_key,
                        "display_name": DISPLAY_NAMES.get(model_key, model_key),
                        "metric": metric,
                        "source_metric": str(source_metric),
                        "value": float(value),
                    }
                )
            missing = sorted(set(SCGPT_METRICS).difference(canonical))
            if missing:
                raise ValueError(
                    f"scIB results for {dataset_key}/{model_key} are missing the "
                    f"scGPT metrics {missing}. Observed columns={list(results.columns)}"
                )
            avg_bio = float(
                np.mean(
                    [
                        canonical["nmi_cell"],
                        canonical["ari_cell"],
                        canonical["asw_cell"],
                    ]
                )
            )
            avg_batch = float(
                np.mean(
                    [canonical["asw_batch"], canonical["graph_connectivity"]]
                )
            )
            summary_rows.append(
                {
                    "dataset": dataset_key,
                    "model": model_key,
                    "display_name": DISPLAY_NAMES.get(model_key, model_key),
                    "avg_bio": avg_bio,
                    "avg_batch": avg_batch,
                    "overall": 0.6 * avg_bio + 0.4 * avg_batch,
                }
            )
        return metric_rows, summary_rows

    def _run_scib(
        self,
        bundle: SingleCellDatasetBundle,
        embedding_keys: list[str],
    ) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
        try:
            from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
        except ImportError as exc:
            raise ImportError(
                "Batch integration requires scib-metrics==0.5.1. Build and use "
                "cluster/scbfm_single_cell.def."
            ) from exc
        benchmarker = Benchmarker(
            bundle.combined,
            batch_key=bundle.batch_key,
            label_key=bundle.label_key,
            embedding_obsm_keys=embedding_keys,
            n_jobs=int(getattr(self.task_cfg, "benchmark_n_jobs", 8)),
            bio_conservation_metrics=BioConservation(
                isolated_labels=False,
                nmi_ari_cluster_labels_kmeans=False,
                nmi_ari_cluster_labels_leiden=True,
                silhouette_label=True,
                clisi_knn=False,
            ),
            batch_correction_metrics=BatchCorrection(
                silhouette_batch=True,
                ilisi_knn=False,
                kbet_per_label=False,
                graph_connectivity=True,
                pcr_comparison=False,
            ),
            pre_integrated_embedding_obsm_key=embedding_keys[0],
        )
        benchmarker.benchmark()
        raw = benchmarker.get_results(min_max_scale=False, clean_names=True)
        metric_rows, summary_rows = self._scgpt_result_rows(
            raw, embedding_keys, bundle.key
        )
        return raw, metric_rows, summary_rows

    def _umap_coordinates(
        self,
        bundle: SingleCellDatasetBundle,
        embedding_keys: list[str],
    ) -> dict[str, np.ndarray]:
        coordinates: dict[str, np.ndarray] = {}
        for key in embedding_keys:
            neighbors_key = f"{key}_neighbors"
            sc.pp.neighbors(
                bundle.combined,
                n_neighbors=int(getattr(self.task_cfg, "umap_n_neighbors", 15)),
                use_rep=key,
                key_added=neighbors_key,
            )
            sc.tl.umap(
                bundle.combined,
                neighbors_key=neighbors_key,
                random_state=int(getattr(self.task_cfg, "random_seed", 42)),
            )
            coordinates[key] = np.asarray(
                bundle.combined.obsm["X_umap"], dtype=np.float32
            ).copy()
        return coordinates

    def _plot_umaps(
        self,
        path: Path,
        bundle: SingleCellDatasetBundle,
        coordinates: dict[str, np.ndarray],
    ) -> None:
        labels = pd.Categorical(bundle.combined.obs[bundle.label_key].astype(str))
        batches = pd.Categorical(bundle.combined.obs[bundle.batch_key].astype(str))
        keys = list(coordinates)
        figure, axes = plt.subplots(
            2, len(keys), figsize=(4.0 * len(keys), 7.5), squeeze=False
        )
        label_cmap = plt.get_cmap("tab20")
        batch_cmap = plt.get_cmap("gist_ncar")
        for column, key in enumerate(keys):
            xy = coordinates[key]
            axes[0, column].scatter(
                xy[:, 0],
                xy[:, 1],
                c=labels.codes,
                cmap=label_cmap,
                s=3,
                alpha=0.7,
                linewidths=0,
            )
            axes[1, column].scatter(
                xy[:, 0],
                xy[:, 1],
                c=batches.codes,
                cmap=batch_cmap,
                s=3,
                alpha=0.7,
                linewidths=0,
            )
            axes[0, column].set_title(DISPLAY_NAMES.get(key, key))
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                axes[row, column].set_xlabel("UMAP 1")
        axes[0, 0].set_ylabel("Cell type\nUMAP 2")
        axes[1, 0].set_ylabel("Batch\nUMAP 2")
        if len(labels.categories) <= 24:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=5,
                    markerfacecolor=label_cmap(
                        index / max(1, len(labels.categories) - 1)
                    ),
                    markeredgecolor="none",
                    label=category,
                )
                for index, category in enumerate(labels.categories)
            ]
            figure.legend(
                handles=handles,
                loc="lower center",
                ncol=min(8, len(handles)),
                frameon=False,
            )
            bottom = 0.12
        else:
            bottom = 0.03
        figure.suptitle(f"{bundle.display_name}: zero-shot batch integration")
        figure.tight_layout(rect=(0, bottom, 1, 0.96))
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
        plot_single_cell_benchmark_summary(
            path,
            rows,
            metric="overall",
            ylabel="Overall integration score",
            title="Zero-shot batch integration: overall score",
        )

    @staticmethod
    def _plot_forgetting(path: Path, rows: list[dict[str, object]]) -> None:
        frame = pd.DataFrame(rows)
        frame = frame[
            (frame["comparison"] == "sc_preadaptation")
            & (frame["metric"] == "overall")
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
        axis.set_title("Batch-integration retention: overall-score change")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)

    def _start_metadata(self, checkpoint_paths: dict[str, str]) -> None:
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "batch_integration_config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8"
        )
        self._run_metadata_path = output_dir / "batch_integration_run_metadata.json"
        start_run_metadata(
            self._run_metadata_path,
            {
                "task": "finetune.batch_integration",
                "protocol": "scgpt_frozen_zero_shot_batch_integration",
                "datasets": self.benchmark.dataset_keys(),
                "label_driven_training": False,
                "cross_validation_folds": 0,
                "embedding": "l2_normalized_final_cls",
                "selected_gene_count": self.benchmark.selected_gene_count,
                "gene_selection": "reference_only_mad",
                "scgpt_metric_definition": {
                    "avg_bio": "mean(nmi_cell, ari_cell, asw_cell)",
                    "avg_batch": "mean(asw_batch, graph_connectivity)",
                    "overall": "0.6 * avg_bio + 0.4 * avg_batch",
                },
                "scib_metrics_version": "0.5.1",
                "world_size": self.benchmark.world_size,
            },
            checkpoint_paths=checkpoint_paths,
            repo_dir=SCBFM_ROOT,
        )

    def run(self) -> dict[str, object]:
        self.benchmark.setup_runtime()
        try:
            checkpoint_paths = self.benchmark.checkpoint_paths()
            output_dir = self._output_dir()
            if self.benchmark.is_master:
                self._start_metadata(checkpoint_paths)

            all_metric_rows: list[dict[str, object]] = []
            all_summary_rows: list[dict[str, object]] = []
            for dataset_key in self.benchmark.dataset_keys():
                bundle = self.benchmark.load_dataset(dataset_key)
                embeddings = self.benchmark.extract_model_embeddings(
                    bundle, checkpoint_paths
                )
                if self.benchmark.is_master:
                    if bool(getattr(self.task_cfg, "include_raw_pca", True)):
                        embeddings[RAW_PCA_MODEL_KEY] = self.benchmark.raw_pca_embeddings(
                            bundle, fit_reference_only=False
                        )
                    embedding_keys = list(embeddings)
                    for key, values in embeddings.items():
                        bundle.combined.obsm[key] = values
                    dataset_dir = output_dir / dataset_key
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    bundle.manifest.to_csv(
                        dataset_dir / "batch_integration_selected_genes.csv", index=False
                    )
                    bundle.combined.uns = {
                        "single_cell_protocol": {
                            "dataset": dataset_key,
                            "batch_key": bundle.batch_key,
                            "label_key": bundle.label_key,
                            "embedding_keys": embedding_keys,
                            "zero_shot": True,
                            "gene_selection_scope": "reference_cells_only",
                        }
                    }
                    bundle.combined.write_h5ad(
                        dataset_dir / "batch_integration_embeddings.h5ad",
                        compression="gzip",
                    )
                    raw, metric_rows, summary_rows = self._run_scib(
                        bundle, embedding_keys
                    )
                    raw.to_csv(dataset_dir / "batch_integration_scib_raw.csv")
                    all_metric_rows.extend(metric_rows)
                    all_summary_rows.extend(summary_rows)
                    coordinates = self._umap_coordinates(bundle, embedding_keys)
                    coordinate_rows: list[dict[str, object]] = []
                    for model, xy in coordinates.items():
                        for index, (x_value, y_value) in enumerate(xy):
                            coordinate_rows.append(
                                {
                                    "dataset": dataset_key,
                                    "model": model,
                                    "cell_id": str(bundle.combined.obs_names[index]),
                                    "batch": str(
                                        bundle.combined.obs.iloc[index][bundle.batch_key]
                                    ),
                                    "cell_type": str(
                                        bundle.combined.obs.iloc[index][bundle.label_key]
                                    ),
                                    "umap_1": float(x_value),
                                    "umap_2": float(y_value),
                                }
                            )
                    self.benchmark.write_rows(
                        dataset_dir / "batch_integration_umap_coordinates.csv",
                        coordinate_rows,
                    )
                    self._plot_umaps(
                        dataset_dir / "batch_integration_umap.png",
                        bundle,
                        coordinates,
                    )
                if self.benchmark.is_distributed:
                    import torch.distributed as dist

                    dist.barrier()

            if not self.benchmark.is_master:
                return {"status": "distributed_worker_complete"}

            summary_rows = self.benchmark.append_macro_average(
                all_summary_rows,
                metric_fields=["avg_bio", "avg_batch", "overall"],
            )
            summary_path = output_dir / "batch_integration_summary.csv"
            metrics_path = output_dir / "batch_integration_metrics.csv"
            self.benchmark.write_rows(metrics_path, all_metric_rows)
            self.benchmark.write_rows(summary_path, summary_rows)
            forgetting_rows = self.benchmark.forgetting_rows(
                summary_rows,
                metric_fields=["avg_bio", "avg_batch", "overall"],
            )
            forgetting_path = output_dir / "batch_integration_forgetting.csv"
            self.benchmark.write_rows(forgetting_path, forgetting_rows)
            self._plot_summary(output_dir / "batch_integration_summary.png", summary_rows)
            self._plot_forgetting(
                output_dir / "batch_integration_forgetting.png", forgetting_rows
            )

            if self._run_metadata_path is None:
                raise RuntimeError("Run metadata was not initialized.")
            update_run_metadata(
                self._run_metadata_path,
                {
                    "mapping": self.benchmark.mapping_stats,
                    "external_scgpt_loading": self.benchmark.external_scgpt_load_reports,
                    "summary_path": str(summary_path.resolve()),
                    "forgetting_path": str(forgetting_path.resolve()),
                },
            )
            complete_run_metadata(self._run_metadata_path, summary_path)
            return {
                "summary": summary_rows,
                "metrics_path": str(metrics_path),
                "summary_path": str(summary_path),
                "forgetting_path": str(forgetting_path),
            }
        finally:
            self.benchmark.close_runtime()
