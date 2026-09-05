"""Build the thesis figure summarizing downstream validation-loss curves."""

from __future__ import annotations

from pathlib import Path
import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd


TASK_SPECS = (
    ("canc_type_class", "C5", False),
    ("canc_type_class_33", "C33", False),
    ("disease_class", "DIS", False),
    ("drug_resp", "DR", False),
    ("gene_essent", "GE", False),
    ("surv_pred", "PCS", False),
    ("surv_pred_binary", "BVS", False),
    ("surv_pred_survboard", "SB", True),
    ("deconv", "DEC", False),
)

MODE_SPECS = (
    ("head_only", "Head"),
    ("adapters", "Adapters"),
    ("full_ft", "Full"),
)

MODEL_STYLES = {
    "random_init": {
        "label": "RI",
        "color": "#6B6B6B",
        "linestyle": ":",
        "marker": "o",
    },
    "pretrain_sc": {
        "label": "PT-sc",
        "color": "#356B9A",
        "linestyle": "-",
        "marker": "o",
    },
    "preadapt_sc": {
        "label": "PA-sc",
        "color": "#356B9A",
        "linestyle": "--",
        "marker": "s",
    },
    "pretrain_bulk": {
        "label": "PT-bulk",
        "color": "#C46A3A",
        "linestyle": "-",
        "marker": "o",
    },
    "preadapt_bulk": {
        "label": "PA-bulk",
        "color": "#C46A3A",
        "linestyle": "--",
        "marker": "s",
    },
}


def _read_curve(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#")
    required = {"epoch", "validation_loss"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing {required - set(frame.columns)} in {path}")
    frame = frame[["epoch", "validation_loss"]].copy()
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["validation_loss"] = pd.to_numeric(
        frame["validation_loss"], errors="coerce"
    )
    frame = frame.dropna(subset=["epoch", "validation_loss"])
    if frame.empty:
        raise ValueError(f"No finite validation losses in {path}")
    return frame


def _load_summary(
    output_root: Path,
    task: str,
    mode: str,
    model: str,
    nested_by_cohort: bool,
) -> pd.DataFrame:
    filename = f"{task}_{mode}_{model}_training_curves.csv"
    if nested_by_cohort:
        paths = sorted(glob.glob(str(output_root / task / "*" / mode / filename)))
    else:
        candidate = output_root / task / mode / filename
        paths = [str(candidate)] if candidate.exists() else []
    if not paths:
        raise FileNotFoundError(
            f"Missing validation-loss curve for {task}/{mode}/{model}"
        )

    if nested_by_cohort:
        # Give each SurvBoard cohort equal weight after averaging its folds.
        cohort_means = []
        for path in paths:
            cohort = (
                _read_curve(path)
                .groupby("epoch", as_index=False)["validation_loss"]
                .mean()
            )
            cohort["cohort"] = Path(path).parents[1].name
            cohort_means.append(cohort)
        frame = pd.concat(cohort_means, ignore_index=True)
    else:
        frame = pd.concat([_read_curve(path) for path in paths], ignore_index=True)

    return (
        frame.groupby("epoch")["validation_loss"]
        .agg(["mean", "std"])
        .reset_index()
        .assign(std=lambda data: data["std"].fillna(0.0))
    )


def plot_downstream_convergence(
    output_root: str | Path,
    figure_dir: str | Path | None = None,
):
    """Plot one task per row and Head, Adapters, and Full by column."""
    output_root = Path(output_root).expanduser()
    summaries = {}
    for task, _task_label, nested_by_cohort in TASK_SPECS:
        for mode, _mode_label in MODE_SPECS:
            for model in MODEL_STYLES:
                summaries[(task, mode, model)] = _load_summary(
                    output_root, task, mode, model, nested_by_cohort
                )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.weight": "normal",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.65,
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            len(TASK_SPECS),
            len(MODE_SPECS),
            figsize=(6.8, 10.5),
            sharex=True,
            squeeze=False,
        )

        for row, (task, task_label, _nested_by_cohort) in enumerate(TASK_SPECS):
            row_lower = []
            row_upper = []
            for mode, _mode_label in MODE_SPECS:
                for model in MODEL_STYLES:
                    summary = summaries[(task, mode, model)]
                    means = summary["mean"].to_numpy(dtype=float)
                    stds = summary["std"].to_numpy(dtype=float)
                    row_lower.extend((means - stds).tolist())
                    row_upper.extend((means + stds).tolist())
            lower = float(np.nanmin(row_lower))
            upper = float(np.nanmax(row_upper))
            padding = max((upper - lower) * 0.06, abs(upper) * 0.005, 1e-8)
            lower -= padding
            upper += padding

            for column, (mode, mode_label) in enumerate(MODE_SPECS):
                ax = axes[row, column]
                for model, style in MODEL_STYLES.items():
                    summary = summaries[(task, mode, model)]
                    epochs = summary["epoch"].to_numpy(dtype=float)
                    means = summary["mean"].to_numpy(dtype=float)
                    stds = summary["std"].to_numpy(dtype=float)
                    ax.plot(
                        epochs,
                        means,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=3.0,
                        markeredgewidth=0,
                        markevery=(0, 4),
                        label=style["label"],
                    )
                    ax.fill_between(
                        epochs,
                        means - stds,
                        means + stds,
                        color=style["color"],
                        alpha=0.08,
                        linewidth=0,
                    )

                ax.set_xlim(1, 20)
                ax.set_ylim(lower, upper)
                ax.set_xticks([1, 5, 10, 15, 20])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                ax.yaxis.set_major_formatter(formatter)
                ax.grid(axis="y", color="#E5E5E5", linewidth=0.65)
                ax.set_axisbelow(True)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#555555")
                ax.spines["bottom"].set_color("#555555")
                ax.spines["left"].set_linewidth(0.8)
                ax.spines["bottom"].set_linewidth(0.8)
                ax.tick_params(
                    width=0.8, length=3.0, pad=1.5, color="#555555"
                )
                if column == 0:
                    ax.set_ylabel(
                        task_label,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=4,
                        fontweight="normal",
                    )
                    ax.yaxis.set_label_coords(-0.30, 0.5)
                else:
                    ax.tick_params(labelleft=False)
                if row == 0:
                    ax.set_title(mode_label, pad=5, fontweight="normal")
                if row < len(TASK_SPECS) - 1:
                    ax.tick_params(labelbottom=False)

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.0,
                markeredgewidth=0,
                linewidth=1.65,
                label=style["label"],
            )
            for style in MODEL_STYLES.values()
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.008),
            ncol=len(handles),
            frameon=False,
            handlelength=2.4,
            columnspacing=1.8,
            prop={"family": "DejaVu Sans", "size": 8.2, "weight": "normal"},
        )
        fig.supxlabel("Epoch", x=0.53, y=0.047, fontsize=9.5)
        fig.text(
            0.012,
            0.5,
            "Validation loss",
            rotation=90,
            va="center",
            ha="left",
            fontsize=9.5,
        )
        fig.subplots_adjust(
            left=0.15,
            right=0.995,
            top=0.985,
            bottom=0.09,
            hspace=0.35,
            wspace=0.12,
        )

        if figure_dir is not None:
            figure_dir = Path(figure_dir).expanduser()
            figure_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = figure_dir / "downstream_finetuning_convergence.pdf"
            png_path = figure_dir / "downstream_finetuning_convergence.png"
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
            fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
            print(f"Saved {pdf_path}")
            print(f"Saved {png_path}")

        return fig
