import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "results" / "ablation_energy_metrics.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "ablation_plots1"

CHAMFER_GT_TO_RECONSTRUCTION = "chamfer_gt_to_reconstruction"
CHAMFER_RECONSTRUCTION_TO_GT = "chamfer_reconstruction_to_gt"
CHAMFER_TOTAL = "chamfer_total"
MISCLASSIFICATION_ERROR = "misclassification_error"
MODEL_COUNT = "model_count"
LOCAL_OPTIMIZATION_ITERATIONS = "local_optimization_iterations"

NUMERIC_COLUMNS = (
    CHAMFER_GT_TO_RECONSTRUCTION,
    CHAMFER_RECONSTRUCTION_TO_GT,
    CHAMFER_TOTAL,
    MISCLASSIFICATION_ERROR,
    MODEL_COUNT,
    LOCAL_OPTIMIZATION_ITERATIONS,
)
REQUIRED_COLUMNS = ("energy", *NUMERIC_COLUMNS)

PREFERRED_ENERGY_ORDER = (
    "GcRansacEnergy",
    "OnlyUnaryGairStrategy",
    "ConstantCoherenceGairEnergy",
    "GairThresholdEnergy",
    "FullGairEnergycoh09",
)
ENERGY_LABELS = {
    "FullGairEnergy": "GAIR(ours)",
    "FullGairEnergycoh09": "GAIR(ours)",
    "ConstantCoherenceGairEnergy": "GAIR C=1",
    "GairThresholdEnergy": "GAIR binary",
    "OnlyUnaryGairStrategy": "GAIR unary",
    "GcRansacEnergy": "GC-RANSAC",
    "OldPaperGairEnergy": "Old-paper GAIR",
}
ENERGY_COLORS = {
    "FullGairEnergy": "#F8961E",
    "FullGairEnergycoh09": "#F8961E",
    "ConstantCoherenceGairEnergy": "#64CCC5",
    "GairThresholdEnergy": "#2A9D8F",
    "OnlyUnaryGairStrategy": "#7A5195",
    "GcRansacEnergy": "#64748B",
    "OldPaperGairEnergy": "#D1495B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots from the GAIR energy ablation metrics.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input metrics CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and the summary CSV are saved.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Plot output format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Resolution used for raster output.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#94A3B8",
            "axes.labelcolor": "#1E293B",
            "axes.titlecolor": "#0F172A",
            "axes.titleweight": "normal",
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.55,
            "font.family": "STIXGeneral",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def load_metrics(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {input_path}")

    frame = pd.read_csv(input_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Metrics CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError(f"Metrics CSV contains no rows: {input_path}")
    if frame["energy"].isna().any():
        raise ValueError("The energy column contains empty values")

    for column in NUMERIC_COLUMNS:
        original_values = frame[column]
        numeric_values = pd.to_numeric(original_values, errors="coerce")
        invalid_values = numeric_values.isna() & original_values.notna()
        if invalid_values.any():
            invalid_rows = (np.flatnonzero(invalid_values.to_numpy()) + 2).tolist()
            raise ValueError(
                f"Column {column} contains invalid values at CSV rows "
                f"{invalid_rows}"
            )
        frame[column] = numeric_values.replace([np.inf, -np.inf], np.nan)

    finite_chamfer_rows = frame[
        [
            CHAMFER_GT_TO_RECONSTRUCTION,
            CHAMFER_RECONSTRUCTION_TO_GT,
            CHAMFER_TOTAL,
        ]
    ].notna().all(axis=1)
    if finite_chamfer_rows.any():
        directional_sum = (
            frame.loc[finite_chamfer_rows, CHAMFER_GT_TO_RECONSTRUCTION]
            + frame.loc[finite_chamfer_rows, CHAMFER_RECONSTRUCTION_TO_GT]
        )
        stored_total = frame.loc[finite_chamfer_rows, CHAMFER_TOTAL]
        if not np.allclose(directional_sum, stored_total):
            raise ValueError(
                "chamfer_total is inconsistent with the two directional distances"
            )

    non_finite_count = int(frame[list(NUMERIC_COLUMNS)].isna().any(axis=1).sum())
    if non_finite_count:
        print(
            f"Warning: {non_finite_count} rows contain non-finite metrics; "
            "those values will be omitted from the affected plots."
        )
    return frame


def get_energy_order(frame: pd.DataFrame) -> list[str]:
    observed_energies = list(dict.fromkeys(frame["energy"].astype(str)))
    preferred = [
        energy
        for energy in PREFERRED_ENERGY_ORDER
        if energy in observed_energies
    ]
    remaining = sorted(set(observed_energies) - set(preferred))
    return preferred + remaining


def get_energy_label(energy: str) -> str:
    return ENERGY_LABELS.get(energy, energy)


def get_energy_colors(energy_order: list[str]) -> dict[str, str | tuple]:
    fallback_colors = plt.get_cmap("tab10").colors
    colors: dict[str, str | tuple] = {}
    for index, energy in enumerate(energy_order):
        colors[energy] = ENERGY_COLORS.get(
            energy,
            fallback_colors[index % len(fallback_colors)],
        )
    return colors


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    output_format: str,
    dpi: int,
) -> Path:
    output_path = output_dir / f"{stem}.{output_format}"
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def plot_metric_distributions(
    frame: pd.DataFrame,
    energy_order: list[str],
    colors: dict[str, str | tuple],
    output_dir: Path,
    output_format: str,
    dpi: int,
) -> Path:
    metric_specs = (
        (CHAMFER_TOTAL, "Total Chamfer distance", "Normalized distance"),
        (
            MISCLASSIFICATION_ERROR,
            "IAE",
            "Fraction of points",
        ),
        (MODEL_COUNT, "Models found", "Count"),
        (
            LOCAL_OPTIMIZATION_ITERATIONS,
            "Local-optimization iterations",
            "Count",
        ),
    )
    labels = [get_energy_label(energy) for energy in energy_order]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    for axis, (metric, title, y_label) in zip(axes.flat, metric_specs):
        grouped_values = [
            frame.loc[frame["energy"] == energy, metric].dropna().to_numpy()
            for energy in energy_order
        ]
        if any(values.size == 0 for values in grouped_values):
            raise ValueError(f"Metric {metric} has no valid value for an energy")

        positions = np.arange(1, len(energy_order) + 1)
        violinplot = axis.violinplot(
            grouped_values,
            positions=positions,
            widths=0.78,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body, energy in zip(violinplot["bodies"], energy_order):
            body.set_facecolor(colors[energy])
            body.set_edgecolor("#475569")
            body.set_linewidth(0.8)
            body.set_alpha(0.68)
        for part_name in ("cmedians", "cbars", "cmins", "cmaxes"):
            violinplot[part_name].set_color("#0F172A")
            violinplot[part_name].set_linewidth(
                1.6 if part_name == "cmedians" else 0.8
            )

        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        if metric == MISCLASSIFICATION_ERROR:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        if metric in (MODEL_COUNT, LOCAL_OPTIMIZATION_ITERATIONS):
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    figure.suptitle(
        "GAIR energy ablation: metric distributions",
        fontsize=16,
        fontweight="normal",
    )
    figure.text(
        0.5,
        0.01,
        "Violin shapes show run distributions. Lower values indicate lower error or lower complexity.",
        ha="center",
        color="#475569",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    return save_figure(
        figure,
        output_dir,
        "metric_distributions",
        output_format,
        dpi,
    )


def mean_and_confidence_interval(
    frame: pd.DataFrame,
    energy_order: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[float] = []
    confidence_intervals: list[float] = []
    for energy in energy_order:
        values = frame.loc[frame["energy"] == energy, metric].dropna().to_numpy()
        if values.size == 0:
            raise ValueError(f"Metric {metric} has no valid value for {energy}")
        means.append(float(np.mean(values)))
        if values.size == 1:
            confidence_intervals.append(0.0)
        else:
            standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
            confidence_intervals.append(1.96 * standard_error)
    return np.asarray(means), np.asarray(confidence_intervals)


def plot_directional_chamfer(
    frame: pd.DataFrame,
    energy_order: list[str],
    output_dir: Path,
    output_format: str,
    dpi: int,
) -> Path:
    gt_mean, gt_ci = mean_and_confidence_interval(
        frame,
        energy_order,
        CHAMFER_GT_TO_RECONSTRUCTION,
    )
    reconstruction_mean, reconstruction_ci = mean_and_confidence_interval(
        frame,
        energy_order,
        CHAMFER_RECONSTRUCTION_TO_GT,
    )

    positions = np.arange(len(energy_order))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12.5, 6.8))
    axis.bar(
        positions - width / 2,
        gt_mean,
        width,
        yerr=gt_ci,
        capsize=4,
        color="#277DA1",
        label="GT to reconstruction (coverage)",
    )
    axis.bar(
        positions + width / 2,
        reconstruction_mean,
        width,
        yerr=reconstruction_ci,
        capsize=4,
        color="#F8961E",
        label="Reconstruction to GT (fidelity)",
    )
    axis.set_xticks(
        positions,
        [get_energy_label(energy) for energy in energy_order],
        rotation=15,
    )
    axis.set_ylabel("Mean normalized distance")
    axis.set_title("Directional Chamfer distance by energy strategy")
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper left")
    axis.text(
        0.99,
        0.98,
        "Error bars: approximate 95% confidence interval\nLower is better",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#475569",
    )
    figure.tight_layout()
    return save_figure(
        figure,
        output_dir,
        "directional_chamfer",
        output_format,
        dpi,
    )


def plot_quality_tradeoffs(
    frame: pd.DataFrame,
    energy_order: list[str],
    colors: dict[str, str | tuple],
    output_dir: Path,
    output_format: str,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    plot_specs = (
        (
            CHAMFER_GT_TO_RECONSTRUCTION,
            CHAMFER_RECONSTRUCTION_TO_GT,
            "Directional Chamfer balance",
            "GT to reconstruction",
            "Reconstruction to GT",
        ),
        (
            CHAMFER_TOTAL,
            MISCLASSIFICATION_ERROR,
            "Geometric error vs IAE",
            "Total Chamfer distance",
            "IAE",
        ),
        (
            MODEL_COUNT,
            CHAMFER_TOTAL,
            "Reconstruction quality vs model count",
            "Models found",
            "Total Chamfer distance",
        ),
    )

    legend_handles = []
    legend_labels = []
    for axis, (x_metric, y_metric, title, x_label, y_label) in zip(
        axes,
        plot_specs,
    ):
        for energy in energy_order:
            energy_rows = frame.loc[
                frame["energy"] == energy,
                [x_metric, y_metric],
            ].dropna()
            points = axis.scatter(
                energy_rows[x_metric],
                energy_rows[y_metric],
                s=48,
                color=colors[energy],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.6,
            )
            axis.scatter(
                energy_rows[x_metric].mean(),
                energy_rows[y_metric].mean(),
                s=145,
                color=colors[energy],
                marker="X",
                edgecolor="#0F172A",
                linewidth=0.8,
                zorder=4,
            )
            if axis is axes[0]:
                legend_handles.append(points)
                legend_labels.append(get_energy_label(energy))

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True)

    directional_values = frame[
        [CHAMFER_GT_TO_RECONSTRUCTION, CHAMFER_RECONSTRUCTION_TO_GT]
    ].to_numpy(dtype=np.float64)
    finite_directional_values = directional_values[np.isfinite(directional_values)]
    if finite_directional_values.size:
        diagonal_max = float(np.max(finite_directional_values)) * 1.05
        axes[0].plot(
            [0.0, diagonal_max],
            [0.0, diagonal_max],
            linestyle="--",
            linewidth=1.0,
            color="#64748B",
            label="Equal directional error",
        )
        axes[0].set_xlim(left=0.0)
        axes[0].set_ylim(bottom=0.0)

    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.suptitle(
        "GAIR energy ablation: quality trade-offs",
        fontsize=16,
        fontweight="normal",
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(len(energy_order), 5),
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.text(
        0.5,
        0.035,
        "Circles represent individual runs; X markers represent strategy means.",
        ha="center",
        color="#475569",
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.94))
    return save_figure(
        figure,
        output_dir,
        "quality_tradeoffs",
        output_format,
        dpi,
    )


def save_summary(
    frame: pd.DataFrame,
    energy_order: list[str],
    output_dir: Path,
) -> Path:
    summary = frame.groupby("energy", sort=False)[list(NUMERIC_COLUMNS)].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    summary = summary.reindex(energy_order)
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary.insert(
        0,
        "energy_label",
        [get_energy_label(energy) for energy in summary.index],
    )
    output_path = output_dir / "ablation_energy_summary.csv"
    summary.to_csv(output_path, index_label="energy")
    return output_path


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")

    configure_plot_style()
    frame = load_metrics(args.input)
    energy_order = get_energy_order(frame)
    colors = get_energy_colors(energy_order)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        plot_metric_distributions(
            frame,
            energy_order,
            colors,
            args.output_dir,
            args.format,
            args.dpi,
        ),
        plot_directional_chamfer(
            frame,
            energy_order,
            args.output_dir,
            args.format,
            args.dpi,
        ),
        plot_quality_tradeoffs(
            frame,
            energy_order,
            colors,
            args.output_dir,
            args.format,
            args.dpi,
        ),
        save_summary(frame, energy_order, args.output_dir),
    ]

    print(f"Loaded {len(frame)} runs from {args.input}")
    for output_path in output_paths:
        print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
