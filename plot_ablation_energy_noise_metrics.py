import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from plot_ablation_energy_metrics import (
    CHAMFER_GT_TO_RECONSTRUCTION,
    CHAMFER_RECONSTRUCTION_TO_GT,
    CHAMFER_TOTAL,
    LOCAL_OPTIMIZATION_ITERATIONS,
    MISCLASSIFICATION_ERROR,
    MODEL_COUNT,
    NUMERIC_COLUMNS,
    configure_plot_style,
    get_energy_colors,
    get_energy_label,
    get_energy_order,
    load_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    (
        "0 degrees",
        PROJECT_ROOT / "data" / "results" / "ablation_energy_metrics.csv",
    ),
    (
        "5 degrees",
        PROJECT_ROOT
        / "data"
        / "results"
        / "ablation_energy_metrics_noise_5.csv",
    ),
    (
        "10 degrees",
        PROJECT_ROOT
        / "data"
        / "results"
        / "ablation_energy_metrics_noise_10.csv",
    ),
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "results" / "ablation_noise_plots"
)

# The current FullGairEnergy implementation uses coherence_min=0.9.
ENERGY_ALIASES = {
    "FullGairEnergy": "FullGairEnergycoh09",
}
EXCLUDED_ENERGIES = {
    "GcRansacEnergy",
}
ENERGY_LABEL_OVERRIDES = {
    "FullGairEnergycoh09": "GAIR(ours)",
}
ENERGY_COLOR_OVERRIDES = {
    "FullGairEnergycoh09": "#F8961E",
    "OnlyUnaryGairStrategy": "#7A5195",
}
METRIC_SPECS = (
    (
        CHAMFER_GT_TO_RECONSTRUCTION,
        "GT to reconstruction",
        "Normalized distance",
    ),
    (
        CHAMFER_RECONSTRUCTION_TO_GT,
        "Reconstruction to GT",
        "Normalized distance",
    ),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GAIR energy strategies across normal-noise levels."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        help=(
            "Noise label and metrics CSV. Repeat this option to add more noise "
            "levels. Defaults to the 0-degree, 5-degree and 10-degree CSV files."
        ),
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


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def get_datasets(args: argparse.Namespace) -> list[tuple[str, Path]]:
    raw_datasets = args.dataset if args.dataset is not None else DEFAULT_DATASETS
    datasets = [
        (str(label), resolve_path(path))
        for label, path in raw_datasets
    ]
    if not datasets:
        raise ValueError("At least one noise dataset is required")

    labels = [label for label, _ in datasets]
    if len(labels) != len(set(labels)):
        raise ValueError("Noise labels must be unique")
    return datasets


def load_noise_metrics(
    datasets: list[tuple[str, Path]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for noise_label, input_path in datasets:
        frame = load_metrics(input_path).copy()
        frame["energy"] = frame["energy"].replace(ENERGY_ALIASES)
        frame = frame.loc[~frame["energy"].isin(EXCLUDED_ENERGIES)].copy()
        frame["noise"] = noise_label
        frame["input_path"] = str(input_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def energy_label(energy: str) -> str:
    return ENERGY_LABEL_OVERRIDES.get(energy, get_energy_label(energy))


def style_metric_axis(
    axis: plt.Axes,
    metric: str,
    y_label: str,
    x_labels: list[str],
    centers: np.ndarray,
) -> None:
    axis.set_ylabel(y_label)
    axis.set_xticks(
        centers,
        x_labels,
    )
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    if metric == MISCLASSIFICATION_ERROR:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    if metric in (MODEL_COUNT, LOCAL_OPTIMIZATION_ITERATIONS):
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))


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


def plot_metric_means(
    frame: pd.DataFrame,
    energy_order: list[str],
    noise_order: list[str],
    colors: dict[str, str],
    output_dir: Path,
    output_format: str,
    dpi: int,
) -> Path:
    positions = np.arange(len(noise_order), dtype=np.float64)
    figure, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    for axis, (metric, title, y_label) in zip(axes.flat, METRIC_SPECS):
        for energy in energy_order:
            means: list[float] = []
            for noise in noise_order:
                values = frame.loc[
                    (frame["energy"] == energy)
                    & (frame["noise"] == noise),
                    metric,
                ].dropna().to_numpy()
                means.append(
                    float(np.mean(values)) if values.size else float("nan")
                )

            if not np.isfinite(means).any():
                continue
            axis.plot(
                positions,
                means,
                marker="o",
                markersize=7,
                linewidth=2.2,
                color=colors[energy],
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=energy_label(energy),
            )

        axis.set_title(title)
        style_metric_axis(axis, metric, y_label, noise_order, positions)
        axis.set_xlabel("Normal-noise level")
        axis.legend(loc="best", fontsize=8)

    figure.suptitle(
        "Mean performance by energy strategy and normal-noise level",
        fontsize=16,
        fontweight="normal",
    )
    figure.text(
        0.5,
        0.025,
        "Each point is the mean across runs for one energy strategy.",
        ha="center",
        color="#475569",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    return save_figure(
        figure,
        output_dir,
        "noise_energy_mean_lines",
        output_format,
        dpi,
    )


def save_summary(
    frame: pd.DataFrame,
    energy_order: list[str],
    noise_order: list[str],
    output_dir: Path,
) -> Path:
    summary = frame.groupby(["energy", "noise"], sort=False)[
        list(NUMERIC_COLUMNS)
    ].agg(["count", "mean", "std", "median", "min", "max"])
    summary = summary.reindex(
        pd.MultiIndex.from_product(
            [energy_order, noise_order],
            names=("energy", "noise"),
        )
    )
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary.insert(
        0,
        "energy_label",
        [energy_label(energy) for energy, _ in summary.index],
    )
    output_path = output_dir / "ablation_energy_noise_summary.csv"
    summary.to_csv(output_path)
    return output_path


def report_missing_groups(
    frame: pd.DataFrame,
    energy_order: list[str],
    noise_order: list[str],
) -> None:
    observed = set(zip(frame["energy"], frame["noise"]))
    missing = [
        (energy, noise)
        for energy in energy_order
        for noise in noise_order
        if (energy, noise) not in observed
    ]
    for energy, noise in missing:
        print(
            "Warning: no rows for "
            f"{energy_label(energy)} at noise level {noise}."
        )


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")

    datasets = get_datasets(args)
    noise_order = [label for label, _ in datasets]
    frame = load_noise_metrics(datasets)
    energy_order = get_energy_order(frame)
    energy_colors = get_energy_colors(energy_order)
    energy_colors.update(ENERGY_COLOR_OVERRIDES)

    report_missing_groups(frame, energy_order, noise_order)
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        plot_metric_means(
            frame,
            energy_order,
            noise_order,
            energy_colors,
            args.output_dir,
            args.format,
            args.dpi,
        ),
        save_summary(
            frame,
            energy_order,
            noise_order,
            args.output_dir,
        ),
    ]

    for noise_label, input_path in datasets:
        row_count = int((frame["noise"] == noise_label).sum())
        print(f"Loaded {row_count} runs for {noise_label} from {input_path}")
    for output_path in output_paths:
        print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
