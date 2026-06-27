import argparse
from pathlib import Path

import pandas as pd

from common_plot_utils import build_method_legend
from common_plot_utils import draw_overlapping_violins
from common_plot_utils import draw_scatter_with_pareto
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = ROOT / "finale" / "FINALSFINALS"
DEFAULT_OUTPUT_PATH = ROOT / "csv_scripts" / "out2" / "figure6_7_multi_model_split_csv.png"
DEFAULT_OPTIMIZATION_METHODS = ["GC-RANSAC", "GAIR-RANSAC"]
PLOT_OUTPUT_SUFFIXES = {
    "misclassification": "misclassification_vs_outliers",
    "local_optimizations": "local_optimizations_vs_outliers",
    "time_vs_misclassification": "time_vs_misclassification",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-glob", default="**/*.csv")
    parser.add_argument("--exclude-name-substring", default="setcover")
    parser.add_argument("--error-methods", nargs="+", default=None)
    parser.add_argument(
        "--scatter-methods",
        "--refinement-methods",
        dest="scatter_methods",
        nargs="+",
        default=None,
        help="Methods to include in the time-vs-misclassification scatter plot.",
    )
    parser.add_argument(
        "--optimization-methods",
        "--local-optimization-methods",
        dest="optimization_methods",
        nargs="+",
        default=DEFAULT_OPTIMIZATION_METHODS,
        help="Methods to include in the local-optimizations violin plot.",
    )
    parser.add_argument("--central-tendency", choices=["mean", "median"], default="mean")
    parser.add_argument("--figure-number", type=int, default=6)
    parser.add_argument("--caption")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--disable-paper-captions", action="store_true")
    return parser.parse_args()


def warn(message):
    print(f"[WARN] {message}")


def resolve_path(path):
    path = path.expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def ensure_columns(df, required_columns, source_label):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_label}: {', '.join(missing)}")


def infer_dataset_label(results_dir):
    name = results_dir.name.strip()
    if name.endswith("_finale"):
        name = name[: -len("_finale")]
    return name or "multi_model"


def slugify(value):
    pieces = []
    for char in str(value).strip():
        if char.isalnum():
            pieces.append(char.lower())
        elif char in {"-", "_"}:
            pieces.append(char)
        elif char.isspace():
            pieces.append("_")
    slug = "".join(pieces).strip("_")
    return slug or "dataset"


def normalize_split_results(df):
    normalized = df.copy()

    if "outlier_ratio" not in normalized.columns and "outlier_frac" in normalized.columns:
        normalized["outlier_ratio"] = normalized["outlier_frac"]

    if "misclassification_error" not in normalized.columns:
        if "classification_rate" in normalized.columns:
            normalized["misclassification_error"] = 1.0 - normalized["classification_rate"]
        elif "accuracy" in normalized.columns:
            normalized["misclassification_error"] = 1.0 - normalized["accuracy"]

    if "execution_time_s" not in normalized.columns and "runtime_s" in normalized.columns:
        normalized["execution_time_s"] = normalized["runtime_s"]

    if "local_optimization_steps" not in normalized.columns and "n_local_opts" in normalized.columns:
        normalized["local_optimization_steps"] = normalized["n_local_opts"]
    if "n_local_opts" not in normalized.columns and "local_optimization_steps" in normalized.columns:
        normalized["n_local_opts"] = normalized["local_optimization_steps"]

    return normalized


def load_split_results(results_dir, csv_glob, exclude_name_substring):
    excluded_token = exclude_name_substring.strip().lower()
    csv_paths = sorted(
        path for path in results_dir.glob(csv_glob)
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and (not excluded_token or excluded_token not in path.name.lower())
    )
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found under {results_dir} matching '{csv_glob}' after exclusions."
        )

    frames = []
    for csv_path in csv_paths:
        df = load_results(csv_path)
        df = normalize_split_results(df)
        relative_path = csv_path.relative_to(results_dir)
        point_cloud_id = relative_path.parts[0] if len(relative_path.parts) > 1 else csv_path.stem
        df["source_csv"] = str(csv_path.relative_to(ROOT)) if csv_path.is_relative_to(ROOT) else str(csv_path)
        df["point_cloud_id"] = point_cloud_id
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined, csv_paths


def prepare_misclassification_dataframe(df):
    prepared = df.copy()
    before_count = len(prepared)
    prepared = prepared.dropna(
        subset=[
            "outlier_ratio",
            "misclassification_error",
            "execution_time_s",
        ]
    ).copy()
    dropped_count = before_count - len(prepared)
    if dropped_count > 0:
        warn(
            f"Dropped {dropped_count} row(s) without usable outlier ratio, "
            "misclassification error, or execution time before plotting."
        )
    if prepared.empty:
        raise ValueError(
            "No rows with outlier_ratio, misclassification_error, and execution_time_s "
            "are available for plotting."
        )
    return prepared


def iter_point_cloud_groups(df):
    if "point_cloud_id" not in df.columns:
        return []
    groups = []
    for point_cloud_id in sorted(df["point_cloud_id"].dropna().unique().tolist()):
        group_df = df[df["point_cloud_id"] == point_cloud_id].copy()
        if not group_df.empty:
            groups.append((str(point_cloud_id), group_df))
    return groups


def build_group_output_path(base_output_path, group_label):
    base_output_path = Path(base_output_path)
    slug = slugify(group_label)

    if base_output_path.suffix:
        return base_output_path.with_name(f"{base_output_path.stem}_{slug}{base_output_path.suffix}")
    return base_output_path / f"{slug}.png"


def build_plot_output_path(base_output_path, plot_key, total=False):
    base_output_path = Path(base_output_path)
    suffix = PLOT_OUTPUT_SUFFIXES[plot_key]
    if base_output_path.suffix:
        stem = f"{base_output_path.stem}_totale" if total else base_output_path.stem
        return base_output_path.with_name(
            f"{stem}_{suffix}{base_output_path.suffix}"
        )
    filename = f"totale_{suffix}.png" if total else f"{suffix}.png"
    return base_output_path / filename


def place_legend_above_axis(ax, handles):
    if not handles:
        return
    ncol = min(max(len(handles), 1), 3)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=ncol,
        frameon=True,
        borderaxespad=0.0,
    )


def render_misclassification_figure(df, args, dataset_label, output_path):
    error_methods = pick_methods(df, args.error_methods)
    if not error_methods:
        raise ValueError("No methods available for misclassification plot.")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    draw_overlapping_violins(
        ax=ax,
        df=df,
        metric_column="misclassification_error",
        methods=error_methods,
        title="Misclassification Error",
        y_label="Misclassification Error",
        central=args.central_tendency,
    )
    place_legend_above_axis(ax, build_method_legend(error_methods))

    fig.subplots_adjust(bottom=0.13, top=0.78)

    save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)


def render_time_vs_misclassification_figure(df, args, dataset_label, output_path):
    scatter_methods = pick_methods(df, args.scatter_methods)
    scatter_methods = [method for method in scatter_methods if method != "RANSAC"]
    if not scatter_methods:
        raise ValueError("No methods available for time-vs-misclassification plot.")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    draw_scatter_with_pareto(
        ax=ax,
        df=df,
        methods=scatter_methods,
        x_col="execution_time_s",
        y_col="misclassification_error",
        x_label="Execution Time [s]",
        y_label="Misclassification Error",
        title="Time vs Misclassification error",
        point_size=12,
        show_pareto=False,
    )
    place_legend_above_axis(ax, build_method_legend(scatter_methods))

    fig.subplots_adjust(bottom=0.13, top=0.78)

    save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)


def prepare_local_optimization_dataframe(df):
    if "local_optimization_steps" not in df.columns:
        warn("No local_optimization_steps or n_local_opts column found; skipping local-optimizations plot.")
        return None

    prepared = df.dropna(subset=["outlier_ratio", "local_optimization_steps"]).copy()
    dropped_count = len(df) - len(prepared)
    if dropped_count > 0:
        warn(
            f"Dropped {dropped_count} row(s) without usable outlier ratio or local "
            "optimization count before plotting local optimizations."
        )
    if prepared.empty:
        warn("No rows with outlier_ratio and local_optimization_steps are available; skipping plot.")
        return None

    return prepared


def render_local_optimization_figure(df, args, dataset_label, output_path):
    local_df = prepare_local_optimization_dataframe(df)
    if local_df is None:
        return

    optimization_methods = pick_methods(local_df, args.optimization_methods)
    if not optimization_methods:
        warn("No requested methods available for local-optimizations plot.")
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    draw_overlapping_violins(
        ax=ax,
        df=local_df,
        metric_column="local_optimization_steps",
        methods=optimization_methods,
        title="Number of Local Optimizations",
        y_label="Number of Local Optimizations",
        central=args.central_tendency,
    )
    ax.set_ylim(bottom=0)
    place_legend_above_axis(ax, build_method_legend(optimization_methods))

    fig.subplots_adjust(bottom=0.13, top=0.78)

    save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)


def render_figures(df, args, dataset_label, output_path, total=False):
    misclassification_output_path = build_plot_output_path(
        output_path,
        "misclassification",
        total=total,
    )
    local_optimization_output_path = build_plot_output_path(
        output_path,
        "local_optimizations",
        total=total,
    )
    time_vs_misclassification_output_path = build_plot_output_path(
        output_path,
        "time_vs_misclassification",
        total=total,
    )

    render_misclassification_figure(
        df,
        args,
        dataset_label,
        misclassification_output_path,
    )
    render_local_optimization_figure(
        df,
        args,
        dataset_label,
        local_optimization_output_path,
    )
    render_time_vs_misclassification_figure(
        df,
        args,
        dataset_label,
        time_vs_misclassification_output_path,
    )


def main():
    args = parse_args()
    results_dir = resolve_path(args.results_dir)
    output_path = resolve_path(args.output)

    df, csv_paths = load_split_results(
        results_dir=results_dir,
        csv_glob=args.csv_glob,
        exclude_name_substring=args.exclude_name_substring,
    )

    ensure_columns(
        df,
        [
            "method",
            "outlier_ratio",
            "misclassification_error",
            "execution_time_s",
        ],
        results_dir,
    )
    df = prepare_misclassification_dataframe(df)

    dataset_label = infer_dataset_label(results_dir)
    warn(f"Loaded {len(csv_paths)} split CSV files from {results_dir}.")
    render_figures(df, args, dataset_label, output_path, total=True)

    point_cloud_groups = iter_point_cloud_groups(df)
    for point_cloud_id, group_df in point_cloud_groups:
        group_output_path = build_group_output_path(output_path, point_cloud_id)
        render_figures(group_df, args, point_cloud_id, group_output_path)
    if point_cloud_groups:
        warn(f"Saved {len(point_cloud_groups)} per-point-cloud plots alongside the aggregate figure.")


if __name__ == "__main__":
    main()
