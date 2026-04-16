import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common_plot_utils import add_figure_caption
from common_plot_utils import add_panel_captions
from common_plot_utils import build_method_legend
from common_plot_utils import draw_overlapping_violins
from common_plot_utils import draw_scatter_with_pareto
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = ROOT / "finale" / "FINALSFINALS"
DEFAULT_OUTPUT_PATH = ROOT / "csv_scripts" / "out2" / "figure6_7_multi_model_split_csv.png"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-glob", default="**/*.csv")
    parser.add_argument("--exclude-name-substring", default="setcover")
    parser.add_argument("--error-methods", nargs="+", default=None)
    parser.add_argument("--time-methods", nargs="+", default=None)
    parser.add_argument("--refinement-methods", nargs="+", default=None)
    parser.add_argument("--central-tendency", choices=["mean", "median"], default="mean")
    parser.add_argument(
        "--refinement-column",
        default="n_local_opts",
        help=(
            "Per-trial column to use for panel (c). "
            "Default: n_local_opts."
        ),
    )
    parser.add_argument("--refinement-title", default="Number of Local Optimizations")
    parser.add_argument("--refinement-y-label", default="Number of Local Optimizations")
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


def ensure_out2_output_path(path):
    path = Path(path)
    if path.suffix:
        if path.parent.name == "out2":
            return path
        if path.parent.name.startswith("out"):
            return path.parent.parent / "out2" / path.name
        return path.parent / "out2" / path.name

    if path.name == "out2":
        return path
    if path.name.startswith("out"):
        return path.parent / "out2"
    return path / "out2"


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


def build_caption(args, dataset_label):
    if args.caption:
        return args.caption
    pretty_label = dataset_label.replace("_", "-")
    return f"Figure {args.figure_number}: Results on {pretty_label} - multi-model."


def render_figure(df, args, dataset_label, output_path):
    error_methods = pick_methods(df, args.error_methods)
    time_methods = pick_methods(df, args.time_methods)
    refinement_methods = pick_methods(df, args.refinement_methods)
    time_methods = [method for method in time_methods if method != "Ransac"]
    refinement_methods = [method for method in refinement_methods if method != "Ransac"]

    if not error_methods:
        raise ValueError("No methods available for panel (a).")
    if not time_methods:
        raise ValueError("No methods available for panel (b).")
    if not refinement_methods:
        raise ValueError("No methods available for panels (c) and (d).")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.8))

    draw_overlapping_violins(
        ax=axes[0, 0],
        df=df,
        metric_column="misclassification_error",
        methods=error_methods,
        title="Misclassification Error",
        y_label="Misclassification Error",
        central=args.central_tendency,
    )
    place_legend_above_axis(axes[0, 0], build_method_legend(error_methods))

    draw_overlapping_violins(
        ax=axes[0, 1],
        df=df,
        metric_column="execution_time_s",
        methods=time_methods,
        title="Execution Time",
        y_label="Execution Time [s]",
        central=args.central_tendency,
    )
    place_legend_above_axis(axes[0, 1], build_method_legend(time_methods))

    draw_overlapping_violins(
        ax=axes[1, 0],
        df=df,
        metric_column=args.refinement_column,
        methods=refinement_methods,
        title=args.refinement_title,
        y_label=args.refinement_y_label,
        central=args.central_tendency,
    )
    place_legend_above_axis(axes[1, 0], build_method_legend(refinement_methods))

    draw_scatter_with_pareto(
        ax=axes[1, 1],
        df=df,
        methods=refinement_methods,
        x_col="execution_time_s",
        y_col="misclassification_error",
        x_label="Execution Time [s]",
        y_label="Misclassification Error",
        title="Time vs Misclassification error",
        point_size=12,
        show_pareto=False,
    )
    place_legend_above_axis(axes[1, 1], build_method_legend(refinement_methods))

    fig.subplots_adjust(bottom=0.22, top=0.82, hspace=0.98, wspace=0.24)

    if not args.disable_paper_captions:
        add_panel_captions(
            fig,
            axes,
            [
                "(a) Misclassification error vs outliers (%)",
                "(b) Time (s) vs outliers (%)",
                "(c) Number of local optimizations",
                "(d) Misclassification error vs time",
            ],
            y_offset=0.05,
        )
        add_figure_caption(fig, build_caption(args, dataset_label), y_position=0.02)

    save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = resolve_path(args.results_dir)
    output_path = ensure_out2_output_path(resolve_path(args.output))

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
            args.refinement_column,
        ],
        results_dir,
    )

    dataset_label = infer_dataset_label(results_dir)
    warn(f"Loaded {len(csv_paths)} split CSV files from {results_dir}.")
    render_figure(df, args, dataset_label, output_path)

    point_cloud_groups = iter_point_cloud_groups(df)
    for point_cloud_id, group_df in point_cloud_groups:
        group_output_path = build_group_output_path(output_path, point_cloud_id)
        render_figure(group_df, args, point_cloud_id, group_output_path)
    if point_cloud_groups:
        warn(f"Saved {len(point_cloud_groups)} per-point-cloud plots alongside the aggregate figure.")


if __name__ == "__main__":
    main()
