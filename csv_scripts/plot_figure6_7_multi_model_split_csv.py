import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common_plot_utils import add_figure_caption
from common_plot_utils import add_panel_captions
from common_plot_utils import build_violin_legend
from common_plot_utils import draw_overlapping_violins
from common_plot_utils import draw_scatter_with_pareto
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = ROOT / "finale" / "fungo"
DEFAULT_OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "figure6_7_multi_model_split_csv.png"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--subdir-glob", default="outliers_*")
    parser.add_argument("--sequential-filename", default="results_sequential.csv")
    parser.add_argument("--error-methods", nargs="+", default=None)
    parser.add_argument("--time-methods", nargs="+", default=None)
    parser.add_argument("--refinement-methods", nargs="+", default=None)
    parser.add_argument("--central-tendency", choices=["mean", "median"], default="mean")
    parser.add_argument(
        "--refinement-column",
        default="n_models",
        help=(
            "Per-trial column to use for panel (c). "
            "Default: n_models, since these split CSVs do not expose local_optimization_steps."
        ),
    )
    parser.add_argument("--refinement-title", default="Number Models Extracted")
    parser.add_argument("--refinement-y-label", default="Number Models Extracted")
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

    return normalized


def load_split_results(results_dir, subdir_glob, sequential_filename):
    csv_paths = sorted(results_dir.glob(f"{subdir_glob}/{sequential_filename}"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No '{sequential_filename}' files found under {results_dir} matching '{subdir_glob}'."
        )

    frames = []
    for csv_path in csv_paths:
        df = load_results(csv_path)
        df = normalize_split_results(df)
        df["source_csv"] = str(csv_path.relative_to(ROOT)) if csv_path.is_relative_to(ROOT) else str(csv_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined, csv_paths


def build_caption(args, dataset_label):
    if args.caption:
        return args.caption
    pretty_label = dataset_label.replace("_", "-")
    return f"Figure {args.figure_number}: Results on {pretty_label} - multi-model."


def render_figure(df, args, dataset_label, output_path):
    error_methods = pick_methods(df, args.error_methods)
    time_methods = pick_methods(df, args.time_methods)
    refinement_methods = pick_methods(df, args.refinement_methods)

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
    axes[0, 0].legend(handles=build_violin_legend(error_methods), loc="upper left", frameon=True)

    draw_overlapping_violins(
        ax=axes[0, 1],
        df=df,
        metric_column="execution_time_s",
        methods=time_methods,
        title="Execution Time",
        y_label="Execution Time [s]",
        central=args.central_tendency,
    )
    axes[0, 1].legend(handles=build_violin_legend(time_methods), loc="upper left", frameon=True)

    draw_overlapping_violins(
        ax=axes[1, 0],
        df=df,
        metric_column=args.refinement_column,
        methods=refinement_methods,
        title=args.refinement_title,
        y_label=args.refinement_y_label,
        central=args.central_tendency,
    )
    axes[1, 0].legend(handles=build_violin_legend(refinement_methods), loc="upper left", frameon=True)

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
    )
    axes[1, 1].legend(loc="upper right", frameon=True)

    fig.subplots_adjust(bottom=0.22, hspace=0.50, wspace=0.22)

    if not args.disable_paper_captions:
        add_panel_captions(
            fig,
            axes,
            [
                "(a) Misclassification error vs outliers (%)",
                "(b) Time (s) vs outliers (%)",
                "(c) Number of extracted models",
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
    output_path = resolve_path(args.output)

    df, csv_paths = load_split_results(
        results_dir=results_dir,
        subdir_glob=args.subdir_glob,
        sequential_filename=args.sequential_filename,
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


if __name__ == "__main__":
    main()
