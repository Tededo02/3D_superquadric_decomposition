import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from common_plot_utils import add_figure_caption
from common_plot_utils import add_panel_captions
from common_plot_utils import build_violin_legend
from common_plot_utils import draw_overlapping_violins
from common_plot_utils import draw_scatter_with_pareto
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "experiments" / "artifacts" / "figure6_7_multi_model" / "results.csv"
OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "figure6_7_multi_model.png"
DEFAULT_ERROR_METHODS = ["Ransac", "Ransac + LO", "Ransac + GAIR"]
DEFAULT_TIME_METHODS = ["Ransac + LO", "Ransac + GAIR"]
DEFAULT_REFINEMENT_METHODS = ["Ransac + LO", "Ransac + GAIR"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--scenario", default="multi_model")
    parser.add_argument("--iteration-budget", type=int, default=500)
    parser.add_argument("--dataset-id", nargs="*")
    parser.add_argument("--error-methods", nargs="+", default=DEFAULT_ERROR_METHODS)
    parser.add_argument("--time-methods", nargs="+", default=DEFAULT_TIME_METHODS)
    parser.add_argument("--refinement-methods", nargs="+", default=DEFAULT_REFINEMENT_METHODS)
    parser.add_argument("--central-tendency", choices=["mean", "median"], default="mean")
    parser.add_argument("--figure-number", type=int, default=6)
    parser.add_argument("--caption")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--disable-paper-captions", action="store_true")
    return parser.parse_args()


def warn(message):
    print(f"[WARN] {message}")


def ensure_columns(df, required_columns, csv_path):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")


def apply_optional_filters(df, args):
    filtered = df.copy()

    if args.scenario is not None:
        if "scenario" in filtered.columns:
            filtered = filtered[filtered["scenario"] == args.scenario]
        else:
            warn(f'Column "scenario" is not present in {args.csv.name}; scenario filter ignored.')

    if args.iteration_budget is not None:
        if "iteration_budget" in filtered.columns:
            filtered = filtered[filtered["iteration_budget"] == args.iteration_budget]
        else:
            warn(f'Column "iteration_budget" is not present in {args.csv.name}; iteration filter ignored.')

    if args.dataset_id:
        if "dataset_id" not in filtered.columns:
            warn(f'Column "dataset_id" is not present in {args.csv.name}; dataset filter ignored.')
        else:
            filtered = filtered[filtered["dataset_id"].isin(args.dataset_id)]

    if filtered.empty:
        raise ValueError("No rows left after applying the requested filters.")

    return filtered


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


def resolve_dataset_label(group_df):
    if "dataset_id" in group_df.columns:
        values = [str(value).strip() for value in group_df["dataset_id"].dropna().unique().tolist() if str(value).strip()]
        if values:
            return values[0]

    if "input_file" in group_df.columns:
        values = [str(value).strip() for value in group_df["input_file"].dropna().unique().tolist() if str(value).strip()]
        if values:
            return Path(values[0]).stem

    return "multi_model"


def iter_groups(df):
    if "dataset_id" not in df.columns:
        yield "multi_model", df.copy()
        return

    for dataset_id in sorted(df["dataset_id"].dropna().unique().tolist()):
        group_df = df[df["dataset_id"] == dataset_id].copy()
        if not group_df.empty:
            yield str(dataset_id), group_df


def build_output_path(base_output_path, dataset_label, multi_output):
    base_output_path = Path(base_output_path)
    suffix = base_output_path.suffix or ".png"

    if not multi_output:
        if base_output_path.suffix:
            return base_output_path
        return base_output_path.with_suffix(suffix)

    stem = base_output_path.stem if base_output_path.suffix else base_output_path.name
    filename = f"{stem}_{slugify(dataset_label)}{suffix}"
    return base_output_path.with_name(filename)


def build_caption(args, dataset_label):
    if args.caption:
        return args.caption
    pretty_label = str(dataset_label).replace("_", "-")
    return f"Figure {args.figure_number}: Results on {pretty_label} - multi-model."


def render_figure(group_df, args, dataset_label, output_path):
    error_methods = pick_methods(group_df, args.error_methods)
    time_methods = pick_methods(group_df, args.time_methods)
    refinement_methods = pick_methods(group_df, args.refinement_methods)

    if not error_methods:
        raise ValueError(f"No error-plot methods available for dataset {dataset_label}.")
    if not time_methods:
        raise ValueError(f"No time-plot methods available for dataset {dataset_label}.")
    if not refinement_methods:
        raise ValueError(f"No refinement methods available for dataset {dataset_label}.")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.8))

    draw_overlapping_violins(
        ax=axes[0, 0],
        df=group_df,
        metric_column="misclassification_error",
        methods=error_methods,
        title="Misclassification Error",
        y_label="Misclassification Error",
        central=args.central_tendency,
    )
    axes[0, 0].legend(handles=build_violin_legend(error_methods), loc="upper left", frameon=True)

    draw_overlapping_violins(
        ax=axes[0, 1],
        df=group_df,
        metric_column="execution_time_s",
        methods=time_methods,
        title="Execution Time",
        y_label="Execution Time [s]",
        central=args.central_tendency,
    )
    axes[0, 1].legend(handles=build_violin_legend(time_methods), loc="upper left", frameon=True)

    draw_overlapping_violins(
        ax=axes[1, 0],
        df=group_df,
        metric_column="local_optimization_steps",
        methods=refinement_methods,
        title="Number Optimization made",
        y_label="Number Optimization made",
        central=args.central_tendency,
    )
    axes[1, 0].legend(handles=build_violin_legend(refinement_methods), loc="upper left", frameon=True)

    draw_scatter_with_pareto(
        ax=axes[1, 1],
        df=group_df,
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
                "(c) # number of times the local refinement is executed",
                "(d) Misclassification error vs time",
            ],
            y_offset=0.05,
        )
        add_figure_caption(fig, build_caption(args, dataset_label), y_position=0.02)

    save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_args()
    csv_path = args.csv.expanduser()
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    df = load_results(csv_path)
    ensure_columns(
        df,
        [
            "method",
            "outlier_ratio",
            "misclassification_error",
            "execution_time_s",
            "local_optimization_steps",
        ],
        csv_path,
    )
    df = apply_optional_filters(df, args)

    groups = list(iter_groups(df))
    multi_output = len(groups) > 1

    for _, group_df in groups:
        dataset_label = resolve_dataset_label(group_df)
        dataset_output_path = build_output_path(output_path, dataset_label, multi_output)
        render_figure(group_df, args, dataset_label, dataset_output_path)


if __name__ == "__main__":
    main()
