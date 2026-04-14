import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common_plot_utils import add_figure_caption
from common_plot_utils import add_panel_captions
from common_plot_utils import build_method_legend
from common_plot_utils import draw_category_violins
from common_plot_utils import draw_scatter_with_pareto
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure


ROOT = Path(__file__).resolve().parents[1]

# Puoi impostare qui una cartella con piu point cloud oppure un singolo results_sequential.csv.
RESULTS_SOURCE = ROOT / "finale" 
SEQUENTIAL_FILENAME = "my_results.csv"
OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "figure6_7_multi_model_chamfer"
DEFAULT_METHODS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_SOURCE,
        help="Path to a results_sequential.csv file or to a directory that contains one or more of them.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sequential-filename", default=SEQUENTIAL_FILENAME)
    parser.add_argument("--dataset-id", nargs="*")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
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


def resolve_results_paths(results_source, sequential_filename):
    results_source = resolve_path(results_source)

    if not results_source.exists():
        raise FileNotFoundError(f"Results source not found: {results_source}")

    if results_source.is_file():
        if results_source.name != sequential_filename:
            raise ValueError(
                f"Expected a '{sequential_filename}' file, got {results_source.name}"
            )
        return results_source, [results_source]

    csv_paths = sorted(path for path in results_source.rglob(sequential_filename) if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(
            f"No '{sequential_filename}' files found under {results_source}."
        )
    return results_source, csv_paths


def infer_dataset_id(csv_path):
    return csv_path.parent.name.strip() or csv_path.stem


def ensure_columns(df, required_columns, source_label):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_label}: {', '.join(missing)}")


def load_combined_results(results_source, sequential_filename):
    source_root, csv_paths = resolve_results_paths(results_source, sequential_filename)

    frames = []
    for csv_path in csv_paths:
        df = load_results(csv_path)
        df["dataset_id"] = infer_dataset_id(csv_path)
        df["source_csv"] = str(csv_path.relative_to(ROOT)) if csv_path.is_relative_to(ROOT) else str(csv_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return source_root, csv_paths, combined


def apply_optional_filters(df, args):
    filtered = df.copy()

    if args.dataset_id:
        filtered = filtered[filtered["dataset_id"].isin(args.dataset_id)]

    if filtered.empty:
        raise ValueError("No rows left after applying the requested filters.")

    return filtered


def iter_groups(df):
    yield "all_datasets", df.copy()

    for dataset_id in sorted(df["dataset_id"].dropna().unique().tolist()):
        group_df = df[df["dataset_id"] == dataset_id].copy()
        if not group_df.empty:
            yield str(dataset_id), group_df


def build_output_path(base_output_path, dataset_label):
    base_output_path = Path(base_output_path)

    if base_output_path.suffix:
        output_dir = base_output_path.parent
        filename = f"{base_output_path.stem}_{slugify(dataset_label)}{base_output_path.suffix}"
    else:
        output_dir = base_output_path
        filename = f"{slugify(dataset_label)}.png"

    return output_dir / filename


def build_caption(args, dataset_label):
    if args.caption:
        return args.caption
    if dataset_label == "all_datasets":
        return f"Figure {args.figure_number}: Chamfer results on all datasets - multi-model."
    pretty_label = dataset_label.replace("_", "-")
    return f"Figure {args.figure_number}: Chamfer results on {pretty_label} - multi-model."


def render_figure(group_df, args, dataset_label, output_path):
    methods = pick_methods(group_df, args.methods)

    if not methods:
        raise ValueError(f"No methods available for dataset {dataset_label}.")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.8))

    draw_category_violins(
        ax=axes[0, 0],
        df=group_df,
        y_col="chamfer",
        methods=methods,
        y_label="Bidirectional Chamfer",
        title="Bidirectional Chamfer per Method",
        central=args.central_tendency,
    )

    draw_category_violins(
        ax=axes[0, 1],
        df=group_df,
        y_col="cd_coverage",
        methods=methods,
        y_label="Chamfer GT -> Estimate",
        title="Unidirectional Chamfer (GT -> Estimate)",
        central=args.central_tendency,
    )

    draw_category_violins(
        ax=axes[1, 0],
        df=group_df,
        y_col="cd_accuracy",
        methods=methods,
        y_label="Chamfer Estimate -> GT",
        title="Unidirectional Chamfer (Estimate -> GT)",
        central=args.central_tendency,
    )

    draw_scatter_with_pareto(
        ax=axes[1, 1],
        df=group_df,
        methods=methods,
        x_col="cd_coverage",
        y_col="cd_accuracy",
        x_label="Chamfer GT -> Estimate",
        y_label="Chamfer Estimate -> GT",
        title="Unidirectional Chamfer Comparison",
        point_size=12,
    )

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.legend(handles=build_method_legend(methods), loc="best", frameon=True)
    axes[1, 1].legend(loc="best", frameon=True)

    fig.subplots_adjust(bottom=0.22, hspace=0.50, wspace=0.24)

    if not args.disable_paper_captions:
        add_panel_captions(
            fig,
            axes,
            [
                "(a) Bidirectional Chamfer per method",
                "(b) Chamfer GT -> estimate per method",
                "(c) Chamfer estimate -> GT per method",
                "(d) GT -> estimate Chamfer vs estimate -> GT Chamfer",
            ],
            y_offset=0.05,
        )
        add_figure_caption(fig, build_caption(args, dataset_label), y_position=0.02)

    save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_args()
    args.output = resolve_path(args.output)

    source_root, csv_paths, df = load_combined_results(args.results, args.sequential_filename)
    ensure_columns(
        df,
        [
            "method",
            "dataset_id",
            "chamfer",
            "cd_coverage",
            "cd_accuracy",
        ],
        source_root,
    )
    df = apply_optional_filters(df, args)

    warn(f"Loaded {len(csv_paths)} results file(s) from {source_root}.")
    for dataset_label, group_df in iter_groups(df):
        output_path = build_output_path(args.output, dataset_label)
        render_figure(group_df, args, dataset_label, output_path)


if __name__ == "__main__":
    main()
