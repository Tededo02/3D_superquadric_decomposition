import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / "csv_scripts" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MY_RESULTS_CSV = ROOT / "finale" / "my_results.csv"
OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "figure6_7_multi_model_avg_time_vs_misclassification.png"
DEFAULT_SCENARIO = "multi_model"
DEFAULT_ITERATION_BUDGET = 500
DEFAULT_METHODS = None

METHOD_ALIASES = {
    "ransac": "Ransac",
    "vanilla": "Ransac",
    "vanilla ransac": "Ransac",
    "vanilla-ransac": "Ransac",
    "lo ransac": "Ransac + LO",
    "lo-ransac": "Ransac + LO",
    "gc ransac": "Ransac + GC",
    "gc-ransac": "Ransac + GC",
    "gair ransac": "Ransac + GAIR",
    "gair-ransac": "Ransac + GAIR",
}

STYLE_MAP = {
    "Ransac": {"line": "#5DAE61", "fill": "#0F8A20"},
    "Ransac + LO": {"line": "#B45AC7", "fill": "#8A0FA8"},
    "Ransac + GAIR": {"line": "#F0A202", "fill": "#F0A202"},
    "Ransac + GC": {"line": "#377EB8", "fill": "#377EB8"},
}

STYLE_ORDER = ["Ransac", "Ransac + LO", "Ransac + GC", "Ransac + GAIR"]
DATASET_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*", "8"]


plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=MY_RESULTS_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--iteration-budget", type=int, default=DEFAULT_ITERATION_BUDGET)
    parser.add_argument("--dataset-id", nargs="*")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--title", default="Average Time vs Misclassification Error")
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


def normalize_method(value):
    key = str(value).strip().lower().replace("_", " ")
    key = " ".join(key.split())
    return METHOD_ALIASES.get(key, str(value).strip())


def parse_float(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_results(csv_path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {csv_path}")

        for raw_row in reader:
            dataset_id = str(raw_row.get("dataset_id", "")).strip()
            if not dataset_id:
                input_file = str(raw_row.get("input_file", "")).strip()
                dataset_id = Path(input_file).stem if input_file else "multi_model"

            method_value = raw_row.get("method", "") or raw_row.get("algo", "")
            rows.append({
                "scenario": str(raw_row.get("scenario", "")).strip(),
                "iteration_budget": parse_float(raw_row.get("iteration_budget")),
                "dataset_id": dataset_id,
                "method": normalize_method(method_value),
                "misclassification_error": parse_float(raw_row.get("misclassification_error")),
                "execution_time_s": parse_float(
                    raw_row.get("execution_time_s", raw_row.get("runtime_s"))
                ),
            })

    if not rows:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")
    return rows


def apply_filters(rows, args, csv_path):
    filtered = list(rows)

    if args.scenario is not None:
        filtered = [row for row in filtered if row["scenario"] == args.scenario]

    if args.iteration_budget is not None:
        filtered = [
            row
            for row in filtered
            if np.isfinite(row["iteration_budget"])
            and np.isclose(row["iteration_budget"], float(args.iteration_budget), atol=1e-12, rtol=0.0)
        ]

    if args.dataset_id:
        allowed = {str(dataset_id).strip() for dataset_id in args.dataset_id}
        filtered = [row for row in filtered if row["dataset_id"] in allowed]

    if not filtered:
        raise ValueError(f"No rows left after applying filters to {csv_path}")
    return filtered


def compute_group_means(rows):
    grouped = defaultdict(lambda: {"time_sum": 0.0, "mis_sum": 0.0, "count": 0})

    for row in rows:
        if not np.isfinite(row["execution_time_s"]) or not np.isfinite(row["misclassification_error"]):
            continue
        key = (row["dataset_id"], row["method"])
        grouped[key]["time_sum"] += float(row["execution_time_s"])
        grouped[key]["mis_sum"] += float(row["misclassification_error"])
        grouped[key]["count"] += 1

    summary_rows = []
    for (dataset_id, method), stats in grouped.items():
        if stats["count"] <= 0:
            continue
        summary_rows.append({
            "dataset_id": dataset_id,
            "method": method,
            "mean_execution_time_s": stats["time_sum"] / stats["count"],
            "mean_misclassification_error": stats["mis_sum"] / stats["count"],
            "sample_count": stats["count"],
        })

    if not summary_rows:
        raise ValueError("No valid rows available to compute grouped means.")
    return summary_rows


def pick_methods(summary_rows, methods=None):
    present = {row["method"] for row in summary_rows}
    if methods:
        wanted = [normalize_method(method) for method in methods]
        return [method for method in wanted if method in present]

    ordered = [method for method in STYLE_ORDER if method in present]
    ordered.extend(
        row["method"]
        for row in summary_rows
        if row["method"] in present and row["method"] not in ordered
    )
    return ordered


def style_for_method(method):
    return STYLE_MAP.get(method, {"line": "#444444", "fill": "#999999"})


def build_dataset_marker_map(summary_rows):
    dataset_ids = sorted({row["dataset_id"] for row in summary_rows})
    return {
        dataset_id: DATASET_MARKERS[idx % len(DATASET_MARKERS)]
        for idx, dataset_id in enumerate(dataset_ids)
    }


def build_method_legend(methods):
    handles = []
    for method in methods:
        style = style_for_method(method)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=style["line"],
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=8,
                label=method,
            )
        )
    return handles


def build_dataset_legend(dataset_marker_map):
    handles = []
    for dataset_id, marker in dataset_marker_map.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=8,
                label=dataset_id,
            )
        )
    return handles


def build_caption(args):
    if args.caption:
        return args.caption
    return (
        f"Figure {args.figure_number}: Mean execution time vs mean misclassification error, "
        "grouped by point cloud and algorithm."
    )


def save_figure(fig, output_path, dpi):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    print(f"[OK] Saved figure to {output_path}")


def render_figure(summary_rows, args, output_path):
    methods = pick_methods(summary_rows, args.methods)
    if not methods:
        raise ValueError("No methods available after filtering.")

    dataset_marker_map = build_dataset_marker_map(summary_rows)
    fig, ax = plt.subplots(figsize=(10.0, 7.4))

    for row in summary_rows:
        if row["method"] not in methods:
            continue
        style = style_for_method(row["method"])
        marker = dataset_marker_map[row["dataset_id"]]
        ax.scatter(
            row["mean_execution_time_s"],
            row["mean_misclassification_error"],
            s=90,
            marker=marker,
            color=style["line"],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xlabel("Mean Execution Time [s]")
    ax.set_ylabel("Mean Misclassification Error")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.22, linewidth=0.7)

    method_legend = ax.legend(
        handles=build_method_legend(methods),
        title="Algorithm",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=build_dataset_legend(dataset_marker_map),
        title="Point Cloud",
        loc="lower right",
        frameon=True,
    )

    fig.subplots_adjust(bottom=0.16)
    if not args.disable_paper_captions:
        fig.text(0.5, 0.03, build_caption(args), ha="center", va="bottom", fontsize=12)

    save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_args()
    csv_path = resolve_path(args.csv)
    output_path = resolve_path(args.output)

    rows = load_results(csv_path)
    rows = apply_filters(rows, args, csv_path)
    summary_rows = compute_group_means(rows)
    render_figure(summary_rows, args, output_path)


if __name__ == "__main__":
    main()
