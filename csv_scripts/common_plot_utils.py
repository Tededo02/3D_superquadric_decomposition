import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
})


METHOD_ALIASES = {
    "ransac": "RANSAC",
    "vanilla": "RANSAC",
    "vanilla ransac": "RANSAC",
    "vanilla-ransac": "RANSAC",
    "lo ransac": "LO-RANSAC",
    "lo-ransac": "LO-RANSAC",
    "ransac + lo": "LO-RANSAC",
    "local opt": "LO-RANSAC",
    "local-opt": "LO-RANSAC",
    "local optimization": "LO-RANSAC",
    "gc ransac": "GC-RANSAC",
    "gc-ransac": "GC-RANSAC",
    "ransac + gc": "GC-RANSAC",
    "gair ransac": "GAIR-RANSAC",
    "gair-ransac": "GAIR-RANSAC",
    "ransac + gair": "GAIR-RANSAC",
}


STYLE_MAP = {
    "RANSAC": {"line": "#5DAE61", "fill": "#0F8A20"},
    "LO-RANSAC": {"line": "#B45AC7", "fill": "#8A0FA8"},
    "GAIR-RANSAC": {"line": "#F0A202", "fill": "#F0A202"},
    "GC-RANSAC": {"line": "#377EB8", "fill": "#377EB8"},
}


STYLE_ORDER = ["RANSAC", "GAIR-RANSAC", "GC-RANSAC", "LO-RANSAC"]
FALLBACK_STYLE_CYCLE = [
    {"line": "#4C78A8", "fill": "#A0CBE8"},
    {"line": "#F58518", "fill": "#FFBF79"},
    {"line": "#54A24B", "fill": "#A7D89A"},
    {"line": "#E45756", "fill": "#F2A7A4"},
    {"line": "#72B7B2", "fill": "#B6E0DC"},
]


def normalize_method(value):
    key = str(value).strip().lower().replace("_", " ")
    key = " ".join(key.split())
    return METHOD_ALIASES.get(key, str(value).strip())


def get_method_style(method):
    if method in STYLE_MAP:
        return STYLE_MAP[method]
    index = abs(hash(method)) % len(FALLBACK_STYLE_CYCLE)
    return FALLBACK_STYLE_CYCLE[index]


def load_results(csv_path):
    df = pd.read_csv(csv_path).copy()
    if "method" not in df.columns and "algo" in df.columns:
        df["method"] = df["algo"]
    df["method"] = df["method"].map(normalize_method)
    for column in df.columns:
        if column not in {"algo", "method"}:
            converted = pd.to_numeric(df[column], errors="coerce")
            if not converted.isna().all():
                df[column] = converted
    return df


def pick_methods(df, methods=None):
    if methods:
        wanted = [normalize_method(method) for method in methods]
        return [method for method in wanted if method in set(df["method"])]

    present = set(df["method"])
    ordered = [method for method in STYLE_ORDER if method in present]
    ordered.extend(method for method in df["method"].tolist() if method in present and method not in ordered)
    return ordered


def build_method_legend(methods):
    handles = []
    for method in methods:
        color = get_method_style(method)["line"]
        handles.append(Line2D([0], [0], color=color, lw=1.0, label=method))
    return handles


def build_violin_legend(methods):
    handles = []
    for method in methods:
        style = get_method_style(method)
        handles.append(Line2D([0], [0], color=style["line"], lw=1.0, label=method))
    for method in methods:
        style = get_method_style(method)
        handles.append(Patch(facecolor=style["fill"], alpha=0.30, edgecolor="none", label=f"{method} Violin"))
    return handles


def compute_pareto_front(x_values, y_values):
    points = sorted(zip(x_values, y_values), key=lambda pair: (pair[0], pair[1]))
    frontier = []
    best_y = math.inf
    for x_value, y_value in points:
        if y_value < best_y:
            frontier.append((x_value, y_value))
            best_y = y_value
    return np.array(frontier, dtype=float)


def draw_overlapping_violins(
    ax,
    df,
    metric_column,
    methods,
    title,
    y_label,
    central="mean",
    violin_width=0.055,
    line_width=0.95,
):
    x_values = sorted(df["outlier_ratio"].dropna().unique().tolist())
    if not x_values:
        return

    for method in methods:
        style = get_method_style(method)
        chunks = []
        positions = []
        centers = []

        for x_value in x_values:
            mask = (df["method"] == method) & np.isclose(df["outlier_ratio"], x_value)
            values = df.loc[mask, metric_column].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue

            positions.append(float(x_value))
            chunks.append(values)
            centers.append(float(np.median(values) if central == "median" else np.mean(values)))

        if not chunks:
            continue

        violin = ax.violinplot(chunks, positions=positions, widths=violin_width, showextrema=True)
        for body in violin["bodies"]:
            body.set_facecolor(style["fill"])
            body.set_edgecolor("none")
            body.set_alpha(0.30)
            body.set_zorder(2)
        for key in ["cbars", "cmins", "cmaxes"]:
            violin[key].set_color(style["line"])
            violin[key].set_linewidth(line_width)
            violin[key].set_alpha(0.95)
            violin[key].set_zorder(3)

        ax.plot(positions, centers, color=style["line"], linewidth=line_width, alpha=0.95, zorder=4)

    min_x = float(x_values[0])
    max_x = float(x_values[-1])
    margin = 0.05 if max_x - min_x < 0.5 else 0.1 * (max_x - min_x)

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{value:g}" for value in x_values])
    ax.set_xlabel("Outlier injection")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)


def draw_numeric_violins(ax, df, x_col, y_col, methods, x_label, y_label, title, central="mean"):
    x_values = sorted(df[x_col].dropna().unique().tolist())
    if not x_values:
        return

    width = 0.18
    if len(x_values) > 1:
        step = min(np.diff(np.array(x_values, dtype=float)))
        width = max(step * 0.22, step * 0.12)

    for method in methods:
        method_df = df[df["method"] == method]
        method_x = []
        chunks = []
        centers = []

        for x_value in x_values:
            values = method_df.loc[np.isclose(method_df[x_col], x_value), y_col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            method_x.append(float(x_value))
            chunks.append(values)
            centers.append(float(np.median(values) if central == "median" else np.mean(values)))

        if not chunks:
            continue

        style = get_method_style(method)
        violin = ax.violinplot(chunks, positions=method_x, widths=width, showextrema=True)
        for body in violin["bodies"]:
            body.set_facecolor(style["fill"])
            body.set_edgecolor("none")
            body.set_alpha(0.28)
        for key in ["cbars", "cmins", "cmaxes"]:
            violin[key].set_color(style["line"])
            violin[key].set_linewidth(0.9)
        ax.plot(method_x, centers, color=style["line"], linewidth=0.95, zorder=4)

    margin = width * 1.6 if len(x_values) == 1 else max(width * 1.8, 0.08 * (x_values[-1] - x_values[0]))
    ax.set_xlim(x_values[0] - margin, x_values[-1] + margin)
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{value:g}" for value in x_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)


def draw_category_violins(ax, df, y_col, methods, y_label, title, central="mean"):
    positions = np.arange(len(methods), dtype=float)

    for pos, method in zip(positions, methods):
        values = df.loc[df["method"] == method, y_col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue

        style = get_method_style(method)
        violin = ax.violinplot([values], positions=[pos], widths=0.55, showextrema=True)
        for body in violin["bodies"]:
            body.set_facecolor(style["fill"])
            body.set_edgecolor("none")
            body.set_alpha(0.28)
        for key in ["cbars", "cmins", "cmaxes"]:
            violin[key].set_color(style["line"])
            violin[key].set_linewidth(0.9)

        center = float(np.median(values) if central == "median" else np.mean(values))
        ax.hlines(center, pos - 0.17, pos + 0.17, color=style["line"], linewidth=1.2, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=12)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)


def draw_scatter_with_pareto(
    ax,
    df,
    methods,
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    point_size=18,
    show_pareto=True,
):
    for method in methods:
        subset = df[df["method"] == method].dropna(subset=[x_col, y_col])
        if subset.empty:
            continue

        style = get_method_style(method)
        x_values = subset[x_col].to_numpy(dtype=float)
        y_values = subset[y_col].to_numpy(dtype=float)
        ax.scatter(x_values, y_values, color=style["line"], s=point_size, marker=".", alpha=0.95, label=method, zorder=3)

        pareto = compute_pareto_front(x_values, y_values)
        if show_pareto and len(pareto) > 0:
            ax.plot(
                pareto[:, 0],
                pareto[:, 1],
                color=style["line"],
                linewidth=0.9,
                marker="x",
                markersize=4,
                label=f"{method} Pareto",
                zorder=4,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)


def add_panel_captions(fig, axes, captions, y_offset=0.06):
    for ax, caption in zip(np.ravel(np.array(axes)), captions):
        box = ax.get_position()
        x_center = box.x0 + box.width / 2.0
        y_position = box.y0 - y_offset
        fig.text(
            x_center,
            y_position,
            caption,
            ha="center",
            va="top",
            fontsize=10,
            fontstyle="italic",
        )


def add_figure_caption(fig, caption, y_position=0.02):
    fig.text(0.5, y_position, caption, ha="center", va="bottom", fontsize=12)


def save_figure(fig, output_path, dpi=300):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    print(f"[OK] Saved figure to {output_path}")
