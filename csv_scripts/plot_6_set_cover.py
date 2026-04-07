from pathlib import Path

import numpy as np

from common_plot_utils import build_method_legend
from common_plot_utils import draw_category_violins
from common_plot_utils import draw_scatter_with_pareto
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure
from common_plot_utils import STYLE_MAP
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

TRIALS_CSV_PATH = ROOT / "experiments" / "artifacts" / "6_set_cover" / "results_trials.csv"
SET_COVER_CSV_PATH = ROOT / "experiments" / "artifacts" / "6_set_cover" / "results_setcover.csv"
TRIALS_OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "6_set_cover_trials.png"
SUMMARY_OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "6_set_cover_summary.png"
METHODS = None
CENTRAL_TENDENCY = "mean"
DPI = 300


df_trials = load_results(TRIALS_CSV_PATH)
methods = pick_methods(df_trials, METHODS)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

draw_category_violins(axes[0, 0], df_trials, "chamfer", methods, "Chamfer", "Trial Chamfer", CENTRAL_TENDENCY)
draw_category_violins(axes[0, 1], df_trials, "hausdorff", methods, "Hausdorff", "Trial Hausdorff", CENTRAL_TENDENCY)
draw_category_violins(axes[1, 0], df_trials, "runtime_s", methods, "Runtime [s]", "Trial Runtime", CENTRAL_TENDENCY)
draw_scatter_with_pareto(axes[1, 1], df_trials, methods, "runtime_s", "chamfer", "Runtime [s]", "Chamfer", "Trial Runtime vs Chamfer")

for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.legend(handles=build_method_legend(methods), loc="best", frameon=True)
axes[1, 1].legend(loc="best", frameon=True)

fig.tight_layout()
save_figure(fig, TRIALS_OUTPUT_PATH, dpi=DPI)
plt.close(fig)


df_cover = load_results(SET_COVER_CSV_PATH)
methods = pick_methods(df_cover, METHODS)
positions = np.arange(len(methods), dtype=float)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))

for ax, metric, title, y_label in [
    (axes[0], "chamfer", "Set-cover Chamfer", "Chamfer"),
    (axes[1], "hausdorff", "Set-cover Hausdorff", "Hausdorff"),
    (axes[2], "accuracy", "Set-cover Accuracy", "Accuracy"),
]:
    for pos, method in zip(positions, methods):
        row = df_cover[df_cover["method"] == method]
        if row.empty:
            continue
        y_value = float(row.iloc[0][metric])
        color = STYLE_MAP[method]["line"]
        ax.scatter([pos], [y_value], color=color, s=75, zorder=3)
        if "k" in row.columns:
            ax.text(pos, y_value, f"  k={int(row.iloc[0]['k'])}", fontsize=8, va="bottom")
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=12)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)

fig.tight_layout()
save_figure(fig, SUMMARY_OUTPUT_PATH, dpi=DPI)
plt.close(fig)
