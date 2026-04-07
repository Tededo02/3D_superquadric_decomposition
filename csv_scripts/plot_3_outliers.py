from pathlib import Path

from common_plot_utils import build_method_legend
from common_plot_utils import draw_numeric_violins
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "experiments" / "artifacts" / "3_outliers" / "results.csv"
OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "3_outliers.png"
METHODS = None
CENTRAL_TENDENCY = "mean"
DPI = 300


df = load_results(CSV_PATH)
methods = pick_methods(df, METHODS)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

draw_numeric_violins(axes[0, 0], df, "n_outliers", "chamfer", methods, "Number of outliers", "Chamfer", "Chamfer vs Outliers", CENTRAL_TENDENCY)
draw_numeric_violins(axes[0, 1], df, "n_outliers", "hausdorff", methods, "Number of outliers", "Hausdorff", "Hausdorff vs Outliers", CENTRAL_TENDENCY)
draw_numeric_violins(axes[1, 0], df, "n_outliers", "accuracy", methods, "Number of outliers", "Accuracy", "Accuracy vs Outliers", CENTRAL_TENDENCY)
draw_numeric_violins(axes[1, 1], df, "n_outliers", "runtime_s", methods, "Number of outliers", "Runtime [s]", "Runtime vs Outliers", CENTRAL_TENDENCY)

for ax in axes.ravel():
    ax.legend(handles=build_method_legend(methods), loc="best", frameon=True)

fig.tight_layout()
save_figure(fig, OUTPUT_PATH, dpi=DPI)
plt.close(fig)
