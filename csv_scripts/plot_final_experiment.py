from pathlib import Path

from common_plot_utils import build_method_legend
from common_plot_utils import draw_category_violins
from common_plot_utils import draw_scatter_with_pareto
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "experiments" / "artifacts" / "final_experiment" / "results_sequential.csv"
OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "final_experiment.png"
METHODS = None
CENTRAL_TENDENCY = "mean"
DPI = 300


df = load_results(CSV_PATH)
methods = pick_methods(df, METHODS)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

draw_category_violins(axes[0, 0], df, "chamfer", methods, "Chamfer", "Chamfer per Method", CENTRAL_TENDENCY)
draw_category_violins(axes[0, 1], df, "hausdorff", methods, "Hausdorff", "Hausdorff per Method", CENTRAL_TENDENCY)
draw_category_violins(axes[1, 0], df, "runtime_s", methods, "Runtime [s]", "Runtime per Method", CENTRAL_TENDENCY)
draw_scatter_with_pareto(axes[1, 1], df, methods, "runtime_s", "chamfer", "Runtime [s]", "Chamfer", "Runtime vs Chamfer")

for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.legend(handles=build_method_legend(methods), loc="best", frameon=True)
axes[1, 1].legend(loc="best", frameon=True)

fig.tight_layout()
save_figure(fig, OUTPUT_PATH, dpi=DPI)
plt.close(fig)
