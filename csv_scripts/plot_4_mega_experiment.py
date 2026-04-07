from pathlib import Path

from common_plot_utils import build_method_legend
from common_plot_utils import draw_numeric_violins
from common_plot_utils import load_results
from common_plot_utils import pick_methods
from common_plot_utils import save_figure
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "experiments" / "artifacts" / "4_mega_experiment" / "results.csv"
OUTPUT_PATH = ROOT / "csv_scripts" / "out" / "4_mega_experiment.png"
METHODS = None
CENTRAL_TENDENCY = "mean"
DPI = 300

SWEEP = "noise_std"
FIXED_NOISE = 0.3
FIXED_OUTLIERS = 2000


df = load_results(CSV_PATH)

if SWEEP == "noise_std":
    x_col = "noise_std"
    x_label = "Noise std"
    subtitle = f"fixed outliers = {FIXED_OUTLIERS:g}"
    df = df[df["n_outliers"] == FIXED_OUTLIERS].copy()
else:
    x_col = "n_outliers"
    x_label = "Number of outliers"
    subtitle = f"fixed noise = {FIXED_NOISE:g}"
    df = df[df["noise_std"] == FIXED_NOISE].copy()

methods = pick_methods(df, METHODS)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

draw_numeric_violins(axes[0, 0], df, x_col, "chamfer", methods, x_label, "Chamfer", f"Chamfer ({subtitle})", CENTRAL_TENDENCY)
draw_numeric_violins(axes[0, 1], df, x_col, "hausdorff", methods, x_label, "Hausdorff", f"Hausdorff ({subtitle})", CENTRAL_TENDENCY)
draw_numeric_violins(axes[1, 0], df, x_col, "accuracy", methods, x_label, "Accuracy", f"Accuracy ({subtitle})", CENTRAL_TENDENCY)
draw_numeric_violins(axes[1, 1], df, x_col, "runtime_s", methods, x_label, "Runtime [s]", f"Runtime ({subtitle})", CENTRAL_TENDENCY)

for ax in axes.ravel():
    ax.legend(handles=build_method_legend(methods), loc="best", frameon=True)

fig.tight_layout()
save_figure(fig, OUTPUT_PATH, dpi=DPI)
plt.close(fig)
