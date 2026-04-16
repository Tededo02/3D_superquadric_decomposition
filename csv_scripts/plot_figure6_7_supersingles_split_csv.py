from pathlib import Path

import plot_figure6_7_multi_model_split_csv as multi_model_plot


ROOT = Path(__file__).resolve().parents[1]


def main():
    multi_model_plot.DEFAULT_RESULTS_DIR = ROOT / "finale" / "supersingles"
    multi_model_plot.DEFAULT_OUTPUT_PATH = ROOT / "csv_scripts" / "out3" / "figure5_supersingles_split_csv.png"
    multi_model_plot.main()


if __name__ == "__main__":
    main()
