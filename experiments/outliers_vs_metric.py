import os
from pathlib import Path

# This script benchmarks one selected metric while the number of outliers grows.
# Each curve uses a different hypothesis budget, so we can compare
# GAIR-RANSAC with few hypotheses against RANSAC with many hypotheses.
# Keep one BLAS thread per worker to avoid oversubscription when using many processes.
BLAS_THREADS_PER_WORKER = 1
if BLAS_THREADS_PER_WORKER is not None:
    os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["NUMEXPR_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)

from benchmark_common import run_outlier_hypotheses_benchmark

# Edit these variables
Y_METRIC = "chamfer"  # "misclassification" or "chamfer"
FIXED_NOISE_STD = 0.2
THRESHOLD = 3.0 * FIXED_NOISE_STD
OUTLIER_VALUES = None  # If None, defaults to range(OUTLIER_START, OUTLIER_STOP + 1, OUTLIER_STEP)
OUTLIER_START = 1000
OUTLIER_STOP = 4000
OUTLIER_STEP = 500
RUNS = 1
N_SURFACE_POINTS = 10000
GRAPH_RADIUS = 0.06  # Relative to the bounding box diagonal.
GAIR_INNER_ITERATIONS = 50 # inner-ransac will compute 150 iteration for worst case scenario
MAX_WORKERS = 4 # None uses all available CPU cores.
USE_MULTIPROCESSING = True
BASE_SEED = 42
HYPOTHESIS_VALUES = [150,450,1000]

if Y_METRIC not in {"misclassification", "chamfer"}:
    raise ValueError("Y_METRIC must be 'misclassification' or 'chamfer'")

OUTPUT_DIR = Path("artifacts") / f"outliers_vs_{Y_METRIC}_hypotheses"

# We choose the current best-looking GAIR setup and compare it to vanilla RANSAC.
CURVES = [
    (f"RANSAC m={m}", "ransac", "first_order", True, Y_METRIC, m)
    for m in HYPOTHESIS_VALUES
] + [
    (f"GAIR-RANSAC m={3}(inner={GAIR_INNER_ITERATIONS})", "gair-ransac", "first_order", True, Y_METRIC, 5)
]


def _metric_title(metric_name: str) -> str:
    if metric_name == "misclassification":
        return "misclassification error"
    if metric_name == "chamfer":
        return "symmetric Chamfer distance"
    return metric_name


def main() -> int:
    return run_outlier_hypotheses_benchmark(
        title=f"Outliers vs {_metric_title(Y_METRIC)} - noise_std = {FIXED_NOISE_STD}, threshold = {THRESHOLD}",
        output_dir=OUTPUT_DIR,
        curves=CURVES,
        fixed_noise_std=FIXED_NOISE_STD,
        outlier_values=OUTLIER_VALUES,
        outlier_start=OUTLIER_START,
        outlier_stop=OUTLIER_STOP,
        outlier_step=OUTLIER_STEP,
        runs=RUNS,
        n_surface_points=N_SURFACE_POINTS,
        threshold_value=THRESHOLD,
        threshold_scale=None,
        graph_radius=GRAPH_RADIUS,
        gair_inner_iterations=GAIR_INNER_ITERATIONS,
        max_workers=MAX_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,
        base_seed=BASE_SEED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
