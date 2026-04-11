import os
from pathlib import Path

# This script benchmarks misclassification error versus noise with a fixed inlier threshold.
# its aim is to show how bad misclassification can get when the noise increases but the threshold is not adapted.
# Keep one BLAS thread per worker to avoid oversubscription when using many processes.
BLAS_THREADS_PER_WORKER = 1
if BLAS_THREADS_PER_WORKER is not None:
    os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["NUMEXPR_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)

from benchmark_common import run_benchmark
# Edit these variables
NOISE_VALUES = None
NOISE_START = 0.1
NOISE_STOP = 1.5
NOISE_STEP = 0.1
RUNS = 1
N_SURFACE_POINTS = 10000
N_OUTLIERS = 1000
THRESHOLD = 3.0 # fixed threshold, not adapted to the noise level.
GRAPH_RADIUS = 0.06  # Relative to the bounding box diagonal.
MAX_ITERATIONS = 50
INNER_ITERATIONS = 50
MAX_WORKERS = 8  # None uses all available CPU cores.
USE_MULTIPROCESSING = True
BASE_SEED = 42
OUTPUT_DIR = Path("artifacts") / "noise_vs_misclassification"
CURVES = [
    # misclassification
    ("RANSAC", "ransac", True, "misclassification"),
    ("GAIR-RANSAC", "gair-ransac", True, "misclassification"),
    # chamfer distance
    ("RANSAC chamfer", "ransac", True, "chamfer"),
    ("GAIR-RANSAC chamfer", "gair-ransac", True, "chamfer"),
    # hausdorff distance
    ("RANSAC hausdorff", "ransac", True, "hausdorff"),
    ("GAIR-RANSAC hausdorff", "gair-ransac", True, "hausdorff"),
]


def main() -> int:
    return run_benchmark(
        title=f"Noise vs misclassification error - fixed threshold = {THRESHOLD}",
        output_dir=OUTPUT_DIR,
        curves=CURVES,
        noise_values=NOISE_VALUES,
        noise_start=NOISE_START,
        noise_stop=NOISE_STOP,
        noise_step=NOISE_STEP,
        runs=RUNS,
        n_surface_points=N_SURFACE_POINTS,
        n_outliers=N_OUTLIERS,
        threshold_value=THRESHOLD,
        threshold_scale=None,
        graph_radius=GRAPH_RADIUS,
        max_iterations=MAX_ITERATIONS,
        inner_iterations=INNER_ITERATIONS,
        max_workers=MAX_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,
        base_seed=BASE_SEED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
