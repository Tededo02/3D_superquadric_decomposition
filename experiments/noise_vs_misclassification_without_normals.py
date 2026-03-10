import os
from pathlib import Path

# This script compares GAIR-RANSAC with and without normal coherence while noise varies
# with epsilon=3*sigma
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
NOISE_STOP = 2.0
NOISE_STEP = 0.1
RUNS = 1
N_SURFACE_POINTS = 15000
N_OUTLIERS = 500
THRESHOLD_SCALE = 3.0
GRAPH_RADIUS = 0.06  # Relative to the bounding box diagonal.
MAX_ITERATIONS = 150
INNER_ITERATIONS = 100
MAX_WORKERS = None  # None uses all available CPU cores.
USE_MULTIPROCESSING = True
BASE_SEED = 42
OUTPUT_DIR = Path("artifacts") / "noise_vs_misclassification_without_normals"
CURVES = [
    ("GAIR-RANSAC with normals", "gair-ransac", "first_order", True),
    ("GAIR-RANSAC without normals", "gair-ransac", "first_order", False),
]


def main() -> int:
    return run_benchmark(
        title=f"Noise vs misclassification error - normals ablation with threshold = {THRESHOLD_SCALE} * noise_std",
        output_dir=OUTPUT_DIR,
        curves=CURVES,
        noise_values=NOISE_VALUES,
        noise_start=NOISE_START,
        noise_stop=NOISE_STOP,
        noise_step=NOISE_STEP,
        runs=RUNS,
        n_surface_points=N_SURFACE_POINTS,
        n_outliers=N_OUTLIERS,
        threshold_value=None,
        threshold_scale=THRESHOLD_SCALE,
        graph_radius=GRAPH_RADIUS,
        max_iterations=MAX_ITERATIONS,
        inner_iterations=INNER_ITERATIONS,
        max_workers=MAX_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,
        base_seed=BASE_SEED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
