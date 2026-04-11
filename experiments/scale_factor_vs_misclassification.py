import os
from pathlib import Path

# This script benchmarks misclassification error versus threshold scale factor at fixed noise.
# Keep one BLAS thread per worker to avoid oversubscription when using many processes.
BLAS_THREADS_PER_WORKER = 1
if BLAS_THREADS_PER_WORKER is not None:
    os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["NUMEXPR_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)


from benchmark_common import run_scale_factor_benchmark



# Edit these variables
FIXED_NOISE_STD = 1.3
SCALE_VALUES = None # If None, it will be generated from SCALE_START, SCALE_STOP and SCALE_STEP
SCALE_START = 0.0
SCALE_STOP = 4.5
SCALE_STEP = 0.2
RUNS = 10
N_SURFACE_POINTS = 10000
N_OUTLIERS = 2000
GRAPH_RADIUS = 0.06  # Relative to the bounding box diagonal.
MAX_ITERATIONS = 10
INNER_ITERATIONS = 10
MAX_WORKERS = 4  # None uses all available CPU cores.
USE_MULTIPROCESSING = True
BASE_SEED = 42
OUTPUT_DIR = Path("artifacts") / "scale_factor_vs_misclassification"
CURVES = [
    ("RANSAC", "ransac", True),
    ("GAIR-RANSAC", "gair-ransac", True, "misclassification"),
]


def main() -> int:
    return run_scale_factor_benchmark(
        title=f"Scale factor vs misclassification error - fixed noise_std = {FIXED_NOISE_STD}",
        output_dir=OUTPUT_DIR,
        curves=CURVES,
        fixed_noise_std=FIXED_NOISE_STD,
        scale_values=SCALE_VALUES,
        scale_start=SCALE_START,
        scale_stop=SCALE_STOP,
        scale_step=SCALE_STEP,
        runs=RUNS,
        n_surface_points=N_SURFACE_POINTS,
        n_outliers=N_OUTLIERS,
        graph_radius=GRAPH_RADIUS,
        max_iterations=MAX_ITERATIONS,
        inner_iterations=INNER_ITERATIONS,
        max_workers=MAX_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,
        base_seed=BASE_SEED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
