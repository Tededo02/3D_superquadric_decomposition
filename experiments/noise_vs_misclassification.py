import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys


# block the use of multiple threads by the function from other libraries (like numpy or scipy) to avoid oversubscription when using multiprocessing
# Keep one BLAS thread per worker to avoid oversubscription when using many processes.
BLAS_THREADS_PER_WORKER = 1
if BLAS_THREADS_PER_WORKER is not None:
    os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
    os.environ["NUMEXPR_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.inner_ransac import inner_ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.superquadrics.superquadric_param import SuperQuadricParams

# Edit these variables
NOISE_VALUES = None
NOISE_START = 0.1
NOISE_STOP = 1.0
NOISE_STEP = 0.1
RUNS = 1
N_SURFACE_POINTS = 10000
N_OUTLIERS = 500
THRESHOLD = 0.6 # Set to None to use THRESHOLD_SCALE * noise_std instead.!!!!!!!!!!!!!!!!!!
THRESHOLD_SCALE = 3.0
GRAPH_RADIUS = 0.06 #factor of the bounding box diagonal
MAX_ITERATIONS = 150 #number of iterations for GAIR-RANSAC (outer loop)
INNER_ITERATIONS = 50 #number of iterations for the inner RANSAC used by both algorithms (for fair comparison)
MAX_WORKERS = None # None uses all available CPU cores
USE_MULTIPROCESSING = True
BASE_SEED = 42
OUTPUT_DIR = Path("artifacts") / "noise_vs_misclassification"
CURVES = [
    ("RANSAC (mix)", "ransac", "mix"),
    ("GAIR-RANSAC (radial)", "gair-ransac", "radial"),
    ("GAIR-RANSAC (first_order)", "gair-ransac", "first_order"),
    ("GAIR-RANSAC (mix)", "gair-ransac", "mix"),
]
TEST_MODEL = SuperQuadricParams(3.0, 3.0, 3.0, 3.5, 2.09, [2.0, 2.0, 1.0], [5.0, 5.0, 5.0])
TEST_MESH = supmesh.superquadric_mesh(TEST_MODEL)


def build_test_cloud(noise_std: float, n_surface_points: int, n_outliers: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_points_noisy, normals_noisy = samp.sampling_sq_noisy([TEST_MESH], n_points=n_surface_points, noise_std=noise_std, clip_k=3.0, seed=seed)
    sampled_points_outliers, normals_outliers = samp.sampling_outliers([TEST_MESH], n_out=n_outliers, margin=0.10, mode="uniform", seed=seed + 10_000)

    surface_points = np.vstack(sampled_points_noisy).astype(np.float32, copy=False)
    surface_normals = np.vstack(normals_noisy).astype(np.float32, copy=False)
    points = np.vstack([surface_points, sampled_points_outliers]).astype(np.float32, copy=False)
    normals = np.vstack([surface_normals, normals_outliers]).astype(np.float32, copy=False)

    gt_inliers = np.zeros(points.shape[0], dtype=bool)
    gt_inliers[:surface_points.shape[0]] = True
    return points, normals, gt_inliers


def run_trial(job: tuple[str, str, str, float, int, int, int, int, float, float | None, float, int, int]) -> dict[str, float | str]:
    label, algorithm, error_metric, noise_std, run_idx, base_seed, n_surface_points, n_outliers, threshold_scale, threshold_value, graph_radius, max_iterations, inner_iterations = job
    seed = base_seed + run_idx + int(round(noise_std * 10_000))
    points, normals, gt_inliers = build_test_cloud(noise_std, n_surface_points, n_outliers, seed)
    if threshold_value is None:
        threshold = max(threshold_scale * noise_std, 1e-3)
    else:
        threshold = float(threshold_value)
    predicted_inliers = np.zeros(points.shape[0], dtype=bool)

    if algorithm == "ransac":
        result = inner_ransac(
            point_cloud=points,
            refined_set_index=np.arange(points.shape[0], dtype=np.int64),
            actual_set_index=np.arange(points.shape[0], dtype=np.int64),
            threshold=threshold,
            error_metric=error_metric,
            n_iters=inner_iterations,
            random_seed=seed,
        )
        if result.best_inlier_count > 0 and result.best_inliers_mask.size == points.shape[0]:
            predicted_inliers = np.asarray(result.best_inliers_mask, dtype=bool)
    else:
        _, inlier_masks = gair_ransac(
            point_cloud=points,
            normals=normals,
            threshold=threshold,
            max_models=1,
            max_iterations=max_iterations,
            radius=graph_radius,
            error_metric=error_metric,
            inner_iterations=inner_iterations,
            random_seed=seed,
        )
        if inlier_masks:
            predicted_inliers = np.asarray(inlier_masks[0], dtype=bool)

    return {
        "curve": label,
        "algorithm": algorithm,
        "error_metric": error_metric,
        "noise_std": noise_std,
        "run_idx": run_idx,
        "misclassification_error": float(np.mean(predicted_inliers != gt_inliers)),
    }


def main() -> int:
    noise_values = NOISE_VALUES
    if noise_values is None:
        if NOISE_STEP <= 0.0:
            raise ValueError("NOISE_STEP must be positive")
        if NOISE_STOP < NOISE_START:
            raise ValueError("noise_stop must be greater than or equal to noise_start")
        noise_values = np.round(
            np.arange(NOISE_START, NOISE_STOP + 0.5 * NOISE_STEP, NOISE_STEP),
            decimals=10,
        ).tolist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            label,
            algorithm,
            error_metric,
            noise_std,
            run_idx,
            BASE_SEED,
            N_SURFACE_POINTS,
            N_OUTLIERS,
            THRESHOLD_SCALE,
            THRESHOLD,
            GRAPH_RADIUS,
            MAX_ITERATIONS,
            INNER_ITERATIONS,
        )
        for label, algorithm, error_metric in CURVES
        for noise_std in noise_values
        for run_idx in range(RUNS)
    ]
    raw_rows: list[dict[str, float | str]] = []
    summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    if USE_MULTIPROCESSING:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            raw_rows = list(executor.map(run_trial, jobs))
    else:
        raw_rows = [run_trial(job) for job in jobs]

    for label, algorithm, error_metric in CURVES:
        means: list[float] = []
        stds: list[float] = []
        for noise_std in noise_values:
            errors = [
                float(row["misclassification_error"])
                for row in raw_rows
                if row["curve"] == label and row["noise_std"] == noise_std
            ]
            means.append(float(np.mean(errors)))
            stds.append(float(np.std(errors)))
        summary[label] = (
            np.asarray(noise_values, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            np.asarray(stds, dtype=np.float64),
        )

    csv_path = OUTPUT_DIR / "raw_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["curve", "algorithm", "error_metric", "noise_std", "run_idx", "misclassification_error"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    plot_path = OUTPUT_DIR / "noise_vs_misclassification.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for label, (noise_std, mean, std) in summary.items():
        plt.plot(noise_std, mean, marker="o", linewidth=2, label=label)
        plt.fill_between(noise_std, mean - std, mean + std, alpha=0.15)
    plt.xlabel("noise_std")
    plt.ylabel("misclassification error")
    plt.title("Noise vs misclassification error")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    for label, (noise_std, mean, _) in summary.items():
        pairs = ", ".join(f"{noise:.2f}:{err:.4f}" for noise, err in zip(noise_std, mean))
        print(f"{label} -> {pairs}")

    print(f"Saved raw results to {csv_path}")
    print(f"Saved plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
