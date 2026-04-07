import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / "experiments" / "artifacts" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_common import build_test_cloud
from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.inner_ransac import inner_ransac
from src.gair_ransac.ransac import ransac as lo_ransac


OUTLIER_RATIOS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
RUNS = 20
N_SURFACE_POINTS = 450
NOISE_STD = 0.20
THRESHOLD = None
THRESHOLD_SCALE = 2.5
ITERATION_BUDGET = 500
INNER_ITERATIONS = 50
GRAPH_RADIUS = 0.06
BASE_SEED = 42
DATASET_ID = "sq_single_01"
SCENARIO = "single_model"
OUTPUT_DIR = ROOT / "experiments" / "artifacts" / "figure5_single_model"
CSV_PATH = OUTPUT_DIR / "results.csv"

METHODS = [
    "Ransac",
    "Ransac + LO",
    "Ransac + GAIR",
]


def get_threshold():
    if THRESHOLD is not None:
        return float(THRESHOLD)
    return max(float(THRESHOLD_SCALE) * float(NOISE_STD), 1e-3)


def ratio_to_outlier_count(outlier_ratio):
    return int(round(N_SURFACE_POINTS * outlier_ratio / (1.0 - outlier_ratio)))


def merge_masks(inlier_masks, n_points):
    if not inlier_masks:
        return np.zeros(n_points, dtype=bool)
    merged = np.asarray(inlier_masks[0], dtype=bool).copy()
    for mask in inlier_masks[1:]:
        merged |= np.asarray(mask, dtype=bool)
    return merged


def run_method(method, points, normals, threshold, seed):
    start = time.perf_counter()

    if method == "Ransac":
        result = inner_ransac(
            point_cloud=points,
            refined_set_index=np.arange(points.shape[0], dtype=np.int64),
            actual_set_index=np.arange(points.shape[0], dtype=np.int64),
            threshold=threshold,
            n_iters=ITERATION_BUDGET,
            random_seed=seed,
        )
        predicted_inliers = np.asarray(result.best_inliers_mask, dtype=bool)
    elif method == "Ransac + LO":
        _, inlier_masks = lo_ransac(
            point_cloud=points,
            threshold=threshold,
            max_models=1,
            max_iterations=ITERATION_BUDGET,
            inner_iterations=INNER_ITERATIONS,
            radius=GRAPH_RADIUS,
            graphcut=False,
            random_seed=seed,
        )
        predicted_inliers = merge_masks(inlier_masks, points.shape[0])
    elif method == "Ransac + GAIR":
        _, inlier_masks, _ = gair_ransac(
            point_cloud=points,
            normals=normals,
            threshold=threshold,
            max_models=1,
            max_iterations=ITERATION_BUDGET,
            inner_iterations=INNER_ITERATIONS,
            radius=GRAPH_RADIUS,
            random_seed=seed,
            use_normal_coherence=True,
        )
        predicted_inliers = merge_masks(inlier_masks, points.shape[0])
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.perf_counter() - start
    return predicted_inliers, runtime


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = get_threshold()
    rows = []

    print(f"noise_std = {NOISE_STD}")
    print(f"threshold = {threshold}")
    print(f"iteration_budget = {ITERATION_BUDGET}")

    for outlier_ratio in OUTLIER_RATIOS:
        n_outliers = ratio_to_outlier_count(outlier_ratio)
        print(f"\noutlier_ratio = {outlier_ratio:.2f} -> n_outliers = {n_outliers}")

        for run_id in range(1, RUNS + 1):
            cloud_seed = BASE_SEED + run_id + int(round(outlier_ratio * 10_000))
            points, normals, gt_inliers = build_test_cloud(NOISE_STD, N_SURFACE_POINTS, n_outliers, cloud_seed)

            for method_idx, method in enumerate(METHODS):
                method_seed = cloud_seed + 100_000 * (method_idx + 1)
                predicted_inliers, runtime = run_method(method, points, normals, threshold, method_seed)
                misclassification_error = float(np.mean(predicted_inliers != gt_inliers))
                num_inliers = int(predicted_inliers.sum())

                print(
                    f"  run={run_id:02d} | {method:14s} | "
                    f"mis={misclassification_error:.4f} | inliers={num_inliers}"
                )

                rows.append({
                    "scenario": SCENARIO,
                    "iteration_budget": ITERATION_BUDGET,
                    "dataset_id": DATASET_ID,
                    "run_id": run_id,
                    "method": method,
                    "outlier_ratio": outlier_ratio,
                    "outlier_count": n_outliers,
                    "noise_std": NOISE_STD,
                    "misclassification_error": misclassification_error,
                    "num_inliers": num_inliers,
                    "execution_time_s": runtime,
                })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "iteration_budget",
                "dataset_id",
                "run_id",
                "method",
                "outlier_ratio",
                "outlier_count",
                "noise_std",
                "misclassification_error",
                "num_inliers",
                "execution_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
