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

import src.gair_ransac.gair_ransac as gair_module
import src.gair_ransac.ransac as lo_module
from src.gair_ransac.vanilla_ransac import vanilla_ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.superquadrics.superquadric_param import SuperQuadricParams


GT_PARAMS = [
    SuperQuadricParams(9.0, 9.0, 9.0, 3.5, 2.09, [2.0, 2.0, 1.0], [5.0, 5.0, 5.0]),
    SuperQuadricParams(3.0, 3.0, 3.0, 0.5, 0.9, [2.0, 2.0, 1.0], [-5.0, -5.0, -5.0]),
    SuperQuadricParams(4.0, 4.0, 4.0, 0.8, 1.1, [1.7, 2.1, 0.9], [13.0, 5.0, 5.0]),
    SuperQuadricParams(2.5, 2.5, 2.5, 0.7, 1.2, [2.1, 1.9, 0.8], [-10.5, -5.0, -5.0]),
]


OUTLIER_RATIOS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
RUNS = 10
NOISE_STD = 0.20
THRESHOLD = None
THRESHOLD_SCALE = 2.5
N_POINTS_PER_MODEL = 120
INNER_ITERATIONS = 50
GRAPH_RADIUS = 0.06
BASE_SEED = 42
SCENARIO = "multi_model"
DATASET_ID = "sq_multi_01"
OUTPUT_DIR = ROOT / "experiments" / "artifacts" / "figure6_7_multi_model"
CSV_PATH = OUTPUT_DIR / "results.csv"

METHODS_BY_BUDGET = {
    500: ["Ransac", "Ransac + LO", "Ransac + GAIR"],
    5000: ["Ransac + LO", "Ransac + GAIR"],
}


GT_MESHES = [supmesh.superquadric_mesh(params) for params in GT_PARAMS]


def get_threshold():
    if THRESHOLD is not None:
        return float(THRESHOLD)
    return max(float(THRESHOLD_SCALE) * float(NOISE_STD), 1e-3)


def ratio_to_outlier_count(outlier_ratio):
    n_inliers = len(GT_PARAMS) * N_POINTS_PER_MODEL
    return int(round(n_inliers * outlier_ratio / (1.0 - outlier_ratio)))


def build_multi_model_cloud(outlier_ratio, seed):
    n_outliers = ratio_to_outlier_count(outlier_ratio)
    sampled_points_noisy, normals_noisy = samp.sampling_sq_noisy(
        GT_MESHES,
        n_points=N_POINTS_PER_MODEL,
        noise_std=NOISE_STD,
        clip_k=3.0,
        seed=seed,
    )
    sampled_points_outliers, normals_outliers = samp.sampling_outliers(
        GT_MESHES,
        n_out=n_outliers,
        margin=0.10,
        mode="uniform",
        seed=seed + 10_000,
    )

    points = np.vstack([*sampled_points_noisy, sampled_points_outliers]).astype(np.float64)
    normals = np.vstack([*normals_noisy, normals_outliers]).astype(np.float64)
    gt_inliers = np.zeros(points.shape[0], dtype=bool)
    gt_inliers[: sum(len(chunk) for chunk in sampled_points_noisy)] = True
    return points, normals, gt_inliers, n_outliers


def merge_masks(inlier_masks, n_points):
    if not inlier_masks:
        return np.zeros(n_points, dtype=bool)
    merged = np.asarray(inlier_masks[0], dtype=bool).copy()
    for mask in inlier_masks[1:]:
        merged |= np.asarray(mask, dtype=bool)
    return merged


def run_with_local_counter(module, fn):
    original = module.inner_ransac
    counter = {"value": 0}

    def wrapped(*args, **kwargs):
        counter["value"] += 1
        return original(*args, **kwargs)

    module.inner_ransac = wrapped
    try:
        result = fn()
    finally:
        module.inner_ransac = original
    return result, counter["value"]


def run_method(method, points, normals, threshold, iteration_budget, seed):
    start = time.perf_counter()

    if method == "Ransac":
        models, inlier_masks = vanilla_ransac(
            points,
            normals,
            threshold=threshold,
            max_models=len(GT_PARAMS),
            max_iterations=iteration_budget,
            consensus_metric="radial",
            random_seed=seed,
        )
        local_steps = 0
    elif method == "Ransac + LO":
        (models, inlier_masks), local_steps = run_with_local_counter(
            lo_module,
            lambda: lo_module.ransac(
                point_cloud=points,
                threshold=threshold,
                max_models=len(GT_PARAMS),
                max_iterations=iteration_budget,
                inner_iterations=INNER_ITERATIONS,
                radius=GRAPH_RADIUS,
                graphcut=False,
                random_seed=seed,
            ),
        )
    elif method == "Ransac + GAIR":
        (models, inlier_masks, _), local_steps = run_with_local_counter(
            gair_module,
            lambda: gair_module.gair_ransac(
                point_cloud=points,
                normals=normals,
                threshold=threshold,
                max_models=len(GT_PARAMS),
                max_iterations=iteration_budget,
                inner_iterations=INNER_ITERATIONS,
                radius=GRAPH_RADIUS,
                consensus_metric="radial",
                random_seed=seed,
                use_normal_coherence=True,
            ),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.perf_counter() - start
    predicted_inliers = merge_masks(inlier_masks, points.shape[0])
    return predicted_inliers, runtime, int(local_steps), len(models)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = get_threshold()
    rows = []

    print(f"noise_std = {NOISE_STD}")
    print(f"threshold = {threshold}")
    print(f"points_per_model = {N_POINTS_PER_MODEL}")

    for iteration_budget, methods in METHODS_BY_BUDGET.items():
        print(f"\n=== iteration_budget = {iteration_budget} ===")
        for outlier_ratio in OUTLIER_RATIOS:
            print(f"\noutlier_ratio = {outlier_ratio:.2f}")

            for run_id in range(1, RUNS + 1):
                cloud_seed = BASE_SEED + run_id + int(round(outlier_ratio * 10_000)) + 1_000_000 * iteration_budget
                points, normals, gt_inliers, n_outliers = build_multi_model_cloud(outlier_ratio, cloud_seed)

                for method_idx, method in enumerate(methods):
                    method_seed = cloud_seed + 100_000 * (method_idx + 1)
                    predicted_inliers, runtime, local_steps, n_models = run_method(
                        method,
                        points,
                        normals,
                        threshold,
                        iteration_budget,
                        method_seed,
                    )
                    misclassification_error = float(np.mean(predicted_inliers != gt_inliers))

                    print(
                        f"  run={run_id:02d} | {method:14s} | "
                        f"mis={misclassification_error:.4f} | time={runtime:.2f}s | lo={local_steps}"
                    )

                    rows.append({
                        "scenario": SCENARIO,
                        "iteration_budget": iteration_budget,
                        "dataset_id": DATASET_ID,
                        "run_id": run_id,
                        "method": method,
                        "outlier_ratio": outlier_ratio,
                        "outlier_count": n_outliers,
                        "noise_std": NOISE_STD,
                        "misclassification_error": misclassification_error,
                        "execution_time_s": runtime,
                        "local_optimization_steps": local_steps,
                        "n_models": n_models,
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
                "execution_time_s",
                "local_optimization_steps",
                "n_models",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
