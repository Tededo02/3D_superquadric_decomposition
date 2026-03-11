import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.inner_ransac import inner_ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.superquadrics.superquadric_param import SuperQuadricParams

TEST_MODEL = SuperQuadricParams(9.0, 9.0, 9.0, 3.5, 2.09, [2.0, 2.0, 1.0], [5.0, 5.0, 5.0])
TEST_MESH = supmesh.superquadric_mesh(TEST_MODEL)


def build_sweep_values(values: list[float] | None, start: float, stop: float, step: float, step_name: str, start_name: str, stop_name: str) -> list[float]:
    if values is not None:
        return list(values)
    if step <= 0.0:
        raise ValueError(f"{step_name} must be positive")
    if stop < start:
        raise ValueError(f"{stop_name} must be greater than or equal to {start_name}")
    return np.round(
        np.arange(start, stop + 0.5 * step, step),
        decimals=10,
    ).tolist()


def build_test_cloud(noise_std: float, n_surface_points: int, n_outliers: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_points_noisy, normals_noisy = samp.sampling_sq_noisy([TEST_MESH], n_points=n_surface_points, noise_std=noise_std,normal_noise_std=0.5, clip_k=3.0, seed=seed)
    sampled_points_outliers, normals_outliers = samp.sampling_outliers([TEST_MESH], n_out=n_outliers, margin=0.10, mode="uniform", seed=seed + 10_000)

    surface_points = np.vstack(sampled_points_noisy).astype(np.float32, copy=False)
    surface_normals = np.vstack(normals_noisy).astype(np.float32, copy=False)
    points = np.vstack([surface_points,sampled_points_outliers]).astype(np.float32, copy=False)
    normals = np.vstack([surface_normals,normals_outliers]).astype(np.float32, copy=False)

    gt_inliers = np.zeros(points.shape[0], dtype=bool)
    gt_inliers[:surface_points.shape[0]] = True
    return points, normals, gt_inliers


def run_trial(job: tuple[str, str, str, bool, float, int, int, int, int, float | None, float | None, float, int, int]) -> dict[str, float | str | bool]:
    label, algorithm, error_metric, use_normal_coherence, noise_std, run_idx, base_seed, n_surface_points, n_outliers, threshold_value, threshold_scale, graph_radius, max_iterations, inner_iterations = job
    seed = base_seed + run_idx + int(round(noise_std * 10_000))
    points, normals, gt_inliers = build_test_cloud(noise_std, n_surface_points, n_outliers, seed)
    if threshold_value is None:
        if threshold_scale is None:
            raise ValueError("THRESHOLD_SCALE must be set when THRESHOLD is None")
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
            use_normal_coherence=use_normal_coherence,
        )
        if inlier_masks:
            predicted_inliers = np.asarray(inlier_masks[0], dtype=bool)

    return {
        "curve": label,
        "algorithm": algorithm,
        "error_metric": error_metric,
        "use_normal_coherence": use_normal_coherence,
        "noise_std": noise_std,
        "threshold_scale": threshold_scale,
        "threshold": threshold,
        "run_idx": run_idx,
        "misclassification_error": float(np.mean(predicted_inliers != gt_inliers)),
    }


def run_benchmark(
    title: str,
    output_dir: Path,
    curves: list[tuple[str, str, str, bool]],
    noise_values: list[float] | None,
    noise_start: float,
    noise_stop: float,
    noise_step: float,
    runs: int,
    n_surface_points: int,
    n_outliers: int,
    threshold_value: float | None,
    threshold_scale: float | None,
    graph_radius: float,
    max_iterations: int,
    inner_iterations: int,
    max_workers: int | None,
    use_multiprocessing: bool,
    base_seed: int,
) -> int:
    noise_grid = build_sweep_values(noise_values, noise_start, noise_stop, noise_step, "NOISE_STEP", "NOISE_START", "NOISE_STOP")
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = [
        (
            label,
            algorithm,
            error_metric,
            use_normal_coherence,
            noise_std,
            run_idx,
            base_seed,
            n_surface_points,
            n_outliers,
            threshold_value,
            threshold_scale,
            graph_radius,
            max_iterations,
            inner_iterations,
        )
        for label, algorithm, error_metric, use_normal_coherence in curves
        for noise_std in noise_grid
        for run_idx in range(runs)
    ]

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            raw_rows = list(executor.map(run_trial, jobs))
    else:
        raw_rows = [run_trial(job) for job in jobs]

    summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, _, _, _ in curves:
        means: list[float] = []
        stds: list[float] = []
        for noise_std in noise_grid:
            errors = [
                float(row["misclassification_error"])
                for row in raw_rows
                if row["curve"] == label and row["noise_std"] == noise_std
            ]
            means.append(float(np.mean(errors)))
            stds.append(float(np.std(errors)))
        summary[label] = (
            np.asarray(noise_grid, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            np.asarray(stds, dtype=np.float64),
        )

    csv_path = output_dir / "raw_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["curve", "algorithm", "error_metric", "use_normal_coherence", "noise_std", "threshold_scale", "threshold", "run_idx", "misclassification_error"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    plot_path = output_dir / "noise_vs_misclassification.png"
    plt.figure(figsize=(9, 5))
    for label, (curve_noise, mean, std) in summary.items():
        plt.plot(curve_noise, mean, marker="o", linewidth=2, label=label)
        plt.fill_between(curve_noise, mean - std, mean + std, alpha=0.15)
    plt.xlabel("noise_std")
    plt.ylabel("misclassification error")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    for label, (curve_noise, mean, _) in summary.items():
        pairs = ", ".join(f"{noise:.2f}:{err:.4f}" for noise, err in zip(curve_noise, mean))
        print(f"{label} -> {pairs}")
    print(f"Saved raw results to {csv_path}")
    print(f"Saved plot to {plot_path}")
    return 0


def run_scale_factor_benchmark(
    title: str,
    output_dir: Path,
    curves: list[tuple[str, str, str, bool]],
    fixed_noise_std: float,
    scale_values: list[float] | None,
    scale_start: float,
    scale_stop: float,
    scale_step: float,
    runs: int,
    n_surface_points: int,
    n_outliers: int,
    graph_radius: float,
    max_iterations: int,
    inner_iterations: int,
    max_workers: int | None,
    use_multiprocessing: bool,
    base_seed: int,
) -> int:
    scale_grid = build_sweep_values(scale_values, scale_start, scale_stop, scale_step, "SCALE_STEP", "SCALE_START", "SCALE_STOP")
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = [
        (
            label,
            algorithm,
            error_metric,
            use_normal_coherence,
            fixed_noise_std,
            run_idx,
            base_seed,
            n_surface_points,
            n_outliers,
            None,
            scale_factor,
            graph_radius,
            max_iterations,
            inner_iterations,
        )
        for label, algorithm, error_metric, use_normal_coherence in curves
        for scale_factor in scale_grid
        for run_idx in range(runs)
    ]

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            raw_rows = list(executor.map(run_trial, jobs))
    else:
        raw_rows = [run_trial(job) for job in jobs]

    summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, _, _, _ in curves:
        means: list[float] = []
        stds: list[float] = []
        for scale_factor in scale_grid:
            errors = [
                float(row["misclassification_error"])
                for row in raw_rows
                if row["curve"] == label and row["threshold_scale"] == scale_factor
            ]
            means.append(float(np.mean(errors)))
            stds.append(float(np.std(errors)))
        summary[label] = (
            np.asarray(scale_grid, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            np.asarray(stds, dtype=np.float64),
        )

    csv_path = output_dir / "raw_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["curve", "algorithm", "error_metric", "use_normal_coherence", "noise_std", "threshold_scale", "threshold", "run_idx", "misclassification_error"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    plot_path = output_dir / "scale_factor_vs_misclassification.png"
    plt.figure(figsize=(9, 5))
    for label, (curve_scale, mean, std) in summary.items():
        plt.plot(curve_scale, mean, marker="o", linewidth=2, label=label)
        plt.fill_between(curve_scale, mean - std, mean + std, alpha=0.15)
    plt.xlabel("threshold scale factor")
    plt.ylabel("misclassification error")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    for label, (curve_scale, mean, _) in summary.items():
        pairs = ", ".join(f"{scale:.2f}:{err:.4f}" for scale, err in zip(curve_scale, mean))
        print(f"{label} -> {pairs}")
    print(f"Saved raw results to {csv_path}")
    print(f"Saved plot to {plot_path}")
    return 0
