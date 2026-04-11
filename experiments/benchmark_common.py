import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

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
_gt_pts_list, _ = samp.sampling_sq_noisy([TEST_MESH], n_points=2000, noise_std=0.0, normal_noise_std=0.0, clip_k=3.0, seed=42)
_GT_POINTS = np.vstack(_gt_pts_list).astype(np.float32)


def _compute_metric(classifier: str, predicted_inliers: np.ndarray, gt_inliers: np.ndarray, fitted_model) -> float:
    if classifier == "misclassification":
        return float(np.mean(predicted_inliers != gt_inliers))
    if fitted_model is None:
        return float("nan")
    try:
        est_mesh = supmesh.superquadric_mesh(fitted_model)
        est_pts_list, _ = samp.sampling_sq_noisy([est_mesh], n_points=2000, noise_std=0.0, normal_noise_std=0.0, clip_k=3.0, seed=0)
        est_pts = np.vstack(est_pts_list).astype(np.float32)
    except Exception:
        return float("nan")
    d_est_to_gt, _ = cKDTree(_GT_POINTS).query(est_pts)
    d_gt_to_est, _ = cKDTree(est_pts).query(_GT_POINTS)
    if classifier == "chamfer":
        return float(np.mean(d_est_to_gt) + np.mean(d_gt_to_est))
    if classifier == "hausdorff":
        return float(max(np.max(d_est_to_gt), np.max(d_gt_to_est)))
    raise ValueError(f"Unknown classifier: {classifier}")


# Build a regular grid when explicit values are not provided.
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


# Build a regular integer grid when explicit values are not provided.
def build_integer_sweep_values(values: list[int] | None, start: int, stop: int, step: int, step_name: str, start_name: str, stop_name: str) -> list[int]:
    if values is not None:
        return [int(v) for v in values]
    if step <= 0:
        raise ValueError(f"{step_name} must be positive")
    if stop < start:
        raise ValueError(f"{stop_name} must be greater than or equal to {start_name}")
    return list(range(int(start), int(stop) + 1, int(step)))


# Build one synthetic single-superquadric cloud with noise and extra outliers.
def build_test_cloud(noise_std: float, n_surface_points: int, n_outliers: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_points_noisy, normals_noisy = samp.sampling_sq_noisy([TEST_MESH], n_points=n_surface_points, noise_std=noise_std, normal_noise_std=0.5, clip_k=3.0, seed=seed)
    sampled_points_outliers, normals_outliers = samp.sampling_outliers([TEST_MESH], n_out=n_outliers, margin=0.10, mode="uniform", seed=seed + 10_000)

    surface_points = np.vstack(sampled_points_noisy).astype(np.float32, copy=False)
    surface_normals = np.vstack(normals_noisy).astype(np.float32, copy=False)
    points = np.vstack([surface_points, sampled_points_outliers]).astype(np.float32, copy=False)
    normals = np.vstack([surface_normals, normals_outliers]).astype(np.float32, copy=False)

    gt_inliers = np.zeros(points.shape[0], dtype=bool)
    gt_inliers[:surface_points.shape[0]] = True
    return points, normals, gt_inliers


def normalize_curves(
    curves: list[tuple[str, str, bool] | tuple[str, str, bool, str]]
) -> list[tuple[str, str, bool, str]]:
    normalized: list[tuple[str, str, bool, str]] = []
    for curve in curves:
        if len(curve) == 3:
            label, algorithm, use_normal_coherence = curve
            normalized.append((label, algorithm, use_normal_coherence, "misclassification"))
        elif len(curve) == 4:
            label, algorithm, use_normal_coherence, classifier = curve
            normalized.append((label, algorithm, use_normal_coherence, classifier))
        else:
            raise ValueError(
                "Each curve must have 3 items "
                "(label, algorithm, use_normal_coherence) "
                "or 4 items including classifier."
            )
    return normalized


# Run one benchmark trial for the noise sweep experiments.
def run_trial(job: tuple) -> dict[str, float | int | str | bool]:
    label, algorithm, use_normal_coherence, classifier, noise_std, run_idx, base_seed, n_surface_points, n_outliers, threshold_value, threshold_scale, graph_radius, max_iterations, inner_iterations = job
    seed = base_seed + run_idx + int(round(noise_std * 10_000))
    points, normals, gt_inliers = build_test_cloud(noise_std, n_surface_points, n_outliers, seed)
    if threshold_value is None:
        if threshold_scale is None:
            raise ValueError("THRESHOLD_SCALE must be set when THRESHOLD is None")
        threshold = max(threshold_scale * noise_std, 1e-3)
    else:
        threshold = float(threshold_value)

    predicted_inliers = np.zeros(points.shape[0], dtype=bool)
    fitted_model = None
    if algorithm == "ransac":
        result = inner_ransac(
            point_cloud=points,
            refined_set_index=np.arange(points.shape[0], dtype=np.int64),
            actual_set_index=np.arange(points.shape[0], dtype=np.int64),
            threshold=threshold,
            n_iters=inner_iterations,
            random_seed=seed,
        )
        if result.best_inlier_count > 0 and result.best_inliers_mask.size == points.shape[0]:
            predicted_inliers = np.asarray(result.best_inliers_mask, dtype=bool)
            fitted_model = result.best_model
    else:
        models, inlier_masks = gair_ransac(
            point_cloud=points,
            normals=normals,
            threshold=threshold,
            max_models=1,
            max_iterations=max_iterations,
            radius=graph_radius,
            inner_iterations=inner_iterations,
            random_seed=seed,
            use_normal_coherence=use_normal_coherence,
        )
        if inlier_masks:
            predicted_inliers = np.asarray(inlier_masks[0], dtype=bool)
            fitted_model = models[0] if models else None

    return {
        "curve": label,
        "algorithm": algorithm,
        "use_normal_coherence": use_normal_coherence,
        "classifier": classifier,
        "noise_std": noise_std,
        "threshold_scale": threshold_scale,
        "threshold": threshold,
        "run_idx": run_idx,
        "metric_value": _compute_metric(classifier, predicted_inliers, gt_inliers, fitted_model),
    }


# Compute mean and std while ignoring NaN failures.
def _nan_summary(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


# Write the aggregated summary table used by all benchmarks.
def _write_summary_csv(summary_rows: list[dict[str, float | int | str | bool]], output_dir: Path) -> Path:
    summary_csv_path = output_dir / "summary_results.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_csv_path


# Run the existing noise sweep benchmark.
def run_benchmark(
    title: str,
    output_dir: Path,
    curves: list[tuple[str, str, bool] | tuple[str, str, bool, str]],
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
    curves = normalize_curves(curves)
    noise_grid = build_sweep_values(noise_values, noise_start, noise_stop, noise_step, "NOISE_STEP", "NOISE_START", "NOISE_STOP")
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = [
        (
            label,
            algorithm,
            use_normal_coherence,
            classifier,
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
        for label, algorithm, use_normal_coherence, classifier in curves
        for noise_std in noise_grid
        for run_idx in range(runs)
    ]

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            raw_rows = list(executor.map(run_trial, jobs))
    else:
        raw_rows = [run_trial(job) for job in jobs]

    classifiers_order: list[str] = []
    for _, _, _, _, cls in curves:
        if cls not in classifiers_order:
            classifiers_order.append(cls)

    summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_rows: list[dict[str, float | int | str | bool]] = []
    for label, algorithm, use_normal_coherence, classifier in curves:
        means: list[float] = []
        stds: list[float] = []
        for noise_std in noise_grid:
            values = [
                float(row["metric_value"])
                for row in raw_rows
                if row["curve"] == label and row["noise_std"] == noise_std
            ]
            mean, std = _nan_summary(values)
            means.append(mean)
            stds.append(std)
            summary_rows.append({
                "curve": label,
                "algorithm": algorithm,
                "use_normal_coherence": use_normal_coherence,
                "classifier": classifier,
                "noise_std": noise_std,
                "mean_metric_value": mean,
                "std_metric_value": std,
            })
        summary[label] = (
            np.asarray(noise_grid, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            np.asarray(stds, dtype=np.float64),
        )

    csv_path = output_dir / "raw_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["curve", "algorithm", "use_normal_coherence", "classifier", "noise_std", "threshold_scale", "threshold", "run_idx", "metric_value"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_csv_path = _write_summary_csv(summary_rows, output_dir)

    plot_paths = []
    for cls in classifiers_order:
        cls_curves = [(label, c) for label, _, _, _, c in curves if c == cls]
        plot_path = output_dir / f"noise_vs_{cls}.png"
        plt.figure(figsize=(9, 5))
        for label, _ in cls_curves:
            curve_noise, mean, std = summary[label]
            plt.plot(curve_noise, mean, marker="o", linewidth=2, label=label)
            plt.fill_between(curve_noise, mean - std, mean + std, alpha=0.15)
        plt.xlabel("noise_std")
        plt.ylabel(cls)
        plt.title(f"{title} [{cls}]")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        plot_paths.append(plot_path)

    for label, (curve_noise, mean, _) in summary.items():
        pairs = ", ".join(f"{noise:.2f}:{val:.4f}" for noise, val in zip(curve_noise, mean))
        print(f"{label} -> {pairs}")
    print(f"Saved raw results to {csv_path}")
    print(f"Saved summary results to {summary_csv_path}")
    for p in plot_paths:
        print(f"Saved plot to {p}")
    return 0


# Run the existing threshold-scale sweep benchmark.
def run_scale_factor_benchmark(
    title: str,
    output_dir: Path,
    curves: list[tuple[str, str, bool] | tuple[str, str, bool, str]],
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
    curves = normalize_curves(curves)
    scale_grid = build_sweep_values(scale_values, scale_start, scale_stop, scale_step, "SCALE_STEP", "SCALE_START", "SCALE_STOP")
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = [
        (
            label,
            algorithm,
            use_normal_coherence,
            classifier,
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
        for label, algorithm, use_normal_coherence, classifier in curves
        for scale_factor in scale_grid
        for run_idx in range(runs)
    ]

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            raw_rows = list(executor.map(run_trial, jobs))
    else:
        raw_rows = [run_trial(job) for job in jobs]

    classifiers_order: list[str] = []
    for _, _, _, _, cls in curves:
        if cls not in classifiers_order:
            classifiers_order.append(cls)

    summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_rows: list[dict[str, float | int | str | bool]] = []
    for label, algorithm, use_normal_coherence, classifier in curves:
        means: list[float] = []
        stds: list[float] = []
        for scale_factor in scale_grid:
            values = [
                float(row["metric_value"])
                for row in raw_rows
                if row["curve"] == label and row["threshold_scale"] == scale_factor
            ]
            mean, std = _nan_summary(values)
            means.append(mean)
            stds.append(std)
            summary_rows.append({
                "curve": label,
                "algorithm": algorithm,
                "use_normal_coherence": use_normal_coherence,
                "classifier": classifier,
                "threshold_scale": scale_factor,
                "mean_metric_value": mean,
                "std_metric_value": std,
            })
        summary[label] = (
            np.asarray(scale_grid, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            np.asarray(stds, dtype=np.float64),
        )

    csv_path = output_dir / "raw_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["curve", "algorithm", "use_normal_coherence", "classifier", "noise_std", "threshold_scale", "threshold", "run_idx", "metric_value"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_csv_path = _write_summary_csv(summary_rows, output_dir)

    plot_paths = []
    for cls in classifiers_order:
        cls_curves = [(label, c) for label, _, _, _, c in curves if c == cls]
        plot_path = output_dir / f"scale_factor_vs_{cls}.png"
        plt.figure(figsize=(9, 5))
        for label, _ in cls_curves:
            curve_scale, mean, std = summary[label]
            plt.plot(curve_scale, mean, marker="o", linewidth=2, label=label)
            plt.fill_between(curve_scale, mean - std, mean + std, alpha=0.15)
        plt.xlabel("threshold scale factor")
        plt.ylabel(cls)
        plt.title(f"{title} [{cls}]")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        plot_paths.append(plot_path)

    for label, (curve_scale, mean, _) in summary.items():
        pairs = ", ".join(f"{scale:.2f}:{val:.4f}" for scale, val in zip(curve_scale, mean))
        print(f"{label} -> {pairs}")
    print(f"Saved raw results to {csv_path}")
    print(f"Saved summary results to {summary_csv_path}")
    for p in plot_paths:
        print(f"Saved plot to {p}")
    return 0


# Run one benchmark trial for the outlier sweep with different hypothesis budgets.
def run_outlier_hypotheses_trial(job: tuple) -> dict[str, float | int | str | bool]:
    label, algorithm, use_normal_coherence, classifier, hypotheses, noise_std, outlier_count, run_idx, base_seed, n_surface_points, threshold_value, threshold_scale, graph_radius, gair_inner_iterations = job
    seed = base_seed + run_idx + 131 * int(outlier_count) + 17 * int(np.log10(max(hypotheses, 1)))
    points, normals, gt_inliers = build_test_cloud(noise_std, n_surface_points, outlier_count, seed)

    if threshold_value is None:
        if threshold_scale is None:
            raise ValueError("THRESHOLD_SCALE must be set when THRESHOLD is None")
        threshold = max(threshold_scale * noise_std, 1e-3)
    else:
        threshold = float(threshold_value)

    predicted_inliers = np.zeros(points.shape[0], dtype=bool)
    fitted_model = None
    if algorithm == "ransac":
        result = inner_ransac(
            point_cloud=points,
            refined_set_index=np.arange(points.shape[0], dtype=np.int64),
            actual_set_index=np.arange(points.shape[0], dtype=np.int64),
            threshold=threshold,
            n_iters=hypotheses,
            random_seed=seed,
        )
        if result.best_inlier_count > 0 and result.best_inliers_mask.size == points.shape[0]:
            predicted_inliers = np.asarray(result.best_inliers_mask, dtype=bool)
            fitted_model = result.best_model
    else:
        models, inlier_masks = gair_ransac(
            point_cloud=points,
            normals=normals,
            threshold=threshold,
            max_models=1,
            max_iterations=hypotheses,
            radius=graph_radius,
            inner_iterations=gair_inner_iterations,
            random_seed=seed,
            use_normal_coherence=use_normal_coherence,
        )
        if inlier_masks:
            predicted_inliers = np.asarray(inlier_masks[0], dtype=bool)
            fitted_model = models[0] if models else None

    return {
        "curve": label,
        "algorithm": algorithm,
        "use_normal_coherence": use_normal_coherence,
        "classifier": classifier,
        "noise_std": noise_std,
        "outlier_count": int(outlier_count),
        "hypotheses": int(hypotheses),
        "threshold_scale": threshold_scale,
        "threshold": threshold,
        "gair_inner_iterations": int(gair_inner_iterations),
        "run_idx": int(run_idx),
        "metric_value": _compute_metric(classifier, predicted_inliers, gt_inliers, fitted_model),
    }


# Benchmark misclassification or chamfer while outliers vary on x and the hypothesis budget varies across curves.
def run_outlier_hypotheses_benchmark(
    title: str,
    output_dir: Path,
    curves: list[tuple[str, str, bool, str, int]],
    fixed_noise_std: float,
    outlier_values: list[int] | None,
    outlier_start: int,
    outlier_stop: int,
    outlier_step: int,
    runs: int,
    n_surface_points: int,
    threshold_value: float | None,
    threshold_scale: float | None,
    graph_radius: float,
    gair_inner_iterations: int,
    max_workers: int | None,
    use_multiprocessing: bool,
    base_seed: int,
) -> int:
    outlier_grid = build_integer_sweep_values(outlier_values, outlier_start, outlier_stop, outlier_step, "OUTLIER_STEP", "OUTLIER_START", "OUTLIER_STOP")
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = [
        (
            label,
            algorithm,
            use_normal_coherence,
            classifier,
            hypotheses,
            fixed_noise_std,
            outlier_count,
            run_idx,
            base_seed,
            n_surface_points,
            threshold_value,
            threshold_scale,
            graph_radius,
            gair_inner_iterations,
        )
        for label, algorithm, use_normal_coherence, classifier, hypotheses in curves
        for outlier_count in outlier_grid
        for run_idx in range(runs)
    ]

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            raw_rows = list(executor.map(run_outlier_hypotheses_trial, jobs))
    else:
        raw_rows = [run_outlier_hypotheses_trial(job) for job in jobs]

    classifiers_order: list[str] = []
    for _, _, _, _, cls, _ in curves:
        if cls not in classifiers_order:
            classifiers_order.append(cls)

    summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_rows: list[dict[str, float | int | str | bool]] = []
    for label, algorithm, use_normal_coherence, classifier, hypotheses in curves:
        means: list[float] = []
        stds: list[float] = []
        for outlier_count in outlier_grid:
            values = [
                float(row["metric_value"])
                for row in raw_rows
                if row["curve"] == label and row["outlier_count"] == outlier_count
            ]
            mean, std = _nan_summary(values)
            means.append(mean)
            stds.append(std)
            summary_rows.append({
                "curve": label,
                "algorithm": algorithm,
                "use_normal_coherence": use_normal_coherence,
                "classifier": classifier,
                "hypotheses": int(hypotheses),
                "outlier_count": int(outlier_count),
                "mean_metric_value": mean,
                "std_metric_value": std,
            })
        summary[label] = (
            np.asarray(outlier_grid, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            np.asarray(stds, dtype=np.float64),
        )

    csv_path = output_dir / "raw_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["curve", "algorithm", "use_normal_coherence", "classifier", "noise_std", "outlier_count", "hypotheses", "threshold_scale", "threshold", "gair_inner_iterations", "run_idx", "metric_value"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_csv_path = _write_summary_csv(summary_rows, output_dir)

    plot_paths = []
    for cls in classifiers_order:
        cls_curves = [(label, c) for label, _, _, _, c, _ in curves if c == cls]
        plot_path = output_dir / f"outliers_vs_{cls}.png"
        plt.figure(figsize=(10, 6))
        for label, _ in cls_curves:
            curve_outliers, mean, std = summary[label]
            plt.plot(curve_outliers, mean, marker="o", linewidth=2, label=label)
            plt.fill_between(curve_outliers, mean - std, mean + std, alpha=0.15)
        plt.xlabel("number of outliers")
        plt.ylabel(cls)
        plt.title(f"{title} [{cls}]")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        plot_paths.append(plot_path)

    for label, (curve_outliers, mean, _) in summary.items():
        pairs = ", ".join(f"{int(outliers)}:{val:.4f}" for outliers, val in zip(curve_outliers, mean))
        print(f"{label} -> {pairs}")
    print(f"Saved raw results to {csv_path}")
    print(f"Saved summary results to {summary_csv_path}")
    for p in plot_paths:
        print(f"Saved plot to {p}")
    return 0
