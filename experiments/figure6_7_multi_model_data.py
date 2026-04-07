import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / "experiments" / "artifacts" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.gair_ransac.gair_ransac as gair_module
import src.gair_ransac.ransac as lo_module
from src.gair_ransac.vanilla_ransac import vanilla_ransac


RUNS = 10
THRESHOLD = None
THRESHOLD_SCALE = 2.5
INNER_ITERATIONS = 50
GRAPH_RADIUS = 0.06
BASE_SEED = 42
MAX_MODELS = 5
SCENARIO = "multi_model"
PC_DIR = ROOT / "test_objects"
OUTPUT_DIR = ROOT / "experiments" / "artifacts" / "figure6_7_multi_model"
CSV_PATH = OUTPUT_DIR / "results.csv"

METHODS_BY_BUDGET = {
    500: ["Ransac", "Ransac + LO", "Ransac + GAIR"],
    5000: ["Ransac + LO", "Ransac + GAIR"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files",
        nargs="*",
        help=(
            "Optional point-cloud .ply files to run. "
            "You can pass bare filenames from --pc-dir or explicit paths."
        ),
    )
    parser.add_argument("--pc-dir", type=Path, default=PC_DIR)
    parser.add_argument("--output-csv", type=Path, default=CSV_PATH)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--max-models", type=int, default=MAX_MODELS)
    return parser.parse_args(argv)


def get_threshold(noise_std):
    if THRESHOLD is not None:
        return float(THRESHOLD)
    return max(float(THRESHOLD_SCALE) * float(noise_std), 1e-3)


def normalize_outlier_ratio(value: str) -> float:
    ratio = float(value)
    if ratio < 0.0:
        raise ValueError(f"outlier ratio must be non-negative, got {ratio}")
    if ratio > 1.0:
        if ratio <= 100.0:
            return ratio / 100.0
        raise ValueError(f"outlier ratio must be in [0, 1] or [0, 100], got {ratio}")
    return ratio


def parse_point_cloud_metadata(path: Path) -> tuple[str, float, float]:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Point cloud filename must end with '_<outlier_ratio>_<noise_std>.ply', got {path.name}"
        )

    dataset_id = "_".join(parts[:-2]).strip()
    if not dataset_id:
        raise ValueError(f"Could not infer dataset_id from {path.name}")

    outlier_ratio = normalize_outlier_ratio(parts[-2])
    noise_std = float(parts[-1])
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    return dataset_id, outlier_ratio, noise_std


def _load_vertices(path: Path) -> np.ndarray:
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        geometry = list(scene.geometry.values())
        if not geometry:
            raise ValueError(f"No geometry found in {path}")
        raw = trimesh.util.concatenate(geometry)
    else:
        raw = scene
    return np.asarray(raw.vertices, dtype=np.float64)


def load_point_cloud_with_normals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = _load_vertices(path)

    normals_path = path.parent / f"normals_{path.name}"
    if normals_path.exists():
        normals = _load_vertices(normals_path)
    else:
        scene = trimesh.load(str(path))
        if isinstance(scene, trimesh.Scene):
            geometry = list(scene.geometry.values())
            if not geometry:
                raise ValueError(f"No geometry found in {path}")
            raw = trimesh.util.concatenate(geometry)
        else:
            raw = scene
        normals = np.asarray(getattr(raw, "vertex_normals", []), dtype=np.float64)
        if normals.shape != points.shape:
            raise FileNotFoundError(
                f"No normals file found for {path.name} and embedded normals are unavailable."
            )

    if points.shape != normals.shape:
        raise ValueError(
            f"Points/normals shape mismatch for {path.name}: points={points.shape}, normals={normals.shape}"
        )

    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(normal_norm > 0.0, normal_norm, 1.0)
    return points, normals


def infer_ground_truth_inliers(n_points: int, outlier_ratio: float) -> tuple[np.ndarray, int]:
    n_outliers = int(round(float(n_points) * float(outlier_ratio)))
    n_outliers = min(max(n_outliers, 0), n_points)

    gt_inliers = np.ones(n_points, dtype=bool)
    if n_outliers > 0:
        # Assumption: when present, outliers are appended at the end of the point cloud.
        gt_inliers[-n_outliers:] = False
    return gt_inliers, n_outliers


def list_point_cloud_files(pc_dir: Path, requested_files: list[str] | None = None) -> list[Path]:
    pc_dir = pc_dir.expanduser().resolve()
    if not pc_dir.exists():
        raise FileNotFoundError(f"Point cloud directory not found: {pc_dir}")

    available_point_clouds = sorted(
        path
        for path in pc_dir.glob("*.ply")
        if path.is_file() and not path.name.startswith("normals_")
    )
    if not available_point_clouds:
        raise FileNotFoundError(f"No point-cloud .ply files found in {pc_dir}")

    if not requested_files:
        return available_point_clouds

    by_name = {path.name: path for path in available_point_clouds}
    selected_files: list[Path] = []
    missing_files: list[str] = []

    for requested in requested_files:
        requested_path = Path(requested).expanduser()
        if not requested_path.is_absolute():
            if requested_path.parent == Path("."):
                candidate = by_name.get(requested_path.name)
                if candidate is None:
                    candidate = (pc_dir / requested_path.name).resolve()
            else:
                candidate = (ROOT / requested_path).resolve()
        else:
            candidate = requested_path.resolve()

        if (
            candidate.exists()
            and candidate.is_file()
            and candidate.suffix.lower() == ".ply"
            and not candidate.name.startswith("normals_")
        ):
            selected_files.append(candidate)
        else:
            missing_files.append(requested)

    if missing_files:
        available_names = ", ".join(path.name for path in available_point_clouds)
        raise FileNotFoundError(
            "Could not resolve the following point-cloud files: "
            f"{', '.join(missing_files)}. Available in {pc_dir}: {available_names}"
        )

    deduped_files: list[Path] = []
    seen: set[Path] = set()
    for path in selected_files:
        if path not in seen:
            deduped_files.append(path)
            seen.add(path)
    return deduped_files


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


def run_method(method, points, normals, threshold, iteration_budget, seed, max_models):
    start = time.perf_counter()

    if method == "Ransac":
        models, inlier_masks = vanilla_ransac(
            points,
            normals,
            threshold=threshold,
            max_models=max_models,
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
                max_models=max_models,
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
                max_models=max_models,
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


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.runs <= 0:
        raise ValueError(f"--runs must be positive, got {args.runs}")
    if args.max_models <= 0:
        raise ValueError(f"--max-models must be positive, got {args.max_models}")

    output_csv = args.output_csv.expanduser()
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    point_cloud_files = list_point_cloud_files(args.pc_dir, args.input_files)
    rows = []

    print(f"point_cloud_dir = {args.pc_dir.expanduser().resolve()}")
    print(f"found {len(point_cloud_files)} input point cloud(s): {[path.name for path in point_cloud_files]}")
    print(f"runs = {args.runs}")
    print(f"max_models = {args.max_models}")
    print("ground-truth assumption = outliers, when present, are stored at the end of each .ply")

    for iteration_budget, methods in METHODS_BY_BUDGET.items():
        print(f"\n=== iteration_budget = {iteration_budget} ===")

        for dataset_idx, pc_file in enumerate(point_cloud_files):
            dataset_id, outlier_ratio, noise_std = parse_point_cloud_metadata(pc_file)
            threshold = get_threshold(noise_std)
            points, normals = load_point_cloud_with_normals(pc_file)
            gt_inliers, n_outliers = infer_ground_truth_inliers(points.shape[0], outlier_ratio)

            print(
                f"\npoint_cloud = {pc_file.name} | dataset_id = {dataset_id} | "
                f"noise_std = {noise_std:.4f} | outlier_ratio = {outlier_ratio:.4f} | "
                f"points = {points.shape[0]} | threshold = {threshold:.4f}"
            )

            for run_id in range(1, args.runs + 1):
                run_seed = (
                    BASE_SEED
                    + 1_000_000 * iteration_budget
                    + 10_000 * dataset_idx
                    + run_id
                )

                for method_idx, method in enumerate(methods):
                    method_seed = run_seed + 100_000 * (method_idx + 1)
                    predicted_inliers, runtime, local_steps, n_models = run_method(
                        method,
                        points,
                        normals,
                        threshold,
                        iteration_budget,
                        method_seed,
                        args.max_models,
                    )
                    misclassification_error = float(np.mean(predicted_inliers != gt_inliers))

                    print(
                        f"  run={run_id:02d} | {method:14s} | "
                        f"mis={misclassification_error:.4f} | time={runtime:.2f}s | "
                        f"lo={local_steps} | models={n_models}"
                    )

                    rows.append({
                        "scenario": SCENARIO,
                        "iteration_budget": iteration_budget,
                        "dataset_id": dataset_id,
                        "input_file": pc_file.name,
                        "run_id": run_id,
                        "method": method,
                        "outlier_ratio": outlier_ratio,
                        "outlier_count": n_outliers,
                        "noise_std": noise_std,
                        "threshold": threshold,
                        "point_count": int(points.shape[0]),
                        "misclassification_error": misclassification_error,
                        "execution_time_s": runtime,
                        "local_optimization_steps": local_steps,
                        "n_models": n_models,
                        "gt_inliers_assumed_from_tail": bool(n_outliers > 0),
                    })

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "iteration_budget",
                "dataset_id",
                "input_file",
                "run_id",
                "method",
                "outlier_ratio",
                "outlier_count",
                "noise_std",
                "threshold",
                "point_count",
                "misclassification_error",
                "execution_time_s",
                "local_optimization_steps",
                "n_models",
                "gt_inliers_assumed_from_tail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
