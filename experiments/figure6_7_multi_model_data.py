import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading

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
PROCESS_WORKERS = 1  # 1 = seriale, intero > 1 = multiprocessing, "max" = tutti i core CPU
SCENARIO = "multi_model"
PC_DIR = ROOT / "test_objects"
OUTPUT_DIR = ROOT / "experiments" / "artifacts" / "figure6_7_multi_model"
CSV_PATH = OUTPUT_DIR / "results.csv"

METHODS_BY_BUDGET = {
    500: ["Ransac", "Ransac + LO", "Ransac + GAIR"],
    5000: ["Ransac + LO", "Ransac + GAIR"],
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_idx: int
    path: str
    input_file: str
    dataset_id: str
    outlier_ratio: float
    noise_std: float
    threshold: float
    point_count: int
    outlier_count: int
    gt_inliers_assumed_from_tail: bool


@dataclass(frozen=True)
class RunJob:
    dataset: DatasetSpec
    iteration_budget: int
    methods: tuple[str, ...]
    run_id: int
    max_models: int


_INNER_RANSAC_COUNTER = threading.local()


def _make_counted_inner_ransac(original):
    def wrapped(*args, **kwargs):
        if getattr(_INNER_RANSAC_COUNTER, "enabled", False):
            _INNER_RANSAC_COUNTER.value = getattr(_INNER_RANSAC_COUNTER, "value", 0) + 1
        return original(*args, **kwargs)

    return wrapped


lo_module.inner_ransac = _make_counted_inner_ransac(lo_module.inner_ransac)
gair_module.inner_ransac = _make_counted_inner_ransac(gair_module.inner_ransac)


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


@contextmanager
def count_inner_ransac_calls():
    previous_enabled = getattr(_INNER_RANSAC_COUNTER, "enabled", False)
    previous_value = getattr(_INNER_RANSAC_COUNTER, "value", 0)
    _INNER_RANSAC_COUNTER.enabled = True
    _INNER_RANSAC_COUNTER.value = 0
    try:
        yield _INNER_RANSAC_COUNTER
    finally:
        _INNER_RANSAC_COUNTER.enabled = previous_enabled
        _INNER_RANSAC_COUNTER.value = previous_value


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
        with count_inner_ransac_calls() as counter:
            models, inlier_masks = lo_module.ransac(
                point_cloud=points,
                threshold=threshold,
                max_models=max_models,
                max_iterations=iteration_budget,
                inner_iterations=INNER_ITERATIONS,
                radius=GRAPH_RADIUS,
                graphcut=False,
                random_seed=seed,
            )
            local_steps = int(getattr(counter, "value", 0))
    elif method == "Ransac + GAIR":
        with count_inner_ransac_calls() as counter:
            models, inlier_masks, _ = gair_module.gair_ransac(
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
            )
            local_steps = int(getattr(counter, "value", 0))
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.perf_counter() - start
    predicted_inliers = merge_masks(inlier_masks, points.shape[0])
    return predicted_inliers, runtime, int(local_steps), len(models)


def build_dataset_specs(point_cloud_files: list[Path]) -> list[DatasetSpec]:
    dataset_specs: list[DatasetSpec] = []
    for dataset_idx, pc_file in enumerate(point_cloud_files):
        dataset_id, outlier_ratio, noise_std = parse_point_cloud_metadata(pc_file)
        threshold = get_threshold(noise_std)
        points, _ = load_point_cloud_with_normals(pc_file)
        _, n_outliers = infer_ground_truth_inliers(points.shape[0], outlier_ratio)
        dataset_specs.append(
            DatasetSpec(
                dataset_idx=dataset_idx,
                path=str(pc_file),
                input_file=pc_file.name,
                dataset_id=dataset_id,
                outlier_ratio=outlier_ratio,
                noise_std=noise_std,
                threshold=threshold,
                point_count=int(points.shape[0]),
                outlier_count=n_outliers,
                gt_inliers_assumed_from_tail=bool(n_outliers > 0),
            )
        )
    return dataset_specs


def build_jobs(dataset_specs: list[DatasetSpec], runs: int, max_models: int) -> list[RunJob]:
    jobs: list[RunJob] = []
    for iteration_budget, methods in METHODS_BY_BUDGET.items():
        for dataset in dataset_specs:
            for run_id in range(1, runs + 1):
                jobs.append(
                    RunJob(
                        dataset=dataset,
                        iteration_budget=iteration_budget,
                        methods=tuple(methods),
                        run_id=run_id,
                        max_models=max_models,
                    )
                )
    return jobs


def resolve_process_workers() -> int:
    value = PROCESS_WORKERS
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "max":
            return max(os.cpu_count() or 1, 1)
        raise ValueError(
            "PROCESS_WORKERS must be 1, an integer > 1, or 'max'. "
            f"Got string value {PROCESS_WORKERS!r}."
        )

    workers = int(value)
    if workers <= 0:
        raise ValueError(f"PROCESS_WORKERS must be positive, got {PROCESS_WORKERS!r}")
    return workers


def describe_job(job: RunJob) -> str:
    return (
        f"{job.dataset.input_file} | budget={job.iteration_budget} | "
        f"run={job.run_id:02d}"
    )


def run_job(job: RunJob) -> tuple[list[dict[str, object]], list[str]]:
    pc_file = Path(job.dataset.path)
    points, normals = load_point_cloud_with_normals(pc_file)
    gt_inliers, n_outliers = infer_ground_truth_inliers(points.shape[0], job.dataset.outlier_ratio)

    run_seed = (
        BASE_SEED
        + 1_000_000 * job.iteration_budget
        + 10_000 * job.dataset.dataset_idx
        + job.run_id
    )

    rows: list[dict[str, object]] = []
    logs: list[str] = []
    for method_idx, method in enumerate(job.methods):
        method_seed = run_seed + 100_000 * (method_idx + 1)
        predicted_inliers, runtime, local_steps, n_models = run_method(
            method,
            points,
            normals,
            job.dataset.threshold,
            job.iteration_budget,
            method_seed,
            job.max_models,
        )
        misclassification_error = float(np.mean(predicted_inliers != gt_inliers))
        rows.append({
            "scenario": SCENARIO,
            "iteration_budget": job.iteration_budget,
            "dataset_id": job.dataset.dataset_id,
            "input_file": job.dataset.input_file,
            "run_id": job.run_id,
            "method": method,
            "outlier_ratio": job.dataset.outlier_ratio,
            "outlier_count": n_outliers,
            "noise_std": job.dataset.noise_std,
            "threshold": job.dataset.threshold,
            "point_count": int(points.shape[0]),
            "misclassification_error": misclassification_error,
            "execution_time_s": runtime,
            "local_optimization_steps": local_steps,
            "n_models": n_models,
            "gt_inliers_assumed_from_tail": job.dataset.gt_inliers_assumed_from_tail,
        })
        logs.append(
            f"{job.dataset.input_file:20s} | budget={job.iteration_budget:<4d} | "
            f"run={job.run_id:02d} | {method:14s} | mis={misclassification_error:.4f} | "
            f"time={runtime:.2f}s | lo={local_steps} | models={n_models}"
        )
    return rows, logs


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
    dataset_specs = build_dataset_specs(point_cloud_files)
    jobs = build_jobs(dataset_specs, args.runs, args.max_models)
    process_workers = resolve_process_workers()
    rows = []

    print(f"point_cloud_dir = {args.pc_dir.expanduser().resolve()}")
    print(f"found {len(point_cloud_files)} input point cloud(s): {[path.name for path in point_cloud_files]}")
    print(f"runs = {args.runs}")
    print(f"max_models = {args.max_models}")
    print(f"process_workers = {PROCESS_WORKERS!r} -> resolved_workers = {process_workers}")
    print("ground-truth assumption = outliers, when present, are stored at the end of each .ply")

    for dataset in dataset_specs:
        print(
            f"dataset = {dataset.input_file} | dataset_id = {dataset.dataset_id} | "
            f"noise_std = {dataset.noise_std:.4f} | outlier_ratio = {dataset.outlier_ratio:.4f} | "
            f"points = {dataset.point_count} | threshold = {dataset.threshold:.4f}"
        )
    print(f"total_jobs = {len(jobs)}")

    if process_workers == 1:
        for job in jobs:
            job_rows, logs = run_job(job)
            for line in logs:
                print(line)
            rows.extend(job_rows)
    else:
        with ProcessPoolExecutor(max_workers=process_workers) as executor:
            futures = {executor.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    job_rows, logs = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Job failed: {describe_job(job)}") from exc
                for line in logs:
                    print(line)
                rows.extend(job_rows)

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
