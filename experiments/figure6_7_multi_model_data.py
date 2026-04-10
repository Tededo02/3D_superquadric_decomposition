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
from scipy.spatial import cKDTree
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


RUNS = 20
THRESHOLD = None
THRESHOLD_FACTOR = 2.5
THRESHOLD_SPACING_FACTOR = 0.09
MIN_THRESHOLD = 0.02
MIN_COVERAGE = 0.0
DEFAULT_NOISE_STD = 0.0
PLOT_OUTLIER_RATIOS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
MIN_SAMPLE_SIZE = 11
MAX_SAMPLE_SIZE = 15
MIN_INLIERS_FLOOR = 11
MAX_INLIERS_CAP = 15
INNER_ITERATIONS = 50
GRAPH_RADIUS = 0.06
OUTLIER_MARGIN = 0.10
BASE_SEED = 94737
MAX_MODELS = 6
PROCESS_WORKERS = "max" # 1 = seriale, intero > 1 = multiprocessing, "max" = tutti i core CPU
SCENARIO = "multi_model"
PC_DIR = ROOT / "test_objects" / "synt_multimodel_valid"
OUTPUT_DIR = ROOT / "experiments" / "artifacts" / "figure6_7_multi_model"
CSV_PATH = OUTPUT_DIR / "results.csv"
BASE_REQUIRED_NOISE_STD = 0.0
BASE_REQUIRED_OUTLIER_RATIO = 0.0
OUTLIER_LABEL_VALUE = -1.0
GT_LABEL_MATCH_ATOL = 1e-12

METHODS_BY_BUDGET = {
    500: [      "Ransac","Ransac + LO","Ransac + GC",
        "Ransac + GAIR",
    ],
    #5000: ["Ransac + LO","Ransac + GC","Ransac + GAIR",],
}


@dataclass(frozen=True)
class PointCloudStats:
    point_count: int
    spatial_extent: float
    median_nn_distance: float


@dataclass(frozen=True)
class RansacTuning:
    sample_size: int
    min_inliers: int
    mss_max_pool_fraction: float


@dataclass(frozen=True)
class DatasetSpec:
    dataset_idx: int
    path: str
    input_file: str
    dataset_id: str
    base_point_count: int
    base_gt_outliers_removed: int


@dataclass(frozen=True)
class RunJob:
    dataset: DatasetSpec
    iteration_budget: int
    methods: tuple[str, ...]
    run_id: int
    max_models: int
    noise_std: float
    outlier_ratio: float


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
            "Optional base point-cloud .ply files to run. "
            "Only files ending with '_0_0.ply' from --pc-dir are accepted."
        ),
    )
    parser.add_argument("--pc-dir", type=Path, default=PC_DIR)
    parser.add_argument("--output-csv", type=Path, default=CSV_PATH)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--max-models", type=int, default=MAX_MODELS)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help=(
            "Gaussian noise std injected on points and normals before running the benchmark. "
            "Default: 0.2."
        ),
    )
    parser.add_argument(
        "--outlier-ratios",
        type=float,
        nargs="+",
        default=list(PLOT_OUTLIER_RATIOS),
        help=(
            "Synthetic outlier ratios to benchmark. "
            "Default: 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4."
        ),
    )
    return parser.parse_args(argv)


def normalize_outlier_ratio(value: float | str) -> float:
    ratio = float(value)
    if ratio < 0.0:
        raise ValueError(f"outlier ratio must be non-negative, got {ratio}")
    if ratio > 1.0:
        if ratio <= 100.0:
            return ratio / 100.0
        raise ValueError(f"outlier ratio must be in [0, 1] or [0, 100], got {ratio}")
    return ratio


def normalize_noise_std(value: float | str) -> float:
    noise_std = float(value)
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    return noise_std


def parse_point_cloud_metadata(path: Path) -> tuple[str, float, float]:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Point cloud filename must end with '_<noise_std>_<outlier_ratio>.ply', got {path.name}"
        )

    dataset_id = "_".join(parts[:-2]).strip()
    if not dataset_id:
        raise ValueError(f"Could not infer dataset_id from {path.name}")

    noise_std = normalize_noise_std(parts[-2])
    outlier_ratio = normalize_outlier_ratio(parts[-1])
    return dataset_id, outlier_ratio, noise_std


def matches_allowed_value(value: float, allowed_values: tuple[float, ...], atol: float = 1e-12) -> bool:
    return any(np.isclose(float(value), float(candidate), atol=atol, rtol=0.0) for candidate in allowed_values)


def is_base_dataset(path: Path) -> bool:
    _, outlier_ratio, noise_std = parse_point_cloud_metadata(path)
    return (
        np.isclose(noise_std, BASE_REQUIRED_NOISE_STD, atol=GT_LABEL_MATCH_ATOL, rtol=0.0)
        and np.isclose(outlier_ratio, BASE_REQUIRED_OUTLIER_RATIO, atol=GT_LABEL_MATCH_ATOL, rtol=0.0)
    )


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


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


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

    return points, normalize_vectors(normals)


def infer_ground_truth_inliers(n_points: int, outlier_ratio: float) -> tuple[np.ndarray, int]:
    n_outliers = int(round(float(n_points) * float(outlier_ratio)))
    n_outliers = min(max(n_outliers, 0), n_points)

    gt_inliers = np.ones(n_points, dtype=bool)
    if n_outliers > 0:
        gt_inliers[-n_outliers:] = False
    return gt_inliers, n_outliers


def load_ground_truth_inliers(path: Path, points: np.ndarray) -> tuple[np.ndarray, int, bool]:
    labels_path = path.parent / f"labels_{path.stem}.npy"
    labels_points_path = path.parent / f"labels_points_{path.stem}.npy"
    labels = None

    if labels_path.exists():
        labels = np.asarray(np.load(labels_path), dtype=np.float64).reshape(-1)
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"Ground-truth labels length mismatch for {path.name}: "
                f"labels={labels.shape[0]} points={points.shape[0]}"
            )

    if labels_points_path.exists():
        labels_points = np.asarray(np.load(labels_points_path), dtype=np.float64)
        if labels_points.ndim != 2 or labels_points.shape[1] < 4:
            raise ValueError(
                f"{labels_points_path.name} must have shape (N, >=4), got {labels_points.shape}"
            )
        if labels_points.shape[0] != points.shape[0]:
            raise ValueError(
                f"Ground-truth labeled points length mismatch for {path.name}: "
                f"labels_points={labels_points.shape[0]} points={points.shape[0]}"
            )
        if not np.allclose(labels_points[:, :3], points, atol=GT_LABEL_MATCH_ATOL, rtol=0.0):
            raise ValueError(
                f"{labels_points_path.name} does not match the point ordering stored in {path.name}"
            )

        point_labels = np.asarray(labels_points[:, -1], dtype=np.float64)
        if labels is None:
            labels = point_labels
        elif not np.allclose(labels, point_labels, atol=GT_LABEL_MATCH_ATOL, rtol=0.0):
            raise ValueError(
                f"{labels_path.name} and {labels_points_path.name} disagree on the ground-truth labels"
            )

    if labels is not None:
        gt_inliers = ~np.isclose(labels, OUTLIER_LABEL_VALUE, atol=GT_LABEL_MATCH_ATOL, rtol=0.0)
        return gt_inliers, int(np.count_nonzero(~gt_inliers)), False

    _, outlier_ratio, _ = parse_point_cloud_metadata(path)
    gt_inliers, n_outliers = infer_ground_truth_inliers(points.shape[0], outlier_ratio)
    return gt_inliers, n_outliers, bool(n_outliers > 0)


def summarize_point_cloud(points: np.ndarray) -> PointCloudStats:
    point_array = np.asarray(points, dtype=np.float64)
    point_count = int(point_array.shape[0])
    if point_count == 0:
        return PointCloudStats(point_count=0, spatial_extent=0.0, median_nn_distance=0.0)

    spans = np.ptp(point_array, axis=0)
    spatial_extent = float(np.linalg.norm(spans))
    if point_count == 1:
        return PointCloudStats(
            point_count=point_count,
            spatial_extent=spatial_extent,
            median_nn_distance=0.0,
        )

    tree = cKDTree(point_array)
    nn_distances = np.asarray(tree.query(point_array, k=2)[0], dtype=np.float64)
    median_nn_distance = float(np.median(nn_distances[:, 1]))
    return PointCloudStats(
        point_count=point_count,
        spatial_extent=spatial_extent,
        median_nn_distance=median_nn_distance,
    )


def tune_ransac_hyperparameters(stats: PointCloudStats) -> RansacTuning:
    point_count = int(stats.point_count)
    if point_count <= 0:
        return RansacTuning(
            sample_size=0,
            min_inliers=0,
            mss_max_pool_fraction=0.25,
        )

    sample_low = min(MIN_SAMPLE_SIZE, point_count)
    sample_high = min(MAX_SAMPLE_SIZE, point_count)
    raw_sample_size = int(round(0.06 * point_count))
    sample_size = int(np.clip(raw_sample_size, sample_low, sample_high))

    min_inliers_low = min(max(sample_size, MIN_INLIERS_FLOOR), point_count)
    min_inliers_high = min(MAX_INLIERS_CAP, point_count)
    raw_min_inliers = int(round(0.15 * point_count))
    min_inliers = int(np.clip(raw_min_inliers, min_inliers_low, min_inliers_high))

    mss_max_pool_fraction = float(
        np.clip((6.0 * sample_size) / max(point_count, 1), 0.18, 0.35)
    )
    return RansacTuning(
        sample_size=sample_size,
        min_inliers=min_inliers,
        mss_max_pool_fraction=mss_max_pool_fraction,
    )


def get_threshold(
    noise_std: float,
    median_nn_distance: float | None = None,
) -> float:
    if THRESHOLD is not None:
        return float(THRESHOLD)

    threshold_from_noise = float(THRESHOLD_FACTOR) * float(noise_std)
    threshold_from_spacing = 0.0
    if median_nn_distance is not None and median_nn_distance > 0.0:
        threshold_from_spacing = float(THRESHOLD_SPACING_FACTOR) * float(median_nn_distance)
    return max(threshold_from_noise, threshold_from_spacing, MIN_THRESHOLD)


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

    base_point_clouds = [path for path in available_point_clouds if is_base_dataset(path)]
    if not base_point_clouds:
        raise FileNotFoundError(
            "No point-cloud .ply files matched the base-dataset filters in "
            f"{pc_dir} (noise_std={BASE_REQUIRED_NOISE_STD:g}, "
            f"outlier_ratio={BASE_REQUIRED_OUTLIER_RATIO:g})"
        )

    if not requested_files:
        return base_point_clouds

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

    unsupported_requested = [path.name for path in deduped_files if not is_base_dataset(path)]
    if unsupported_requested:
        raise ValueError(
            "The following requested point clouds are not valid base clouds. "
            "Only files with filename suffix '_0_0.ply' are accepted here: "
            f"{', '.join(unsupported_requested)}"
        )

    return deduped_files


def load_clean_base_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    points, normals = load_point_cloud_with_normals(path)
    gt_inliers, n_outliers, _ = load_ground_truth_inliers(path, points)
    if gt_inliers.shape[0] != points.shape[0]:
        raise ValueError(f"Ground-truth mask length mismatch for {path.name}")

    clean_points = points[gt_inliers]
    clean_normals = normals[gt_inliers]
    if clean_points.shape[0] == 0:
        raise ValueError(f"No inlier points available in base dataset {path.name}")
    return clean_points, clean_normals, int(n_outliers)


def build_synthetic_cloud(
    base_points: np.ndarray,
    base_normals: np.ndarray,
    noise_std: float,
    outlier_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    points = np.asarray(base_points, dtype=np.float64).copy()
    normals = np.asarray(base_normals, dtype=np.float64).copy()

    if noise_std > 0.0:
        points = points + rng.normal(0.0, noise_std, size=points.shape)
        normals = normals + rng.normal(0.0, noise_std, size=normals.shape)
        normals = normalize_vectors(normals)

    total_point_count = int(points.shape[0])
    n_outliers = int(round(float(total_point_count) * float(outlier_ratio)))
    n_outliers = min(max(n_outliers, 0), total_point_count)
    n_inliers = total_point_count - n_outliers
    if n_inliers <= 0:
        raise ValueError(
            f"Synthetic corruption would remove all inliers: total={total_point_count}, "
            f"outlier_ratio={outlier_ratio}"
        )

    if n_outliers == 0:
        gt_inliers = np.ones(total_point_count, dtype=bool)
        return points, normals, gt_inliers, 0

    keep_indices = np.sort(rng.choice(total_point_count, size=n_inliers, replace=False))
    inlier_points = points[keep_indices]
    inlier_normals = normals[keep_indices]

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    spans = bbox_max - bbox_min
    margin = OUTLIER_MARGIN * np.where(spans > 0.0, spans, 1.0)
    outlier_points = rng.uniform(bbox_min - margin, bbox_max + margin, size=(n_outliers, 3))
    outlier_normals = normalize_vectors(rng.normal(0.0, 1.0, size=(n_outliers, 3)))

    synthetic_points = np.vstack([inlier_points, outlier_points])
    synthetic_normals = np.vstack([inlier_normals, outlier_normals])
    gt_inliers = np.zeros(total_point_count, dtype=bool)
    gt_inliers[:n_inliers] = True
    return synthetic_points, synthetic_normals, gt_inliers, int(n_outliers)


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


def run_method(
    method: str,
    points: np.ndarray,
    normals: np.ndarray,
    threshold: float,
    iteration_budget: int,
    seed: int,
    max_models: int,
    tuning: RansacTuning,
) -> tuple[np.ndarray, float, int, int]:
    start = time.perf_counter()

    if method == "Ransac":
        models, inlier_masks = vanilla_ransac(
            points,
            normals,
            threshold=threshold,
            max_models=max_models,
            max_iterations=iteration_budget,
            sample_size=tuning.sample_size,
            min_inliers=tuning.min_inliers,
            consensus_metric="radial",
            random_seed=seed,
            min_coverage=MIN_COVERAGE,
        )
        local_steps = 0
    elif method == "Ransac + LO":
        with count_inner_ransac_calls() as counter:
            models, inlier_masks = lo_module.ransac(
                point_cloud=points,
                threshold=threshold,
                max_models=max_models,
                max_iterations=iteration_budget,
                sample_size=tuning.sample_size,
                min_inliers=tuning.min_inliers,
                inner_iterations=INNER_ITERATIONS,
                radius=GRAPH_RADIUS,
                graphcut=False,
                normals=normals,
                random_seed=seed,
                use_normal_guided_mss=True,
                mss_max_pool_fraction=tuning.mss_max_pool_fraction,
            )
            local_steps = int(getattr(counter, "value", 0))
    elif method == "Ransac + GC":
        with count_inner_ransac_calls() as counter:
            models, inlier_masks, _ = gair_module.gair_ransac(
                point_cloud=points,
                normals=normals,
                threshold=threshold,
                max_models=max_models,
                max_iterations=iteration_budget,
                sample_size=tuning.sample_size,
                min_inliers=tuning.min_inliers,
                inner_iterations=INNER_ITERATIONS,
                radius=GRAPH_RADIUS,
                consensus_metric="radial",
                random_seed=seed,
                use_normal_coherence=False,
                min_coverage=MIN_COVERAGE,
                use_normal_guided_mss=True,
                mss_max_pool_fraction=tuning.mss_max_pool_fraction,
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
                sample_size=tuning.sample_size,
                min_inliers=tuning.min_inliers,
                inner_iterations=INNER_ITERATIONS,
                radius=GRAPH_RADIUS,
                consensus_metric="radial",
                random_seed=seed,
                use_normal_coherence=True,
                min_coverage=MIN_COVERAGE,
                use_normal_guided_mss=True,
                mss_max_pool_fraction=tuning.mss_max_pool_fraction,
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
        dataset_id, _, _ = parse_point_cloud_metadata(pc_file)
        clean_points, _, removed_outliers = load_clean_base_cloud(pc_file)
        dataset_specs.append(
            DatasetSpec(
                dataset_idx=dataset_idx,
                path=str(pc_file),
                input_file=pc_file.name,
                dataset_id=dataset_id,
                base_point_count=int(clean_points.shape[0]),
                base_gt_outliers_removed=removed_outliers,
            )
        )
    return dataset_specs


def build_jobs(
    dataset_specs: list[DatasetSpec],
    runs: int,
    max_models: int,
    noise_std: float,
    outlier_ratios: list[float],
) -> list[RunJob]:
    jobs: list[RunJob] = []
    for iteration_budget, methods in METHODS_BY_BUDGET.items():
        for dataset in dataset_specs:
            for outlier_ratio in outlier_ratios:
                for run_id in range(1, runs + 1):
                    jobs.append(
                        RunJob(
                            dataset=dataset,
                            iteration_budget=iteration_budget,
                            methods=tuple(methods),
                            run_id=run_id,
                            max_models=max_models,
                            noise_std=noise_std,
                            outlier_ratio=outlier_ratio,
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
        f"{job.dataset.input_file} | noise={job.noise_std:g} | "
        f"outliers={job.outlier_ratio:g} | budget={job.iteration_budget} | run={job.run_id:02d}"
    )


def build_job_seed(job: RunJob) -> int:
    return (
        BASE_SEED
        + 1_000_000 * job.iteration_budget
        + 100_000 * job.dataset.dataset_idx
        + 10_000 * job.run_id
        + int(round(job.noise_std * 10_000))
        + int(round(job.outlier_ratio * 100_000))
    )


def run_job(job: RunJob) -> tuple[list[dict[str, object]], list[str]]:
    pc_file = Path(job.dataset.path)
    base_points, base_normals, base_removed_outliers = load_clean_base_cloud(pc_file)
    cloud_seed = build_job_seed(job)
    points, normals, gt_inliers, n_outliers = build_synthetic_cloud(
        base_points,
        base_normals,
        noise_std=job.noise_std,
        outlier_ratio=job.outlier_ratio,
        seed=cloud_seed,
    )
    cloud_stats = summarize_point_cloud(points)
    threshold = get_threshold(job.noise_std, median_nn_distance=cloud_stats.median_nn_distance)
    ransac_tuning = tune_ransac_hyperparameters(cloud_stats)

    synthetic_input_file = f"{job.dataset.dataset_id}_{job.noise_std:g}_{job.outlier_ratio:g}.synthetic.ply"
    rows: list[dict[str, object]] = []
    logs: list[str] = []
    for method_idx, method in enumerate(job.methods):
        method_seed = cloud_seed + 1_000 * (method_idx + 1)
        predicted_inliers, runtime, local_steps, n_models = run_method(
            method,
            points,
            normals,
            threshold,
            job.iteration_budget,
            method_seed,
            job.max_models,
            ransac_tuning,
        )
        misclassification_error = float(np.mean(predicted_inliers != gt_inliers))
        rows.append({
            "scenario": SCENARIO,
            "iteration_budget": job.iteration_budget,
            "dataset_id": job.dataset.dataset_id,
            "input_file": synthetic_input_file,
            "source_input_file": job.dataset.input_file,
            "run_id": job.run_id,
            "method": method,
            "outlier_ratio": job.outlier_ratio,
            "outlier_count": n_outliers,
            "noise_std": job.noise_std,
            "threshold": threshold,
            "threshold_factor": THRESHOLD_FACTOR if THRESHOLD is None else "",
            "threshold_spacing_factor": THRESHOLD_SPACING_FACTOR if THRESHOLD is None else "",
            "median_nn_distance": cloud_stats.median_nn_distance,
            "spatial_extent": cloud_stats.spatial_extent,
            "point_count": int(points.shape[0]),
            "base_point_count": int(job.dataset.base_point_count),
            "base_gt_outliers_removed": int(base_removed_outliers),
            "sample_size": ransac_tuning.sample_size,
            "min_inliers": ransac_tuning.min_inliers,
            "mss_max_pool_fraction": ransac_tuning.mss_max_pool_fraction,
            "inner_iterations": INNER_ITERATIONS,
            "misclassification_error": misclassification_error,
            "execution_time_s": runtime,
            "local_optimization_steps": local_steps,
            "n_models": n_models,
            "synthetic_outliers_appended_at_tail": bool(n_outliers > 0),
            "gt_inliers_assumed_from_tail": False,
        })
        logs.append(
            f"{job.dataset.dataset_id:12s} | noise={job.noise_std:<4.2f} | "
            f"out={job.outlier_ratio:<4.2f} | budget={job.iteration_budget:<4d} | "
            f"run={job.run_id:02d} | {method:14s} | mis={misclassification_error:.4f} | "
            f"time={runtime:.2f}s | lo={local_steps} | models={n_models} | "
            f"thr={threshold:.4f} | mss={ransac_tuning.sample_size}"
        )
    return rows, logs


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.runs <= 0:
        raise ValueError(f"--runs must be positive, got {args.runs}")
    if args.max_models <= 0:
        raise ValueError(f"--max-models must be positive, got {args.max_models}")

    noise_std = normalize_noise_std(args.noise_std)
    outlier_ratios = [normalize_outlier_ratio(value) for value in args.outlier_ratios]
    if not outlier_ratios:
        raise ValueError("At least one outlier ratio must be provided.")

    output_csv = args.output_csv.expanduser()
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    point_cloud_files = list_point_cloud_files(args.pc_dir, args.input_files)
    dataset_specs = build_dataset_specs(point_cloud_files)
    jobs = build_jobs(
        dataset_specs,
        runs=args.runs,
        max_models=args.max_models,
        noise_std=noise_std,
        outlier_ratios=outlier_ratios,
    )
    process_workers = resolve_process_workers()
    rows = []

    print(f"point_cloud_dir = {args.pc_dir.expanduser().resolve()}")
    print(f"found {len(point_cloud_files)} base point cloud(s): {[path.name for path in point_cloud_files]}")
    print(f"runs = {args.runs}")
    print(f"max_models = {args.max_models}")
    print(f"process_workers = {PROCESS_WORKERS!r} -> resolved_workers = {process_workers}")
    print(
        "base_dataset_filters = "
        f"noise_std={BASE_REQUIRED_NOISE_STD:g} | outlier_ratio={BASE_REQUIRED_OUTLIER_RATIO:g}"
    )
    print(
        "synthetic_corruption = "
        f"noise_std={noise_std:.4f} | outlier_ratio sweep={outlier_ratios} | "
        "points/normals gaussian noise + uniform bbox outliers"
    )
    print(
        "threshold_rule = "
        f"max({THRESHOLD_FACTOR:.4f} * noise_std, "
        f"{THRESHOLD_SPACING_FACTOR:.4f} * median_nn_distance, "
        f"{MIN_THRESHOLD:.4f})"
    )
    print(
        "ransac_tuning_rule = "
        "sample_size=clip(round(0.06*N), 12, 20) | "
        "min_inliers=clip(round(0.15*N), max(sample_size,12), 30) | "
        "mss_max_pool_fraction=clip(6*sample_size/N, 0.18, 0.35)"
    )
    print(
        "methods_by_budget = "
        + ", ".join(f"{budget}:{list(methods)}" for budget, methods in METHODS_BY_BUDGET.items())
    )

    for dataset in dataset_specs:
        print(
            f"base_dataset = {dataset.input_file} | dataset_id = {dataset.dataset_id} | "
            f"base_points = {dataset.base_point_count} | removed_gt_outliers = {dataset.base_gt_outliers_removed}"
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
                "source_input_file",
                "run_id",
                "method",
                "outlier_ratio",
                "outlier_count",
                "noise_std",
                "threshold",
                "threshold_factor",
                "threshold_spacing_factor",
                "median_nn_distance",
                "spatial_extent",
                "point_count",
                "base_point_count",
                "base_gt_outliers_removed",
                "sample_size",
                "min_inliers",
                "mss_max_pool_fraction",
                "inner_iterations",
                "misclassification_error",
                "execution_time_s",
                "local_optimization_steps",
                "n_models",
                "synthetic_outliers_appended_at_tail",
                "gt_inliers_assumed_from_tail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
