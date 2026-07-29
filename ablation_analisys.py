# FullGairEnergy, with labels y_i in {outlier, inlier}:
# E(y) = sum_i U_i(y_i) + sum_(i,j) w_ij [y_i != y_j].
# U_i(inlier) = clip(d_i / eps, 0, 1) + clip((1 - <n_i, n_model_i>) / 2, 0, 1).
# U_i(outlier) = 1 + 1/2 sum_j p_ij, where p_ij = c_ij(1 - (rho_i + rho_j) / 2).
# c_ij = (1 + <n_i, n_j>) / 2, rho_i = clip(d_i / (outlier_scale eps), 0, 1).
# w_ij = c_ij - p_ij / 2.
# ConstantCoherenceGairEnergy uses the same formula and unary term, but fixes c_ij = 1.
# GairThresholdEnergy filters with c_ij, then fixes c_ij = 1 on retained edges.
# OnlyUnaryGairStrategy uses the model-normal unary term and the GC-RANSAC pairwise term.

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from src.gair_ransac.energy_strategies import (
    ComposedGairEnergy,
    ConstantCoherenceGairEnergy,
    FullGairEnergy,
    GairThresholdEnergy,
    GairEnergyStrategy,
    GcRansacEnergy,
    OldPaperGairEnergy,
    OnlyUnaryGairStrategy,
)
from src.gair_ransac.gair_ransac import gair_ransac
from src.superquadrics import superquadric_mesh as superquadric_mesh
from src.superquadrics import superquadric_sampling


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_GEOMETRY_PATH = (
    PROJECT_ROOT / "test_objects" / "99992.stl"
)
RESULTS_CSV_PATH = (
    PROJECT_ROOT / "data" / "results" / "ablation_energy_metrics.csv"
)

ENERGY_STRATEGIES: tuple[GairEnergyStrategy, ...] = (
    GairThresholdEnergy(),
    ConstantCoherenceGairEnergy(),
    OnlyUnaryGairStrategy(),
    FullGairEnergy(),
    GcRansacEnergy(),
)
ITERATIONS_PER_ENERGY = 8
RUN_STRATEGIES_CONCURRENTLY = True
MAX_STRATEGY_THREADS = len(ENERGY_STRATEGIES)
# Input meshes use this count; input point clouds are never subsampled.
SAMPLED_POINT_COUNT = 8000
# This count is used for mesh-based GT and reconstructed model sampling.
EVALUATION_POINT_COUNT = 16000
OUTLIER_RATIO = 0.10
OUTLIER_MARGIN = 0.10
OUTLIER_MODE = "uniform"
DEFAULT_BASE_SEED = 2345679
EVALUATION_SEED = 421
NOISE_STD = 0.0
NOISE_NORMAL_STD =0.0 #0.14  per 10    #0.07 per 5 gradi
THRESHOLD = 0.018
SAMPLE_SIZE = 30
MIN_INLIERS = 40
MAX_MODELS = 8
MAX_ITERATIONS = 20
INNER_ITERATIONS = 40
M_NEIGHBORS = 6
MIN_COVERAGE = 0.14

CSV_FIELD_NAMES = (
    "energy",
    "unary",
    "pairwise",
    "chamfer_gt_to_reconstruction",
    "chamfer_reconstruction_to_gt",
    "chamfer_total",
    "misclassification_error",
    "model_count",
    "local_optimization_iterations",
)
CSV_WRITE_LOCK = Lock()


def resolve_input_path(path: str | Path) -> Path:
    input_path = Path(path).expanduser()
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Input geometry not found: {input_path}")
    return input_path


def load_input_geometry(
    path: Path,
) -> trimesh.Trimesh | trimesh.PointCloud:
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        if not geometry.geometry:
            raise ValueError(f"Scene has no geometry: {path}")
        scene_geometries = list(geometry.geometry.values())
        if all(isinstance(item, trimesh.Trimesh) for item in scene_geometries):
            geometry = trimesh.util.concatenate(scene_geometries)
        elif len(scene_geometries) == 1:
            geometry = scene_geometries[0]
        else:
            raise TypeError(f"Scene contains incompatible geometries: {path}")

    if not isinstance(geometry, (trimesh.Trimesh, trimesh.PointCloud)):
        raise TypeError(f"Unsupported input geometry: {path}")
    return geometry


def extract_point_cloud_data(
    point_cloud: trimesh.PointCloud,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(point_cloud.vertices, dtype=np.float64)
    ply_raw = point_cloud.metadata.get("_ply_raw", {})
    vertex_data = ply_raw.get("vertex", {}).get("data")
    property_names = vertex_data.dtype.names if vertex_data is not None else None
    if not property_names or not {"nx", "ny", "nz"}.issubset(property_names):
        raise ValueError(
            "The input point cloud must contain the nx, ny and nz vertex properties"
        )

    normals = np.column_stack(
        (vertex_data["nx"], vertex_data["ny"], vertex_data["nz"])
    ).astype(np.float64, copy=False)
    if points.shape != normals.shape:
        raise ValueError(
            f"Point and normal shapes do not match: {points.shape} and {normals.shape}"
        )
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError(f"Point cloud contains non-finite values: {path}")

    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(normal_lengths <= 1e-12):
        raise ValueError(f"Point cloud contains zero-length normals: {path}")
    return points, normals / normal_lengths


def sample_experiment_data(
    geometry: trimesh.Trimesh | trimesh.PointCloud,
    input_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if isinstance(geometry, trimesh.PointCloud):
        original_points, original_normals = extract_point_cloud_data(
            geometry,
            input_path,
        )
        ground_truth = original_points.copy()
        outlier_geometry = geometry
        input_kind = "point cloud"
    else:
        sampled_points, sampled_normals = (
            superquadric_sampling.sampling_sq_noisy(
                [geometry],
                n_points=SAMPLED_POINT_COUNT,
                noise_std=NOISE_STD,
                normal_noise_std=NOISE_NORMAL_STD,
                seed=DEFAULT_BASE_SEED,
            )
        )
        ground_truth_samples, _ = superquadric_sampling.sampling_sq_random(
            [geometry],
            n_points=EVALUATION_POINT_COUNT,
            seed=EVALUATION_SEED,
        )
        original_points = np.vstack(sampled_points)
        original_normals = np.vstack(sampled_normals)
        ground_truth = np.vstack(ground_truth_samples)
        outlier_geometry = geometry
        input_kind = "mesh"

    if not 0.0 <= OUTLIER_RATIO < 1.0:
        raise ValueError("OUTLIER_RATIO must be in the range [0, 1)")
    outlier_point_count = int(round(original_points.shape[0] * OUTLIER_RATIO))
    outlier_points, outlier_normals = superquadric_sampling.sampling_outliers(
        [outlier_geometry],
        n_out=outlier_point_count,
        margin=OUTLIER_MARGIN,
        mode=OUTLIER_MODE,
        seed=DEFAULT_BASE_SEED,
    )

    bb_min, bb_max = original_points.min(axis=0), original_points.max(axis=0)
    bb_size = bb_max - bb_min
    scale = float(np.max(bb_size))
    if scale <= 0.0:
        scale = 1.0

    original_points = (original_points - bb_min) / scale
    outlier_points = (outlier_points - bb_min) / scale
    ground_truth = (ground_truth - bb_min) / scale
    print(
        "  bounding box after normalization: "
        f"min={original_points.min(axis=0)}  "
        f"max={original_points.max(axis=0)}"
    )
    print(
        "  runtime data: "
        f"input={input_kind} "
        f"inliers={original_points.shape[0]} "
        f"outliers={outlier_points.shape[0]} "
        f"outlier_ratio={OUTLIER_RATIO:.1%} "
        f"ground_truth={ground_truth.shape[0]}"
    )

    point_cloud = np.vstack((original_points, outlier_points))
    normals = np.vstack((original_normals, outlier_normals))
    return point_cloud, normals, ground_truth, original_points.shape[0]


def combine_inlier_masks(
    inlier_masks: list[np.ndarray],
    point_count: int,
) -> np.ndarray:
    combined_mask = np.zeros(point_count, dtype=bool)
    for inlier_mask in inlier_masks:
        mask = np.asarray(inlier_mask, dtype=bool)
        if mask.shape != combined_mask.shape:
            raise ValueError(
                f"Inlier mask must have shape {combined_mask.shape}, "
                f"got {mask.shape}"
            )
        combined_mask |= mask
    return combined_mask


def compute_chamfer_metrics(
    ground_truth_points: np.ndarray,
    reconstruction_points: np.ndarray,
) -> tuple[float, float, float]:
    if reconstruction_points.shape[0] == 0:
        return float("inf"), float("inf"), float("inf")

    reconstruction_tree = cKDTree(reconstruction_points)
    ground_truth_tree = cKDTree(ground_truth_points)
    gt_to_reconstruction = float(
        reconstruction_tree.query(ground_truth_points, k=1)[0].mean()
    )
    reconstruction_to_gt = float(
        ground_truth_tree.query(reconstruction_points, k=1)[0].mean()
    )
    return (
        gt_to_reconstruction,
        reconstruction_to_gt,
        gt_to_reconstruction + reconstruction_to_gt,
    )


def compute_misclassification_error(
    predicted_inliers: np.ndarray,
    original_point_count: int,
) -> float:
    ground_truth_inliers = np.zeros(predicted_inliers.shape[0], dtype=bool)
    ground_truth_inliers[:original_point_count] = True
    return float(np.mean(predicted_inliers != ground_truth_inliers))


def get_energy_names(
    energy_strategy: GairEnergyStrategy,
) -> tuple[str, str, str]:
    energy_name = type(energy_strategy).__name__
    if not isinstance(energy_strategy, ComposedGairEnergy):
        return energy_name, "", ""
    return (
        energy_name,
        type(energy_strategy.unary).__name__,
        type(energy_strategy.pairwise).__name__,
    )


def save_energy_metrics(
    energy_strategy: GairEnergyStrategy,
    models: list,
    inlier_masks: list[np.ndarray],
    point_count: int,
    original_point_count: int,
    ground_truth_points: np.ndarray,
    local_optimization_iterations: int,
) -> dict[str, str | int | float]:
    if models:
        meshes = [
            superquadric_mesh.superquadric_mesh(model)
            for model in models
        ]
        sampled_reconstruction, _ = (
            superquadric_sampling.sampling_sq_random(
                meshes,
                n_points=EVALUATION_POINT_COUNT,
                seed=EVALUATION_SEED,
            )
        )
        reconstruction_points = np.vstack(sampled_reconstruction)
    else:
        reconstruction_points = np.empty((0, 3), dtype=np.float64)

    chamfer_gt_to_reconstruction, chamfer_reconstruction_to_gt, chamfer_total = (
        compute_chamfer_metrics(
            ground_truth_points,
            reconstruction_points,
        )
    )
    predicted_inliers = combine_inlier_masks(inlier_masks, point_count)
    misclassification_error = compute_misclassification_error(
        predicted_inliers,
        original_point_count,
    )
    energy_name, unary_name, pairwise_name = get_energy_names(energy_strategy)
    row: dict[str, str | int | float] = {
        "energy": energy_name,
        "unary": unary_name,
        "pairwise": pairwise_name,
        "chamfer_gt_to_reconstruction": chamfer_gt_to_reconstruction,
        "chamfer_reconstruction_to_gt": chamfer_reconstruction_to_gt,
        "chamfer_total": chamfer_total,
        "misclassification_error": misclassification_error,
        "model_count": len(models),
        "local_optimization_iterations": local_optimization_iterations,
    }

    with CSV_WRITE_LOCK:
        RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not RESULTS_CSV_PATH.exists()
            or RESULTS_CSV_PATH.stat().st_size == 0
        )
        with RESULTS_CSV_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELD_NAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    return row


def run_energy_strategy(
    energy_strategy: GairEnergyStrategy,
    point_cloud: np.ndarray,
    normals: np.ndarray,
    ground_truth_points: np.ndarray,
    original_point_count: int,
    random_seed: int,
    iteration: int,
) -> tuple:
    result = gair_ransac(
        point_cloud=point_cloud,
        normals=normals,
        threshold=THRESHOLD,
        max_models=MAX_MODELS,
        max_iterations=MAX_ITERATIONS,
        sample_size=SAMPLE_SIZE,
        min_inliers=MIN_INLIERS,
        inner_iterations=INNER_ITERATIONS,
        m_neighbors=M_NEIGHBORS,
        min_coverage=MIN_COVERAGE,
        energy_strategy=energy_strategy,
        random_seed=random_seed,
    )
    models, inlier_masks, _, local_optimization_iterations = result
    metrics = save_energy_metrics(
        energy_strategy=energy_strategy,
        models=models,
        inlier_masks=inlier_masks,
        point_count=point_cloud.shape[0],
        original_point_count=original_point_count,
        ground_truth_points=ground_truth_points,
        local_optimization_iterations=local_optimization_iterations,
    )
    print(
        f"Saved metrics for {metrics['energy']} iteration {iteration} "
        f"to {RESULTS_CSV_PATH}"
    )
    return result


def run_strategy_iteration(
    point_cloud: np.ndarray,
    normals: np.ndarray,
    ground_truth_points: np.ndarray,
    original_point_count: int,
    random_seed: int,
    iteration: int,
) -> list[tuple]:
    if not RUN_STRATEGIES_CONCURRENTLY:
        return [
            run_energy_strategy(
                energy_strategy=energy_strategy,
                point_cloud=point_cloud,
                normals=normals,
                ground_truth_points=ground_truth_points,
                original_point_count=original_point_count,
                random_seed=random_seed,
                iteration=iteration,
            )
            for energy_strategy in ENERGY_STRATEGIES
        ]

    if MAX_STRATEGY_THREADS <= 0:
        raise ValueError("MAX_STRATEGY_THREADS must be positive")

    worker_count = min(MAX_STRATEGY_THREADS, len(ENERGY_STRATEGIES))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_energy_strategy,
                energy_strategy=energy_strategy,
                point_cloud=point_cloud,
                normals=normals,
                ground_truth_points=ground_truth_points,
                original_point_count=original_point_count,
                random_seed=random_seed,
                iteration=iteration,
            )
            for energy_strategy in ENERGY_STRATEGIES
        ]
        return [future.result() for future in futures]


def call_gair_ransac_with_multiple_energy(
    point_cloud: np.ndarray,
    normals: np.ndarray,
    ground_truth_points: np.ndarray,
    original_point_count: int,
) -> list[tuple]:
    results: list[tuple] = []
    for iteration in range(1, ITERATIONS_PER_ENERGY + 1):
        results.extend(
            run_strategy_iteration(
                point_cloud=point_cloud,
                normals=normals,
                ground_truth_points=ground_truth_points,
                original_point_count=original_point_count,
                random_seed=DEFAULT_BASE_SEED + iteration,
                iteration=iteration,
            )
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_geometry", nargs="?", default=INPUT_GEOMETRY_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    input_path = resolve_input_path(args.input_geometry)
    geometry = load_input_geometry(input_path)
    point_cloud, normals, ground_truth_points, original_point_count = (
        sample_experiment_data(geometry, input_path)
    )

    call_gair_ransac_with_multiple_energy(
        point_cloud=point_cloud,
        normals=normals,
        ground_truth_points=ground_truth_points,
        original_point_count=original_point_count,
    )


if __name__ == "__main__":
    main()
