import csv
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from src.gair_ransac.energy_strategies import (
    ComposedGairEnergy,
    FullGairEnergy,
    GairEnergyStrategy,
)
from src.gair_ransac.gair_ransac import gair_ransac
from src.superquadrics import superquadric_mesh as superquadric_mesh
from src.superquadrics import superquadric_sampling


PROJECT_ROOT = Path(__file__).resolve().parent
POINT_CLOUD_PATH = PROJECT_ROOT / "test_objects" / "cartoon_character.glb"
RESULTS_CSV_PATH = (
    PROJECT_ROOT / "data" / "results" / "ablation_energy_metrics.csv"
)

ENERGY_STRATEGIES: tuple[GairEnergyStrategy, ...] = (
    FullGairEnergy(),
)
SAMPLED_POINT_COUNT = 30000
EVALUATION_POINT_COUNT = 30000
OUTLIER_POINT_COUNT = 3000
OUTLIER_MARGIN = 0.10
OUTLIER_MODE = "uniform"
DEFAULT_BASE_SEED = 12345679
EVALUATION_SEED = 42
NOISE_STD = 0.0
THRESHOLD = 0.02
NOISE_NORMAL_STD = 0.0
SAMPLE_SIZE = 40
MIN_INLIERS = 40
MAX_MODELS = 2
MAX_ITERATIONS = 60
INNER_ITERATIONS = 40
M_NEIGHBORS = 6

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


def load_mesh(path: Path) -> trimesh.Trimesh:
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        if not geometry.geometry:
            raise ValueError(f"Scene has no geometry: {path}")
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))

    if not isinstance(geometry, trimesh.Trimesh):
        raise TypeError(f"Input is not a mesh: {path}")
    return geometry


def sample_experiment_data(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    sampled_points, sampled_normals = superquadric_sampling.sampling_sq_noisy(
        [mesh],
        n_points=SAMPLED_POINT_COUNT,
        noise_std=NOISE_STD,
        normal_noise_std=NOISE_NORMAL_STD,
        seed=DEFAULT_BASE_SEED,
    )
    ground_truth_samples, _ = superquadric_sampling.sampling_sq_random(
        [mesh],
        n_points=EVALUATION_POINT_COUNT,
        seed=EVALUATION_SEED,
    )
    outlier_points, outlier_normals = superquadric_sampling.sampling_outliers(
        [mesh],
        n_out=OUTLIER_POINT_COUNT,
        margin=OUTLIER_MARGIN,
        mode=OUTLIER_MODE,
        seed=DEFAULT_BASE_SEED,
    )

    original_points = np.vstack(sampled_points)
    original_normals = np.vstack(sampled_normals)
    point_cloud = np.vstack((original_points, outlier_points))
    normals = np.vstack((original_normals, outlier_normals))
    ground_truth = np.vstack(ground_truth_samples)
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

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not RESULTS_CSV_PATH.exists()
        or RESULTS_CSV_PATH.stat().st_size == 0
    )
    with RESULTS_CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELD_NAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row


def call_gair_ransac_with_multiple_energy(
    point_cloud: np.ndarray,
    normals: np.ndarray,
    ground_truth_points: np.ndarray,
    original_point_count: int,
) -> list[tuple]:
    results = []
    for energy_strategy in ENERGY_STRATEGIES:
        result = gair_ransac(
            point_cloud=point_cloud,
            normals=normals,
            threshold=THRESHOLD,
            max_models=MAX_MODELS,
            max_iterations=MAX_ITERATIONS,
            sample_size=SAMPLE_SIZE,
            min_inliers=MIN_INLIERS,
            inner_iterations=INNER_ITERATIONS,
            random_seed=DEFAULT_BASE_SEED,
            m_neighbors=M_NEIGHBORS,
            energy_strategy=energy_strategy,
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
            f"Saved metrics for {metrics['energy']} "
            f"to {RESULTS_CSV_PATH}"
        )
        results.append(result)
    return results


def main():
    mesh = load_mesh(POINT_CLOUD_PATH)
    point_cloud, normals, ground_truth_points, original_point_count = (
        sample_experiment_data(mesh)
    )

    call_gair_ransac_with_multiple_energy(
        point_cloud=point_cloud,
        normals=normals,
        ground_truth_points=ground_truth_points,
        original_point_count=original_point_count,
    )


if __name__ == "__main__":
    main()
