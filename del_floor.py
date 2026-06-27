from __future__ import annotations

"""Preprocess a raw point cloud and save a reduced, object-focused point cloud.

The script applies the following operations, in order:

1. Estimates the dominant plane with RANSAC, refines it using all its supporting
   points, orients its normal toward the side containing most of the scene, and
   removes the plane together with every point closer than ``FLOOR_CLEARANCE``.
2. When ``APPLY_VOXEL_AND_COMPONENT_FILTERING`` is enabled, voxel-downsamples
   the remaining cloud by retaining one representative point for each cube
   whose side length is ``VOXEL_SIZE``.
3. When the same flag is enabled, builds a radius-neighbor graph and retains
   only its largest connected component, removing isolated points and smaller
   pieces of scene clutter.

Every filtering step also updates the indices referring to the original point
cloud. These indices are used when saving the result so that vertex colors stay
aligned with the retained points. The source point cloud is never modified:
the processed cloud is exported to ``OUTPUT_POINT_CLOUD``. All configuration is
defined through the global variables below; the script accepts no command-line
arguments.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_POINT_CLOUD = PROJECT_ROOT / "test_objects" / "real_pc" / "etp_no_floor.ply"
OUTPUT_POINT_CLOUD = PROJECT_ROOT / "test_objects" / "real_pc" / "etp_no_floor_2.ply"

RANSAC_ITERATIONS = 400
RANSAC_SAMPLE_SIZE = 50000
RANSAC_THRESHOLD = 0.008
FLOOR_CLEARANCE = 0.015
RANDOM_SEED = 1234

# Set to False to perform only floor removal.
APPLY_VOXEL_AND_COMPONENT_FILTERING = False
VOXEL_SIZE = 0.008
COMPONENT_RADIUS_FACTOR = 2.0


@dataclass(frozen=True)
class FloorRemovalResult:
    points: np.ndarray
    keep_mask: np.ndarray
    floor_normal: np.ndarray
    floor_offset: float


@dataclass(frozen=True)
class PointCloudPreprocessingResult:
    points: np.ndarray
    keep_indices: np.ndarray
    floor_normal: np.ndarray
    floor_offset: float
    original_count: int
    above_floor_count: int
    voxelized_count: int


def voxel_downsample(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    point_array = np.asarray(points, dtype=np.float64)
    voxel_coordinates = np.floor(point_array / float(voxel_size)).astype(np.int64)
    _, selected_indices = np.unique(voxel_coordinates, axis=0, return_index=True)
    return point_array[selected_indices], selected_indices


def largest_radius_component(
    points: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    point_array = np.asarray(points, dtype=np.float64)
    pairs = cKDTree(point_array).query_pairs(r=float(radius), output_type="ndarray")
    if pairs.size == 0:
        return point_array, np.ones(point_array.shape[0], dtype=bool)

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    columns = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adjacency = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, columns)),
        shape=(point_array.shape[0], point_array.shape[0]),
    )
    _, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels)
    keep_mask = labels == int(np.argmax(component_sizes))
    return point_array[keep_mask], keep_mask


def fit_dominant_plane(
    points: np.ndarray,
    *,
    ransac_iterations: int = RANSAC_ITERATIONS,
    ransac_sample_size: int = RANSAC_SAMPLE_SIZE,
    distance_threshold: float = RANSAC_THRESHOLD,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, float]:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {point_array.shape}")
    if point_array.shape[0] < 3:
        raise ValueError("At least three points are required to estimate a plane")
    if ransac_iterations <= 0 or ransac_sample_size <= 0 or distance_threshold <= 0.0:
        raise ValueError("RANSAC iterations, sample size, and distance threshold must be positive")

    rng = np.random.default_rng(random_seed)
    sample_size = min(int(ransac_sample_size), point_array.shape[0])
    sample = point_array[rng.choice(point_array.shape[0], size=sample_size, replace=False)]

    best_normal = None
    best_offset = 0.0
    best_count = 0
    for _ in range(int(ransac_iterations)):
        triplet = sample[rng.choice(sample.shape[0], size=3, replace=False)]
        normal = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            continue

        normal /= normal_norm
        offset = -float(normal @ triplet[0])
        count = int(np.count_nonzero(np.abs(sample @ normal + offset) < distance_threshold))
        if count > best_count:
            best_normal = normal
            best_offset = offset
            best_count = count

    if best_normal is None:
        raise RuntimeError("Could not estimate a dominant plane")

    support_mask = np.abs(point_array @ best_normal + best_offset) < distance_threshold
    support = point_array[support_mask]
    center = support.mean(axis=0)
    covariance = (support - center).T @ (support - center)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    offset = -float(normal @ center)

    signed_distances = point_array @ normal + offset
    if np.quantile(signed_distances, 0.99) < abs(np.quantile(signed_distances, 0.01)):
        normal = -normal
        offset = -offset
    return normal, offset


def remove_floor_plane(
    points: np.ndarray,
    *,
    clearance: float = FLOOR_CLEARANCE,
    ransac_iterations: int = RANSAC_ITERATIONS,
    ransac_sample_size: int = RANSAC_SAMPLE_SIZE,
    distance_threshold: float = RANSAC_THRESHOLD,
    random_seed: int = RANDOM_SEED,
) -> FloorRemovalResult:
    if clearance < 0.0:
        raise ValueError(f"clearance must be non-negative, got {clearance}")

    point_array = np.asarray(points, dtype=np.float64)
    floor_normal, floor_offset = fit_dominant_plane(
        point_array,
        ransac_iterations=ransac_iterations,
        ransac_sample_size=ransac_sample_size,
        distance_threshold=distance_threshold,
        random_seed=random_seed,
    )
    signed_distances = point_array @ floor_normal + floor_offset
    keep_mask = signed_distances > float(clearance)
    return FloorRemovalResult(
        points=point_array[keep_mask],
        keep_mask=keep_mask,
        floor_normal=floor_normal,
        floor_offset=floor_offset,
    )


def preprocess_point_cloud(
    points: np.ndarray,
    *,
    clearance: float = FLOOR_CLEARANCE,
    ransac_iterations: int = RANSAC_ITERATIONS,
    ransac_sample_size: int = RANSAC_SAMPLE_SIZE,
    distance_threshold: float = RANSAC_THRESHOLD,
    voxel_size: float = VOXEL_SIZE,
    component_radius_factor: float = COMPONENT_RADIUS_FACTOR,
    random_seed: int = RANDOM_SEED,
) -> PointCloudPreprocessingResult:
    if APPLY_VOXEL_AND_COMPONENT_FILTERING and component_radius_factor <= 0.0:
        raise ValueError(
            f"component_radius_factor must be positive, got {component_radius_factor}"
        )

    point_array = np.asarray(points, dtype=np.float64)
    keep_indices = np.arange(point_array.shape[0])

    floor_result = remove_floor_plane(
        point_array,
        clearance=clearance,
        ransac_iterations=ransac_iterations,
        ransac_sample_size=ransac_sample_size,
        distance_threshold=distance_threshold,
        random_seed=random_seed,
    )
    point_array = floor_result.points
    keep_indices = keep_indices[floor_result.keep_mask]
    above_floor_count = int(point_array.shape[0])

    if APPLY_VOXEL_AND_COMPONENT_FILTERING:
        point_array, voxel_indices = voxel_downsample(point_array, voxel_size)
        keep_indices = keep_indices[voxel_indices]
        voxelized_count = int(point_array.shape[0])

        point_array, component_mask = largest_radius_component(
            point_array,
            radius=float(component_radius_factor) * float(voxel_size),
        )
        keep_indices = keep_indices[component_mask]
    else:
        voxelized_count = above_floor_count

    return PointCloudPreprocessingResult(
        points=point_array,
        keep_indices=keep_indices,
        floor_normal=floor_result.floor_normal,
        floor_offset=floor_result.floor_offset,
        original_count=int(np.asarray(points).shape[0]),
        above_floor_count=above_floor_count,
        voxelized_count=voxelized_count,
    )


def _load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    raw = trimesh.load(str(path))
    if not isinstance(raw, trimesh.PointCloud):
        raise ValueError(f"{path} is not a point cloud")

    points = np.asarray(raw.vertices, dtype=np.float64)
    colors = np.asarray(getattr(raw, "colors", []))
    if colors.shape[0] != points.shape[0]:
        colors = None
    return points, colors


def save_preprocessed_point_cloud() -> PointCloudPreprocessingResult:
    source = INPUT_POINT_CLOUD.expanduser().resolve()
    destination = OUTPUT_POINT_CLOUD.expanduser().resolve()
    points, colors = _load_point_cloud(source)
    result = preprocess_point_cloud(points)

    filtered_colors = None if colors is None else colors[result.keep_indices]
    destination.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(result.points, colors=filtered_colors).export(destination)
    return result


def main() -> int:
    result = save_preprocessed_point_cloud()
    print(
        f"Saved preprocessed point cloud to {OUTPUT_POINT_CLOUD.resolve()} | "
        f"original={result.original_count} "
        f"above_floor={result.above_floor_count} "
        f"voxelized={result.voxelized_count} "
        f"largest_component={result.points.shape[0]} "
        f"voxel_and_component_filtering={APPLY_VOXEL_AND_COMPONENT_FILTERING} "
        f"(floor_normal={np.array2string(result.floor_normal, precision=4)}, "
        f"clearance={FLOOR_CLEARANCE:.4f}, "
        f"voxel_size={VOXEL_SIZE:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
