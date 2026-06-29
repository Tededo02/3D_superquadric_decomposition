from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NormalEstimationResult:
    normals: np.ndarray
    method: str


def _normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


def _orient_normals_global_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    center = np.median(points, axis=0)
    radial = points - center
    score = np.einsum("ij,ij->i", normals, radial, optimize=True)
    finite_score = score[np.isfinite(score)]
    if finite_score.size and float(np.median(finite_score)) < 0.0:
        return -normals
    return normals


def estimate_normals_pca_center_oriented(
    points: np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    point_count = int(point_array.shape[0])
    if point_count < 3:
        raise ValueError(f"Need at least 3 points to estimate normals, got {point_count}")

    k = min(max(int(n_neighbors), 3), point_count)
    tree = cKDTree(point_array)
    neighbor_indices = tree.query(point_array, k=k)[1]
    normals = np.empty_like(point_array, dtype=np.float64)
    cloud_center = np.mean(point_array, axis=0)

    for index, indices in enumerate(neighbor_indices):
        neighborhood = point_array[indices]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if np.dot(normal, point_array[index] - cloud_center) < 0.0:
            normal = -normal
        normals[index] = normal

    return _normalized(normals)


def _estimate_normals_open3d_consistent(
    points: np.ndarray,
    n_neighbors: int,
    orient_neighbors: int,
) -> np.ndarray | None:
    try:
        import open3d as o3d
    except ImportError:
        return None

    point_array = np.asarray(points, dtype=np.float64)
    point_count = int(point_array.shape[0])
    if point_count < 3:
        raise ValueError(f"Need at least 3 points to estimate normals, got {point_count}")

    estimate_k = min(max(int(n_neighbors), 3), point_count)
    orient_k = min(max(int(orient_neighbors), 3), point_count)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=estimate_k)
    )
    pcd.orient_normals_consistent_tangent_plane(orient_k)

    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape != point_array.shape or not np.isfinite(normals).all():
        return None

    normals = _normalized(normals)
    return _orient_normals_global_outward(point_array, normals)


def estimate_normals_consistent(
    points: np.ndarray,
    n_neighbors: int = 20,
    orient_neighbors: int | None = None,
    prefer_open3d: bool = True,
) -> NormalEstimationResult:
    orient_k = int(n_neighbors if orient_neighbors is None else orient_neighbors)
    if prefer_open3d:
        normals = _estimate_normals_open3d_consistent(points, n_neighbors, orient_k)
        if normals is not None:
            return NormalEstimationResult(
                normals=normals,
                method=(
                    "open3d estimate_normals + "
                    f"orient_normals_consistent_tangent_plane(k={orient_k})"
                ),
            )

    normals = estimate_normals_pca_center_oriented(points, n_neighbors=n_neighbors)
    return NormalEstimationResult(
        normals=normals,
        method="local PCA + per-point center-outward fallback",
    )
