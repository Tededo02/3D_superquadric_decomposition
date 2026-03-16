from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def point_cloud_spatial_extent(points: np.ndarray) -> float:
    # Use the bounding-box diagonal as a global scale for the cloud.
    if points.size == 0:
        return 0.0

    spans = np.ptp(points, axis=0)
    return float(np.linalg.norm(spans))


# Return the neighbors within the chosen radius for each point and the edge list (i, j) with i < j.
def build_radius_graph(
    points: np.ndarray,
    m_neighbors: int = 12,
    radius: float = 0.06,
    radius_is_relative: bool = True,
) -> tuple[list[list[int]], np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"points must be a 2D array, got shape {points.shape}")

    n_points = int(points.shape[0])
    if n_points == 0:
        return [], np.empty((0, 2), dtype=np.int64)
    if m_neighbors <= 0:
        return [[] for _ in range(n_points)], np.empty((0, 2), dtype=np.int64)
    if radius_is_relative:
        # The paper sets the graph radius as a function of the point-cloud spatial extension.
        extent = point_cloud_spatial_extent(points)
        effective_radius = float(radius) * extent
    else:
        effective_radius = float(radius)
    if effective_radius <= 0.0:
        return [[] for _ in range(n_points)], np.empty((0, 2), dtype=np.int64)

    tree = cKDTree(points)
    max_query_k = min(n_points, int(m_neighbors) + 1)
    dists, idxs = tree.query(points, k=max_query_k, distance_upper_bound=effective_radius)
    dists = np.asarray(dists, dtype=np.float64).reshape(n_points, max_query_k)
    idxs = np.asarray(idxs, dtype=np.int64).reshape(n_points, max_query_k)

    neighbors: list[list[int]] = []
    edges: list[tuple[int, int]] = []
    for i in range(n_points):
        row_idx = idxs[i]
        row_dist = dists[i]
        valid = np.isfinite(row_dist) & (row_idx != i) & (row_idx < n_points)
        row_neighbors = row_idx[valid].tolist()
        neighbors.append(row_neighbors)
        for j in row_neighbors:
            if j > i:
                edges.append((i, j))

    if not edges:
        return neighbors, np.empty((0, 2), dtype=np.int64)

    edges_arr = np.asarray(edges, dtype=np.int64)
    return neighbors, edges_arr
