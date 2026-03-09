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
    # The paper sets the graph radius as a function of the point-cloud spatial extension.
    extent = point_cloud_spatial_extent(points)
    effective_radius = float(radius) * extent
    tree = cKDTree(points)
    # query_ball_point also returns the point itself, which we remove below.
    neigh_lists = tree.query_ball_point(points, r=effective_radius)

    neighbors: list[list[int]] = []
    edges: list[tuple[int, int]] = []
    for i, idxs in enumerate(neigh_lists):
        # Keep at most the first m spatial neighbors inside the radius.
        idxs = [j for j in idxs if j != i]
        if len(idxs) > m_neighbors:
            d2 = np.sum((points[idxs] - points[i]) ** 2, axis=1)
            order = np.argsort(d2)[:m_neighbors]
            idxs = [idxs[k] for k in order]
        neighbors.append(idxs)
        for j in idxs:
            if j > i:
                edges.append((i, j))

    if not edges:
        return neighbors, np.empty((0, 2), dtype=np.int64)

    edges_arr = np.asarray(edges, dtype=np.int64)
    return neighbors, edges_arr
