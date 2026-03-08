# gair.py (o utility interna)
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


# return neighbors within radius for each point, and the list of edges (i,j) with i<j
def build_radius_graph(points: np.ndarray, m_neighbors: int = 12, radius: float =1)-> tuple[list[list[int]], np.ndarray]:
    N = points.shape[0]
    tree = cKDTree(points)
    # list of lists indices within radius (including self)
    neigh_lists = tree.query_ball_point(points, r=radius)

    neighbors: list[list[int]] = []
    edges = []
    for i, idxs in enumerate(neigh_lists):
        # remove self and cap to m_neighbors
        idxs = [j for j in idxs if j != i]
        if len(idxs) > m_neighbors:
            # opzionale: tieni i più vicini
            d2 = np.sum((points[idxs] - points[i])**2, axis=1)
            order = np.argsort(d2)[:m_neighbors]
            idxs = [idxs[k] for k in order]
        neighbors.append(idxs)
        for j in idxs:
            if j > i:
                edges.append((i, j))

    edges_arr = np.asarray(edges)
    return neighbors, edges_arr