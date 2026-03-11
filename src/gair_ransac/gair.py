import numpy as np
import maxflow
from src.superquadrics.superquadric_param import SuperQuadricParams
from .consensus import distance_err
# building energy and graph cut

def gair(
    points: np.ndarray,
    edges: np.ndarray,
    normals: np.ndarray,
    model: SuperQuadricParams,
    eps: float,
    error_metric: str = "mix",
    use_normal_coherence: bool = True,
    pair_weight: float = 1.0,
    coh_min: float = 0.05,
    outlier_scale: float = 3.0,
) -> np.ndarray:
    # GAIR: Geometric Aware Inlier Refinement via Graph-Cut.
    # points: (N, 3)
    # edges: (E, 2) undirected edges with i < j
    # normals: (N, 3) unit normals
    # model: current superquadric
    # eps: inlier threshold epsilon
    # error_metric: residual used for unary costs
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    N = points.shape[0]
    if normals.shape != (N, 3):
        raise ValueError(f"normals must have shape {(N,3)}, got {normals.shape}")

    # distances/residuals
    dist = distance_err(model, points, error_metric=error_metric)  # (N,)
    d_unary = np.clip(dist / (eps + 1e-12), 0.0, 1.0)
    d_pair = np.clip(dist / (outlier_scale * eps + 1e-12), 0.0, 1.0)

    # unary costs (paper eq. (5)):
    # cost(inlier) = err (normalized), cost(outlier) = 1
    cost_inlier = d_unary.copy()
    cost_outlier = np.zeros(N, dtype=np.float64)

    # We'll accumulate extra outlier unary costs derived from pairwise E00
    outlier_add = np.zeros(N, dtype=np.float64)

    # Allocate maxflow graph
    g = maxflow.Graph[float](N, int(edges.shape[0] * 2))
    nodeids = g.add_nodes(N)

    # Pairwise term (paper eq. (6)-(8)) encoded as:
    # - add Potts edge with weight w
    # - add outlier unary correction a/2 on both endpoints
    for (p, q) in edges:
        # Use the normal-driven coherence term only when requested.
        if use_normal_coherence:
            C = 0.5 * (1.0 + float(np.dot(normals[p], normals[q])))
            if C <= coh_min:
                continue
        else:
            C = 1.0

        # E00 = C*(1 - (d_p + d_q)/2)
        a = C * (1.0 - 0.5 * (d_pair[p] + d_pair[q]))
        a = max(a, 0.0)  # numerical safety

        # Potts weight w = C - a/2
        w = C - 0.5 * a
        w = max(w, 0.0)

        # accumulate unary outlier correction
        outlier_add[p] += 0.5 * a
        outlier_add[q] += 0.5 * a

        # add undirected (symmetric) edge: penalizes different labels
        ww = pair_weight * w
        g.add_edge(nodeids[p], nodeids[q], ww, ww)

    # apply outlier unary correction
    cost_outlier += pair_weight * outlier_add

    # IMPORTANT: choose mapping for labels:
    # We'll interpret:
    #   SOURCE side  -> outlier label (0)
    #   SINK side    -> inlier label (1)
    #
    # In PyMaxflow: add_tedge(node, cs, ct) adds:
    #   source->node capacity cs
    #   node->sink capacity ct
    # If node ends in SINK, we pay cs; if node ends in SOURCE, we pay ct.
    #
    # So to make:
    #   cost(inlier)  = cost_inlier
    #   cost(outlier) = cost_outlier
    # we set:
    #   cs = cost_inlier, ct = cost_outlier
    for i in range(N):
        g.add_tedge(nodeids[i], float(cost_inlier[i]), float(cost_outlier[i]))

    g.maxflow()

    # segment: 0 -> SOURCE(outlier), 1 -> SINK(inlier)
    inliers_mask = np.fromiter((g.get_segment(nid) == 1 for nid in nodeids), dtype=bool, count=N)
    return inliers_mask
