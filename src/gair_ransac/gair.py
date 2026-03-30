import numpy as np
import maxflow
from src.superquadrics.superquadric_param import SuperQuadricParams
from .consensus import distance_err, normal_alignment_score

# building energy and graph cut
COH_MIN: float = 0.9
OUTLIER_SCALE: float = 3.0,




def gair(
    points: np.ndarray,
    edges: np.ndarray,
    normals: np.ndarray,
    model: SuperQuadricParams,
    eps: float,
    error_metric: str = "radial",
    use_normal_coherence: bool = True,
    use_model_normal_agreement: bool = False,
    model_normal_weight: float = 1.0,
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
    edges = np.asarray(edges, dtype=np.int64)
    N = points.shape[0]
    if normals.shape != (N, 3):
        raise ValueError(f"normals must have shape {(N,3)}, got {normals.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {edges.shape}")

    # distances/residuals
    eps_denom = float(eps) + 1e-12
    pair_denom = float(OUTLIER_SCALE) * float(eps) + 1e-12
    dist = distance_err(model, points, error_metric=error_metric)  # (N,)
    d_unary = np.clip(dist / eps_denom, 0.0, 1.0)
    d_pair = np.clip(dist / pair_denom, 0.0, 1.0)

    # unary costs (paper eq. (5)):
    # cost(inlier) = err (normalized), cost(outlier) = 1
    cost_inlier = d_unary.copy()
    cost_outlier = np.ones(N, dtype=np.float64)
    # Optionally add a normal agreement penalty to the inlier cost, based on the angle between the point normal and the model normal.
    # The penalty is in [0, model_normal_weight], where 0 means perfect agreement and model_normal_weight means orthogonal (or opposite) normals.
    # the function used is 0.5*(1 - cos(angle)) which is 0 when angle=0 and 1 when angle=180, scaled by model_normal_weight and clipped to [0, model_normal_weight].
    if use_model_normal_agreement and model_normal_weight != 0.0:
        alignment = normal_alignment_score(model, points, normals)
        normal_penalty = np.clip(0.5 * (1.0 - alignment), 0.0, 1.0)
        cost_inlier += model_normal_weight * normal_penalty

    # Allocate maxflow graph
    g = maxflow.Graph[float](N, int(edges.shape[0] * 2))
    nodeids = g.add_nodes(N)

    # Pairwise term (paper eq. (6)-(8)) encoded as:
    # - add Potts edge with weight w
    # - add outlier unary correction a/2 on both endpoints
    if edges.size:
        edge_p = edges[:, 0]
        edge_q = edges[:, 1]

        if use_normal_coherence:
            coherence = 0.5 * (
                1.0 + np.einsum("ij,ij->i", normals[edge_p], normals[edge_q], optimize=True)
            )
            valid_edges = coherence > COH_MIN
            edge_p = edge_p[valid_edges]
            edge_q = edge_q[valid_edges]
            coherence = coherence[valid_edges]
        else:
            coherence = np.ones(edges.shape[0], dtype=np.float64)

        if edge_p.size:
            # E00 = C*(1 - (d_p + d_q)/2)
            a = coherence * (1.0 - 0.5 * (d_pair[edge_p] + d_pair[edge_q]))
            np.maximum(a, 0.0, out=a)  # numerical safety

            # Potts weight w = C - a/2
            w = coherence - 0.5 * a
            np.maximum(w, 0.0, out=w)

            # accumulate unary outlier correction in edge order
            outlier_add = np.zeros(N, dtype=np.float64)
            half_a = 0.5 * a
            np.add.at(outlier_add, edge_p, half_a)
            np.add.at(outlier_add, edge_q, half_a)
            cost_outlier += outlier_add

            # add undirected (symmetric) edges: penalize different labels
            ww = w
            g.add_edges(nodeids[edge_p], nodeids[edge_q], ww, ww)

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
    g.add_grid_tedges(nodeids, cost_inlier, cost_outlier)

    g.maxflow()

    # segment: 0 -> SOURCE(outlier), 1 -> SINK(inlier)
    return g.get_grid_segments(nodeids)
