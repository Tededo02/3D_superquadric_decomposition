"""
RansaCov: MAXIMUM COVERAGE ILP for multi-model selection.
Based on eq. (5) of "RansaCov: Multi-model fitting with coverage constraints" (Magri & Fusiello).

Given a pool of candidate models and a point cloud, selects at most k models
that cover the maximum number of inlier points (residual < threshold).
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def ransacov(candidates, points, k, threshold, residual_fn):
    """
    Parameters
    ----------
    candidates  : list of model objects (SuperQuadricParams)
    points      : (N, 3) float array — clean point cloud to cover
    k           : int — maximum number of models to select
    threshold   : float — inlier distance threshold (epsilon)
    residual_fn : callable(model, points) -> (N,) residuals (unsigned)

    Returns
    -------
    selected_indices : list[int] — indices into `candidates` of selected models
    n_covered        : int — number of points covered by the solution
    """
    n_pts = len(points)
    m = len(candidates)

    if m == 0:
        return [], 0

    # ── Build binary coverage matrix P[i,j]
    # P[i,j] = 1 if point i is inlier of model j
    P = np.zeros((n_pts, m), dtype=np.float64)
    for j, model in enumerate(candidates):
        residuals = residual_fn(model, points)
        P[:, j] = (residuals < threshold).astype(np.float64)

    # Sort by cardinality descending; drop Sj if fully subsumed by union of larger sets.
    order = np.argsort(-P.sum(axis=0))
    keep = []
    covered_so_far = np.zeros(n_pts, dtype=bool)
    for j in order:
        col = P[:, j].astype(bool)
        if not np.all(col <= covered_so_far):
            keep.append(int(j))
            covered_so_far |= col

    if not keep:
        return [], 0

    P_kept = P[:, keep]
    m_kept = len(keep)

    # ── MILP formulation (scipy.optimize.milp) ────────────────────────────────
    # Variables: [z_0 .. z_{m_kept-1},  y_0 .. y_{n_pts-1}]
    # minimize  -Σ yi            (maximise coverage)
    # subject to:
    #   (1)  Σ zj <= k
    #   (2)  Σ_j P[i,j]*zj - yi >= 0   ∀i   (yi can only be 1 if covered)
    #   zj ∈ {0,1},  0 <= yi <= 1
    n_vars = m_kept + n_pts

    c = np.zeros(n_vars)
    c[m_kept:] = -1.0          # -yi

    integrality = np.zeros(n_vars)
    integrality[:m_kept] = 1   # zj are integer (binary)

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

    # Constraint (1): sum(zj) <= k
    row0 = np.zeros(n_vars)
    row0[:m_kept] = 1.0

    # Constraints (2): P[i,:] z - yi >= 0
    A_coverage = np.zeros((n_pts, n_vars))
    A_coverage[:, :m_kept] = P_kept
    A_coverage[np.arange(n_pts), m_kept + np.arange(n_pts)] = -1.0

    A = np.vstack([row0, A_coverage])
    lb_con = np.concatenate([[-np.inf], np.zeros(n_pts)])
    ub_con = np.concatenate([[float(k)], np.full(n_pts, np.inf)])

    result = milp(
        c=c,
        constraints=LinearConstraint(A, lb_con, ub_con),
        integrality=integrality,
        bounds=bounds,
    )

    if not result.success:
        print(f"  [ransacov] ILP solver failed: {result.message}")
        return [], 0

    z = result.x[:m_kept]
    selected_local = np.where(z > 0.5)[0].tolist()
    selected_global = [keep[i] for i in selected_local]
    n_covered = int(round(result.x[m_kept:].sum()))

    return selected_global, n_covered
