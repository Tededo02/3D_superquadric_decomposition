import numpy as np
from .superquadric_param import SuperQuadricParams

# compute gradient
def _superquadric_gradient_canonical_from_pc(
    model: SuperQuadricParams,
    pc: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    a1, a2, a3 = model.a1, model.a2, model.a3
    e1, e2 = model.e1, model.e2

    p_xy = 2.0 / e2
    p_z = 2.0 / e1
    k = e2 / e1

    ax = np.maximum(np.abs(x / a1), eps)
    ay = np.maximum(np.abs(y / a2), eps)
    az = np.maximum(np.abs(z / a3), eps)

    u = ax ** p_xy + ay ** p_xy
    u = np.maximum(u, eps)

    du_dx = p_xy * (ax ** (p_xy - 1.0)) * (np.sign(x) / a1)
    du_dy = p_xy * (ay ** (p_xy - 1.0)) * (np.sign(y) / a2)
    dv_du = k * (u ** (k - 1.0))

    dfdx = dv_du * du_dx
    dfdy = dv_du * du_dy
    dfdz = p_z * (az ** (p_z - 1.0)) * (np.sign(z) / a3)
    return np.stack([dfdx, dfdy, dfdz], axis=1)

# compute the implicit function value for each point
# return an array of shape (N,) with the distance from each point to the superquadric surface
# the distance is computed as the absolute value of the implicit function value, which is zero on the surface and positive outside, negative inside
def implicit_f_superquadric_residual(model: SuperQuadricParams, points: np.ndarray) -> np.ndarray:
    R = model.rotation_matrix()
    points_canonical = (points - model.t) @ R
    x = points_canonical[:, 0]
    y = points_canonical[:, 1]
    z = points_canonical[:, 2]
    a1, a2, a3, e1, e2 = model.a1, model.a2, model.a3, model.e1, model.e2
    f = ((np.abs(x / a1) ** (2 / e2) + np.abs(y / a2) ** (2 / e2)) ** (e2 / e1) + np.abs(z / a3) ** (2 / e1)) - 1.0
    return f

# compute the first order residual for each point, which is the implicit function value divided by the norm of the gradient of the implicit function
# this gives a better approximation of the distance to the surface, especially for points close to the surface, and is more stable for optimization
def superquadric_first_order_residual(model: SuperQuadricParams, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    R = model.rotation_matrix()
    pc = (points - model.t) @ R
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    a1, a2, a3 = model.a1, model.a2, model.a3
    e1, e2 = model.e1, model.e2
    #build the function and its gradient in a way that is stable for optimization
    # powers
    p_xy = 2.0 / e2
    p_z = 2.0 / e1
    k = e2 / e1

    # safe abs to avoid 0**(neg) when exponents go < 1
    ax = np.maximum(np.abs(x / a1), eps)
    ay = np.maximum(np.abs(y / a2), eps)
    az = np.maximum(np.abs(z / a3), eps)
    u = ax ** p_xy + ay ** p_xy
    u = np.maximum(u, eps)

    v = u ** k
    w = az ** p_z
    f = v + w - 1.0

    grad_canonical = _superquadric_gradient_canonical_from_pc(model, pc, eps)
    grad_norm = np.linalg.norm(grad_canonical, axis=1)
    grad_norm = np.maximum(grad_norm, 1e-9)
    return f / grad_norm

# return the normal vector of the superquadric at each point, computed as the normalized gradient of the implicit function
def superquadric_normal_world(
    model: SuperQuadricParams,
    points: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    R = model.rotation_matrix()
    pc = (points - model.t) @ R
    grad_canonical = _superquadric_gradient_canonical_from_pc(model, pc, eps)
    grad_world = grad_canonical @ R.T
    grad_norm = np.linalg.norm(grad_world, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-9)
    return grad_world / grad_norm

# computes the ray from the center of the superquadric to each point in space
# then finds the respective point on the surface
# then computes the length of the segment from the surface to the point
def superquadric_radial_residual(model: SuperQuadricParams, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    R = model.rotation_matrix()
    pc = (points - model.t) @ R
    r = np.maximum(np.sqrt(pc[:, 0]**2 + pc[:, 1]**2 + pc[:, 2]**2), eps)
    f = implicit_f_superquadric_residual(model, points)
    r_surface = r * np.maximum(f + 1.0, eps) ** (-model.e1 / 2.0)
    return r - r_surface

# uses first-order residual for points whose direction is close to one of the 3 axes
# (horizontal/vertical), and radial residual from center for diagonal points.
# axis-alignment is measured in ellipsoid-normalised coordinates: if one component
# dominates (> AXIS_THRESHOLD after normalisation) the point is considered axis-aligned.
def superquadric_combo(model: SuperQuadricParams, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    R = model.rotation_matrix()
    a1, a2, a3 = model.a1, model.a2, model.a3
    AXIS_THRESHOLD = 0.7 # sensitivity of the axis distance (1 is full radial, 0 is full first order)

    pc = (points - model.t) @ R 

    # direction normalised by semi-axes so the measure is shape-aware
    n = np.stack([np.abs(pc[:, 0] / a1),
                  np.abs(pc[:, 1] / a2),
                  np.abs(pc[:, 2] / a3)], axis=1)
    n_norm = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), eps)
    n = n / n_norm  # each row is a unit vector; component > threshold → near that axis

    use_axis   = n.max(axis=1) > AXIS_THRESHOLD
    use_center = ~use_axis

    residuals = np.zeros(len(points), dtype=np.float64)

    if use_center.any():
        residuals[use_center] = superquadric_radial_residual(model, points[use_center], eps)

    if use_axis.any():
        residuals[use_axis] = superquadric_first_order_residual(model, points[use_axis], eps)

    return residuals


def superquadric_residual_vector(
    model: SuperQuadricParams,
    points: np.ndarray,
    metric: str = "mix",
    eps: float = 1e-12,
) -> np.ndarray:
    metric_name = metric.lower().replace("-", "_").replace(" ", "_")

    if metric_name == "radial":
        return superquadric_radial_residual(model, points, eps)

    if metric_name == "first_order":
        return superquadric_first_order_residual(model, points, eps)

    if metric_name == "mix":
        return superquadric_combo(model, points, eps)

    raise ValueError(f"Unsupported residual metric: {metric}")
