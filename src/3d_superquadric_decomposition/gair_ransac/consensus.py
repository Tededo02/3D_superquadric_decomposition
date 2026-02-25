import numpy as np
from superquadric_param import SuperQuadricParams

def algebraic_err(model: SuperQuadricParams, points: np.ndarray) -> np.ndarray:
    from .inner_ransac import implicit_f_superquadric_residual
    f=implicit_f_superquadric_residual(model, points)
    return np.abs(f)


def compute_consensus(model: SuperQuadricParams, points: np.ndarray, threshold: float) -> np.ndarray:
    # compute the implicit function value for each point
    err = algebraic_err(model, points)
    # compute the inliers as the points where the absolute value of the implicit function is less than the threshold
    inliers = np.abs(err) < threshold
    return inliers
