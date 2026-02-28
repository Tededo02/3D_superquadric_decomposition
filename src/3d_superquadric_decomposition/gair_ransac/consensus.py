import numpy as np
from superquadric_param import SuperQuadricParams
from .superquadric_residual import *

def distance_err(model: SuperQuadricParams, points: np.ndarray) -> np.ndarray:
    d = superquadric_radial_residual(model, points)
    return np.abs(d)


def compute_consensus(model: SuperQuadricParams, points: np.ndarray, threshold: float) -> np.ndarray:
    err = distance_err(model, points)
    inliers = err < threshold
    return inliers
