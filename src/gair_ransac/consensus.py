import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_residual_vector


def distance_err(model: SuperQuadricParams, points: np.ndarray, error_metric: str = "first_order") -> np.ndarray:
    d = superquadric_residual_vector(model, points, metric="radial"if error_metric == "first_order" else error_metric)
    return np.abs(d)


def compute_consensus(model: SuperQuadricParams, points: np.ndarray, threshold: float, error_metric: str = "mix") -> np.ndarray[bool]:
    err = distance_err(model, points, error_metric=error_metric)
    inliers = err < threshold
    return inliers


def expanded_removal_mask(model: SuperQuadricParams, points: np.ndarray, threshold: float, factor: float = 1.5, error_metric: str = "mix") -> np.ndarray:
    err = distance_err(model, points, error_metric=error_metric)
    return err <= factor * threshold
