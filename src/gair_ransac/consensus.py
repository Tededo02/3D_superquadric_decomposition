import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import *

def distance_err(model: SuperQuadricParams, points: np.ndarray) -> np.ndarray:
    d = superquadric_radial_residual(model, points)
    return np.abs(d)


def compute_consensus(model: SuperQuadricParams, points: np.ndarray, threshold: float) -> np.ndarray[bool]:
    err = distance_err(model, points)
    inliers = err < threshold
    return inliers

def expanded_removal_mask(model: SuperQuadricParams, points: np.ndarray, threshold: float, factor: float = 1.5) -> np.ndarray:
    err = distance_err(model, points)
    return err <= factor * threshold
