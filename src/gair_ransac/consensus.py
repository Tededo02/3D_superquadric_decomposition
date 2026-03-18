import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_normal_world,superquadric_residual_vector



DEFAULT_NORMAL_COS_THRESHOLD = 0.0


def distance_err(model: SuperQuadricParams, points: np.ndarray, error_metric: str = "first_order") -> np.ndarray:
    d = superquadric_residual_vector(model, points, metric="radial"if error_metric == "first_order" else error_metric)
    return np.abs(d)

# computes the cosine of the angle between the normal of the superquadric at each point and the normal of the point cloud,
#  as a measure of alignment (1 is perfectly aligned, -1 is opposite, 0 is orthogonal)
def normal_alignment_score(
    model: SuperQuadricParams,
    points: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    model_normals = superquadric_normal_world(model, points)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_norm = np.maximum(normal_norm, 1e-9)
    normals_unit = normals / normal_norm
    return np.clip(np.einsum("ij,ij->i", model_normals, normals_unit), -1.0, 1.0)


def compute_consensus(
    model: SuperQuadricParams,
    points: np.ndarray,
    threshold: float,
    error_metric: str = "mix",
    normals: np.ndarray | None = None,
    normal_cos_threshold: float | None = None,
) -> np.ndarray[bool]:
    err = distance_err(model, points, error_metric=error_metric)
    inliers = err < threshold
    if normals is not None:
        cos_threshold = DEFAULT_NORMAL_COS_THRESHOLD if normal_cos_threshold is None else float(normal_cos_threshold)
        inliers &= normal_alignment_score(model, points, normals) >= cos_threshold
    return inliers


def expanded_removal_mask(
    model: SuperQuadricParams,
    points: np.ndarray,
    threshold: float,
    factor: float = 1.5,
    error_metric: str = "mix",
    normals: np.ndarray | None = None,
    normal_cos_threshold: float | None = None,
) -> np.ndarray:
    err = distance_err(model, points, error_metric=error_metric)
    remove_mask = err <= factor * threshold
    if normals is not None:
        cos_threshold = DEFAULT_NORMAL_COS_THRESHOLD if normal_cos_threshold is None else float(normal_cos_threshold)
        remove_mask &= normal_alignment_score(model, points, normals) >= cos_threshold
    return remove_mask
