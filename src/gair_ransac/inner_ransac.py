from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from scipy.optimize import least_squares
from .consensus import compute_consensus
from src.superquadrics.superquadric_residual import superquadric_radial_residual_and_jacobian,superquadric_residual_vector

AXIS_UPPER_FACTOR = 1.6
AXIS_UPPER_FLOOR_FACTOR = 0.18
TRANSLATION_MARGIN_FACTOR = 0.15
MIN_TRANSLATION_MARGIN_FACTOR = 0.05
MODEL_DIAGONAL_SUPPORT_FACTOR = 1.6
MODEL_SUPPORT_SLACK_FACTOR = 4.0


@dataclass
class InnerRansacResult:
    best_model: SuperQuadricParams
    best_inlier_count: int
    best_inliers_mask: np.ndarray


def pca_initialization(points: np.ndarray) -> SuperQuadricParams:
    point_array = np.asarray(points, dtype=np.float64)

    # Estimate translation and principal directions from the point sample.
    center = point_array.mean(axis=0)
    centered_points = point_array - center
    covariance = centered_points.T @ centered_points / max(point_array.shape[0] - 1, 1)
    _, principal_axes = np.linalg.eigh(covariance)
    rotation_matrix = principal_axes[:, ::-1]
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1.0

    # Estimate the semi-axes from robust quantiles in the PCA frame.
    pca_coordinates = centered_points @ rotation_matrix
    lower_quantile = np.percentile(pca_coordinates, 5.0, axis=0)
    upper_quantile = np.percentile(pca_coordinates, 95.0, axis=0)
    semi_axes = np.maximum(0.525 * (upper_quantile - lower_quantile), 1e-3)

    # Convert the PCA rotation matrix into the yaw-pitch-roll convention used here.
    sin_pitch = np.clip(-rotation_matrix[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < 1e-8:
        yaw = 0.0
        roll = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
    else:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return SuperQuadricParams(
        a1=float(semi_axes[0]),
        a2=float(semi_axes[1]),
        a3=float(semi_axes[2]),
        e1=1.0,
        e2=1.0,
        rot=np.array([yaw, pitch, roll], dtype=np.float64),
        t=center.astype(np.float64, copy=False),
    )


def _reference_axis_upper_bounds(
    reference_points: np.ndarray,
    reference_diagonal: float,
    min_axis_length: float,
    global_max_axis_length: float,
) -> np.ndarray:
    reference_model = pca_initialization(reference_points)
    reference_axes = np.array(
        [reference_model.a1, reference_model.a2, reference_model.a3],
        dtype=np.float64,
    )
    axis_floor = max(min_axis_length, AXIS_UPPER_FLOOR_FACTOR * reference_diagonal)
    upper_bounds = np.maximum(AXIS_UPPER_FACTOR * reference_axes, axis_floor)
    return np.clip(upper_bounds, min_axis_length, global_max_axis_length)


def model_matches_support_scale(
    model: SuperQuadricParams,
    support_points: np.ndarray,
    threshold: float = 0.0,
) -> bool:
    support_array = np.asarray(support_points, dtype=np.float64)
    if support_array.shape[0] == 0:
        return False

    axes = np.array([model.a1, model.a2, model.a3], dtype=np.float64)
    support_spans = np.ptp(support_array, axis=0)
    support_diagonal = float(np.linalg.norm(support_spans))
    slack = max(
        MODEL_SUPPORT_SLACK_FACTOR * float(threshold),
        0.02 * support_diagonal,
        1e-3,
    )

    model_diagonal = float(np.linalg.norm(2.0 * axes))
    return bool(model_diagonal <= MODEL_DIAGONAL_SUPPORT_FACTOR * support_diagonal + slack)


def _normalize_index_input(index_input: np.ndarray, n_points: int, name: str) -> np.ndarray:
    index_array = np.asarray(index_input)
    if index_array.dtype == bool:
        if index_array.shape != (n_points,):
            raise ValueError(f"{name} boolean mask must have shape ({n_points},), got {index_array.shape}")
        return np.flatnonzero(index_array).astype(np.int64)
    return np.asarray(index_input, dtype=np.int64)


# Fit a superquadric with bounded non-linear least squares.
def fit_superquadric_ls(
    points: np.ndarray,
    error_metric: str = "radial",
    bounds_reference_points: np.ndarray | None = None,
) -> SuperQuadricParams:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape[0] < 11:
        raise ValueError("too few points for a stable superqadric fit")

    # Build a robust initial guess from PCA.
    initial_model = pca_initialization(point_array)

    # Use the requested reference points to define stable optimization bounds.
    reference_points = point_array if bounds_reference_points is None else np.asarray(bounds_reference_points, dtype=np.float64)
    if reference_points.shape[0] == 0:
        raise ValueError("bounds_reference_points must contain at least one point")

    reference_min = reference_points.min(axis=0)
    reference_max = reference_points.max(axis=0)
    reference_span = reference_max - reference_min
    reference_diagonal = float(np.linalg.norm(reference_max - reference_min) + 1e-12)
    min_axis_length = max(1e-3, 5e-2 * reference_diagonal)
    global_max_axis_length = max(1e-2, 1.2 * reference_diagonal)
    max_axis_lengths = _reference_axis_upper_bounds(
        reference_points,
        reference_diagonal,
        min_axis_length,
        global_max_axis_length,
    )
    exponent_min, exponent_max = 0.08, 4.0
    angle_min, angle_max = -np.pi, np.pi
    min_translation_margin = MIN_TRANSLATION_MARGIN_FACTOR * reference_diagonal
    translation_margin = np.maximum(
        TRANSLATION_MARGIN_FACTOR * reference_span,
        min_translation_margin,
    )

    lower_bounds = np.array(
        [
            min_axis_length,
            min_axis_length,
            min_axis_length,
            exponent_min,
            exponent_min,
            angle_min,
            angle_min,
            angle_min,
            reference_min[0] - translation_margin[0],
            reference_min[1] - translation_margin[1],
            reference_min[2] - translation_margin[2],
        ],
        dtype=np.float64,
    )
    upper_bounds = np.array(
        [
            max_axis_lengths[0],
            max_axis_lengths[1],
            max_axis_lengths[2],
            exponent_max,
            exponent_max,
            angle_max,
            angle_max,
            angle_max,
            reference_max[0] + translation_margin[0],
            reference_max[1] + translation_margin[1],
            reference_max[2] + translation_margin[2],
        ],
        dtype=np.float64,
    )

    initial_parameters = np.array(
        [
            initial_model.a1,
            initial_model.a2,
            initial_model.a3,
            initial_model.e1,
            initial_model.e2,
            initial_model.rot[0],
            initial_model.rot[1],
            initial_model.rot[2],
            initial_model.t[0],
            initial_model.t[1],
            initial_model.t[2],
        ],
        dtype=np.float64,
    )
    np.clip(initial_parameters, lower_bounds, upper_bounds, out=initial_parameters)

    normalized_metric = error_metric.lower().replace("-", "_").replace(" ", "_")

    # SciPy uses the keyword fun for the residual function to minimize.
    if normalized_metric == "radial":
        radial_cache: dict[str, np.ndarray | None] = {"parameters": None, "residuals": None, "jacobian": None}

        def radial_residuals(parameters: np.ndarray) -> np.ndarray:
            cached_parameters = radial_cache["parameters"]
            if cached_parameters is None or not np.array_equal(cached_parameters, parameters):
                current_model = SuperQuadricParams(
                    a1=parameters[0],
                    a2=parameters[1],
                    a3=parameters[2],
                    e1=parameters[3],
                    e2=parameters[4],
                    rot=np.array(parameters[5:8], dtype=np.float64),
                    t=np.array(parameters[8:11], dtype=np.float64),
                )
                residuals, jacobian = superquadric_radial_residual_and_jacobian(current_model, point_array)
                radial_cache["parameters"] = np.array(parameters, dtype=np.float64, copy=True)
                radial_cache["residuals"] = residuals
                radial_cache["jacobian"] = jacobian
            return radial_cache["residuals"]

        def radial_jacobian(parameters: np.ndarray) -> np.ndarray:
            radial_residuals(parameters)
            return radial_cache["jacobian"]

        optimization_result = least_squares(
            fun=radial_residuals,
            jac=radial_jacobian,
            x0=initial_parameters,
            method="trf",
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=250,
        )
    else:
        def residuals(parameters: np.ndarray) -> np.ndarray:
            current_model = SuperQuadricParams(
                a1=parameters[0],
                a2=parameters[1],
                a3=parameters[2],
                e1=parameters[3],
                e2=parameters[4],
                rot=np.array(parameters[5:8], dtype=np.float64),
                t=np.array(parameters[8:11], dtype=np.float64),
            )
            return superquadric_residual_vector(current_model, point_array, metric=error_metric)

        optimization_result = least_squares(
            fun=residuals,
            x0=initial_parameters,
            method="trf",
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=250,
        )

    if not optimization_result.success:
        raise RuntimeError(f"least_squares failed: {optimization_result.message}")

    a1, a2, a3, e1, e2, yaw, pitch, roll, px, py, pz = optimization_result.x
    return SuperQuadricParams(
        a1=a1,
        a2=a2,
        a3=a3,
        e1=e1,
        e2=e2,
        rot=np.array([yaw, pitch, roll], dtype=np.float64),  # (yaw=z, pitch=y, roll=x)
        t=np.array([px, py, pz], dtype=np.float64),
    )


def inner_ransac(
    point_cloud: np.ndarray,
    refined_set_index: np.ndarray,
    actual_set_index: np.ndarray | None,
    threshold: float,
    normals: np.ndarray | None = None,
    error_metric: str = "mix",
    consensus_metric: str | None = None,
    n_iters: int = 50,
    random_seed: int | None = None,
    sample_size: int = 30,
) -> InnerRansacResult:
    point_array = np.asarray(point_cloud, dtype=np.float64)
    candidate_indices = _normalize_index_input(refined_set_index, point_array.shape[0], "refined_set_index")
    evaluation_indices = (
        None
        if actual_set_index is None
        else _normalize_index_input(actual_set_index, point_array.shape[0], "actual_set_index")
    )
    evaluation_points = point_array if evaluation_indices is None else point_array[evaluation_indices]
    if normals is None:
        evaluation_normals = None
    else:
        normal_array = np.asarray(normals, dtype=np.float64)
        evaluation_normals = normal_array if evaluation_indices is None else normal_array[evaluation_indices]
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")

    # Use the current GAIR set as the reference support for bounded fitting.
    fit_reference_points = point_array[candidate_indices]
    sampled_point_count = min(candidate_indices.size, int(sample_size))
    effective_consensus_metric = error_metric if consensus_metric is None else consensus_metric

    best_model: Optional[SuperQuadricParams] = None
    best_inlier_mask = np.empty((0,), dtype=bool)
    best_inlier_count = -1
    rng = np.random.default_rng(random_seed)

    # Repeatedly sample from the refined set and keep the model with the largest consensus.
    for _ in range(n_iters):
        sampled_indices = rng.choice(candidate_indices, size=sampled_point_count, replace=False, shuffle=False)
        sampled_points = point_array[sampled_indices]
        try:
            candidate_model = fit_superquadric_ls(
                sampled_points,
                error_metric=error_metric,
                bounds_reference_points=fit_reference_points,
            )
        except Exception:
            continue

        candidate_inlier_mask = compute_consensus(
            candidate_model,
            evaluation_points,
            threshold,
            error_metric=effective_consensus_metric,
            normals=evaluation_normals,
        )
        candidate_inlier_count = int(np.count_nonzero(candidate_inlier_mask))
        if candidate_inlier_count > best_inlier_count:
            best_model = candidate_model
            best_inlier_count = candidate_inlier_count
            best_inlier_mask = candidate_inlier_mask.astype(bool, copy=False)

    # Return an empty result if no valid fit was found.
    if best_inlier_count < 0 or best_model is None:
        return InnerRansacResult(
            best_model=SuperQuadricParams(1, 1, 1, 1, 1, [0, 0, 0], [0, 0, 0]),
            best_inlier_count=0,
            best_inliers_mask=np.empty((0,), dtype=bool),
        )

    # Refit once on the full best consensus set for better final accuracy.
    best_inlier_points = evaluation_points[best_inlier_mask]
    if best_inlier_points.shape[0] >= 11:
        try:
            refined_model = fit_superquadric_ls(
                best_inlier_points,
                error_metric=error_metric,
                bounds_reference_points=best_inlier_points,
            )
            refined_inlier_mask = compute_consensus(
                refined_model,
                evaluation_points,
                threshold,
                error_metric=effective_consensus_metric,
                normals=evaluation_normals,
            )
            refined_inlier_count = int(np.count_nonzero(refined_inlier_mask))
            if refined_inlier_count >= best_inlier_count:
                best_model = refined_model
                best_inlier_count = refined_inlier_count
                best_inlier_mask = refined_inlier_mask.astype(bool, copy=False)
        except Exception:
            pass

    return InnerRansacResult(
        best_model=best_model,
        best_inlier_count=best_inlier_count,
        best_inliers_mask=best_inlier_mask,
    )
