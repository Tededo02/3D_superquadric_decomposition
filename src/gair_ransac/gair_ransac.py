from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.superquadrics.superquadric_param import SuperQuadricParams

from .consensus import compute_consensus, expanded_removal_mask
from .gair import gair
from .initgraph import build_radius_graph
from .inner_ransac import InnerRansacResult, fit_superquadric_ls, inner_ransac, model_matches_support_scale
from .mss import (
    adaptive_local_fps_mss,
    build_adaptive_local_fps_sampler_context,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
IMAGES_DIR = Path(__file__).resolve().parents[2] / "images"


def compare_consensus(prev_mask: np.ndarray, new_mask: np.ndarray, min_gain: int = 1) -> bool:
    return int(new_mask.sum()) >= int(prev_mask.sum()) + min_gain


def _point_subset_mask(points: FloatArray, subset_points: np.ndarray, atol: float = 1e-12) -> BoolArray:
    mask = np.zeros(points.shape[0], dtype=bool)
    if subset_points.size == 0:
        return mask

    subset_points = np.asarray(subset_points)
    if subset_points.dtype == bool:
        if subset_points.shape[0] != points.shape[0]:
            raise ValueError(f"Boolean mask must have length {points.shape[0]}, got {subset_points.shape[0]}")
        return subset_points.astype(bool, copy=False)

    subset_points = np.asarray(subset_points, dtype=np.float64).reshape(-1, 3)
    for sample_point in subset_points:
        mask |= np.all(np.isclose(points, sample_point, atol=atol, rtol=0.0), axis=1)
    return mask


def _sample_superquadric_surface(model: "SuperQuadricParams", n: int = 1000) -> FloatArray:
    eta = np.linspace(-np.pi / 2, np.pi / 2, int(n ** 0.5) + 1)
    omega = np.linspace(-np.pi, np.pi, int(n ** 0.5) + 1)
    eta, omega = np.meshgrid(eta, omega)
    eta, omega = eta.ravel(), omega.ravel()

    def _sp(val, exp):
        return np.sign(val) * (np.abs(val) ** exp)

    x = model.a1 * _sp(np.cos(eta), model.e1) * _sp(np.cos(omega), model.e2)
    y = model.a2 * _sp(np.cos(eta), model.e1) * _sp(np.sin(omega), model.e2)
    z = model.a3 * _sp(np.sin(eta), model.e1)
    pts = np.stack([x, y, z], axis=1)
    rotation = model.rotation_matrix()
    return (pts @ rotation.T) + model.t


def gair_ransac(
    point_cloud: np.ndarray,
    normals: np.ndarray,
    threshold: float,
    max_models: int = 1,
    max_iterations: int = 300,
    m_neighbors: int = 12,
    radius: float = 0.06,
    radius_is_relative: bool = True,
    sample_size: int = 30,
    min_inliers: int = 30,
    min_gain: int = 1,
    inner_iterations: int = 50,
    random_seed: int | None = None,
    use_normal_coherence: bool = True,
    min_coverage: float = 0.0,
    use_normal_guided_mss: bool = True,
    mss_max_pool_fraction: float | None = 0.25,
) -> tuple[list[SuperQuadricParams], list[BoolArray], FloatArray | None]:
    total_best_mss_used: FloatArray | None = None

    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    n_points = point_cloud.shape[0]
    remaining_indices: IntArray = np.arange(n_points, dtype=np.int64)
    models_set: list[SuperQuadricParams] = []
    inliers_set: list[BoolArray] = []

    for _ in range(max_models):
        if remaining_indices.size < max(sample_size, min_inliers):
            break

        current_point_cloud: FloatArray = point_cloud[remaining_indices]
        current_normals: FloatArray = normals[remaining_indices]
        best_model: Optional[SuperQuadricParams] = None
        best_inliers: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)

        _, edge = build_radius_graph(
            current_point_cloud,
            m_neighbors=m_neighbors,
            radius=radius,
            radius_is_relative=radius_is_relative,
        )
        edge = np.asarray(edge, dtype=np.int64)

        mss_normals = current_normals if use_normal_guided_mss else None
        sampler_context = build_adaptive_local_fps_sampler_context(current_point_cloud, mss_normals)
        best_mss_used: FloatArray | None = None

        for _ in range(max_iterations):
            sample_points = np.asarray(
                adaptive_local_fps_mss(
                    current_point_cloud,
                    mss_normals,
                    sample_size=sample_size,
                    seed_tries=12,
                    candidate_multiplier=20.0,
                    initial_k=m_neighbors,
                    rng=rng,
                    sampler_context=sampler_context,
                    max_pool_fraction=mss_max_pool_fraction,
                ),
                dtype=np.float64,
            )

            try:
                candidate_model = fit_superquadric_ls(sample_points)
            except Exception:
                continue

            candidate_inliers = np.asarray(
                compute_consensus(
                    candidate_model,
                    current_point_cloud,
                    threshold,
                    normals=current_normals,
                ),
                dtype=bool,
            )
            candidate_count = int(np.count_nonzero(candidate_inliers))
            best_count = int(np.count_nonzero(best_inliers))

            if candidate_count < best_count + 1:
                continue

            current_model = candidate_model
            current_inliers = candidate_inliers.copy()
            current_count = candidate_count
            terminate = False

            while not terminate:
                refined_inliers = np.asarray(
                    gair(
                        points=current_point_cloud,
                        edges=edge,
                        normals=current_normals,
                        model=current_model,
                        eps=threshold,
                        use_normal_coherence=use_normal_coherence,
                        use_model_normal_agreement=True,
                    ),
                    dtype=bool,
                )

                refined_count = int(np.count_nonzero(refined_inliers))
                if refined_count < min_inliers:
                    terminate = True
                    continue

                refined_set_index = np.flatnonzero(refined_inliers).astype(np.int64)
                inner_result: InnerRansacResult = inner_ransac(
                    current_point_cloud,
                    refined_set_index,
                    None,
                    threshold,
                    normals=current_normals,
                    n_iters=inner_iterations,
                    random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
                if inner_result.best_inlier_count <= 0:
                    terminate = True
                    continue

                new_inliers = np.asarray(inner_result.best_inliers_mask, dtype=bool)
                if compare_consensus(current_inliers, new_inliers, min_gain=min_gain):
                    current_inliers = new_inliers
                    current_count = int(np.count_nonzero(new_inliers))
                    current_model = inner_result.best_model
                else:
                    terminate = True

            if current_count > int(np.count_nonzero(best_inliers)):
                if current_count >= min_inliers:
                    current_points = current_point_cloud[current_inliers]
                    if not model_matches_support_scale(current_model, current_points, threshold):
                        continue
                best_model = current_model
                best_inliers = current_inliers
                best_mss_used = sample_points.copy()

        if best_model is None:
            break

        best_count = int(np.count_nonzero(best_inliers))
        if best_count < min_inliers:
            break

        best_points = current_point_cloud[best_inliers]
        if best_points.shape[0] >= 11:
            try:
                refit_model = fit_superquadric_ls(
                    best_points,
                    bounds_reference_points=best_points,
                )
                refit_inliers = np.asarray(
                    compute_consensus(
                        refit_model,
                        current_point_cloud,
                        threshold,
                        normals=current_normals,
                    ),
                    dtype=bool,
                )
                refit_count = int(np.count_nonzero(refit_inliers))
                if refit_count >= best_count and model_matches_support_scale(
                    refit_model,
                    current_point_cloud[refit_inliers],
                    threshold,
                ):
                    best_model = refit_model
                    best_inliers = refit_inliers
                    best_count = refit_count
            except Exception:
                pass

        if min_coverage > 0.0:
            from scipy.spatial import cKDTree as _cKDTree

            surface_samples = _sample_superquadric_surface(best_model, n=1000)
            inlier_points = current_point_cloud[best_inliers]
            tree_inliers = _cKDTree(inlier_points)
            dists, _ = tree_inliers.query(surface_samples, k=1)
            coverage = float((dists < threshold).mean())
            if coverage < min_coverage:
                remaining_indices = remaining_indices[~best_inliers]
                continue

        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True

        models_set.append(best_model)
        inliers_set.append(global_inliers)
        if best_mss_used is not None:
            total_best_mss_used = (
                best_mss_used
                if total_best_mss_used is None
                else np.vstack((total_best_mss_used, best_mss_used))
            )

        remove_mask = expanded_removal_mask(
            best_model,
            current_point_cloud,
            threshold,
            factor=1.0,
            normals=current_normals,
        )
        remaining_indices = remaining_indices[~remove_mask]

    return models_set, inliers_set, total_best_mss_used
