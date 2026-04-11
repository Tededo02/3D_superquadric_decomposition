from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.superquadrics.superquadric_param import SuperQuadricParams

from .consensus import compute_consensus, expanded_removal_mask
from .gair import gair
from .initgraph import build_radius_graph
from .inner_ransac import InnerRansacResult, fit_superquadric_ls, inner_ransac, model_matches_support_scale
from .mss import adaptive_local_fps_mss

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def compare_consensus(prev_mask: np.ndarray, new_mask: np.ndarray, min_gain: int = 1) -> bool:
    return int(new_mask.sum()) >= int(prev_mask.sum()) + min_gain


def ransac(
    point_cloud: np.ndarray,
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
    graphcut: bool = True,
    normals: np.ndarray | None = None,
    use_normal_guided_mss: bool | None = None,
    mss_max_pool_fraction: float | None = 0.25,
) -> tuple[list[SuperQuadricParams], list[BoolArray]]:
    point_cloud: FloatArray = np.asarray(point_cloud, dtype=np.float64)
    provided_normals: FloatArray | None = None
    if normals is not None:
        provided_normals = np.asarray(normals, dtype=np.float64)
        if provided_normals.shape != point_cloud.shape:
            raise ValueError(
                f"normals must have shape {point_cloud.shape}, got {provided_normals.shape}"
            )

    dummy_normals: FloatArray = np.zeros((point_cloud.shape[0], 3), dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    n_points: int = point_cloud.shape[0]
    remaining_indices: IntArray = np.arange(n_points, dtype=np.int64)
    models_set: list[SuperQuadricParams] = []
    inliers_set: list[BoolArray] = []

    for _ in range(max_models):
        if remaining_indices.size < max(sample_size, min_inliers):
            break

        current_point_cloud: FloatArray = point_cloud[remaining_indices]
        current_normals: FloatArray = dummy_normals[remaining_indices]
        best_model: Optional[SuperQuadricParams] = None
        best_inliers: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)

        if graphcut:
            _, edge = build_radius_graph(
                current_point_cloud,
                m_neighbors=m_neighbors,
                radius=radius,
                radius_is_relative=radius_is_relative,
            )
            edge = np.asarray(edge, dtype=np.int64)

        for _ in range(max_iterations):
            sampler_normals: FloatArray | None = None
            if use_normal_guided_mss is not None:
                if use_normal_guided_mss and provided_normals is not None:
                    sampler_normals = provided_normals[remaining_indices]

            if graphcut or use_normal_guided_mss is not None:
                sample_points: FloatArray = np.asarray(
                    adaptive_local_fps_mss(
                        current_point_cloud,
                        normals=sampler_normals,
                        sample_size=sample_size,
                        seed_tries=12,
                        candidate_multiplier=20.0,
                        initial_k=512,
                        rng=rng,
                        max_pool_fraction=mss_max_pool_fraction,
                    ),
                    dtype=np.float64,
                )
            else:
                idx = rng.choice(
                    current_point_cloud.shape[0],
                    size=min(sample_size, current_point_cloud.shape[0]),
                    replace=False,
                )
                sample_points = current_point_cloud[idx]

            try:
                candidate_model = fit_superquadric_ls(sample_points)
            except Exception:
                continue

            candidate_inliers: BoolArray = np.asarray(
                compute_consensus(candidate_model, current_point_cloud, threshold),
                dtype=bool,
            )
            candidate_count = int(np.count_nonzero(candidate_inliers))
            best_count = int(np.count_nonzero(best_inliers))

            if candidate_count < best_count + 1:
                continue

            current_model = candidate_model
            current_inliers = candidate_inliers.copy()
            current_count = candidate_count

            if graphcut:
                terminate = False
                while not terminate:
                    refined_inliers: BoolArray = np.asarray(
                        gair(
                            points=current_point_cloud,
                            edges=edge,
                            normals=current_normals,
                            model=current_model,
                            eps=threshold,
                            use_normal_coherence=False,
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
                        normals=None,
                        n_iters=inner_iterations,
                        random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    )
                    if inner_result.best_inlier_count <= 0:
                        terminate = True
                        continue

                    new_inliers_mask: BoolArray = np.asarray(inner_result.best_inliers_mask, dtype=bool)
                    if compare_consensus(current_inliers, new_inliers_mask, min_gain=min_gain):
                        current_inliers = new_inliers_mask
                        current_count = int(np.count_nonzero(new_inliers_mask))
                        current_model = inner_result.best_model
                    else:
                        terminate = True
            else:
                refined_set_index = np.flatnonzero(candidate_inliers).astype(np.int64)
                if refined_set_index.size >= min_inliers:
                    inner_result = inner_ransac(
                        current_point_cloud,
                        refined_set_index,
                        None,
                        threshold,
                        normals=None,
                        n_iters=inner_iterations,
                        random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    )
                    if inner_result.best_inlier_count > 0:
                        current_inliers = np.asarray(inner_result.best_inliers_mask, dtype=bool)
                        current_count = int(np.count_nonzero(current_inliers))
                        current_model = inner_result.best_model

            if current_count > int(np.count_nonzero(best_inliers)):
                if current_count >= min_inliers:
                    current_points = current_point_cloud[current_inliers]
                    if not model_matches_support_scale(current_model, current_points, threshold):
                        continue
                best_model = current_model
                best_inliers = current_inliers

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
                    compute_consensus(refit_model, current_point_cloud, threshold),
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

        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True

        models_set.append(best_model)
        inliers_set.append(global_inliers)

        remove_mask = expanded_removal_mask(
            best_model,
            current_point_cloud,
            threshold,
            factor=1.3,
        )
        remaining_indices = remaining_indices[~remove_mask]

    return models_set, inliers_set
