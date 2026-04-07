from typing import Optional
import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from .consensus import compute_consensus, expanded_removal_mask
from .inner_ransac import inner_ransac, fit_superquadric_ls, InnerRansacResult
from .mss import adaptive_local_fps_mss
from .gair import gair
from .initgraph import build_radius_graph
from numpy.typing import NDArray

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
    error_metric: str = "mix",
    consensus_metric: str = "radial",
    inner_iterations: int = 50,
    random_seed: int | None = None,
    graphcut: bool = True,
    normals: np.ndarray | None = None,
    use_normal_guided_mss: bool | None = None,
) -> tuple[list[SuperQuadricParams], list[BoolArray]]:
    point_cloud: FloatArray = np.asarray(point_cloud, dtype=np.float64)
    provided_normals: FloatArray | None = None
    if normals is not None:
        provided_normals = np.asarray(normals, dtype=np.float64)
        if provided_normals.shape != point_cloud.shape:
            raise ValueError(
                f"normals must have shape {point_cloud.shape}, got {provided_normals.shape}"
            )
    # dummy normals: normal coherence is disabled, but gair still requires shape (N,3)
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

        # Build radius graph once for the current residual (only needed for graphcut)
        if graphcut:
            _, edge = build_radius_graph(current_point_cloud, m_neighbors=m_neighbors, radius=radius, radius_is_relative=radius_is_relative)
            edge: IntArray = np.asarray(edge, dtype=np.int64)

        for _ in range(max_iterations):
            sampler_normals: FloatArray | None = None
            if use_normal_guided_mss is not None:
                if use_normal_guided_mss and provided_normals is not None:
                    sampler_normals = provided_normals[remaining_indices]
                else:
                    sampler_normals = None

            if graphcut:
                sample_pts: FloatArray = np.asarray(
                    adaptive_local_fps_mss(
                        current_point_cloud,
                        normals=sampler_normals,
                        sample_size=sample_size,
                        seed_tries=12,
                        candidate_multiplier=20.0,
                        initial_k=512,
                        rng=rng,
                    ),
                    dtype=np.float64,
                )
            else:
                if use_normal_guided_mss is None:
                    idx = rng.choice(
                        current_point_cloud.shape[0],
                        size=min(sample_size, current_point_cloud.shape[0]),
                        replace=False,
                    )
                    sample_pts = current_point_cloud[idx]
                else:
                    sample_pts = np.asarray(
                        adaptive_local_fps_mss(
                            current_point_cloud,
                            normals=sampler_normals,
                            sample_size=sample_size,
                            seed_tries=12,
                            candidate_multiplier=20.0,
                            initial_k=512,
                            rng=rng,
                        ),
                        dtype=np.float64,
                    )

            try:
                H_j: SuperQuadricParams = fit_superquadric_ls(sample_pts, error_metric="first_order")
            except Exception:
                continue

            candidate_inliers: BoolArray = np.asarray(
                compute_consensus(H_j, current_point_cloud, threshold, error_metric=consensus_metric),
                dtype=bool,
            )
            candidate_count: int = int(np.count_nonzero(candidate_inliers))
            best_count: int = int(np.count_nonzero(best_inliers))

            if candidate_count < best_count + 1:
                continue

            current_model: SuperQuadricParams = H_j
            current_inliers: BoolArray = candidate_inliers.copy()
            current_count: int = candidate_count

            if graphcut:
                # Local optimization: GC (no normal coherence) + inner RANSAC
                terminate: bool = False
                while not terminate:
                    refined_inliers: BoolArray = np.asarray(
                        gair(
                            points=current_point_cloud,
                            edges=edge,
                            normals=current_normals,
                            model=current_model,
                            eps=threshold,
                            error_metric=consensus_metric,
                            use_normal_coherence=False,
                        ),
                        dtype=bool,
                    )
                    refined_count: int = int(np.count_nonzero(refined_inliers))

                    if refined_count < min_inliers:
                        terminate = True
                        continue

                    refined_set_index: IntArray = np.flatnonzero(refined_inliers).astype(np.int64)

                    inner_result: InnerRansacResult = inner_ransac(
                        current_point_cloud,
                        refined_set_index,
                        None,
                        threshold,
                        normals=None,
                        error_metric=error_metric,
                        consensus_metric=consensus_metric,
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
                # Simple inner RANSAC on candidate inliers, no graph
                refined_set_index: IntArray = np.flatnonzero(candidate_inliers).astype(np.int64)

                if refined_set_index.size >= min_inliers:
                    inner_result: InnerRansacResult = inner_ransac(
                        current_point_cloud,
                        refined_set_index,
                        None,
                        threshold,
                        normals=None,
                        error_metric=error_metric,
                        consensus_metric=consensus_metric,
                        n_iters=inner_iterations,
                        random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    )

                    if inner_result.best_inlier_count > 0:
                        new_inliers_mask: BoolArray = np.asarray(inner_result.best_inliers_mask, dtype=bool)
                        current_inliers = new_inliers_mask
                        current_count = int(np.count_nonzero(new_inliers_mask))
                        current_model = inner_result.best_model

            if current_count > int(np.count_nonzero(best_inliers)):
                best_model = current_model
                best_inliers = current_inliers

        if best_model is None:
            break

        best_count: int = int(np.count_nonzero(best_inliers))
        if best_count < min_inliers:
            break

        # Final refit on all inliers
        best_points = current_point_cloud[best_inliers]
        if best_points.shape[0] >= 11:
            try:
                refit_model = fit_superquadric_ls(
                    best_points,
                    error_metric=error_metric,
                    bounds_reference_points=best_points,
                )
                refit_inliers = np.asarray(
                    compute_consensus(refit_model, current_point_cloud, threshold, error_metric=consensus_metric),
                    dtype=bool,
                )
                refit_count = int(np.count_nonzero(refit_inliers))
                if refit_count >= best_count:
                    best_model = refit_model
                    best_inliers = refit_inliers
                    best_count = refit_count
            except Exception:
                pass

        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True

        models_set.append(best_model)
        inliers_set.append(global_inliers)

        remove_mask = expanded_removal_mask(best_model, current_point_cloud, threshold, factor=1.3, error_metric=consensus_metric)
        remaining_indices = remaining_indices[~remove_mask]

    return models_set, inliers_set
