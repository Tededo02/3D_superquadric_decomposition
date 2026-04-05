from pathlib import Path
from typing import Optional
import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from src.visualizations.visualization import save_point_cloud_inlier_view
from .consensus import compute_consensus, expanded_removal_mask
from .inner_ransac import inner_ransac, fit_superquadric_ls
from .gair import gair
from .initgraph import build_radius_graph
from .mss import spatial_walk_mss , adaptive_local_fps_mss, uniform_partition_mss
from .inner_ransac import InnerRansacResult
from numpy.typing import NDArray

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
    """Sample n points on the superquadric surface parametrically."""
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
    R = model.rotation_matrix()
    return (pts @ R.T) + model.t


def gair_ransac(point_cloud: np.ndarray, normals: np.ndarray | None = None, threshold: float = 0.1, max_models: int = 1, max_iterations: int = 300, m_neighbors: int = 12, radius: float = 0.06, radius_is_relative: bool = True, sample_size: int = 30, min_inliers: int = 30, min_gain: int = 1, error_metric: str = "radial", consensus_metric: str = "first_order", inner_iterations: int = 50, random_seed: int | None = None, min_coverage: float = 0.0, use_normal_coherence: bool | None = None) -> tuple[list[SuperQuadricParams], list[BoolArray], FloatArray | None]:
    total_best_mss_used: FloatArray | None = None

    point_cloud: FloatArray = np.asarray(point_cloud, dtype=np.float64)
    # if not explicitly set, derive from whether normals are provided
    if use_normal_coherence is None:
        use_normal_coherence = normals is not None
    # MSS always gets real normals when available (independent of GAIR coherence flag)
    mss_normals: FloatArray | None = np.asarray(normals, dtype=np.float64) if normals is not None else None
    normals: FloatArray = np.asarray(normals, dtype=np.float64) if normals is not None else np.zeros((point_cloud.shape[0], 3), dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    # Store total number of points in the original point cloud
    n_points: int = point_cloud.shape[0]
    # Initialize the set of indices still available for sequential extraction
    remaining_indices: IntArray = np.arange(n_points, dtype=np.int64)
    # These lists store the final models and their global inlier masks
    models_set: list[SuperQuadricParams] = []
    inliers_set: list[BoolArray] = []
    # extract up to max_models 
    for k in range(max_models):
        if remaining_indices.size < max(sample_size, min_inliers):
            break
        # Build the current residual point cloud
        current_point_cloud: FloatArray = point_cloud[remaining_indices]
        V: FloatArray = normals[remaining_indices]
        V_mss: FloatArray | None = mss_normals[remaining_indices] if mss_normals is not None else None
        # Initialize best-so-far model and inlier set for the current residual
        best_model: Optional[SuperQuadricParams] = None
        best_inliers: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)
        # Build the graph once for the current residual using a scale-aware radius.
        _, edge = build_radius_graph(current_point_cloud,m_neighbors=m_neighbors,radius=radius,radius_is_relative=radius_is_relative,)
        edge: IntArray = np.asarray(edge, dtype=np.int64)
        # Temporary debug mode: fit a single hypothesis using the whole residual cloud.
        # SETTARE A TRUE PER I TEST CON TUTTA POINT CLOUD---------------------------
        use_full_cloud_hypothesis = False
        n_hypotheses = 1 if use_full_cloud_hypothesis else max_iterations
        # Main RANSAC loop over m hypotheses-------------EXTERNAL RANSAC----------------
        best_mss_used: FloatArray | None = None
        for j in range(n_hypotheses):
            # Draw a non-minimal sample set and estimate a candidate model
            if use_full_cloud_hypothesis:
                M_j: FloatArray = current_point_cloud.copy()
            else:
                M_j = np.asarray(
                    adaptive_local_fps_mss(
                        current_point_cloud,
                        V_mss,
                        sample_size=sample_size,
                        seed_tries=12,
                        candidate_multiplier=20.0,
                        initial_k=512,
                        rng=rng,
                    ),
                    dtype=np.float64,
                )
            save_point_cloud_inlier_view(
                current_point_cloud,
                _point_subset_mask(current_point_cloud, M_j),
                IMAGES_DIR / f"gair_ransac_model_{k + 1:02d}_hyp_{j + 1:03d}_mss.png",
            )
            try:
                H_j: SuperQuadricParams = fit_superquadric_ls(M_j, error_metric="first_order")
            except Exception:
                continue
            # Compute the standard consensus set of the candidate model
            candidate_inliers: BoolArray = np.asarray(
                compute_consensus(
                    H_j,
                    current_point_cloud,
                    threshold,
                    error_metric=consensus_metric,
                    normals=V if use_normal_coherence else None,
                ),
                dtype=bool,
            )
            # Count current candidate inliers and current best inliers
            candidate_count: int = int(np.count_nonzero(candidate_inliers))
            best_count: int = int(np.count_nonzero(best_inliers))
            # Start local optimization only if the new hypothesis improves the best-so-far
            if candidate_count < best_count + 1:
                continue
            # Initialize local optimization state
            current_model: SuperQuadricParams = H_j
            current_inliers: BoolArray = candidate_inliers.copy()
            current_count: int = candidate_count
            terminate: bool = False
            local_iteration: int = 0
            # Local optimization loop: GAIR + inner RANSAC + consensus comparison
            while not terminate:
                # Refine the current inlier set with GAIR
                refined_inliers: BoolArray = np.asarray(
                    gair(
                        points=current_point_cloud,
                        edges=edge,
                        normals=V,
                        model=current_model,
                        eps=threshold,
                        error_metric=consensus_metric,
                        use_normal_coherence=use_normal_coherence,
                        use_model_normal_agreement=use_normal_coherence,
                    ),
                    dtype=bool,
                )
                local_iteration += 1
                save_point_cloud_inlier_view(
                    current_point_cloud,
                    refined_inliers,
                    IMAGES_DIR / f"gair_ransac_model_{k + 1:02d}_hyp_{j + 1:03d}_gair_{local_iteration:02d}.png",
                )
                refined_count: int = int(np.count_nonzero(refined_inliers))
                # Stop if the refined set is too small
                if refined_count < min_inliers:
                    terminate = True
                    continue
                # Sample from the refined set, but validate on the whole residual cloud
                # so the model can grow from a local patch to the full object support.
                refined_set_index: IntArray = np.flatnonzero(refined_inliers).astype(np.int64)
                # Run inner RANSAC: sample from I_hat and evaluate over the current residual.
                inner_result: InnerRansacResult = inner_ransac(
                    current_point_cloud,
                    refined_set_index,
                    None,
                    threshold,
                    normals=V if use_normal_coherence else None,
                    error_metric=error_metric,
                    consensus_metric=consensus_metric,
                    n_iters=inner_iterations,
                    random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
                # Stop if inner RANSAC fails
                if inner_result.best_inlier_count <= 0:
                    terminate = True
                    continue
                # The inlier mask returned by inner_ransac is defined on actual_set_index
                # Update the current solution only if consensus improves
                # if compare_consensus(current_inliers, refined_inliers, min_gain=min_gain):
                # It does not work well. The paper does not use c_hat, but with c_hat
                # the update is much more stable in this implementation.
                # from here
                new_inliers_mask_c_hat: BoolArray = np.asarray(inner_result.best_inliers_mask, dtype=bool)

                if compare_consensus(current_inliers, new_inliers_mask_c_hat, min_gain=min_gain):
                    current_inliers = new_inliers_mask_c_hat
                    current_count = int(np.count_nonzero(new_inliers_mask_c_hat))
                    current_model = inner_result.best_model
                else:
                    terminate = True


                # The paper does not use c_hat, but with c_hat
                # to here
            # Update the best-so-far solution for the current residual
            if current_count > int(np.count_nonzero(best_inliers)):
                best_model = current_model
                best_inliers = current_inliers
                best_mss_used = M_j.copy()
        
        # Stop if no valid model was found
        if best_model is None:
            break

        # Count inliers of the best model found on the current residual
        best_count: int = int(np.count_nonzero(best_inliers))

        # Stop if the model is not supported by enough points
        if best_count < min_inliers:
            break

        # Final refit on the best residual-wide inlier set before extracting the model.
        best_points = current_point_cloud[best_inliers]
        if best_points.shape[0] >= 11:
            try:
                refit_model = fit_superquadric_ls(
                    best_points,
                    error_metric=error_metric,
                    bounds_reference_points=best_points,
                )
                refit_inliers = np.asarray(
                    compute_consensus(
                        refit_model,
                        current_point_cloud,
                        threshold,
                        error_metric=consensus_metric,
                        normals=V if use_normal_coherence else None,
                    ),
                    dtype=bool,
                )
                refit_count = int(np.count_nonzero(refit_inliers))
                if refit_count >= best_count:
                    best_model = refit_model
                    best_inliers = refit_inliers
                    best_count = refit_count
            except Exception:
                pass

        # Coverage check: reject oversized models whose surface is sparsely supported
        if min_coverage > 0.0:
            from scipy.spatial import cKDTree as _cKDTree
            surface_samples = _sample_superquadric_surface(best_model, n=1000)
            inlier_pts = current_point_cloud[best_inliers]
            tree_inliers = _cKDTree(inlier_pts)
            dists, _ = tree_inliers.query(surface_samples, k=1)
            coverage = float((dists < threshold).mean())
            if coverage < min_coverage:
                remaining_indices = remaining_indices[~best_inliers]
                continue

        # Convert the local inlier mask into a global mask over the original point cloud
        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True

        # Save the extracted model and its global inlier mask
        models_set.append(best_model)
        inliers_set.append(global_inliers)
        if best_mss_used is not None:
            total_best_mss_used = (
                best_mss_used
                if total_best_mss_used is None
                else np.vstack((total_best_mss_used, best_mss_used))
            )
        
        # Remove the inliers of the extracted model from the residual set and all the points too close to it
        remove_mask = expanded_removal_mask(
            best_model,
            current_point_cloud,
            threshold,
            factor=1.3,
            error_metric=consensus_metric,
            normals=V if use_normal_coherence else None,
        )
        remaining_indices = remaining_indices[~remove_mask]
        
        # Return all extracted models and their global inlier masks
    return models_set, inliers_set, total_best_mss_used
