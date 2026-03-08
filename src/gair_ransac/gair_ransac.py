from typing import Optional
import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from .consensus import compute_consensus, expanded_removal_mask
from .inner_ransac import inner_ransac, fit_superquadric_ls
from .gair import gair
from .initgraph import build_radius_graph
from .mss import spatial_walk_mss
from .inner_ransac import InnerRansacResult
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def compare_consensus(prev_mask: np.ndarray, new_mask: np.ndarray, min_gain: int = 1) -> bool:
    return int(new_mask.sum()) >= int(prev_mask.sum()) + min_gain


def gair_ransac(point_cloud: np.ndarray, normals: np.ndarray, threshold: float, max_models: int = 2, max_iterations: int = 300, m_neighbors: int = 12, radius: float = 0.5, sample_size: int = 30, min_inliers: int = 30, min_gain: int = 1) -> tuple[list[SuperQuadricParams], list[BoolArray]]:
    # Convert inputs to standard float arrays
    point_cloud: FloatArray = np.asarray(point_cloud, dtype=np.float64)
    normals: FloatArray = np.asarray(normals, dtype=np.float64)
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
        # Initialize best-so-far model and inlier set for the current residual
        best_model: Optional[SuperQuadricParams] = None
        best_inliers: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)
        # Build the graph once for the current residual
        G, edge = build_radius_graph(current_point_cloud, m_neighbors=m_neighbors, radius=radius)
        edge: IntArray = np.asarray(edge, dtype=np.int64)
        # Main RANSAC loop over m hypotheses-------------EXTERNAL RANSAC----------------
        for j in range(max_iterations):
            # Draw a non-minimal sample set and estimate a candidate model
            try:
                M_j: FloatArray = np.asarray(spatial_walk_mss(current_point_cloud, V, sample_size=sample_size), dtype=np.float64)
                H_j: SuperQuadricParams = fit_superquadric_ls(M_j)
            except Exception:
                continue
            # Compute the standard consensus set of the candidate model
            candidate_inliers: BoolArray = np.asarray(compute_consensus(H_j, current_point_cloud, threshold), dtype=bool)
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
            # Local optimization loop: GAIR + inner RANSAC + consensus comparison
            while not terminate:
                # Refine the current inlier set with GAIR
                refined_inliers: BoolArray = np.asarray(gair(points=current_point_cloud, edges=edge, normals=V, model=current_model, eps=threshold), dtype=bool)
                refined_count: int = int(np.count_nonzero(refined_inliers))
                # Stop if the refined set is too small
                if refined_count < min_inliers:
                    terminate = True
                    continue
                # Build I_hat and I index sets
                actual_set_index: IntArray = np.flatnonzero(current_inliers).astype(np.int64)
                refined_set_index: IntArray = np.flatnonzero(refined_inliers).astype(np.int64)
                # if validate on the whole residual cloud, not only on current_inliers
                # Run inner RANSAC: sample from I_hat and evaluate over I
                # actual_set_index is all_current_index because inner_ransac will compute consensus on the whole residual cloud, not only on the refined set. 
                # with some test, this way is better because it allows inner_ransac to find a better model that fits more points in the residual
                # anyway, stick to the paper so use actual_set_index=refined_set_index
                inner_result: InnerRansacResult = inner_ransac(current_point_cloud, refined_set_index, actual_set_index, threshold)
                # Stop if inner RANSAC fails
                if inner_result.best_inlier_count <= 0:
                    terminate = True
                    continue
                # The inlier mask returned by inner_ransac is defined on actual_set_index
                # Update the current solution only if consensus improves
                """if compare_consensus(current_inliers, refined_inliers, min_gain=min_gain):"""
                # it doesn't work well. the paper for some reason doesn't use c_hat... but with c_hat it works much better.
                # so let's use it
                new_inliers_mask: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)
                new_inliers_mask[actual_set_index[inner_result.best_inliers_mask]] = True
                new_count: int = int(np.count_nonzero(new_inliers_mask))
                if new_count >= current_count + min_gain:
                    current_inliers = new_inliers_mask
                    current_count = new_count
                    current_model = inner_result.best_model
                else:
                    terminate = True
            # Update the best-so-far solution for the current residual
            if current_count > int(np.count_nonzero(best_inliers)):
                best_model = current_model
                best_inliers = current_inliers

        # Stop if no valid model was found
        if best_model is None:
            break

        # Count inliers of the best model found on the current residual
        best_count: int = int(np.count_nonzero(best_inliers))

        # Stop if the model is not supported by enough points
        if best_count < min_inliers:
            break

        # Convert the local inlier mask into a global mask over the original point cloud
        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True

        # Save the extracted model and its global inlier mask
        models_set.append(best_model)
        inliers_set.append(global_inliers)

        # Remove the inliers of the extracted model from the residual set and all the points too close to it
        remove_mask = expanded_removal_mask(best_model, current_point_cloud, threshold, factor=1.3)
        remaining_indices = remaining_indices[~remove_mask]
        # Return all extracted models and their global inlier masks
    return models_set, inliers_set
