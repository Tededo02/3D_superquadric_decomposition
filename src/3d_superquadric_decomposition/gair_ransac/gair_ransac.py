import numpy as np
from superquadric_param import SuperQuadricParams
from scipy.sparse import csr

# GAIR-RANSAC algorithm to fit multiple superquadrics to a point cloud.
"""
Args:
    points: (N, 3) array of input points
    n_superquadrics: number of superquadrics to fit
    max_iterations: maximum number of RANSAC iterations
    inlier_threshold: distance threshold to consider a point as an inlier
Returns:
    List of fitted SuperQuadricParams, one for each detected superquadric.
"""
def gair_ransac(point_cloud: np.ndarray, k_superquadrics: int, max_iterations: int = 1000, inlier_threshold: float = 0.01) -> list[SuperQuadricParams]:
    superquadric_list: list[SuperQuadricParams] = []
    inlier_sets: list[np.ndarray] = []  # List of boolean masks for inliers of each detected superquadric
    graph: csr 
    normals: list[np.ndarray] # Nx3 
    for j in range(max_iterations):
        best_superquadric: SuperQuadricParams = None
        sample_points =estimate_minimal_sample_set(point_cloud)
        
        


        if best_superquadric is not None:
            superquadric_list.append(best_superquadric)
            inlier_sets.append(best_inliers_mask)
            # Remove inliers of the detected superquadric from remaining points
            remaining_points = remaining_points[~best_inliers_mask]
        else:
            break  # No valid superquadric found, stop early
    return superquadric_list