import sys
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.visualizations import visualization as vis
import numpy as np
from src.superquadrics import superquadric_sampling as samp
from src.gair_ransac.inner_ransac import InnerRansacResult, inner_ransac, fit_superquadric_ls
from src.gair_ransac.ransac import ransac
from point_cloud_utils import k_nearest_neighbors, chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.gair_ransac import gair_ransac

NOISE_STD = 0.2
THRESHOLD = 3*NOISE_STD

def create_and_estimate_supq():
    # lightblue = ground truth, lightgreen = estimated
    # add/remove rows here to change the scene
    gt_params = [
    SuperQuadricParams(9.0, 9.0, 9.0, 3.5, 2.09, [2.0, 2.0, 1.0], [5.0, 5.0, 5.0]),
    SuperQuadricParams(3.0, 3.0, 3.0, 0.5, 0.9, [2.0, 2.0, 1.0], [-5.0, -5.0, -5.0]),
    SuperQuadricParams(4.0, 4.0, 4.0, 0.8, 1.1, [1.7, 2.1, 0.9], [13.0, 5.0, 5.0]),
    SuperQuadricParams(2.5, 2.5, 2.5, 0.7, 1.2, [2.1, 1.9, 0.8], [-10.5, -5.0, -5.0])
    ]

    list_mesh = [supmesh.superquadric_mesh(p) for p in gt_params]
    colors = ["lightblue"] * len(list_mesh)

    #----those functions return a set of sample and their normals-----------
    #sampled_points,normals = samp.sampling_sq(list_mesh, n_points=1000)
    sampled_points_noisy, normals_sp_noisy = samp.sampling_sq_noisy(list_mesh, n_points=30000, noise_std=NOISE_STD, clip_k=3.0, seed=42)
    sampled_points_random, _ = samp.sampling_sq_random(list_mesh, n_points=4000, seed=42)
    sampled_points_random = np.vstack(sampled_points_random)
    sampled_points_outliers, normals_sp_outliers = samp.sampling_outliers(list_mesh, n_out=1, margin=0.10, mode="uniform", seed=42)
    
    
    
    algorithm="gair-ransac"
    total_best_mss_used = None
    #------choose here which kind of points to use for fitting the superquadric------
    sampled_points = np.vstack([*sampled_points_noisy, sampled_points_outliers])
    normals = np.vstack([*normals_sp_noisy, normals_sp_outliers])
    del sampled_points_noisy, normals_sp_noisy, sampled_points_outliers, normals_sp_outliers
    n_gt = len(list_mesh)
    mesh_estimated = None

    if algorithm == "ls":
        small_sample = sampled_points[:30] # just for testing, should be sampled from the gair set
        theta0 = fit_superquadric_ls(small_sample)
        mesh_estimated = supmesh.superquadric_mesh(theta0)
        list_mesh.append(mesh_estimated)
        colors.append("lightgreen")
    elif algorithm == "inner-ransac":
        theta0:InnerRansacResult = inner_ransac(sampled_points, refined_set_index=np.arange(sampled_points.shape[0]), actual_set_index=None, threshold=THRESHOLD)
        mesh_estimated = supmesh.superquadric_mesh(theta0.best_model)
        list_mesh.append(mesh_estimated)
        colors.append("lightgreen")
    elif algorithm == "ransac":
        graph_radius = 0.08
        models, inliers_masks = ransac(sampled_points, threshold=THRESHOLD, max_models=len(list_mesh), max_iterations=10, inner_iterations=40, radius=graph_radius, graphcut=True)
        if not models:
            raise RuntimeError("ransac did not return any model")
        for model in models:
            mesh_estimated = supmesh.superquadric_mesh(model)
            list_mesh.append(mesh_estimated)
            colors.append("lightgreen")
    elif algorithm == "gair-ransac":
        # Radius is expressed as a fraction of the point cloud bounding-box diagonal.
        graph_radius = 0.08
        models, inliers_masks, total_best_mss_used = gair_ransac(sampled_points, normals, threshold=THRESHOLD, max_models=len(list_mesh),max_iterations=10,inner_iterations=40, radius=graph_radius)
        if not models:
            raise RuntimeError("gair_ransac did not return any model")
        for model in models:
            mesh_estimated = supmesh.superquadric_mesh(model)
            list_mesh.append(mesh_estimated)
            colors.append("lightgreen")


    # use chamfer distance(one side) on ground-truth= sampled points(noisy) and the points from our supq estimated
    estimated_meshes = list_mesh[n_gt:]
    sampled_estimated, _ = samp.sampling_sq_random(estimated_meshes, n_points=4000, seed=50)
    sample_from_supq_estimated = np.vstack(sampled_estimated)
    _, index_sample_to_estimate = k_nearest_neighbors(sampled_points, sample_from_supq_estimated, k=1)
    index_sample_to_estimate = np.asarray(index_sample_to_estimate).reshape(-1)
    # One-sided distance: for each ground-truth point, measure the distance to its nearest point on the estimated shape.
    cd_one_side = np.linalg.norm(sample_from_supq_estimated[index_sample_to_estimate] - sampled_points,axis=1,ord=2,).mean()
    print(f"one-side chamfer distance = {cd_one_side:.4f}")
    # symmetric chamfer distance
    cd=chamfer_distance(sampled_points,sample_from_supq_estimated)
    print(f"symmetric chamfer distance = {cd:.4f}")
    cd=chamfer_distance(sampled_points_random,sample_from_supq_estimated)
    print(f"symmetric chamfer distance from 2 exact mesh = {cd:.4f}")

    # hausdorff distance
    tree_est = cKDTree(sample_from_supq_estimated)
    tree_gt  = cKDTree(sampled_points_random)
    d_gt_to_est = tree_est.query(sampled_points_random, k=1)[0]
    d_est_to_gt = tree_gt.query(sample_from_supq_estimated, k=1)[0]
    hausdorff = max(d_gt_to_est.max(), d_est_to_gt.max())
    print(f"Hausdorff distance = {hausdorff:.4f}")
    
    # f1 score
    tau = 0.05
    precision = (d_est_to_gt < tau).mean()
    recall    = (d_gt_to_est < tau).mean()
    fs = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"F1-score (tau={tau}) = {fs:.4f}")

    if algorithm == "inner-ransac":
        inlier_mask = theta0.best_inliers_mask
    elif algorithm == "ransac":
        inlier_mask = None
        if inliers_masks:
            inlier_mask = inliers_masks[0].copy()
            for mask in inliers_masks[1:]:
                inlier_mask |= mask
    elif algorithm == "gair-ransac":
        inlier_mask = None
        if inliers_masks:
            inlier_mask = inliers_masks[0].copy()
            for mask in inliers_masks[1:]:
                inlier_mask |= mask
    else:
        inlier_mask = None
    if inlier_mask is not None:
        n_inliers = inlier_mask.sum()
        n_outliers = len(inlier_mask) - n_inliers
        print(f"Inliers: {n_inliers} | Outliers: {n_outliers} | Total: {len(inlier_mask)} | Outlier ratio: {n_outliers/len(inlier_mask):.2%}")
    vis.show_mesh_and_points(
        list_mesh,
        pts=sampled_points,
        point_size=5,
        show_bounds=True,
        colors=colors,
        inlier_mask=inlier_mask,
        mss_used=total_best_mss_used if algorithm == "gair-ransac" else None,
    )

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    create_and_estimate_supq()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
