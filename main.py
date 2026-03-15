import sys
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.visualizations import visualization as vis
import numpy as np
from src.superquadrics import superquadric_sampling as samp
from src.gair_ransac.inner_ransac import InnerRansacResult, inner_ransac, fit_superquadric_ls
from point_cloud_utils import k_nearest_neighbors, chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.gair_ransac import gair_ransac

NOISE_STD = 0.6
THRESHOLD = 3*NOISE_STD

def create_and_estimate_supq():
    colors: list[str] = []
    # add a superquadric to the scene, sample points then try to fit a superquadric to the points, visualize the results
    # lightblue is the original superquadric, lightgreen is the estimated superquadric
    test = SuperQuadricParams(9.0, 9.0, 9.0, 0.1, 0.1, [2.0, 2.0, 1.0], [5.0, 5.0, 5.0])
    test2 = SuperQuadricParams(3.0, 3.0, 3.0, 0.5, 0.9, [2.0, 2.0, 1.0], [-5.0, -5.0, -5.0])
    test4 = SuperQuadricParams(2.5, 2.5, 2.5, 0.1, 0.1, [2.1, 1.9, 0.8], [-10.5, -5.0, -5.0])

    mesh = supmesh.superquadric_mesh(test)

    list_mesh = []
    list_mesh.append(mesh)
    colors.append("lightblue")


    #----those functions return a set of sample and their normals-----------
    #sampled_points,normals = samp.sampling_sq(list_mesh, n_points=1000)
    sampled_points_noisy,normals_sp_noisy = samp.sampling_sq_noisy(list_mesh, n_points=30000, noise_std=NOISE_STD, clip_k=3.0, seed=42)#list 
    sampled_points_random, normals_sp_random = samp.sampling_sq_random(list_mesh, n_points=4000, seed=42)
    sampled_points_random = np.vstack(sampled_points_random).astype(np.float32, copy=False)
    sampled_points_outliers,normals_sp_outliers = samp.sampling_outliers(list_mesh, n_out=1, margin=0.10, mode="uniform", seed=42) #array 2D (N_out, 3)
    algorithm="ransac"
    #------choose here which kind of points to use for fitting the superquadric------
    sampled_points = np.vstack([*sampled_points_noisy, sampled_points_outliers]).astype(np.float32, copy=False)
    normals = np.vstack([*normals_sp_noisy, normals_sp_outliers]).astype(np.float32, copy=False)
    mesh_estimated = None
    if algorithm == "ls":

        small_sample = sampled_points[:30] # just for testing, should be sampled from the gair set
        theta0 = fit_superquadric_ls(small_sample)
        mesh_estimated = supmesh.superquadric_mesh(theta0)
        list_mesh.append(mesh_estimated)
        colors.append("lightgreen")
    elif algorithm == "ransac":
        theta0:InnerRansacResult = inner_ransac(sampled_points, refined_set_index=np.arange(sampled_points.shape[0]), actual_set_index=np.arange(sampled_points.shape[0]), threshold=THRESHOLD)
        mesh_estimated = supmesh.superquadric_mesh(theta0.best_model)
        list_mesh.append(mesh_estimated)
        colors.append("lightgreen")
    elif algorithm == "gair-ransac":
        # Radius is expressed as a fraction of the point cloud bounding-box diagonal.
        graph_radius = 0.08
        models, inliers_masks = gair_ransac(sampled_points, normals, threshold=THRESHOLD, max_models=len(list_mesh),max_iterations=10,inner_iterations=40, radius=graph_radius)
        if not models:
            raise RuntimeError("gair_ransac did not return any model")
        for model in models:
            mesh_estimated = supmesh.superquadric_mesh(model)
            list_mesh.append(mesh_estimated)
            colors.append("lightgreen")


    # use chamfer distance(one side) on ground-truth= sampled points(noisy) and the points from our supq estimated
    sampled_estimated, _ = samp.sampling_sq_random([mesh_estimated], n_points=4000, seed=50)
    sample_from_supq_estimated = np.asarray(sampled_estimated[0], dtype=np.float32)
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

    if algorithm == "ransac":
        inlier_mask = theta0.best_inliers_mask
    elif algorithm == "gair-ransac":
        inlier_mask = np.any(np.stack(inliers_masks), axis=0) if inliers_masks else None
    else:
        inlier_mask = None
    if inlier_mask is not None:
        n_inliers = inlier_mask.sum()
        n_outliers = len(inlier_mask) - n_inliers
        print(f"Inliers: {n_inliers} | Outliers: {n_outliers} | Total: {len(inlier_mask)} | Outlier ratio: {n_outliers/len(inlier_mask):.2%}")
    vis.show_mesh_and_points(list_mesh, pts=sampled_points, point_size=5, show_bounds=True, colors=colors, inlier_mask=inlier_mask)

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    create_and_estimate_supq()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
