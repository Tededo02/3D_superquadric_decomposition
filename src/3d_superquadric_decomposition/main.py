import sys
import superquadric_mesh as supmesh
from superquadric_param import SuperQuadricParams
import visualization as vis
import numpy as np
import superquadric_sampling as samp
from gair_ransac.inner_ransac import InnerRansacResult, inner_ransac, fit_superquadric_ls
from point_cloud_utils import k_nearest_neighbors, chamfer_distance

def create_and_estimate_supq():
    colors: list[str] = []
    # add a superquadric to the scene, sample points then try to fit a superquadric to the points, visualize the results
    # lightblue is the original superquadric, lightgreen is the estimated superquadric
    test = SuperQuadricParams(1.2,5.2,0.7,0.5,2,[2,2,1],[5,5,5])
    mesh = supmesh.superquadric_mesh(test)

    list_mesh = []
    list_mesh.append(mesh)
    colors.append("lightblue")
    #list_mesh.append(mesh2)
    #sampled_points = samp.sampling_sq(list_mesh, n_points=1000)
    sampled_points_noisy = samp.sampling_sq_noisy(list_mesh, n_points=2000, noise_std=0.07, clip_k=3.0, seed=42)#list 
    sampled_points_random = samp.sampling_sq_random(list_mesh, n_points=2000, seed=42) #list 
    sampled_points_outliers = samp.sampling_outliers(list_mesh, n_out=400, margin=0.10, mode="uniform", seed=42) #array 2D (N_out, 3)
    algorithm="ransac"
    #------choose here which kind of points to use for fitting
    sampled_points = np.vstack([*sampled_points_noisy]).astype(np.float32, copy=False)

    if algorithm == "ls":

        small_sample = sampled_points[:30] # just for testing, should be sampled from the gair set
        theta0 = fit_superquadric_ls(small_sample)
        mesh_estimated = supmesh.superquadric_mesh(theta0)

    else:
        theta0:InnerRansacResult = inner_ransac(sampled_points, refined_set_index=np.arange(sampled_points.shape[0]), actual_set_index=np.arange(sampled_points.shape[0]), threshold=0.1)
        mesh_estimated = supmesh.superquadric_mesh(theta0.best_model)


    list_mesh.append(mesh_estimated)
    colors.append("lightgreen")

    # use chamfer distance(one side) on ground-truth= sampled points(noisy) and the points from our supq estimated
    sample_from_supq_estimated = np.asarray(samp.sampling_sq_random([mesh_estimated], n_points=2000, seed=50)[0],dtype=np.float32,)
    _, index_sample_to_estimate = k_nearest_neighbors(sampled_points, sample_from_supq_estimated, k=1)
    # One-sided distance: for each ground-truth point, measure the distance to its nearest point on the estimated shape.
    cd_one_side = np.linalg.norm(sample_from_supq_estimated[index_sample_to_estimate] - sampled_points,axis=-1,ord=2,).mean()
    print(f"one-side chamfer distance = {cd_one_side:.4f}")
    # symmetric chamfer distance
    cd=chamfer_distance(sampled_points,sample_from_supq_estimated)
    print(f"symmetric chamfer distance = {cd:.4f}")

    vis.show_mesh_and_points(list_mesh, pts=sampled_points, point_size=5, show_bounds=True, colors=colors)

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    
    create_and_estimate_supq()

    # GAIR-RANSAC: /TODO
    superquadric_list: list[SuperQuadricParams] = []

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
