import sys
import superquadric_mesh as supmesh
from superquadric_param import SuperQuadricParams
import visualization as vis
import numpy as np
import superquadric_sampling as samp
from gair_ransac.inner_ransac import InnerRansacResult, inner_ransac, fit_superquadric_ls

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    
    colors: np.ndarray = []
    # add a superquadric to the scene, sample points then try to fit a superquadric to the points, visualize the results
    # lightblue is the original superquadric, lightgreen is the estimated superquadric
    test = SuperQuadricParams(1.2,5.2,0.7,0.5,2,[2,2,1],[5,5,5])
    mesh = supmesh.superquadric_mesh(test)

    list_mesh = []
    list_mesh.append(mesh)
    colors.append("lightblue")
    #list_mesh.append(mesh2)
    #sampled_points = samp.sampling_sq(list_mesh, n_points=1000)
    sampled_points_noisy = samp.sampling_sq_noisy(list_mesh, n_points=1000, noise_std=0.05, clip_k=3.0, seed=42)#list 
    sampled_points_random = samp.sampling_sq_random(list_mesh, n_points=1000, seed=42) #list 
    sampled_points_outliers = samp.sampling_outliers(list_mesh, n_out=400, margin=0.10, mode="uniform", seed=42) #array 2D (N_out, 3)

    #------choose here which kind of points to use for fitting
    sampled_points = np.vstack([*sampled_points_noisy])
    #theta0:InnerRansacResult = inner_ransac(sampled_points, refined_set_index=np.arange(sampled_points.shape[0]), actual_set_index=np.arange(sampled_points.shape[0]), threshold=0.1)
    #mesh_estimated = supmesh.superquadric_mesh(theta0.best_model)

    small_sample = sampled_points_noisy[0][:20] # just for testing, should be sampled from the gair set
    theta0:InnerRansacResult = fit_superquadric_ls(small_sample)
    mesh_estimated = supmesh.superquadric_mesh(theta0)
    list_mesh.append(mesh_estimated)
    colors.append("lightgreen")
    vis.show_mesh_and_points(list_mesh, pts=sampled_points, point_size=5, show_bounds=True, colors=colors)
    

    # GAIR-RANSAC: /TODO
    superquadric_list: list[SuperQuadricParams] = []
  #  superquadric_list=

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
