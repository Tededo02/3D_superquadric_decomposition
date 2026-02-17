import sys
import superquadric_mesh as supmesh
from superquadric_param import SuperQuadricParams
import visualization as vis
import numpy as np
import superquadric_sampling as samp

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]



    # TODO: pipeline    
    test = SuperQuadricParams(1.2,1.2,0.7,0.5,4,[2,2,1],[5,5,5])
    mesh = supmesh.superquadric_mesh(test)
        
    test2 = SuperQuadricParams(1.2,1.2,0.7,0.5,4,[1,1,1],[4,4,4])
    mesh2 = supmesh.superquadric_mesh(test2)
    list_mesh = []
    list_mesh.append(mesh)
    list_mesh.append(mesh2)
    #sampled_points = samp.sampling_sq(list_mesh, n_points=1000)
    sampled_points_noisy = samp.sampling_sq_noisy(list_mesh, n_points=1000, noise_std=0.05, clip_k=3.0, seed=42)#list
    sampled_points_random = samp.sampling_sq_random(list_mesh, n_points=1000, seed=42) #list
    sampled_points_outliers = samp.sampling_outliers(list_mesh, n_out=400, margin=0.10, mode="uniform", seed=42) #array
    spn=np.array(sampled_points_noisy) # convert list to array
    spr=np.array(sampled_points_random) # convert list to array
    sampled_points = np.vstack([spn,spr, sampled_points_outliers]) # concatenate the points from both meshes and the outliers
    vis.show_mesh_and_points(list_mesh, pts=sampled_points, point_size=5, show_bounds=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
