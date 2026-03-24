import sys
from pathlib import Path
import trimesh
import numpy as np
from src.superquadrics import superquadric_mesh as supmesh
from src.visualizations import visualization as vis
from src.superquadrics import superquadric_sampling as samp
from src.gair_ransac.inner_ransac import InnerRansacResult, inner_ransac, fit_superquadric_ls
from src.gair_ransac.ransac import ransac
from point_cloud_utils import k_nearest_neighbors, chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.gair_ransac import gair_ransac

THRESHOLD = 0.1
PC_FILE = Path("src/point_clouds/mushroom.glb")  # <-- change this

def create_and_estimate_supq():
    # --- load point cloud from file ---
    scene = trimesh.load(str(PC_FILE))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene
    sampled_points = np.asarray(mesh_raw.vertices, dtype=np.float64)
    normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)
    print(f"Loaded {sampled_points.shape[0]} points from {PC_FILE.name}")

    list_mesh = []
    colors = []
    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    n_gt = 0  # no ground truth meshes
    total_best_mss_used = None

    algorithm = "gair-ransac"
    max_models = 50 # <-- how many superquadrics to find

    if algorithm == "ls":
        small_sample = sampled_points[:30]
        theta0 = fit_superquadric_ls(small_sample)
        list_mesh.append(supmesh.superquadric_mesh(theta0))
        colors.append("lightgreen")
    elif algorithm == "inner-ransac":
        theta0: InnerRansacResult = inner_ransac(sampled_points, refined_set_index=np.arange(sampled_points.shape[0]), actual_set_index=None, threshold=THRESHOLD)
        list_mesh.append(supmesh.superquadric_mesh(theta0.best_model))
        colors.append("lightgreen")
    elif algorithm == "ransac":
        graph_radius = 0.08
        models, inliers_masks = ransac(sampled_points, threshold=THRESHOLD, max_models=max_models, max_iterations=10, inner_iterations=40, radius=graph_radius, graphcut=True)
        if not models:
            raise RuntimeError("ransac did not return any model")
        for i, model in enumerate(models):
            list_mesh.append(supmesh.superquadric_mesh(model))
            colors.append(palette[i % len(palette)])
    elif algorithm == "gair-ransac":
        graph_radius = 0.08
        models, inliers_masks, total_best_mss_used = gair_ransac(sampled_points, normals, threshold=THRESHOLD, max_models=max_models, max_iterations=50, inner_iterations=80, radius=graph_radius, use_normal_coherence=True, min_coverage=0.4)
        if not models:
            raise RuntimeError("gair_ransac did not return any model")
        for i, model in enumerate(models):
            list_mesh.append(supmesh.superquadric_mesh(model))
            colors.append(palette[i % len(palette)])

    if algorithm == "inner-ransac":
        inlier_mask = theta0.best_inliers_mask
    elif algorithm in ("ransac", "gair-ransac"):
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

    if list_mesh:
        sampled_estimated, _ = samp.sampling_sq_random(list_mesh, n_points=4000, seed=50)
        sample_from_supq_estimated = np.vstack(sampled_estimated)
        cd = chamfer_distance(sampled_points, sample_from_supq_estimated)
        print(f"reconstruction chamfer = {cd:.4f}")

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
