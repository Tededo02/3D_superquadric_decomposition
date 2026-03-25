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
PROJECT_ROOT = Path(__file__).resolve().parent
PC_FILE = PROJECT_ROOT / "test_objects" / "anthropomorphic_mushroom_character.glb"
N_POINTS: int | None = None


def resolve_input_mesh_path(pc_file: str | Path) -> Path:
    mesh_path = Path(pc_file).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = PROJECT_ROOT / mesh_path

    if mesh_path.exists():
        return mesh_path

    available_meshes = sorted(
        path.relative_to(PROJECT_ROOT)
        for pattern in ("*.glb", "*.stl", "*.obj", "*.ply")
        for path in PROJECT_ROOT.rglob(pattern)
    )
    available_hint = ", ".join(str(path) for path in available_meshes[:10])
    raise FileNotFoundError(
        f"Input mesh not found: {mesh_path}\n"
        f"Available meshes in repo: {available_hint}"
    )


def create_and_estimate_supq(pc_file: str | Path = PC_FILE):
    # --- load point cloud from file ---
    mesh_path = resolve_input_mesh_path(pc_file)
    scene = trimesh.load(str(mesh_path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene

    if N_POINTS is None:
        sampled_points = np.asarray(mesh_raw.vertices, dtype=np.float64)
        normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)
    else:
        sampled_points, face_idx = trimesh.sample.sample_surface(mesh_raw, N_POINTS)
        sampled_points = np.asarray(sampled_points, dtype=np.float64)
        normals = np.asarray(mesh_raw.face_normals[face_idx], dtype=np.float64)

    print(f"Loaded {sampled_points.shape[0]} points from {mesh_path.name}")

    list_mesh = []
    colors = []
    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    n_gt = 0  # no ground truth meshes
    total_best_mss_used = None

    algorithm = "gair-ransac"
    max_models = 1 # <-- how many superquadrics to find

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
        models, inliers_masks, total_best_mss_used = gair_ransac(sampled_points, normals, threshold=THRESHOLD, max_models=max_models, max_iterations=2, inner_iterations=80, radius=graph_radius, use_normal_coherence=True, min_coverage=0.4)
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
        models=models,
        treshold=THRESHOLD
    )

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    input_mesh = argv[0] if argv else PC_FILE
    create_and_estimate_supq(input_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
