from pathlib import Path

import numpy as np
import trimesh

from src.gair_ransac.gair_ransac import gair_ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.visualizations.visualization import show_mesh_and_points


NOISE_STD = 0.01
THRESHOLD = 3 * NOISE_STD
N_POINTS = 30000
MAX_MODELS = 5
MAX_ITERATIONS = 10
INNER_ITERATIONS = 40
GRAPH_RADIUS = 0.08
RANDOM_SEED = 53
DEFAULT_STL_PATH = Path(__file__).resolve().parent / "test_objects" / "131969.stl"


def load_mesh_and_build_noisy_cloud(
    stl_path: str | Path,
    n_points: int = N_POINTS,
    noise_std: float = NOISE_STD,
    seed: int | None = RANDOM_SEED,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(stl_path)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Empty mesh scene: {stl_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.fix_normals()
    sampled_points_list, normals_list = samp.sampling_sq_noisy([mesh],n_points=n_points,noise_std=noise_std,clip_k=3.0,seed=seed,)
    return mesh, sampled_points_list[0], normals_list[0]


def merge_inlier_masks(inliers_masks: list[np.ndarray]) -> np.ndarray | None:
    if not inliers_masks:
        return None

    merged = np.asarray(inliers_masks[0], dtype=bool).copy()
    # Merge inliers from all masks using logical OR (a point is an inlier if it's an inlier in any mask)
    for mask in inliers_masks[1:]:
        merged = merged | np.asarray(mask, dtype=bool)
    return merged


def create_and_estimate_from_stl(
    stl_path: str | Path = DEFAULT_STL_PATH,
    n_points: int = N_POINTS,
    noise_std: float = NOISE_STD,
) -> None:
    original_mesh, sampled_points, normals = load_mesh_and_build_noisy_cloud(
        stl_path,
        n_points=n_points,
        noise_std=noise_std,
        seed=RANDOM_SEED,
    )

    models, inliers_masks, _ = gair_ransac(
        sampled_points,
        normals,
        threshold=3 * noise_std,
        max_models=MAX_MODELS,
        max_iterations=MAX_ITERATIONS,
        inner_iterations=INNER_ITERATIONS,
        radius=GRAPH_RADIUS,
        random_seed=RANDOM_SEED,
    )
    if not models:
        raise RuntimeError("gair_ransac did not return any model")

    estimated_meshes = [supmesh.superquadric_mesh(model) for model in models]
    meshes_to_show = [original_mesh, *estimated_meshes]
    colors = ["lightblue", *(["lightgreen"] * len(estimated_meshes))]
    inlier_mask = merge_inlier_masks(inliers_masks)

    if inlier_mask is not None:
        n_inliers = int(np.count_nonzero(inlier_mask))
        n_outliers = int(inlier_mask.size - n_inliers)
        print(
            f"Estimated models: {len(models)} | "
            f"Inliers: {n_inliers} | Outliers: {n_outliers} | Total: {inlier_mask.size}"
        )

    show_mesh_and_points(
        meshes_to_show,
        pts=sampled_points,
        point_size=5,
        show_bounds=True,
        colors=colors,
        inlier_mask=inlier_mask,
    )


def main() -> int:
    create_and_estimate_from_stl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
