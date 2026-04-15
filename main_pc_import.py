import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from src.superquadrics import superquadric_mesh as supmesh
from src.visualizations import visualization as vis
from src.superquadrics import superquadric_sampling as samp
from src.gair_ransac.inner_ransac import inner_ransac, fit_superquadric_ls
from src.gair_ransac.ransac import ransac
from point_cloud_utils import chamfer_distance
from src.gair_ransac.gair_ransac import gair_ransac


THRESHOLD_SCALE = 3.0
THRESHOLD_SPACING_FACTOR = 0.5
NOISE_STD=0.0
THRESHOLD = 0.02#THRESHOLD_SCALE * NOISE_STD
NOISE_NORMAL_STD=0.0
GRAPH_RADIUS = 0.06
SAMPLE_SIZE = 30
MIN_INLIERS = 30
PROJECT_ROOT = Path(__file__).resolve().parent
"""anthropomorphic_mushroom_character.glb"""
PC_FILE = PROJECT_ROOT / "test_objects" / "cartoon_character.glb"
# Single knob for how many points are sampled from the input mesh
# and from the reconstructed superquadrics for evaluation.
SAMPLED_POINT_COUNT = 30000
DEFAULT_BASE_SEED = 1234567932
MAX_MODEL = 10
ALGORITHM = "gair-ransac" # options: "ls", "inner-ransac", "ransac", "gair-ransac", "gc-ransac"

@dataclass(frozen=True)
class RunSeeds:
    base: int
    input_sampling: int
    algorithm: int
    evaluation_sampling: int


def _seed_from_sequence(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def build_run_seeds(base_seed: int = DEFAULT_BASE_SEED) -> RunSeeds:
    child_sequences = np.random.SeedSequence(base_seed).spawn(3)
    return RunSeeds(
        base=int(base_seed),
        input_sampling=_seed_from_sequence(child_sequences[0]),
        algorithm=_seed_from_sequence(child_sequences[1]),
        evaluation_sampling=_seed_from_sequence(child_sequences[2]),
    )


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


def sample_input_mesh(
    mesh: trimesh.Trimesh,
    n_points: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_points <= 0:
        raise ValueError(f"SAMPLED_POINT_COUNT must be positive, got {n_points}")

    sampled_points_list, normals_list = samp.sampling_sq(
        [mesh],
        n_points=int(n_points / 2),
        seed=seed,
    )
    noisy_sampled, noisy_normal = samp.sampling_sq_noisy(
        [mesh],
        n_points=n_points,
        noise_std=NOISE_STD,
        normal_noise_std=NOISE_NORMAL_STD,
        seed=seed,
    )
    sampled_points = np.vstack([np.asarray(sampled_points_list[0], dtype=np.float64),np.asarray(noisy_sampled[0], dtype=np.float64),])
    normals = np.vstack([np.asarray(normals_list[0], dtype=np.float64),np.asarray(noisy_normal[0], dtype=np.float64),])
    return sampled_points, normals


def estimate_point_spacing(points: np.ndarray) -> float:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape[0] < 2:
        return 0.0

    tree = cKDTree(point_array)
    nn_dists, _ = tree.query(point_array, k=2)
    nn_dists = np.asarray(nn_dists, dtype=np.float64)
    return float(np.median(nn_dists[:, 1]))


def compute_effective_threshold(
    points: np.ndarray,
    min_threshold: float = THRESHOLD,
    spacing_factor: float = THRESHOLD_SPACING_FACTOR,
) -> tuple[float, float]:
    point_spacing = estimate_point_spacing(points)
    effective_threshold = max(float(min_threshold), float(spacing_factor) * point_spacing)
    return effective_threshold, point_spacing


def create_and_estimate_supq(
    pc_file: str | Path = PC_FILE,
    base_seed: int = DEFAULT_BASE_SEED,
):
    # --- load point cloud from file ---
    run_seeds = build_run_seeds(base_seed)
    mesh_path = resolve_input_mesh_path(pc_file)
    scene = trimesh.load(str(mesh_path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene

    # Always sample the input mesh through the shared sampling utilities module.
    sampled_points, normals = sample_input_mesh(
        mesh_raw,
        n_points=SAMPLED_POINT_COUNT,
        seed=run_seeds.input_sampling,
    )
    effective_threshold, point_spacing = compute_effective_threshold(sampled_points)

    print(f"Loaded {sampled_points.shape[0]} points from {mesh_path.name}")
    print(
        "Seeds | "
        f"base={run_seeds.base} "
        f"input_sampling={run_seeds.input_sampling} "
        f"algorithm={run_seeds.algorithm} "
        f"evaluation_sampling={run_seeds.evaluation_sampling}"
    )
    print(f"Sampling | sampled_point_count={SAMPLED_POINT_COUNT}")
    print(
        "Consensus | "
        f"min_threshold={THRESHOLD:.4f} "
        f"point_spacing={point_spacing:.4f} "
        f"effective_threshold={effective_threshold:.4f}"
    )

    list_mesh = []
    colors = []
    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    n_gt = 0  # no ground truth meshes
    total_best_mss_used = None

    algorithm = ALGORITHM
    max_models = MAX_MODEL # <-- how many superquadrics to find

    if algorithm == "ls":
        small_sample = sampled_points[:30]
        theta0 = fit_superquadric_ls(small_sample)
        models = [theta0]
        list_mesh.append(supmesh.superquadric_mesh(theta0))
        colors.append("lightgreen")
    elif algorithm == "inner-ransac":
        theta0 = inner_ransac(
            sampled_points,
            refined_set_index=np.arange(sampled_points.shape[0]),
            actual_set_index=None,
            threshold=effective_threshold,
            random_seed=run_seeds.algorithm,
        )
        models = [theta0.best_model]
        list_mesh.append(supmesh.superquadric_mesh(theta0.best_model))
        colors.append("lightgreen")
    elif algorithm == "ransac":
        graph_radius = GRAPH_RADIUS
        models, inliers_masks = ransac(
            sampled_points,
            threshold=effective_threshold,
            max_models=max_models,
            max_iterations=20,
            inner_iterations=100,
            radius=graph_radius,
            graphcut=True,
            random_seed=run_seeds.algorithm,
        )
        if not models:
            raise RuntimeError("ransac did not return any model")
        for i, model in enumerate(models):
            list_mesh.append(supmesh.superquadric_mesh(model))
            colors.append(palette[i % len(palette)])
    elif algorithm in ("gair-ransac", "gc-ransac"):
        graph_radius = GRAPH_RADIUS
        models, inliers_masks, total_best_mss_used = gair_ransac(
            sampled_points,
            normals,
            threshold=effective_threshold,
            max_models=max_models,
            max_iterations=40,
            inner_iterations=200,
            radius=graph_radius,
            use_normal_coherence=(algorithm == "gair-ransac"),
            min_coverage=0.0,
            random_seed=run_seeds.algorithm,
            sample_size=SAMPLE_SIZE,
            min_inliers=MIN_INLIERS,

        )
        if not models:
            raise RuntimeError("gair_ransac did not return any model")
        for i, model in enumerate(models):
            list_mesh.append(supmesh.superquadric_mesh(model))
            colors.append(palette[i % len(palette)])
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if algorithm == "inner-ransac":
        inlier_mask = theta0.best_inliers_mask
    elif algorithm in ("ransac", "gair-ransac", "gc-ransac"):
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
        sampled_estimated, _ = samp.sampling_sq(
            list_mesh,
            n_points=SAMPLED_POINT_COUNT,
            seed=run_seeds.evaluation_sampling,
        )
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
        mss_used=total_best_mss_used if algorithm in ("gair-ransac", "gc-ransac") else None,
        models=models,
        treshold=effective_threshold
    )

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", nargs="?", default=PC_FILE)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    create_and_estimate_supq(args.input_mesh, base_seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
