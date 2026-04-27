import argparse
import sys
import time
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

#THRESHOLD = 0.03
M_NEIGHBORS = 4
THRESHOLD_FACTOR = 3
THRESHOLD_SPACING_FACTOR = 0.5
MIN_THRESHOLD = 0.09
MIN_COVERAGE = 0.2
NOISE_STD = 0.015
# Set to a value > 0 to append synthetic bbox outliers to mesh-sampled inputs.
N_OUTLIERS = 600
OUTLIER_MARGIN = 0.10
OUTLIER_MODE = "uniform"
OUTLIER_SEED_OFFSET = 10000
DEBUG_V = False
PROJECT_ROOT = Path(__file__).resolve().parent
OBJECT_VIEWS_DIR = PROJECT_ROOT / "im_obj"
"""anthropomorphic_mushroom_character.glb"""
PC_FILE = PROJECT_ROOT / "test_objects" / "131969.stl"
# Single knob for how many points are sampled from the input mesh
# and from the reconstructed superquadrics for evaluation.
SAMPLED_POINT_COUNT = 4000
DEFAULT_BASE_SEED = 12345#1234
MAX_MODELS = 10
MIN_SAMPLE_SIZE = 10
MAX_SAMPLE_SIZE = 20
MIN_INLIERS_FLOOR = 5
MAX_INLIERS_CAP = 12
SUPPORTED_ALGORITHMS = (
    "ls",
    "inner-ransac",
    "ransac",
    "gair-ransac",
    "gc-ransac",
    "gair-mss-no-normals",
    "gc-mss-no-normals",
)
PYVISTA_CAMERA_POSITION = [
    (9.38230319524604, 3.7942846282426563, 6.294340545028417),
    (0.0, 0.0, 0.0),
    (-0.18140538052646699, 0.9380912113821817, -0.2950880665895499),
]


@dataclass(frozen=True)
class RunSeeds:
    base: int
    input_sampling: int
    algorithm: int
    evaluation_sampling: int


@dataclass(frozen=True)
class LoadedInput:
    points: np.ndarray
    normals: np.ndarray
    kind: str
    noise_std: float
    noise_std_source: str
    gt_inlier_mask: np.ndarray | None = None
    gt_outlier_ratio: float | None = None
    gt_noise_std: float | None = None
    gt_inliers_assumed_from_tail: bool = False


@dataclass(frozen=True)
class PointCloudStats:
    point_count: int
    spatial_extent: float
    median_nn_distance: float


@dataclass(frozen=True)
class RansacTuning:
    sample_size: int
    min_inliers: int
    mss_max_pool_fraction: float


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


def normalize_algorithm_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "gair": "gair-ransac",
        "gc": "gc-ransac",
        "gair-nomss": "gair-mss-no-normals",
        "gc-nomss": "gc-mss-no-normals",
        "gair-no-mss": "gair-mss-no-normals",
        "gc-no-mss": "gc-mss-no-normals",
        "lo": "ransac",
        "lo-ransac": "ransac",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: {name}. "
            f"Supported values: {', '.join(SUPPORTED_ALGORITHMS)}"
        )
    return normalized


def normalize_outlier_ratio(value: str) -> float:
    ratio = float(value)
    if ratio < 0.0:
        raise ValueError(f"outlier ratio must be non-negative, got {ratio}")
    if ratio > 1.0:
        if ratio <= 100.0:
            return ratio / 100.0
        raise ValueError(f"outlier ratio must be in [0, 1] or [0, 100], got {ratio}")
    return ratio


def parse_point_cloud_metadata(path: Path) -> tuple[float, float]:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Point cloud filename must end with '_<noise_std>_<outlier_ratio>.ply', got {path.name}"
        )

    noise_std = float(parts[-2])
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    outlier_ratio = normalize_outlier_ratio(parts[-1])

    return outlier_ratio, noise_std


def infer_ground_truth_inliers(
    n_points: int,
    outlier_ratio: float,
) -> tuple[np.ndarray, int]:
    n_outliers = int(round(float(n_points) * float(outlier_ratio)))
    n_outliers = min(max(n_outliers, 0), n_points)

    gt_inliers = np.ones(n_points, dtype=bool)
    if n_outliers > 0:
        # Assumption shared with the experiment scripts: outliers are appended at the end.
        gt_inliers[-n_outliers:] = False
    return gt_inliers, n_outliers


def summarize_point_cloud(points: np.ndarray) -> PointCloudStats:
    point_array = np.asarray(points, dtype=np.float64)
    point_count = int(point_array.shape[0])
    if point_count == 0:
        return PointCloudStats(point_count=0, spatial_extent=0.0, median_nn_distance=0.0)

    spans = np.ptp(point_array, axis=0)
    spatial_extent = float(np.linalg.norm(spans))
    if point_count == 1:
        return PointCloudStats(
            point_count=point_count,
            spatial_extent=spatial_extent,
            median_nn_distance=0.0,
        )

    tree = cKDTree(point_array)
    nn_distances = np.asarray(tree.query(point_array, k=2)[0], dtype=np.float64)
    median_nn_distance = float(np.median(nn_distances[:, 1]))
    return PointCloudStats(
        point_count=point_count,
        spatial_extent=spatial_extent,
        median_nn_distance=median_nn_distance,
    )


def tune_ransac_hyperparameters(stats: PointCloudStats) -> RansacTuning:
    point_count = int(stats.point_count)
    if point_count <= 0:
        return RansacTuning(
            sample_size=0,
            min_inliers=0,
            mss_max_pool_fraction=0.25,
        )

    sample_low = min(MIN_SAMPLE_SIZE, point_count)
    sample_high = min(MAX_SAMPLE_SIZE, point_count)
    raw_sample_size = int(round(0.06 * point_count))
    sample_size = int(np.clip(raw_sample_size, sample_low, sample_high))

    min_inliers_low = min(max(sample_size, MIN_INLIERS_FLOOR), point_count)
    min_inliers_high = min(MAX_INLIERS_CAP, point_count)
    raw_min_inliers = int(round(0.15 * point_count))
    min_inliers = int(np.clip(raw_min_inliers, min_inliers_low, min_inliers_high))

    mss_max_pool_fraction = float(
        np.clip((6.0 * sample_size) / max(point_count, 1), 0.18, 0.35)
    )
    return RansacTuning(
        sample_size=sample_size,
        min_inliers=min_inliers,
        mss_max_pool_fraction=mss_max_pool_fraction,
    )


def get_threshold(
    noise_std: float,
    median_nn_distance: float | None = None,
) -> float:
    threshold_from_noise = float(THRESHOLD_FACTOR) * float(noise_std)
    threshold_from_spacing = 0.0
    if median_nn_distance is not None and median_nn_distance > 0.0:
        threshold_from_spacing = float(THRESHOLD_SPACING_FACTOR) * float(median_nn_distance)
    return max(float(MIN_THRESHOLD), threshold_from_noise, threshold_from_spacing)


def resolve_input_path(pc_file: str | Path) -> Path:
    input_path = Path(pc_file).expanduser()
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if input_path.exists():
        return input_path

    available_inputs = sorted(
        path.relative_to(PROJECT_ROOT)
        for pattern in ("*.glb", "*.stl", "*.obj", "*.ply")
        for path in PROJECT_ROOT.rglob(pattern)
    )
    available_hint = ", ".join(str(path) for path in available_inputs[:10])
    raise FileNotFoundError(
        f"Input file not found: {input_path}\n"
        f"Available inputs in repo: {available_hint}"
    )


def load_trimesh_asset(path: Path):
    asset = trimesh.load(str(path))
    if isinstance(asset, trimesh.Scene):
        geometry = list(asset.geometry.values())
        if not geometry:
            raise ValueError(f"No geometry found in {path}")
        return trimesh.util.concatenate(geometry)
    return asset


def load_point_cloud_with_normals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = load_trimesh_asset(path)
    points = np.asarray(raw.vertices, dtype=np.float64)

    normals_path = path.parent / f"normals_{path.name}"
    if normals_path.exists():
        normals_raw = load_trimesh_asset(normals_path)
        normals = np.asarray(normals_raw.vertices, dtype=np.float64)
    else:
        normals = np.asarray(getattr(raw, "vertex_normals", []), dtype=np.float64)
        if normals.shape != points.shape:
            raise FileNotFoundError(
                f"No normals file found for {path.name} and embedded normals are unavailable."
            )

    if points.shape != normals.shape:
        raise ValueError(
            f"Points/normals shape mismatch for {path.name}: "
            f"points={points.shape}, normals={normals.shape}"
        )

    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(normal_norm > 0.0, normal_norm, 1.0)
    return points, normals


def sample_input_mesh(
    mesh: trimesh.Trimesh,
    n_points: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if n_points <= 0:
        raise ValueError(f"SAMPLED_POINT_COUNT must be positive, got {n_points}")
    if N_OUTLIERS < 0:
        raise ValueError(f"N_OUTLIERS must be non-negative, got {N_OUTLIERS}")

    sampled_points_list, normals_list = samp.sampling_sq(
        [mesh],
        n_points=int(n_points / 2),
        seed=seed,
    )
    noisy_sampled, noisy_normal = samp.sampling_sq_noisy(
        [mesh],
        n_points=n_points,
        noise_std=NOISE_STD,
        normal_noise_std=0.0,
        seed=seed,
    )
    sampled_points = np.vstack([np.asarray(sampled_points_list[0], dtype=np.float64),np.asarray(noisy_sampled[0], dtype=np.float64),])
    normals = np.vstack([np.asarray(normals_list[0], dtype=np.float64),np.asarray(noisy_normal[0], dtype=np.float64),])
    gt_inlier_mask = None

    if N_OUTLIERS > 0:
        outlier_seed = None if seed is None else int(seed) + OUTLIER_SEED_OFFSET
        outlier_points, outlier_normals = samp.sampling_outliers(
            [mesh],
            n_out=int(N_OUTLIERS),
            margin=OUTLIER_MARGIN,
            mode=OUTLIER_MODE,
            seed=outlier_seed,
        )
        n_inliers = int(sampled_points.shape[0])
        sampled_points = np.vstack(
            [sampled_points, np.asarray(outlier_points, dtype=np.float64)]
        )
        normals = np.vstack([normals, np.asarray(outlier_normals, dtype=np.float64)])
        gt_inlier_mask = np.concatenate(
            [
                np.ones(n_inliers, dtype=bool),
                np.zeros(int(outlier_points.shape[0]), dtype=bool),
            ]
        )

    return sampled_points, normals, gt_inlier_mask


def load_input_points_and_normals(
    input_path: Path,
    sample_seed: int,
) -> LoadedInput:
    raw = load_trimesh_asset(input_path)
    if isinstance(raw, trimesh.PointCloud):
        points, normals = load_point_cloud_with_normals(input_path)
        gt_inlier_mask = None
        gt_outlier_ratio = None
        gt_noise_std = None
        gt_inliers_assumed_from_tail = False
        try:
            gt_outlier_ratio, gt_noise_std = parse_point_cloud_metadata(input_path)
        except ValueError:
            pass
        else:
            gt_inlier_mask, n_gt_outliers = infer_ground_truth_inliers(
                points.shape[0],
                gt_outlier_ratio,
            )
            gt_inliers_assumed_from_tail = bool(n_gt_outliers > 0)

        return LoadedInput(
            points=points,
            normals=normals,
            kind="point-cloud",
            noise_std=float(NOISE_STD if gt_noise_std is None else gt_noise_std),
            noise_std_source=(
                "global NOISE_STD fallback"
                if gt_noise_std is None
                else "point-cloud metadata"
            ),
            gt_inlier_mask=gt_inlier_mask,
            gt_outlier_ratio=gt_outlier_ratio,
            gt_noise_std=gt_noise_std,
            gt_inliers_assumed_from_tail=gt_inliers_assumed_from_tail,
        )

    points, normals, gt_inlier_mask = sample_input_mesh(
        raw,
        n_points=SAMPLED_POINT_COUNT,
        seed=sample_seed,
    )
    gt_n_outliers = (
        0
        if gt_inlier_mask is None
        else int(len(gt_inlier_mask) - int(gt_inlier_mask.sum()))
    )
    return LoadedInput(
        points=points,
        normals=normals,
        kind="mesh",
        noise_std=float(NOISE_STD),
        noise_std_source="global NOISE_STD",
        gt_inlier_mask=gt_inlier_mask,
        gt_outlier_ratio=(
            None if gt_inlier_mask is None else float(gt_n_outliers / len(gt_inlier_mask))
        ),
        gt_noise_std=None if gt_inlier_mask is None else float(NOISE_STD),
        gt_inliers_assumed_from_tail=bool(gt_n_outliers > 0),
    )


def create_and_estimate_supq(
    pc_file: str | Path = PC_FILE,
    algorithm: str = "gair-ransac",
    base_seed: int = DEFAULT_BASE_SEED,
    visualize: bool = True,
    save_debug_views: bool = False,
    return_artifacts: bool = False,
):
    save_debug_views = bool(save_debug_views or DEBUG_V)
    # --- load point cloud from file ---
    algorithm = normalize_algorithm_name(algorithm)
    run_seeds = build_run_seeds(base_seed)
    input_path = resolve_input_path(pc_file)
    loaded_input = load_input_points_and_normals(
        input_path,
        sample_seed=run_seeds.input_sampling,
    )
    sampled_points = loaded_input.points
    normals = loaded_input.normals
    gt_inlier_mask = loaded_input.gt_inlier_mask
    algorithm_noise_std = float(loaded_input.noise_std)
    cloud_stats = summarize_point_cloud(sampled_points)
    threshold = get_threshold(
        algorithm_noise_std,
        median_nn_distance=cloud_stats.median_nn_distance,
    )
    ransac_tuning = tune_ransac_hyperparameters(cloud_stats)

    print(
        f"Loaded {sampled_points.shape[0]} points from {input_path.name} "
        f"(input_kind={loaded_input.kind})"
    )
    print(
        "Seeds | "
        f"base={run_seeds.base} "
        f"input_sampling={run_seeds.input_sampling} "
        f"algorithm={run_seeds.algorithm} "
        f"evaluation_sampling={run_seeds.evaluation_sampling}"
    )
    if loaded_input.kind == "mesh":
        synthetic_outliers = 0
        if loaded_input.gt_inlier_mask is not None:
            synthetic_outliers = int(
                len(loaded_input.gt_inlier_mask) - int(loaded_input.gt_inlier_mask.sum())
            )
        print(
            "Sampling | "
            f"target_sample_count={SAMPLED_POINT_COUNT} "
            f"actual_point_count={sampled_points.shape[0]} "
            f"synthetic_outliers={synthetic_outliers} "
            f"outlier_mode={OUTLIER_MODE} "
            f"outlier_margin={OUTLIER_MARGIN:.2f}"
        )
    else:
        print(f"Sampling | point_cloud_count={sampled_points.shape[0]} (used directly)")
    print(
        "Point-cloud stats | "
        f"extent={cloud_stats.spatial_extent:.4f} "
        f"median_nn={cloud_stats.median_nn_distance:.4f}"
    )
    print(
        "Threshold | "
        f"threshold={threshold:.4f} "
        f"noise_std={algorithm_noise_std:.4f} "
        f"threshold_factor={THRESHOLD_FACTOR:.4f} "
        f"spacing_factor={THRESHOLD_SPACING_FACTOR:.4f} "
        f"source={loaded_input.noise_std_source}"
    )
    print(
        "RANSAC tuning | "
        f"sample_size={ransac_tuning.sample_size} "
        f"min_inliers={ransac_tuning.min_inliers} "
        f"mss_max_pool_fraction={ransac_tuning.mss_max_pool_fraction:.2f} "
        f"min_coverage={MIN_COVERAGE:.2f}"
    )
    """
    if visualize:
        vis.show_point_cloud(
            sampled_points,
            point_size=5,
            camera_position=PYVISTA_CAMERA_POSITION,
            print_camera_on_close=True,
        )
    """
    gt_n_inliers = None
    gt_n_outliers = None
    gt_classification_rate = None
    gt_misclassification_error = None
    if gt_inlier_mask is not None:
        gt_n_inliers = int(gt_inlier_mask.sum())
        gt_n_outliers = int(len(gt_inlier_mask) - gt_n_inliers)
        gt_suffix = ""
        if loaded_input.gt_inliers_assumed_from_tail:
            gt_suffix = " | assumption=outliers appended at tail"
        print(
            "Ground truth | "
            f"inliers={gt_n_inliers} "
            f"outliers={gt_n_outliers} "
            f"outlier_ratio={loaded_input.gt_outlier_ratio:.2%} "
            f"noise_std={loaded_input.gt_noise_std:.4f}"
            f"{gt_suffix}"
        )
    elif loaded_input.kind == "point-cloud":
        print(
            "Ground truth | unavailable "
            f"for {input_path.name} "
            "(expected suffix '_<noise_std>_<outlier_ratio>.ply')"
        )

    list_mesh = []
    colors = []
    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    n_gt = 0  # no ground truth meshes
    total_best_mss_used = None
    max_models = MAX_MODELS
    models = []
    inliers_masks = []
    algorithm_t0 = time.perf_counter()
    print(f"Running algorithm '{algorithm}'...")
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
            threshold=threshold,
            random_seed=run_seeds.algorithm,
            sample_size=ransac_tuning.sample_size,
        )
        models = [theta0.best_model]
        list_mesh.append(supmesh.superquadric_mesh(theta0.best_model))
        colors.append("lightgreen")
    elif algorithm == "ransac":
        graph_radius = 0.06
        models, inliers_masks = ransac(
            sampled_points,
            threshold=threshold,
            max_models=max_models,
            max_iterations=40,
            inner_iterations=150,
            radius=graph_radius,
            graphcut=True,
            random_seed=run_seeds.algorithm,
            sample_size=ransac_tuning.sample_size,
            min_inliers=ransac_tuning.min_inliers,
            use_normal_guided_mss=True,
            mss_max_pool_fraction=ransac_tuning.mss_max_pool_fraction,
        )
        if not models:
            raise RuntimeError("ransac did not return any model")
        for i, model in enumerate(models):
            list_mesh.append(supmesh.superquadric_mesh(model))
            colors.append(palette[i % len(palette)])
    elif algorithm in ("gair-ransac", "gc-ransac", "gair-mss-no-normals", "gc-mss-no-normals"):
        graph_radius = 0.06
        use_normal_guided_mss = algorithm in ("gair-ransac", "gc-ransac")
        use_normal_coherence = algorithm in ("gair-ransac", "gair-mss-no-normals")
        models, inliers_masks, total_best_mss_used = gair_ransac(
            sampled_points,
            normals,
            threshold=threshold,
            max_models=max_models,
            max_iterations=3,
            inner_iterations=100,
            radius=graph_radius,
            use_normal_coherence=use_normal_coherence,
            min_coverage=MIN_COVERAGE,
            random_seed=run_seeds.algorithm,
            use_normal_guided_mss=use_normal_guided_mss,
            sample_size=ransac_tuning.sample_size,
            min_inliers=ransac_tuning.min_inliers,
            mss_max_pool_fraction=ransac_tuning.mss_max_pool_fraction,
            save_debug_views=save_debug_views,
            m_neighbors=M_NEIGHBORS
        )
        if not models:
            print("gair_ransac did not return any model")
        for i, model in enumerate(models):
            list_mesh.append(supmesh.superquadric_mesh(model))
            colors.append(palette[i % len(palette)])
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    runtime_s = time.perf_counter() - algorithm_t0

    if algorithm == "inner-ransac":
        inlier_mask = theta0.best_inliers_mask
    elif algorithm in ("ransac", "gair-ransac", "gc-ransac", "gair-mss-no-normals", "gc-mss-no-normals"):
        inlier_mask = None
        if inliers_masks:
            inlier_mask = inliers_masks[0].copy()
            for mask in inliers_masks[1:]:
                inlier_mask |= mask
    else:
        inlier_mask = None

    n_inliers = None
    n_outliers = None
    outlier_ratio = None
    if inlier_mask is not None:
        n_inliers = int(inlier_mask.sum())
        n_outliers = int(len(inlier_mask) - n_inliers)
        outlier_ratio = float(n_outliers / len(inlier_mask))
        print(f"Inliers: {n_inliers} | Outliers: {n_outliers} | Total: {len(inlier_mask)} | Outlier ratio: {n_outliers/len(inlier_mask):.2%}")
        if gt_inlier_mask is not None:
            if len(gt_inlier_mask) != len(inlier_mask):
                raise ValueError(
                    f"Predicted/ground-truth mask length mismatch: "
                    f"predicted={len(inlier_mask)} gt={len(gt_inlier_mask)}"
                )
            gt_classification_rate = float(np.mean(inlier_mask == gt_inlier_mask))
            gt_misclassification_error = float(1.0 - gt_classification_rate)
            print(
                "Against ground truth | "
                f"classification_rate={gt_classification_rate:.2%} "
                f"misclassification_error={gt_misclassification_error:.4f}"
            )
    elif gt_inlier_mask is not None:
        print(
            "Against ground truth | unavailable "
            f"because algorithm '{algorithm}' does not produce an inlier mask"
        )

    cd = None
    one_sided_cd = None
    saved_object_view_paths: list[Path] = []
    if list_mesh:
        sampled_estimated, _ = samp.sampling_sq(
            list_mesh,
            n_points=SAMPLED_POINT_COUNT,
            seed=run_seeds.evaluation_sampling,
        )
        sample_from_supq_estimated = np.vstack(sampled_estimated)
        cd = chamfer_distance(sampled_points, sample_from_supq_estimated)
        print(f"reconstruction chamfer = {cd:.4f}")
        one_sided_cd = float(cKDTree(sample_from_supq_estimated).query(sampled_points, k=1)[0].mean())
        print(f"reconstruction one-sided chamfer = {one_sided_cd:.4f}")
        output_stem = f"{input_path.stem}_{algorithm}_seed{run_seeds.base}"
        saved_object_view_paths = vis.save_mesh_view_triplet(
            list_mesh,
            output_dir=OBJECT_VIEWS_DIR,
            output_stem=output_stem,
            pts=sampled_points,
            point_size=5,
            colors=colors,
            inlier_mask=inlier_mask,
            mss_used=(
                total_best_mss_used
                if algorithm in ("gair-ransac", "gc-ransac", "gair-mss-no-normals", "gc-mss-no-normals")
                else None
            ),
            models=models,
            treshold=threshold,
        )
        print(
            f"Saved {len(saved_object_view_paths)} fitted-object views in "
            f"{saved_object_view_paths[0].parent}"
        )

    if visualize:
        vis.show_mesh_and_points(
            list_mesh,
            pts=sampled_points,
            point_size=5,
            show_bounds=True,
            colors=colors,
            inlier_mask=inlier_mask,
            mss_used=total_best_mss_used if algorithm in ("gair-ransac", "gc-ransac", "gair-mss-no-normals", "gc-mss-no-normals") else None,
            models=models,
            treshold=threshold,
            print_camera_on_close=True,
            camera_position=PYVISTA_CAMERA_POSITION,
        )

    result = {
        "input_mesh": str(input_path),
        "algorithm": algorithm,
        "base_seed": int(run_seeds.base),
        "input_sampling_seed": int(run_seeds.input_sampling),
        "algorithm_seed": int(run_seeds.algorithm),
        "evaluation_sampling_seed": int(run_seeds.evaluation_sampling),
        "sampled_point_count": int(sampled_points.shape[0]),
        "n_models": int(len(models)),
        "runtime_s": float(runtime_s),
        "threshold": float(threshold),
        "input_noise_std": float(algorithm_noise_std),
        "configured_n_outliers": int(N_OUTLIERS),
        "outlier_sampling_mode": OUTLIER_MODE,
        "outlier_sampling_margin": float(OUTLIER_MARGIN),
        "chamfer": None if cd is None else float(cd),
        "one_sided_chamfer": None if one_sided_cd is None else float(one_sided_cd),
        "saved_object_views_dir": str(OBJECT_VIEWS_DIR) if saved_object_view_paths else None,
        "saved_object_views_count": int(len(saved_object_view_paths)),
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "outlier_ratio": outlier_ratio,
        "gt_n_inliers": gt_n_inliers,
        "gt_n_outliers": gt_n_outliers,
        "gt_outlier_ratio": None if loaded_input.gt_outlier_ratio is None else float(loaded_input.gt_outlier_ratio),
        "gt_noise_std": None if loaded_input.gt_noise_std is None else float(loaded_input.gt_noise_std),
        "gt_classification_rate": None if gt_classification_rate is None else float(gt_classification_rate),
        "gt_misclassification_error": None if gt_misclassification_error is None else float(gt_misclassification_error),
        "gt_inliers_assumed_from_tail": bool(loaded_input.gt_inliers_assumed_from_tail),
    }

    if return_artifacts:
        result.update(
            {
                "models": models,
                "sampled_points": sampled_points,
                "normals": normals,
                "inliers_masks": inliers_masks,
            }
        )

    return result

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", nargs="?", default=PC_FILE)
    parser.add_argument(
        "--algorithm",
        default="gair-ransac",
        help="Algorithm to run. Accepted aliases: gair, gc, lo, plus underscore variants.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip the PyVista visualization window.",
    )
    parser.add_argument(
        "--save-debug-views",
        action="store_true",
        help=(
            "Save intermediate PyVista debug images during GAIR-RANSAC. "
            "Disabled by default because off-screen rendering can emit noisy "
            "macOS CoreAnalytics warnings."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    result = create_and_estimate_supq(
        args.input_mesh,
        algorithm=args.algorithm,
        base_seed=args.seed,
        visualize=not args.no_visualize,
        save_debug_views=args.save_debug_views,
    )
    summary_parts = [
        "Summary |",
        f"algorithm={result['algorithm']}",
        f"seed={result['base_seed']}",
        f"models={result['n_models']}",
        f"runtime_s={result['runtime_s']:.3f}",
        f"chamfer={result['chamfer']:.4f}",
        f"one_sided_chamfer={result['one_sided_chamfer']:.4f}",
    ]
    if result["gt_misclassification_error"] is not None:
        summary_parts.append(f"gt_misclassification={result['gt_misclassification_error']:.4f}")
        summary_parts.append(f"gt_classification_rate={result['gt_classification_rate']:.4f}")
    print(" ".join(summary_parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
