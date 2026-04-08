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
THRESHOLD_FACTOR = 2.5
NOISE_STD=0.0
PROJECT_ROOT = Path(__file__).resolve().parent
"""anthropomorphic_mushroom_character.glb"""
PC_FILE = PROJECT_ROOT / "test_objects" / "cat1_0_0.2.ply"
# Single knob for how many points are sampled from the input mesh
# and from the reconstructed superquadrics for evaluation.
SAMPLED_POINT_COUNT = None 
DEFAULT_BASE_SEED = 876534 #12345
SUPPORTED_ALGORITHMS = (
    "ls",
    "inner-ransac",
    "ransac",
    "gair-ransac",
    "gc-ransac",
    "gair-mss-no-normals",
    "gc-mss-no-normals",
)


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


def get_threshold(noise_std: float) -> float:
    return max(float(THRESHOLD_FACTOR) * float(noise_std), 0.3)


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
        seed=seed,
    )
    sampled_points = np.vstack([np.asarray(sampled_points_list[0], dtype=np.float64),np.asarray(noisy_sampled[0], dtype=np.float64),])
    normals = np.vstack([np.asarray(normals_list[0], dtype=np.float64),np.asarray(noisy_normal[0], dtype=np.float64),])
    return sampled_points, normals


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

    points, normals = sample_input_mesh(
        raw,
        n_points=SAMPLED_POINT_COUNT,
        seed=sample_seed,
    )
    return LoadedInput(
        points=points,
        normals=normals,
        kind="mesh",
        noise_std=float(NOISE_STD),
        noise_std_source="global NOISE_STD",
    )


def create_and_estimate_supq(
    pc_file: str | Path = PC_FILE,
    algorithm: str = "gair-ransac",
    base_seed: int = DEFAULT_BASE_SEED,
    visualize: bool = True,
):
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
    threshold = get_threshold(algorithm_noise_std)

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
        print(
            "Sampling | "
            f"target_sample_count={SAMPLED_POINT_COUNT} "
            f"actual_point_count={sampled_points.shape[0]}"
        )
    else:
        print(f"Sampling | point_cloud_count={sampled_points.shape[0]} (used directly)")
    print(
        "Threshold | "
        f"threshold={threshold:.4f} "
        f"noise_std={algorithm_noise_std:.4f} "
        f"threshold_factor={THRESHOLD_FACTOR:.4f} "
        f"source={loaded_input.noise_std_source}"
    )
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
    max_models = 4 # <-- how many superquadrics to find
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
            max_iterations=10,
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
    elif algorithm in ("gair-ransac", "gc-ransac", "gair-mss-no-normals", "gc-mss-no-normals"):
        graph_radius = 0.06
        use_normal_guided_mss = algorithm in ("gair-ransac", "gc-ransac")
        use_normal_coherence = algorithm in ("gair-ransac", "gair-mss-no-normals")
        models, inliers_masks, total_best_mss_used = gair_ransac(
            sampled_points,
            normals,
            threshold=threshold,
            max_models=max_models,
            max_iterations=80,
            inner_iterations=150,
            radius=graph_radius,
            use_normal_coherence=use_normal_coherence,
            min_coverage=0.4,
            random_seed=run_seeds.algorithm,
            use_normal_guided_mss=use_normal_guided_mss,
        )
        if not models:
            raise RuntimeError("gair_ransac did not return any model")
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
            treshold=threshold
        )

    return {
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
        "chamfer": None if cd is None else float(cd),
        "one_sided_chamfer": None if one_sided_cd is None else float(one_sided_cd),
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
