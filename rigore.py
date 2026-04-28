import argparse
import ast
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from point_cloud_utils import chamfer_distance
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = ROOT / "experiments" / "artifacts" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.ransacov import ransacov
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.superquadrics import superquadric_sampling as samp
from src.superquadrics.superquadric_param import SuperQuadricParams


NOISE_VALUES = (0.1, 0.2, 0.4)
N_OUTLIERS = 5000
SURFACE_POINTS_PER_SUPERQUADRIC = 6000
ESTIMATED_POINTS_PER_MODEL = 6000
THRESHOLD_SCALE = 2.5
GRAPH_RADIUS = 0.08
MAX_ITERATIONS = 10
INNER_ITERATIONS = 50
TRIALS = 10
MAX_PROCESSES = 8
BASE_SEED = 42
EVAL_SEED = 50
OUT_DIR = ROOT / "experiments" / "artifacts" / "rigore"

ALGORITHMS = {
    "gc-ransac": "GC-RANSAC",
    "gair-ransac": "GAIR-RANSAC (ours)",
}

RAW_FIELDNAMES = [
    "sigma",
    "n_outliers",
    "algorithm",
    "algorithm_key",
    "trial",
    "seed",
    "threshold",
    "status",
    "chamfer",
    "hausdorff",
    "accuracy",
    "misclassification_error",
    "runtime_s",
    "n_models",
    "point_count",
    "gt_inliers",
    "gt_outliers",
    "predicted_inliers",
]

SUMMARY_FIELDNAMES = [
    "sigma",
    "n_outliers",
    "algorithm",
    "algorithm_key",
    "trials",
    "chamfer_mean",
    "chamfer_std",
    "hausdorff_mean",
    "hausdorff_std",
    "accuracy_mean",
    "accuracy_std",
    "misclassification_error_mean",
    "misclassification_error_std",
    "runtime_s_mean",
    "runtime_s_std",
]

RANSACOV_FIELDNAMES = [
    "sigma",
    "n_outliers",
    "algorithm",
    "algorithm_key",
    "selection",
    "source_trials",
    "threshold",
    "status",
    "n_candidates",
    "ransacov_k",
    "selected_count",
    "selected_candidate_indices",
    "selected_origins",
    "n_covered",
    "coverage_ratio",
    "chamfer",
    "hausdorff",
    "accuracy",
    "misclassification_error",
    "runtime_s",
    "point_count",
    "gt_inliers",
    "gt_outliers",
    "predicted_inliers",
]


@dataclass(frozen=True)
class SyntheticScene:
    points: np.ndarray
    normals: np.ndarray
    surface_points: np.ndarray
    gt_inlier_mask: np.ndarray


@dataclass(frozen=True)
class TrialJob:
    sigma: float
    trial: int
    base_seed: int
    gt_params: tuple[SuperQuadricParams, ...]
    max_models: int
    max_iterations: int
    inner_iterations: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic table data for GC-RANSAC and GAIR-RANSAC, then run "
            "RansaCov over the trial candidates for each noise level and algorithm."
        )
    )
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--noise-values", type=float, nargs="+", default=list(NOISE_VALUES))
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--inner-iterations", type=int, default=INNER_ITERATIONS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--max-processes", type=int, default=MAX_PROCESSES)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def _find_gt_params_node(tree: ast.Module) -> ast.List:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "create_and_estimate_supq":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "gt_params" for target in child.targets):
                if not isinstance(child.value, ast.List):
                    raise ValueError("main.py: gt_params must be a list")
                return child.value
    raise ValueError("Could not find gt_params inside create_and_estimate_supq() in main.py")


def load_gt_params_from_main(main_path: Path = ROOT / "main.py") -> list[SuperQuadricParams]:
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    gt_params_node = _find_gt_params_node(tree)
    gt_params: list[SuperQuadricParams] = []

    for item in gt_params_node.elts:
        if not isinstance(item, ast.Call):
            raise ValueError("main.py: every gt_params entry must be a SuperQuadricParams(...) call")

        call_name = item.func.id if isinstance(item.func, ast.Name) else None
        if call_name != "SuperQuadricParams":
            raise ValueError(f"main.py: unsupported gt_params entry {ast.unparse(item.func)!r}")

        args = [ast.literal_eval(arg) for arg in item.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in item.keywords if kw.arg is not None}
        gt_params.append(SuperQuadricParams(*args, **kwargs))

    if not gt_params:
        raise ValueError("main.py: gt_params is empty")
    return gt_params


def build_scene(gt_meshes, sigma: float, seed: int) -> SyntheticScene:
    sampled_points_noisy, normals_noisy = samp.sampling_sq_noisy(
        gt_meshes,
        n_points=SURFACE_POINTS_PER_SUPERQUADRIC,
        noise_std=sigma,
        clip_k=3.0,
        seed=seed,
    )
    sampled_points_outliers, normals_outliers = samp.sampling_outliers(
        gt_meshes,
        n_out=N_OUTLIERS,
        margin=0.10,
        mode="uniform",
        seed=seed,
    )

    surface_points = np.vstack(sampled_points_noisy)
    surface_normals = np.vstack(normals_noisy)
    points = np.vstack([surface_points, sampled_points_outliers])
    normals = np.vstack([surface_normals, normals_outliers])

    gt_inlier_mask = np.zeros(points.shape[0], dtype=bool)
    gt_inlier_mask[: surface_points.shape[0]] = True

    return SyntheticScene(
        points=points,
        normals=normals,
        surface_points=surface_points,
        gt_inlier_mask=gt_inlier_mask,
    )


def merge_inlier_masks(inlier_masks: list[np.ndarray], n_points: int) -> np.ndarray:
    merged = np.zeros(n_points, dtype=bool)
    for mask in inlier_masks:
        merged |= np.asarray(mask, dtype=bool)
    return merged


def evaluate_models(models, target_points: np.ndarray) -> tuple[float, float]:
    if not models:
        return float("nan"), float("nan")

    estimated_meshes = [supmesh.superquadric_mesh(model) for model in models]
    sampled_estimated, _ = samp.sampling_sq_random(
        estimated_meshes,
        n_points=ESTIMATED_POINTS_PER_MODEL,
        seed=EVAL_SEED,
    )
    estimated_points = np.vstack(sampled_estimated)

    cd = float(chamfer_distance(target_points, estimated_points))

    tree_est = cKDTree(estimated_points)
    tree_gt = cKDTree(target_points)
    d_gt_to_est = tree_est.query(target_points, k=1)[0]
    d_est_to_gt = tree_gt.query(estimated_points, k=1)[0]
    hd = float(max(d_gt_to_est.max(), d_est_to_gt.max()))

    return cd, hd


def unsigned_radial_residual(model: SuperQuadricParams, points: np.ndarray) -> np.ndarray:
    return np.abs(superquadric_radial_residual(model, points))


def model_union_inliers(
    models: list[SuperQuadricParams],
    points: np.ndarray,
    threshold: float,
    threshold_multiplier: float = 1.0,
) -> np.ndarray:
    union = np.zeros(points.shape[0], dtype=bool)
    effective_threshold = threshold_multiplier * threshold
    for model in models:
        union |= unsigned_radial_residual(model, points) < effective_threshold
    return union


def format_origin(origin: dict[str, int]) -> str:
    return (
        f"trial={origin['trial']},"
        f"seed={origin['seed']},"
        f"model={origin['model_idx']}"
    )


def classification_ratio(predicted_inliers: np.ndarray, gt_inlier_mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(predicted_inliers, dtype=bool) == np.asarray(gt_inlier_mask, dtype=bool)))


def run_algorithm(
    algorithm_key: str,
    scene: SyntheticScene,
    sigma: float,
    seed: int,
    max_models: int,
    max_iterations: int,
    inner_iterations: int,
) -> dict[str, object]:
    threshold = THRESHOLD_SCALE * sigma
    start = time.perf_counter()

    if algorithm_key == "gc-ransac":
        use_normal_coherence = False
    elif algorithm_key == "gair-ransac":
        use_normal_coherence = True
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_key}")

    models, inlier_masks, _ = gair_ransac(
        scene.points,
        scene.normals,
        threshold=threshold,
        max_models=max_models,
        max_iterations=max_iterations,
        inner_iterations=inner_iterations,
        radius=GRAPH_RADIUS,
        use_normal_coherence=use_normal_coherence,
        m_neighbors=4,
        random_seed=seed,
    )

    runtime = time.perf_counter() - start
    predicted_inliers = merge_inlier_masks(inlier_masks, scene.points.shape[0])
    misclassification_error = classification_ratio(predicted_inliers, scene.gt_inlier_mask)
    chamfer, hausdorff = evaluate_models(models, scene.surface_points)

    return {
        "status": "ok" if models else "no_models",
        "chamfer": chamfer,
        "hausdorff": hausdorff,
        # In this experiment the requested "misclassification_error" is the
        # correctly classified ratio: correct / total.
        "accuracy": misclassification_error,
        "misclassification_error": misclassification_error,
        "runtime_s": runtime,
        "n_models": len(models),
        "predicted_inliers": int(predicted_inliers.sum()),
        "models": models,
    }


def run_trial_job(job: TrialJob) -> dict[str, object]:
    gt_meshes = [supmesh.superquadric_mesh(params) for params in job.gt_params]
    scene_seed = job.base_seed + job.trial * 1000 + int(round(job.sigma * 10000))
    scene = build_scene(gt_meshes, sigma=job.sigma, seed=scene_seed)
    threshold = THRESHOLD_SCALE * job.sigma
    gt_outliers = int(np.count_nonzero(~scene.gt_inlier_mask))
    gt_inliers = int(np.count_nonzero(scene.gt_inlier_mask))

    rows: list[dict[str, object]] = []
    candidate_bank: dict[str, list[SuperQuadricParams]] = {
        algorithm_key: [] for algorithm_key in ALGORITHMS
    }
    origin_bank: dict[str, list[dict[str, int]]] = {
        algorithm_key: [] for algorithm_key in ALGORITHMS
    }

    for algorithm_index, (algorithm_key, algorithm_name) in enumerate(ALGORITHMS.items()):
        run_seed = scene_seed + 100 * (algorithm_index + 1)
        result = run_algorithm(
            algorithm_key,
            scene,
            sigma=job.sigma,
            seed=run_seed,
            max_models=job.max_models,
            max_iterations=job.max_iterations,
            inner_iterations=job.inner_iterations,
        )
        models = list(result.pop("models"))
        for model_idx, model in enumerate(models):
            candidate_bank[algorithm_key].append(model)
            origin_bank[algorithm_key].append(
                {
                    "trial": int(job.trial),
                    "seed": int(run_seed),
                    "model_idx": int(model_idx),
                }
            )

        rows.append(
            {
                "sigma": job.sigma,
                "n_outliers": N_OUTLIERS,
                "algorithm": algorithm_name,
                "algorithm_key": algorithm_key,
                "trial": job.trial,
                "seed": run_seed,
                "threshold": threshold,
                "point_count": int(scene.points.shape[0]),
                "gt_inliers": gt_inliers,
                "gt_outliers": gt_outliers,
                **result,
            }
        )

    return {
        "trial": job.trial,
        "rows": rows,
        "candidate_bank": candidate_bank,
        "origin_bank": origin_bank,
        "selection_scene": scene if job.trial == 0 else None,
    }


def run_ransacov_selection(
    algorithm_key: str,
    algorithm_name: str,
    scene: SyntheticScene,
    sigma: float,
    candidates: list[SuperQuadricParams],
    origins: list[dict[str, int]],
    max_models: int,
) -> dict[str, object]:
    threshold = THRESHOLD_SCALE * sigma
    gt_outliers = int(np.count_nonzero(~scene.gt_inlier_mask))
    gt_inliers = int(np.count_nonzero(scene.gt_inlier_mask))
    source_trials = len({origin["trial"] for origin in origins})
    start = time.perf_counter()

    if not candidates:
        return {
            "sigma": sigma,
            "n_outliers": N_OUTLIERS,
            "algorithm": algorithm_name,
            "algorithm_key": algorithm_key,
            "selection": "ransacov",
            "source_trials": source_trials,
            "threshold": threshold,
            "status": "no_candidates",
            "n_candidates": 0,
            "ransacov_k": max_models,
            "selected_count": 0,
            "selected_candidate_indices": "",
            "selected_origins": "",
            "n_covered": 0,
            "coverage_ratio": 0.0,
            "chamfer": float("nan"),
            "hausdorff": float("nan"),
            "accuracy": float("nan"),
            "misclassification_error": float("nan"),
            "runtime_s": time.perf_counter() - start,
            "point_count": int(scene.points.shape[0]),
            "gt_inliers": gt_inliers,
            "gt_outliers": gt_outliers,
            "predicted_inliers": 0,
        }

    selected_indices, n_covered = ransacov(
        candidates,
        scene.points,
        k=min(max_models, len(candidates)),
        threshold=threshold,
        residual_fn=unsigned_radial_residual,
    )
    selected_models = [candidates[index] for index in selected_indices]
    runtime = time.perf_counter() - start

    chamfer, hausdorff = evaluate_models(selected_models, scene.surface_points)
    # src.gair_ransac.ransacov currently applies a 2x threshold internally.
    predicted_inliers = model_union_inliers(
        selected_models,
        scene.points,
        threshold=threshold,
        threshold_multiplier=2.0,
    )
    misclassification_error = classification_ratio(predicted_inliers, scene.gt_inlier_mask)
    selected_origin_text = ";".join(format_origin(origins[index]) for index in selected_indices)

    return {
        "sigma": sigma,
        "n_outliers": N_OUTLIERS,
        "algorithm": algorithm_name,
        "algorithm_key": algorithm_key,
        "selection": "ransacov",
        "source_trials": source_trials,
        "threshold": threshold,
        "status": "ok" if selected_models else "no_selection",
        "n_candidates": len(candidates),
        "ransacov_k": min(max_models, len(candidates)),
        "selected_count": len(selected_models),
        "selected_candidate_indices": ",".join(str(index) for index in selected_indices),
        "selected_origins": selected_origin_text,
        "n_covered": int(n_covered),
        "coverage_ratio": float(n_covered / scene.points.shape[0]),
        "chamfer": chamfer,
        "hausdorff": hausdorff,
        "accuracy": misclassification_error,
        "misclassification_error": misclassification_error,
        "runtime_s": runtime,
        "point_count": int(scene.points.shape[0]),
        "gt_inliers": gt_inliers,
        "gt_outliers": gt_outliers,
        "predicted_inliers": int(predicted_inliers.sum()),
    }


def write_header(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def finite_values(rows: list[dict[str, object]], metric: str) -> np.ndarray:
    values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def mean_std(rows: list[dict[str, object]], metric: str) -> tuple[float, float]:
    values = finite_values(rows, metric)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    noise_values = sorted({float(row["sigma"]) for row in rows})
    for sigma in noise_values:
        for algorithm_key, algorithm_name in ALGORITHMS.items():
            group = [
                row
                for row in rows
                if np.isclose(float(row["sigma"]), sigma) and row["algorithm_key"] == algorithm_key
            ]
            if not group:
                continue

            chamfer_mean, chamfer_std = mean_std(group, "chamfer")
            hausdorff_mean, hausdorff_std = mean_std(group, "hausdorff")
            accuracy_mean, accuracy_std = mean_std(group, "accuracy")
            runtime_mean, runtime_std = mean_std(group, "runtime_s")
            summaries.append(
                {
                    "sigma": sigma,
                    "n_outliers": N_OUTLIERS,
                    "algorithm": algorithm_name,
                    "algorithm_key": algorithm_key,
                    "trials": len(group),
                    "chamfer_mean": chamfer_mean,
                    "chamfer_std": chamfer_std,
                    "hausdorff_mean": hausdorff_mean,
                    "hausdorff_std": hausdorff_std,
                    "accuracy_mean": accuracy_mean,
                    "accuracy_std": accuracy_std,
                    "misclassification_error_mean": accuracy_mean,
                    "misclassification_error_std": accuracy_std,
                    "runtime_s_mean": runtime_mean,
                    "runtime_s_std": runtime_std,
                }
            )
    return summaries


def format_pm(mean_value: float, std_value: float) -> str:
    if not np.isfinite(mean_value) or not np.isfinite(std_value):
        return "nan"
    return f"{mean_value:.4f}+/-{std_value:.4f}"


def print_table(summaries: list[dict[str, object]]) -> None:
    print("\nValori per la tabella (media +/- std)")
    print("Nota: accuracy e misclassification_error contengono corretti/totale, quindi piu alto e meglio.")
    print(f"{'(sigma, Nout)':<16} {'Algorithm':<22} {'CD':>18} {'HD':>18} {'Acc/Mis':>18}")
    print("-" * 94)
    for row in summaries:
        key = f"({float(row['sigma']):g}, {int(row['n_outliers'])})"
        cd = format_pm(float(row["chamfer_mean"]), float(row["chamfer_std"]))
        hd = format_pm(float(row["hausdorff_mean"]), float(row["hausdorff_std"]))
        acc = format_pm(float(row["accuracy_mean"]), float(row["accuracy_std"]))
        print(f"{key:<16} {str(row['algorithm']):<22} {cd:>18} {hd:>18} {acc:>18}")


def print_ransacov_table(rows: list[dict[str, object]]) -> None:
    print("\nRisultati RansaCov sulle iterazioni raccolte per ogni noise/algoritmo")
    print(f"{'(sigma, Nout)':<16} {'Algorithm':<24} {'Selection':<10} {'Cand':>6} {'Sel':>4} {'CD':>10} {'HD':>10} {'Acc/Mis':>10}")
    print("-" * 94)
    for row in rows:
        key = f"({float(row['sigma']):g}, {int(row['n_outliers'])})"
        cd = float(row["chamfer"])
        hd = float(row["hausdorff"])
        acc = float(row["accuracy"])
        cd_text = "nan" if not np.isfinite(cd) else f"{cd:.4f}"
        hd_text = "nan" if not np.isfinite(hd) else f"{hd:.4f}"
        acc_text = "nan" if not np.isfinite(acc) else f"{acc:.4f}"
        print(
            f"{key:<16} {str(row['algorithm']):<24} {str(row['selection']):<10} "
            f"{int(row['n_candidates']):>6} {int(row['selected_count']):>4} "
            f"{cd_text:>10} {hd_text:>10} {acc_text:>10}"
        )


def resolve_max_processes(requested: int, n_trials: int) -> int:
    if requested <= 0:
        raise ValueError(f"--max-processes must be positive, got {requested}")
    return max(1, min(int(requested), int(n_trials)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trials <= 0:
        raise ValueError(f"--trials must be positive, got {args.trials}")
    if args.max_iterations <= 0:
        raise ValueError(f"--max-iterations must be positive, got {args.max_iterations}")
    if args.inner_iterations <= 0:
        raise ValueError(f"--inner-iterations must be positive, got {args.inner_iterations}")
    max_processes = resolve_max_processes(args.max_processes, args.trials)

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "results.csv"
    summary_csv = output_dir / "summary.csv"
    table_txt = output_dir / "table.txt"
    ransacov_csv = output_dir / "ransacov_results.csv"
    ransacov_table_txt = output_dir / "ransacov_table.txt"

    gt_params = load_gt_params_from_main()
    gt_meshes = [supmesh.superquadric_mesh(params) for params in gt_params]
    max_models = len(gt_meshes)
    gt_params_tuple = tuple(gt_params)

    write_header(raw_csv, RAW_FIELDNAMES)
    write_header(ransacov_csv, RANSACOV_FIELDNAMES)
    rows: list[dict[str, object]] = []
    ransacov_rows: list[dict[str, object]] = []

    print(f"Rigore: {len(gt_meshes)} gt_params letti da main.py")
    print(
        f"Sampling come main.py: {SURFACE_POINTS_PER_SUPERQUADRIC} punti noisy per SQ, "
        f"{N_OUTLIERS} outlier fissi. CD/HD usano gli stessi punti di superficie, senza outlier."
    )
    print(f"Trials per noise/algoritmo: {args.trials}")
    print(f"Processi massimi per i trial: {max_processes}")
    print(f"CSV raw: {raw_csv}")
    print(f"CSV RansaCov: {ransacov_csv}")

    for sigma in args.noise_values:
        sigma = float(sigma)
        selection_scene: SyntheticScene | None = None
        candidate_bank: dict[str, list[SuperQuadricParams]] = {
            algorithm_key: [] for algorithm_key in ALGORITHMS
        }
        origin_bank: dict[str, list[dict[str, int]]] = {
            algorithm_key: [] for algorithm_key in ALGORITHMS
        }

        print(
            f"\nNoise sigma={sigma:g}: lancio {args.trials} trial "
            f"con max {max_processes} processi.",
            flush=True,
        )
        jobs = [
            TrialJob(
                sigma=sigma,
                trial=trial,
                base_seed=args.base_seed,
                gt_params=gt_params_tuple,
                max_models=max_models,
                max_iterations=args.max_iterations,
                inner_iterations=args.inner_iterations,
            )
            for trial in range(args.trials)
        ]
        trial_results: dict[int, dict[str, object]] = {}
        if max_processes == 1:
            for job in jobs:
                trial_results[job.trial] = run_trial_job(job)
                print(f"  completato trial {job.trial + 1}/{args.trials}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=max_processes) as executor:
                future_to_trial = {
                    executor.submit(run_trial_job, job): job.trial
                    for job in jobs
                }
                for future in as_completed(future_to_trial):
                    trial = future_to_trial[future]
                    trial_results[trial] = future.result()
                    print(f"  completato trial {trial + 1}/{args.trials}", flush=True)

        for trial in range(args.trials):
            result = trial_results[trial]
            if selection_scene is None and result["selection_scene"] is not None:
                selection_scene = result["selection_scene"]

            for algorithm_key in ALGORITHMS:
                candidate_bank[algorithm_key].extend(result["candidate_bank"][algorithm_key])
                origin_bank[algorithm_key].extend(result["origin_bank"][algorithm_key])

            for row in result["rows"]:
                rows.append(row)
                append_row(raw_csv, RAW_FIELDNAMES, row)
                print(
                    f"Fine  | sigma={float(row['sigma']):g} trial={int(row['trial']) + 1}/{args.trials} "
                    f"{row['algorithm']} | "
                    f"CD={float(row['chamfer']):.4f} "
                    f"HD={float(row['hausdorff']):.4f} "
                    f"Acc/Mis={float(row['accuracy']):.4f} "
                    f"runtime={float(row['runtime_s']):.2f}s "
                    f"models={int(row['n_models'])} "
                    f"status={row['status']}",
                    flush=True,
                )

        if selection_scene is None:
            continue

        print(f"\nRansaCov | sigma={sigma:g} sui candidati raccolti dalle {args.trials} iterazioni")
        for algorithm_key, algorithm_name in ALGORITHMS.items():
            ransacov_row = run_ransacov_selection(
                algorithm_key=algorithm_key,
                algorithm_name=algorithm_name,
                scene=selection_scene,
                sigma=sigma,
                candidates=candidate_bank[algorithm_key],
                origins=origin_bank[algorithm_key],
                max_models=max_models,
            )
            ransacov_rows.append(ransacov_row)
            append_row(ransacov_csv, RANSACOV_FIELDNAMES, ransacov_row)
            print(
                f"  {ransacov_row['algorithm']} | selection={ransacov_row['selection']} "
                f"candidates={ransacov_row['n_candidates']} "
                f"selected={ransacov_row['selected_count']} "
                f"covered={ransacov_row['n_covered']}/{ransacov_row['point_count']} "
                f"CD={float(ransacov_row['chamfer']):.4f} "
                f"HD={float(ransacov_row['hausdorff']):.4f} "
                f"Acc/Mis={float(ransacov_row['accuracy']):.4f} "
                f"status={ransacov_row['status']}",
                flush=True,
            )

    summaries = summarize_rows(rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summaries)

    table_lines: list[str] = []
    for row in summaries:
        table_lines.append(
            "\t".join(
                [
                    f"({float(row['sigma']):g}, {int(row['n_outliers'])})",
                    str(row["algorithm"]),
                    format_pm(float(row["chamfer_mean"]), float(row["chamfer_std"])),
                    format_pm(float(row["hausdorff_mean"]), float(row["hausdorff_std"])),
                    format_pm(float(row["accuracy_mean"]), float(row["accuracy_std"])),
                ]
            )
        )
    table_txt.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    ransacov_table_lines: list[str] = []
    for row in ransacov_rows:
        ransacov_table_lines.append(
            "\t".join(
                [
                    f"({float(row['sigma']):g}, {int(row['n_outliers'])})",
                    str(row["algorithm"]),
                    str(row["selection"]),
                    str(row["n_candidates"]),
                    str(row["selected_count"]),
                    str(row["n_covered"]),
                    f"{float(row['chamfer']):.6f}",
                    f"{float(row['hausdorff']):.6f}",
                    f"{float(row['accuracy']):.6f}",
                ]
            )
        )
    ransacov_table_txt.write_text("\n".join(ransacov_table_lines) + "\n", encoding="utf-8")

    print_table(summaries)
    print_ransacov_table(ransacov_rows)
    print(f"\nSalvati: {raw_csv}")
    print(f"Salvati: {summary_csv}")
    print(f"Salvati: {table_txt}")
    print(f"Salvati: {ransacov_csv}")
    print(f"Salvati: {ransacov_table_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
