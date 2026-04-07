import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / "experiments" / "artifacts" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from point_cloud_utils import chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.gair_ransac import gair_ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.superquadrics.superquadric_param import SuperQuadricParams

GT_PARAMS = [
    SuperQuadricParams(9.0, 9.0, 9.0, 3.5, 2.09, [2.0, 2.0, 1.0], [5.0, 5.0, 5.0]),
    SuperQuadricParams(3.0, 3.0, 3.0, 0.5, 0.9, [2.0, 2.0, 1.0], [-5.0, -5.0, -5.0]),
    SuperQuadricParams(4.0, 4.0, 4.0, 0.8, 1.1, [1.7, 2.1, 0.9], [13.0, 5.0, 5.0]),
    SuperQuadricParams(2.5, 2.5, 2.5, 0.7, 1.2, [2.1, 1.9, 0.8], [-10.5, -5.0, -5.0]),
]

# ── config ────────────────────────────────────────────────────────────────────
VISUALIZE  = False   # set to False to skip visualization and run all trials headlessly
GRAPH_RADIUS = 0.06
MAX_MODELS = len(GT_PARAMS)
MAX_ITER   = 5
INNER_ITER = 100
N_TRIALS   = 10
K = 5
NOISE_STD = 0.2
THRESHOLD  = 2.5 * NOISE_STD
SAMPLED_POINT_COUNT = 10
NOISY_POINTS_PER_MESH = 10000
REFERENCE_POINTS_PER_MESH = 10
OUTLIER_COUNT = 2500
OUTLIER_MARGIN = 0.10
INPUT_SAMPLING_SEED = 42
EVAL_SEED  = 42          # fixed so chamfer is comparable across trials
OUT_DIR    = ROOT / "experiments" / "artifacts" / "final_experiment"
# ──────────────────────────────────────────────────────────────────────────────

_WORKER_POINTS = None
_WORKER_NORMALS = None
_WORKER_REFERENCE_POINTS = None


def load_point_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_meshes = [supmesh.superquadric_mesh(p) for p in GT_PARAMS]
    sampled_points_noisy, normals_sp_noisy = samp.sampling_sq_noisy(
        gt_meshes,
        n_points=NOISY_POINTS_PER_MESH,
        noise_std=NOISE_STD,
        clip_k=3.0,
        seed=INPUT_SAMPLING_SEED,
    )
    sampled_points_random, _ = samp.sampling_sq_random(
        gt_meshes,
        n_points=REFERENCE_POINTS_PER_MESH,
        seed=INPUT_SAMPLING_SEED,
    )
    sampled_points_outliers, normals_sp_outliers = samp.sampling_outliers(
        gt_meshes,
        n_out=OUTLIER_COUNT,
        margin=OUTLIER_MARGIN,
        mode="uniform",
        seed=INPUT_SAMPLING_SEED,
    )
    sampled_points = np.vstack([*sampled_points_noisy, sampled_points_outliers]).astype(np.float64)
    normals = np.vstack([*normals_sp_noisy, normals_sp_outliers]).astype(np.float64)
    reference_points = np.vstack(sampled_points_random).astype(np.float64)
    return sampled_points, normals, reference_points


def show_mesh_and_points(*args, **kwargs):
    from src.visualizations import visualization as vis

    vis.show_mesh_and_points(*args, **kwargs)


def run_one(
    points: np.ndarray,
    normals: np.ndarray,
    reference_points: np.ndarray | None,
    algorithm: str,
    seed: int,
):
    t0 = time.perf_counter()
    use_normal_coherence = (algorithm == "gair-ransac")
    models, inliers_masks, _ = gair_ransac(
        points,
        normals,
        threshold=THRESHOLD,
        max_models=MAX_MODELS,
        max_iterations=MAX_ITER,
        inner_iterations=INNER_ITER,
        radius=GRAPH_RADIUS,
        use_normal_coherence=use_normal_coherence,
        min_coverage=0.4,
        random_seed=seed,
    )
    runtime = time.perf_counter() - t0

    if not models:
        return None

    meshes = [supmesh.superquadric_mesh(m) for m in models]
    sampled_est, _ = samp.sampling_sq(meshes, n_points=SAMPLED_POINT_COUNT, seed=EVAL_SEED)
    est_pts = np.vstack(sampled_est)
    metric_points = points if reference_points is None else reference_points

    cd = chamfer_distance(metric_points, est_pts)

    tree_est = cKDTree(est_pts)
    tree_inp = cKDTree(metric_points)
    hd = max(tree_est.query(metric_points, k=1)[0].max(),
             tree_inp.query(est_pts, k=1)[0].max())

    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    colors = [palette[i % len(palette)] for i in range(len(meshes))]
    inlier_mask = None
    if inliers_masks:
        inlier_mask = inliers_masks[0].copy()
        for mask in inliers_masks[1:]:
            inlier_mask |= mask
    if VISUALIZE:
        show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, inlier_mask=inlier_mask, models=models)

    return {"chamfer": cd, "hausdorff": hd, "runtime_s": runtime, "n_models": len(models), "models": models}


def _init_trial_worker(points, normals, reference_points):
    global _WORKER_POINTS, _WORKER_NORMALS, _WORKER_REFERENCE_POINTS
    _WORKER_POINTS = points
    _WORKER_NORMALS = normals
    _WORKER_REFERENCE_POINTS = reference_points


def _run_trial_job(job):
    algorithm, trial, seed = job
    if _WORKER_POINTS is None or _WORKER_NORMALS is None:
        raise RuntimeError("trial worker not initialized")
    return trial, seed, run_one(_WORKER_POINTS, _WORKER_NORMALS, _WORKER_REFERENCE_POINTS, algorithm, seed)


def _print_trial_result(trial: int, result):
    if result is None:
        print(f"  trial {trial}: no model found, skipping")
        return
    print(
        f"  trial {trial} | CD={result['chamfer']:.4f}  HD={result['hausdorff']:.4f}  "
        f"RT={result['runtime_s']:.1f}s  models={result['n_models']}"
    )


def run_trials(
    points: np.ndarray,
    normals: np.ndarray,
    reference_points: np.ndarray | None,
    algorithm: str,
    n_trials: int,
    rng: np.random.Generator,
):
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_trials)]
    max_workers = min(n_trials, os.cpu_count() or 1)

    if VISUALIZE or max_workers <= 1:
        ordered_results = []
        for trial, seed in enumerate(seeds):
            result = run_one(points, normals, reference_points, algorithm, seed)
            _print_trial_result(trial, result)
            ordered_results.append((trial, seed, result))
        return ordered_results

    ordered_results = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_trial_worker,
        initargs=(points, normals, reference_points),
    ) as executor:
        future_to_trial = {
            executor.submit(_run_trial_job, (algorithm, trial, seed)): trial
            for trial, seed in enumerate(seeds)
        }
        for future in as_completed(future_to_trial):
            trial = future_to_trial[future]
            try:
                completed_trial, seed, result = future.result()
            except Exception as exc:
                raise RuntimeError(f"{algorithm} trial {trial} failed") from exc
            _print_trial_result(completed_trial, result)
            ordered_results[completed_trial] = (seed, result)

    return [(trial, *ordered_results[trial]) for trial in range(n_trials)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic point cloud from superquadrics ...")
    points, normals, reference_points = load_point_cloud()
    print(f"  scene superquadrics: {len(GT_PARAMS)}")
    print(f"  fitting points: {points.shape[0]}")
    print(f"  reference points: {reference_points.shape[0]}")

    rng = np.random.default_rng()
    algorithms = ["gair-ransac","gc-ransac"]
    all_results = []
    all_candidates = {algo: [] for algo in algorithms}
    max_workers = min(N_TRIALS, os.cpu_count() or 1)

    if VISUALIZE:
        print("Visualization enabled: running trials sequentially")
    elif max_workers > 1:
        print(f"Running {N_TRIALS} trials in parallel with up to {max_workers} workers per algorithm")

    for algo in algorithms:
        print(f"\n=== {algo} ===")
        ordered_trials = run_trials(points, normals, reference_points, algo, N_TRIALS, rng)
        for trial, seed, result in ordered_trials:
            if result is None:
                continue
            all_results.append({"algo": algo, "trial": trial, "seed": seed, **{k: v for k, v in result.items() if k != "models"}})
            all_candidates[algo].extend(result["models"])

    # summary
    print("\n=== summary ===")
    for algo in algorithms:
        rows = [r for r in all_results if r["algo"] == algo]
        if not rows:
            continue
        for metric in ("chamfer", "hausdorff", "runtime_s"):
            vals = [r[metric] for r in rows]
            print(f"  {algo:12s}  {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # save csv
    csv_path = OUT_DIR / "results_sequential.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "trial", "seed", "chamfer", "hausdorff", "runtime_s", "n_models"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")
"""
    # ── set-cover ─────────────────────────────────────────────────────────────
    print(f"\n=== set-cover (k_max={K}) ===")
    cover_results = []
    for algo in algorithms:
        candidates = all_candidates[algo]
        if not candidates:
            print(f"  {algo}: no candidates, skipping")
            continue
        print(f"\n  {algo} — {len(candidates)} candidates")
        selected_idx = exhaustive_best_cover(candidates, points, k_max=K)
        selected_models = [candidates[i] for i in selected_idx]

        meshes = [supmesh.superquadric_mesh(m) for m in selected_models]
        sampled_est, _ = samp.sampling_sq(meshes, n_points=SAMPLED_POINT_COUNT, seed=EVAL_SEED)
        est_pts = np.vstack(sampled_est)

        cd = chamfer_distance(points, est_pts)
        tree_est = cKDTree(est_pts)
        tree_inp = cKDTree(points)
        hd = max(tree_est.query(points, k=1)[0].max(),
                 tree_inp.query(est_pts, k=1)[0].max())

        print(f"  {algo} | k={len(selected_models)}  CD={cd:.4f}  HD={hd:.4f}")
        cover_results.append({"algo": algo, "k": len(selected_models), "chamfer": round(cd, 6), "hausdorff": round(hd, 6), "n_candidates": len(candidates)})

        if VISUALIZE:
            palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
            colors = [palette[i % len(palette)] for i in range(len(meshes))]
            show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, models=selected_models)

    csv_cover = OUT_DIR / "results_setcover.csv"
    with open(csv_cover, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "k", "chamfer", "hausdorff", "n_candidates"])
        writer.writeheader()
        writer.writerows(cover_results)
    print(f"Set-cover results saved to {csv_cover}")

"""
def score_combo(combo_indices, candidates, points):
    meshes = [supmesh.superquadric_mesh(candidates[i]) for i in combo_indices]
    sampled, _ = samp.sampling_sq(meshes, n_points=SAMPLED_POINT_COUNT, seed=EVAL_SEED)
    return chamfer_distance(points, np.vstack(sampled))


def exhaustive_best_cover(candidates, points, k_max):
    n = len(candidates)
    overall_best_score = np.inf
    overall_best_combo = None
    overall_best_k = None
    for k in range(1, k_max + 1):
        n_combos = math.comb(n, k)
        print(f"  k={k}  ({n_combos} combinations)")
        best_score = np.inf
        best_combo = None
        for combo in combinations(range(n), k):
            score = score_combo(combo, candidates, points)
            if score < best_score:
                best_score = score
                best_combo = combo
        print(f"  best combo={best_combo}  cd={best_score:.4f}")
        if best_score < overall_best_score:
            overall_best_score = best_score
            overall_best_combo = best_combo
            overall_best_k = k
    print(f"\n  => best K={overall_best_k}  combo={overall_best_combo}  cd={overall_best_score:.4f}")
    return list(overall_best_combo)


if __name__ == "__main__":
    main()
