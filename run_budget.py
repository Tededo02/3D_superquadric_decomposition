"""
run_budget.py — same experiment as run.py but each trial gets a fixed time
budget. All algorithms stop at the same wall-clock mark so results are
directly comparable at equal compute cost.
"""
import csv
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from point_cloud_utils import chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.ransacov import ransacov
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.ransac import ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp

# ── config ────────────────────────────────────────────────────────────────────
TIME_BUDGET_S     = 120.0  # seconds each algorithm gets per trial
PC_DIR            = ROOT / "data" / "point_clouds"
THRESHOLD         = 0.9
GRAPH_RADIUS      = 0.1
MAX_MODELS        = 10
PER_MODEL_BUDGET  = TIME_BUDGET_S / MAX_MODELS  # seconds per model slot
INNER_ITER        = 25
N_TRIALS       = 16
K              = 1
EVAL_SEED      = 42
OUT_DIR        = ROOT / "data" / "results_budget"
MAX_LOAD_PTS   = 8000
MAX_COVER_ITER = 1000
PC_CONFIGS_PATH = Path(__file__).parent / "data" / "pc_configs.json"
# ──────────────────────────────────────────────────────────────────────────────


def _load_pc_config(pc_stem: str) -> dict:
    if PC_CONFIGS_PATH.exists():
        with open(PC_CONFIGS_PATH) as f:
            all_configs = json.load(f)
        cfg = dict(all_configs.get("default", {}))
        cfg.update(all_configs.get(pc_stem, {}))
        return cfg
    return {}


def _subsample(points: np.ndarray, max_pts: int | None):
    if max_pts is None or len(points) <= max_pts:
        return points, np.arange(len(points))
    idx = np.random.default_rng().choice(len(points), max_pts, replace=False)
    idx.sort()
    return points[idx], idx


def _drop_background_color(points: np.ndarray, colors_rgba: np.ndarray):
    from collections import Counter
    rgb = np.asarray(colors_rgba, dtype=np.uint8)[:, :3]
    bg_color, bg_count = Counter(map(tuple, rgb)).most_common(1)[0]
    if bg_count / len(points) <= 0.5:
        return points, np.ones(len(points), dtype=bool)
    mask = np.array([tuple(c) != bg_color for c in rgb])
    return points[mask], mask


def load_point_cloud(path: Path):
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene

    points = np.asarray(mesh_raw.vertices, dtype=np.float64)

    if isinstance(mesh_raw, trimesh.PointCloud):
        colors = getattr(mesh_raw, "colors", None)
        if colors is not None and len(colors):
            points, keep = _drop_background_color(points, colors)
        else:
            keep = np.ones(len(points), dtype=bool)
        points, sub_idx = _subsample(points, MAX_LOAD_PTS)
        normals_path = path.parent / f"normals_{path.name}"
        if normals_path.exists():
            all_normals = np.asarray(trimesh.load(str(normals_path)).vertices, dtype=np.float64)
            normals = all_normals[keep][sub_idx]
        else:
            import point_cloud_utils as pcu
            _, normals = pcu.estimate_point_cloud_normals_knn(points.astype(np.float32), num_neighbors=10)
            normals = normals.astype(np.float64)
    else:
        normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)
        points, sub_idx = _subsample(points, MAX_LOAD_PTS)
        normals = normals[sub_idx]

    if points.shape[0] == 0:
        raise ValueError(f"Point cloud is empty after loading {path.name}")

    bb_min, bb_max = points.min(axis=0), points.max(axis=0)
    bb_size = bb_max - bb_min
    bb_size[bb_size == 0] = 1.0
    points = (points - bb_min) / bb_size
    return points, normals


def corrupt_point_cloud(points: np.ndarray, normals: np.ndarray,
                        rng: np.random.Generator, n_outliers: int):
    if n_outliers > 0:
        lo, hi = points.min(axis=0), points.max(axis=0)
        outlier_pts = rng.uniform(lo, hi, (n_outliers, 3))
        outlier_nrm = rng.normal(0, 1, (n_outliers, 3))
        outlier_nrm /= np.linalg.norm(outlier_nrm, axis=1, keepdims=True)
        points  = np.vstack([points,  outlier_pts])
        normals = np.vstack([normals, outlier_nrm])
    return points, normals


def score_combo(combo_indices, candidates, points):
    meshes = [supmesh.superquadric_mesh(candidates[i]) for i in combo_indices]
    sampled, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    return chamfer_distance(points, np.vstack(sampled))


def exhaustive_best_cover(candidates, points, k_max):
    n = len(candidates)
    overall_best_score = np.inf
    overall_best_combo = None
    overall_best_k = None
    total_evals = 0
    for k in range(1, k_max + 1):
        n_combos = math.comb(n, k)
        print(f"  k={k}  ({n_combos} combinations, cap={MAX_COVER_ITER})")
        best_score = np.inf
        best_combo = None
        for i, combo in enumerate(combinations(range(n), k)):
            if i >= MAX_COVER_ITER:
                break
            score = score_combo(combo, candidates, points)
            total_evals += 1
            if score < best_score:
                best_score = score
                best_combo = combo
        if best_score < overall_best_score:
            overall_best_score = best_score
            overall_best_combo = best_combo
            overall_best_k = k
    print(f"  => best K={overall_best_k}  combo={overall_best_combo}  cd={overall_best_score:.4f}")
    return list(overall_best_combo), total_evals


def _find_max_iters(algorithm: str, points: np.ndarray, normals: np.ndarray,
                    pc_cfg: dict, seed: int) -> int:
    """Search for the iteration count whose actual runtime is closest to PER_MODEL_BUDGET."""
    threshold = pc_cfg.get("threshold", THRESHOLD)

    def _run(n, s) -> float:
        t0 = time.perf_counter()
        if algorithm in ("vanilla", "lo-ransac"):
            ransac(points, threshold=threshold, max_models=1, max_iterations=n,
                   inner_iterations=INNER_ITER, random_seed=s,
                   local_optimization=(algorithm == "lo-ransac"))
        else:
            gair_ransac(points, normals if algorithm == "gair-ransac" else None,
                        threshold=threshold, max_models=1, max_iterations=n,
                        inner_iterations=INNER_ITER, radius=GRAPH_RADIUS, min_coverage=0.4,
                        random_seed=s, sample_size=pc_cfg.get("mss_sample_size", 50))
        return time.perf_counter() - t0

    _run(1, seed ^ 0xDEAD)   # warm-up

    best_n, best_diff = 1, float("inf")
    n = 1
    while True:
        t = _run(n, seed)
        diff = abs(t - PER_MODEL_BUDGET)
        if diff < best_diff:
            best_diff, best_n = diff, n
        if t >= PER_MODEL_BUDGET:
            break
        n_next = max(n + 1, round(n * PER_MODEL_BUDGET / t))
        n = n_next
    return best_n


def run_one(points, normals, clean_points, n_clean, algorithm: str, seed: int,
            pc_cfg: dict, max_iter: int):
    threshold = pc_cfg.get("threshold", THRESHOLD)

    t0 = time.perf_counter()
    if algorithm in ("vanilla", "lo-ransac"):
        models, inliers_masks, n_local_opts = ransac(
            points,
            threshold=threshold,
            max_models=MAX_MODELS,
            max_iterations=max_iter,
            inner_iterations=INNER_ITER,
            random_seed=seed,
            local_optimization=(algorithm == "lo-ransac"),
        )
    else:
        models, inliers_masks, _mss_used, n_local_opts = gair_ransac(
            points,
            normals if algorithm == "gair-ransac" else None,
            threshold=threshold,
            max_models=MAX_MODELS,
            max_iterations=max_iter,
            inner_iterations=INNER_ITER,
            radius=GRAPH_RADIUS,
            min_coverage=0.4,
            random_seed=seed,
            sample_size=pc_cfg.get("mss_sample_size", 50),
        )
    runtime = time.perf_counter() - t0

    if not models:
        return None

    meshes = [supmesh.superquadric_mesh(m) for m in models]
    sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    est_pts = np.vstack(sampled_est)

    cd           = chamfer_distance(clean_points, est_pts)
    tree_est     = cKDTree(est_pts)
    tree_inp     = cKDTree(clean_points)
    d_inp_to_est = tree_est.query(clean_points, k=1)[0]
    d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]
    cd_coverage  = float(d_inp_to_est.mean())
    cd_accuracy  = float(d_est_to_inp.mean())
    hd           = max(d_inp_to_est.max(), d_est_to_inp.max())

    inlier_mask = None
    if inliers_masks:
        inlier_mask = inliers_masks[0].copy()
        for mask in inliers_masks[1:]:
            inlier_mask |= mask

    if inlier_mask is not None:
        true_inliers_correct  = int(inlier_mask[:n_clean].sum())
        true_outliers_correct = int((~inlier_mask[n_clean:]).sum())
        correctly_classified  = true_inliers_correct + true_outliers_correct
        classification_rate   = correctly_classified / len(points)
    else:
        correctly_classified = 0
        classification_rate  = 0.0

    n_inliers = int(inlier_mask[:n_clean].sum()) if inlier_mask is not None else 0
    return {
        "chamfer": cd, "cd_coverage": cd_coverage, "cd_accuracy": cd_accuracy,
        "hausdorff": hd, "classification_rate": classification_rate,
        "runtime_s": runtime, "n_models": len(models),
        "n_inliers": n_inliers, "n_local_opts": n_local_opts, "models": models,
    }


def run_for_pc(pc_file: Path, rng: np.random.Generator):
    print(f"\n{'='*60}")
    print(f"Point cloud: {pc_file.name}  |  budget: {TIME_BUDGET_S}s/trial")
    print(f"{'='*60}")

    points_clean, normals_clean = load_point_cloud(pc_file)
    n_clean      = len(points_clean)
    clean_points = points_clean.copy()
    pc_cfg       = _load_pc_config(pc_file.stem)
    print(f"  {n_clean} points  |  config: {pc_cfg}")

    trial_seeds   = [int(rng.integers(0, 2**31)) for _ in range(N_TRIALS)]
    outlier_fracs = [round(x * 0.05, 2) for x in range(9)]
    algorithms    = ["gair-ransac", "gc-ransac", "lo-ransac", "vanilla"]

    pc_out_dir = OUT_DIR / pc_file.stem
    pc_out_dir.mkdir(parents=True, exist_ok=True)

    for out_frac in outlier_fracs:
        n_outliers = int(n_clean * out_frac)
        print(f"\n{'#'*60}")
        print(f"Outlier fraction: {out_frac:.2f}  ({n_outliers} outliers)")
        print(f"{'#'*60}")

        points_c, normals_c = corrupt_point_cloud(
            points_clean.copy(), normals_clean.copy(), rng, n_outliers
        )

        frac_out_dir = pc_out_dir / f"outliers_{out_frac:.2f}"
        frac_out_dir.mkdir(parents=True, exist_ok=True)

        all_results    = []
        all_candidates = {algo: [] for algo in algorithms}

        print("  calibrating...")
        max_iters = {}
        for algo in algorithms:
            max_iters[algo] = _find_max_iters(algo, points_c, normals_c, pc_cfg, seed=EVAL_SEED)
            print(f"    [{algo}] → {max_iters[algo]} iters/slot")

        for trial in range(N_TRIALS):
            for algo in algorithms:
                seed = trial_seeds[trial]
                print(f"\n  --- trial {trial} | {algo} | seed={seed} ---")
                result = run_one(points_c, normals_c, clean_points, n_clean, algo, seed, pc_cfg, max_iters[algo])
                if result is None:
                    print(f"  trial {trial} {algo}: no model found")
                    continue
                print(
                    f"  trial {trial:3d} {algo:12s}  seed={seed}"
                    f"  CD={result['chamfer']:.4f}"
                    f"  CLF={result['classification_rate']:.3f}"
                    f"  RT={result['runtime_s']:.1f}s"
                    f"  models={result['n_models']}"
                )
                all_results.append({
                    "algo": algo, "trial": trial, "seed": seed,
                    "outlier_frac": out_frac, "budget_s": TIME_BUDGET_S,
                    **{k: v for k, v in result.items() if k != "models"},
                })
                all_candidates[algo].extend(result["models"])

        print("\n=== summary ===")
        for algo in algorithms:
            rows = [r for r in all_results if r["algo"] == algo]
            if not rows:
                continue
            for metric in ("chamfer", "classification_rate", "runtime_s", "n_models"):
                vals = [r[metric] for r in rows]
                print(f"  {algo:12s}  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        csv_path = frac_out_dir / "results_sequential.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "algo", "trial", "seed", "outlier_frac", "budget_s",
                "chamfer", "cd_coverage", "cd_accuracy", "hausdorff",
                "classification_rate", "runtime_s", "n_models", "n_inliers", "n_local_opts",
            ])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results saved to {csv_path}")

        # ── set-cover ─────────────────────────────────────────────────────────
        cover_threshold = pc_cfg.get("threshold", THRESHOLD)
        cover_results   = []

        for algo in algorithms:
            candidates = all_candidates[algo]
            if not candidates:
                continue
            for method in ("exhaustive", "ransacov"):
                print(f"\n=== set-cover [{method}]  {algo} — {len(candidates)} candidates ===")
                if method == "exhaustive":
                    selected_idx, n_evals = exhaustive_best_cover(candidates, clean_points, k_max=K)
                    if selected_idx:
                        inlier_mask = np.zeros(len(clean_points), dtype=bool)
                        for idx in selected_idx:
                            res = np.abs(superquadric_radial_residual(candidates[idx], clean_points))
                            inlier_mask |= (res < cover_threshold)
                        n_covered = int(inlier_mask.sum())
                    else:
                        n_covered = 0
                else:
                    selected_idx, n_covered = ransacov(
                        candidates, clean_points, k=K,
                        threshold=cover_threshold,
                        residual_fn=lambda m, pts: np.abs(superquadric_radial_residual(m, pts)),
                    )
                    n_evals = len(candidates)

                if not selected_idx:
                    print("  no models selected")
                    continue

                selected_models = [candidates[i] for i in selected_idx]
                meshes = [supmesh.superquadric_mesh(m) for m in selected_models]
                sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
                est_pts = np.vstack(sampled_est)

                cd           = chamfer_distance(clean_points, est_pts)
                tree_est     = cKDTree(est_pts)
                tree_inp     = cKDTree(clean_points)
                d_inp_to_est = tree_est.query(clean_points, k=1)[0]
                d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]
                cd_coverage  = float(d_inp_to_est.mean())
                cd_accuracy  = float(d_est_to_inp.mean())
                hd           = max(d_inp_to_est.max(), d_est_to_inp.max())

                print(
                    f"  {algo} [{method}] | k={len(selected_models)}"
                    f"  n_covered={n_covered}  n_evals={n_evals}"
                    f"  CD={cd:.4f}  COV={cd_coverage:.4f}  ACC={cd_accuracy:.4f}  HD={hd:.4f}"
                )
                cover_results.append({
                    "algo": algo, "method": method, "outlier_frac": out_frac,
                    "budget_s": TIME_BUDGET_S,
                    "k": len(selected_models), "n_covered": n_covered, "n_evals": n_evals,
                    "chamfer": round(cd, 6), "cd_coverage": round(cd_coverage, 6),
                    "cd_accuracy": round(cd_accuracy, 6), "hausdorff": round(hd, 6),
                    "n_candidates": len(candidates),
                })

        csv_cover = frac_out_dir / "results_setcover.csv"
        with open(csv_cover, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "algo", "method", "outlier_frac", "budget_s", "k",
                "n_covered", "n_evals", "chamfer", "cd_coverage",
                "cd_accuracy", "hausdorff", "n_candidates",
            ])
            writer.writeheader()
            writer.writerows(cover_results)
        print(f"Set-cover results saved to {csv_cover}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pc_files = sorted(
        f for f in PC_DIR.iterdir()
        if f.is_file() and not f.name.startswith("normals_") and not f.name.startswith(".")
    )
    if not pc_files:
        print(f"No files found in {PC_DIR}")
        return
    print(f"Found {len(pc_files)} point cloud(s): {[f.name for f in pc_files]}")
    print(f"Time budget per trial: {TIME_BUDGET_S}s")
    rng = np.random.default_rng()
    for pc_file in pc_files:
        run_for_pc(pc_file, rng)


if __name__ == "__main__":
    main()
