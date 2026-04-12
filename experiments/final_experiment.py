import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
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
from src.visualizations import visualization as vis

# ── config ────────────────────────────────────────────────────────────────────
VISUALIZE  = True   # set to False to ski p visualization and run all trials headlessly
PC_DIR     = ROOT / "src" / "point_clouds"
THRESHOLD  = 0.9
GRAPH_RADIUS = 0.06
MAX_MODELS = 5
MAX_ITER   = 20
INNER_ITER = 20
N_TRIALS   = 25
K = 3
EVAL_SEED  = 42          # fixed so chamfer is comparable across trials
OUT_DIR    = ROOT / "experiments" / "artifacts" / "final_experiment"
N_OUTLIERS = 130           # number of random uniform outliers to inject (0 = none)
NOISE      = 0.05        # gaussian noise std applied to positions and normals (0.0 = none)
PC_CONFIGS_PATH = Path(__file__).parent / "pc_configs.json"
# ──────────────────────────────────────────────────────────────────────────────

def _load_pc_config(pc_stem: str) -> dict:
    if PC_CONFIGS_PATH.exists():
        with open(PC_CONFIGS_PATH) as f:
            all_configs = json.load(f)
        cfg = dict(all_configs.get("default", {}))
        cfg.update(all_configs.get(pc_stem, {}))
        return cfg
    return {}


def corrupt_point_cloud(points, normals, rng):
    # additive gaussian noise on positions and normals
    if NOISE > 0.0:
        points  = points  + rng.normal(0, NOISE, points.shape)
        normals = normals + rng.normal(0, NOISE, normals.shape)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 0, norms, 1.0)

    # random uniform outliers spanning the bounding box of the cloud
    if N_OUTLIERS > 0:
        lo, hi = points.min(axis=0), points.max(axis=0)
        outlier_pts = rng.uniform(lo, hi, (N_OUTLIERS, 3))
        outlier_nrm = rng.normal(0, 1, (N_OUTLIERS, 3))
        outlier_nrm /= np.linalg.norm(outlier_nrm, axis=1, keepdims=True)
        points  = np.vstack([points,  outlier_pts])
        normals = np.vstack([normals, outlier_nrm])

    return points, normals


def load_point_cloud(path: Path):
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene

    points = np.asarray(mesh_raw.vertices, dtype=np.float64)

    if isinstance(mesh_raw, trimesh.PointCloud):
        normals_path = path.parent / f"normals_{path.name}"
        if not normals_path.exists():
            raise FileNotFoundError(f"No normals file found for {path.name} (expected {normals_path.name})")
        normals = np.asarray(trimesh.load(str(normals_path)).vertices, dtype=np.float64)
    else:
        normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)

    return points, normals


def _run_one_worker(args):
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    trial, algo, seed, points, normals, clean_points, n_clean, pc_cfg = args
    result = run_one(points, normals, clean_points, n_clean, algo, seed, pc_cfg)
    return trial, algo, seed, result


def run_one(points, normals, clean_points, n_clean, algorithm: str, seed: int, pc_cfg: dict = {}):
    t0 = time.perf_counter()
    threshold = pc_cfg.get("threshold", THRESHOLD)
    mss_used = None
    if algorithm in ("vanilla", "lo-ransac"):
        models, inliers_masks, n_local_opts = ransac(
            points,
            threshold=threshold,
            max_models=MAX_MODELS,
            max_iterations=MAX_ITER,
            inner_iterations=INNER_ITER,
            random_seed=seed,
            local_optimization=(algorithm == "lo-ransac"),
        )
    else:
        models, inliers_masks, mss_used, n_local_opts = gair_ransac(
            points,
            normals if algorithm == "gair-ransac" else None,
            threshold=threshold,
            max_models=MAX_MODELS,
            max_iterations=MAX_ITER,
            inner_iterations=INNER_ITER,
            radius=GRAPH_RADIUS,
            min_coverage=0.4,
            random_seed=seed,
            mss_pts_per_patch=pc_cfg.get("mss_pts_per_patch", 5),
            mss_voxel_spacing_factor=pc_cfg.get("mss_voxel_spacing_factor", 10),
            sample_size=pc_cfg.get("mss_sample_size", 50),
        )
    runtime = time.perf_counter() - t0

    if not models:
        return None

    meshes = [supmesh.superquadric_mesh(m) for m in models]
    sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    est_pts = np.vstack(sampled_est)

    cd = chamfer_distance(clean_points, est_pts)

    tree_est = cKDTree(est_pts)
    tree_inp = cKDTree(clean_points)
    d_inp_to_est = tree_est.query(clean_points, k=1)[0]   # each input pt → nearest estimated pt
    d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]   # each estimated pt → nearest input pt
    cd_coverage = float(d_inp_to_est.mean())               # coverage:  did estimate cover the shape?
    cd_accuracy = float(d_est_to_inp.mean())               # accuracy:  are superquadrics on the surface?
    hd = max(d_inp_to_est.max(), d_est_to_inp.max())

    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    colors = [palette[i % len(palette)] for i in range(len(meshes))]
    inlier_mask = None
    if inliers_masks:
        inlier_mask = inliers_masks[0].copy()
        for mask in inliers_masks[1:]:
            inlier_mask |= mask
    if inlier_mask is not None:
        true_inliers_correct  = int(inlier_mask[:n_clean].sum())        # clean pts correctly called inlier
        true_outliers_correct = int((~inlier_mask[n_clean:]).sum())     # injected pts correctly called outlier
        correctly_classified  = true_inliers_correct + true_outliers_correct
        classification_rate   = correctly_classified / len(points)
    else:
        correctly_classified = 0
        classification_rate  = 0.0
    print(f"    correctly_classified={correctly_classified}/{len(points)}  ({classification_rate*100:.1f}%)")
    if VISUALIZE and algorithm in ("gair-ransac", "gc-ransac"):
        vis.show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, inlier_mask=inlier_mask, mss_used=mss_used, models=models)

    n_inliers = int(inlier_mask[:n_clean].sum()) if inlier_mask is not None else 0
    return {"chamfer": cd, "cd_coverage": cd_coverage, "cd_accuracy": cd_accuracy, "hausdorff": hd, "classification_rate": classification_rate, "runtime_s": runtime, "n_models": len(models), "n_inliers": n_inliers, "n_local_opts": n_local_opts, "models": models}


def run_for_pc(pc_file: Path, rng):
    print(f"\n{'='*60}")
    print(f"Point cloud: {pc_file.name}")
    print(f"{'='*60}")
    points_clean, normals_clean = load_point_cloud(pc_file)
    n_clean = len(points_clean)
    clean_points = points_clean.copy()
    pc_cfg = _load_pc_config(pc_file.stem)
    print(f"  {n_clean} points loaded  |  config: {pc_cfg}")

    # fixed seeds across all outlier levels for comparability
    trial_seeds = [int(rng.integers(0, 2**31)) for _ in range(N_TRIALS)]

    outlier_fracs = [round(x * 0.05, 2) for x in range(9)]  # 0.0, 0.05, ..., 0.40

    pc_out_dir = OUT_DIR / pc_file.stem
    pc_out_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["gair-ransac", "gc-ransac", "lo-ransac", "vanilla"]

    for out_frac in outlier_fracs:
        n_outliers = int(n_clean * out_frac)
        print(f"\n{'#'*60}")
        print(f"Outlier fraction: {out_frac:.2f}  ({n_outliers} outliers injected)")
        print(f"{'#'*60}")

        points, normals = corrupt_point_cloud(points_clean.copy(), normals_clean.copy(), rng)
        if n_outliers > 0:
            lo, hi = points_clean.min(axis=0), points_clean.max(axis=0)
            outlier_pts = rng.uniform(lo, hi, (n_outliers, 3))
            outlier_nrm = rng.normal(0, 1, (n_outliers, 3))
            outlier_nrm /= np.linalg.norm(outlier_nrm, axis=1, keepdims=True)
            points  = np.vstack([points,  outlier_pts])
            normals = np.vstack([normals, outlier_nrm])

        frac_out_dir = pc_out_dir / f"outliers_{out_frac:.2f}"
        frac_out_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        all_candidates = {algo: [] for algo in algorithms}

        tasks = [
            (trial, algo, trial_seeds[trial], points, normals, clean_points, n_clean, pc_cfg)
            for trial in range(N_TRIALS)
            for algo in algorithms
        ]
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_run_one_worker, t): t for t in tasks}
            for future in as_completed(futures):
                trial, algo, seed, result = future.result()
                if result is None:
                    print(f"  trial {trial} {algo}: no model found, skipping")
                    continue
                print(f"  trial {trial} {algo}  seed={seed}  CD={result['chamfer']:.4f}  CD_COV={result['cd_coverage']:.4f}  CD_ACC={result['cd_accuracy']:.4f}  HD={result['hausdorff']:.4f}  CLF={result['classification_rate']:.3f}  RT={result['runtime_s']:.1f}s  models={result['n_models']}")
                all_results.append({"algo": algo, "trial": trial, "seed": seed, "outlier_frac": out_frac, **{k: v for k, v in result.items() if k != "models"}})
                all_candidates[algo].extend(result["models"])

        # summary
        print("\n=== summary ===")
        for algo in algorithms:
            rows = [r for r in all_results if r["algo"] == algo]
            if not rows:
                continue
            for metric in ("chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "classification_rate", "runtime_s"):
                vals = [r[metric] for r in rows]
                print(f"  {algo:12s}  {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        # save csv
        csv_path = frac_out_dir / "results_sequential.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["algo", "trial", "seed", "outlier_frac", "chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "classification_rate", "runtime_s", "n_models", "n_inliers", "n_local_opts"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")

        # ── set-cover ─────────────────────────────────────────────────────────
        cover_threshold = pc_cfg.get("threshold", THRESHOLD)
        cover_results = []
        for algo in algorithms:
            candidates = all_candidates[algo]
            if not candidates:
                continue

            for method in ["exhaustive", "ransacov"]:
                print(f"\n=== set-cover [{method}]  {algo} — {len(candidates)} candidates ===")
                if method == "exhaustive":
                    selected_idx, n_evals = exhaustive_best_cover(candidates, clean_points, k_max=K)
                    # count inliers: points within threshold of any selected model
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
                    n_evals = len(candidates)  # all candidates evaluated to build P matrix

                if not selected_idx:
                    print(f"  no models selected")
                    continue
                selected_models = [candidates[i] for i in selected_idx]

                meshes = [supmesh.superquadric_mesh(m) for m in selected_models]
                sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
                est_pts = np.vstack(sampled_est)

                cd = chamfer_distance(clean_points, est_pts)
                tree_est = cKDTree(est_pts)
                tree_inp = cKDTree(clean_points)
                d_inp_to_est = tree_est.query(clean_points, k=1)[0]
                d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]
                cd_coverage = float(d_inp_to_est.mean())
                cd_accuracy = float(d_est_to_inp.mean())
                hd = max(d_inp_to_est.max(), d_est_to_inp.max())

                print(f"  {algo} [{method}] | k={len(selected_models)}  n_covered={n_covered}  n_evals={n_evals}  CD={cd:.4f}  COV={cd_coverage:.4f}  ACC={cd_accuracy:.4f}  HD={hd:.4f}")
                cover_results.append({"algo": algo, "method": method, "outlier_frac": out_frac, "k": len(selected_models), "n_covered": n_covered, "n_evals": n_evals, "chamfer": round(cd, 6), "cd_coverage": round(cd_coverage, 6), "cd_accuracy": round(cd_accuracy, 6), "hausdorff": round(hd, 6), "n_candidates": len(candidates)})

                if VISUALIZE:
                    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
                    colors = [palette[i % len(palette)] for i in range(len(meshes))]
                    vis.show_mesh_and_points(meshes, pts=clean_points, point_size=5, colors=colors, models=selected_models)

        csv_cover = frac_out_dir / "results_setcover.csv"
        with open(csv_cover, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["algo", "method", "outlier_frac", "k", "n_covered", "n_evals", "chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "n_candidates"])
            writer.writeheader()
            writer.writerows(cover_results)
        print(f"Set-cover results saved to {csv_cover}")



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pc_files = sorted(f for f in PC_DIR.iterdir() if not f.name.startswith("normals_"))
    if not pc_files:
        print(f"No files found in {PC_DIR}")
        return
    print(f"Found {len(pc_files)} point cloud(s): {[f.name for f in pc_files]}")

    rng = np.random.default_rng()
    for pc_file in pc_files:
        run_for_pc(pc_file, rng)


def score_combo(combo_indices, candidates, points):
    meshes = [supmesh.superquadric_mesh(candidates[i]) for i in combo_indices]
    sampled, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    return chamfer_distance(points, np.vstack(sampled))


MAX_COVER_ITER = 1000


def exhaustive_best_cover(candidates, points, k_max):
    """Returns (selected_indices, n_evals)."""
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
        print(f"  best combo={best_combo}  cd={best_score:.4f}")
        if best_score < overall_best_score:
            overall_best_score = best_score
            overall_best_combo = best_combo
            overall_best_k = k
    print(f"\n  => best K={overall_best_k}  combo={overall_best_combo}  cd={overall_best_score:.4f}")
    return list(overall_best_combo), total_evals


if __name__ == "__main__":
    main()