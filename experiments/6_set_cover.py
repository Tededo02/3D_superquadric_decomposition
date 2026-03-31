import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from point_cloud_utils import chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.gair_ransac import gair_ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_residual as supres
from src.superquadrics import superquadric_sampling as samp
from src.visualizations import visualization as vis

# ── config ─────────────────────────────────────────────────────────────────────
VISUALIZE    = False
PC_FILE      = ROOT / "src" / "point_clouds" / "mushroom.glb"
THRESHOLD    = 0.1
GRAPH_RADIUS = 0.06
MAX_MODELS   = 2
MAX_ITER     = 5
INNER_ITER   = 100
N_RUNS       = 5       # subsampled RANSAC runs per trial (candidates collection)
N_TRIALS     = 5
SUBSAMPLE    = 0.8
K            = 5       # max models for set-cover
EVAL_SEED    = 42
OUT_DIR      = ROOT / "experiments" / "artifacts" / "6_set_cover"
# ───────────────────────────────────────────────────────────────────────────────

CAMERA_POSITIONS = [
    ( 6.0,  0.0,  0.5),
    (-6.0,  0.0,  0.5),
    ( 0.0,  6.0,  0.5),
    ( 0.0, -6.0,  0.5),
    ( 4.0,  4.0,  6.0),
]
ANGLE_NAMES = ["front", "back", "left", "right", "top"]
PALETTE = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]


def load_point_cloud(path: Path):
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene
    points  = np.asarray(mesh_raw.vertices, dtype=np.float64)
    normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)
    return points, normals


def run_one(points, normals, algorithm: str, seed: int):
    """Exactly like experiment 5: one trial = one sequential RANSAC run."""
    t0 = time.perf_counter()
    use_normal_coherence = (algorithm == "gair-ransac")
    models, _, _ = gair_ransac(
        points, normals,
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
    sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    est_pts = np.vstack(sampled_est)

    cd = chamfer_distance(points, est_pts)
    tree_est = cKDTree(est_pts)
    tree_inp = cKDTree(points)
    hd = max(tree_est.query(points, k=1)[0].max(),
             tree_inp.query(est_pts, k=1)[0].max())

    return {"chamfer": cd, "hausdorff": hd, "runtime_s": runtime, "n_models": len(models), "models": models}


def score_combo(combo_indices, candidates, points):
    meshes = [supmesh.superquadric_mesh(candidates[i]) for i in combo_indices]
    sampled, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    return chamfer_distance(points, np.vstack(sampled))


def exhaustive_best_cover(candidates, points, k_max):
    from itertools import combinations
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


def save_screenshots(tag: str, meshes, points, colors):
    for cam_pos, angle in zip(CAMERA_POSITIONS, ANGLE_NAMES):
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
        pl.set_background("white")
        for mesh, color in zip(meshes, colors):
            faces = np.hstack([
                np.full((len(mesh.faces), 1), 3, dtype=np.int64),
                mesh.faces.astype(np.int64),
            ]).ravel()
            pl.add_mesh(pv.PolyData(mesh.vertices, faces), smooth_shading=True, opacity=0.7, color=color)
        pl.add_points(points, render_points_as_spheres=True, point_size=4, color="black", opacity=0.3)
        pl.enable_eye_dome_lighting()
        pl.camera_position = [cam_pos, (0, 0, 0), (0, 0, 1)]
        img_path = OUT_DIR / f"{tag}_{angle}.png"
        pl.screenshot(str(img_path))
        pl.close()
        print(f"  saved {img_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading point cloud from {PC_FILE.name} ...")
    points, normals = load_point_cloud(PC_FILE)
    print(f"  {points.shape[0]} points loaded")

    rng = np.random.default_rng()
    algorithms = ["gc-ransac", "gair-ransac"]
    all_results = []
    all_candidates = {algo: [] for algo in algorithms}

    for algo in algorithms:
        print(f"\n=== {algo} ===")
        for trial in range(N_TRIALS):
            seed = int(rng.integers(0, 2**31))
            result = run_one(points, normals, algo, seed)
            if result is None:
                print(f"  trial {trial}: no model found, skipping")
                continue
            print(f"  trial {trial} | CD={result['chamfer']:.4f}  HD={result['hausdorff']:.4f}  RT={result['runtime_s']:.1f}s  models={result['n_models']}")
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

    # save CSV (trials)
    csv_path = OUT_DIR / "results_trials.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "trial", "seed", "chamfer", "hausdorff", "runtime_s", "n_models"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nTrial results saved to {csv_path}")

    # ── set-cover on all collected candidates ─────────────────────────────────
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
        sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
        est_pts = np.vstack(sampled_est)

        cd = chamfer_distance(points, est_pts)
        tree_est = cKDTree(est_pts)
        tree_inp = cKDTree(points)
        hd = max(tree_est.query(points, k=1)[0].max(),
                 tree_inp.query(est_pts, k=1)[0].max())
        union_mask = np.zeros(len(points), dtype=bool)
        for m in selected_models:
            residuals = np.abs(supres.superquadric_radial_residual(m, points))
            union_mask |= (residuals < THRESHOLD)
        accuracy = float(union_mask.sum()) / len(points)

        print(f"  {algo} | k={len(selected_models)}  CD={cd:.4f}  HD={hd:.4f}  ACC={accuracy:.3f}")
        cover_results.append({"algo": algo, "k": len(selected_models), "chamfer": round(cd, 6), "hausdorff": round(hd, 6), "accuracy": round(accuracy, 6), "n_candidates": len(candidates)})

        colors = [PALETTE[i % len(PALETTE)] for i in range(len(meshes))]
        save_screenshots(f"{algo}_k{len(selected_models)}", meshes, points, colors)

        if VISUALIZE:
            vis.show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, models=selected_models)

    csv_cover = OUT_DIR / "results_setcover.csv"
    with open(csv_cover, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "k", "chamfer", "hausdorff", "accuracy", "n_candidates"])
        writer.writeheader()
        writer.writerows(cover_results)
    print(f"Set-cover results saved to {csv_cover}")


if __name__ == "__main__":
    main()
