"""
6_set_cover.py
==============
Runs sequential RANSAC N times with random subsampling, collects all candidate
models, re-evaluates their inliers on the full point cloud, then applies a
greedy set-cover to select either:
  - the minimum number of models that covers all points  (mode="min")
  - the best K models that cover the most points         (mode="topk")
"""

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from point_cloud_utils import chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.consensus import compute_consensus
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.visualizations import visualization as vis

# ── config ─────────────────────────────────────────────────────────────────────
VISUALIZE    = True
PC_FILE      = ROOT / "src" / "point_clouds" / "mushroom.glb"
THRESHOLD    = 0.05
GRAPH_RADIUS = 0.06
MAX_MODELS   = 5       # max models per RANSAC run
MAX_ITER     = 5
INNER_ITER   = 100
N_RUNS       = 3      # how many times to run sequential RANSAC
SUBSAMPLE    = 0.8     # fraction of points used per run (perturbation); 1.0 = no subsample
K            = 5       # max models to consider; algorithm stops early if gain <= 0
EVAL_SEED    = 42
ALGORITHM    = "gair-ransac"  # "gair-ransac" or "gc-ransac"
# ───────────────────────────────────────────────────────────────────────────────


def load_point_cloud(path: Path):
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene
    points  = np.asarray(mesh_raw.vertices, dtype=np.float64)
    normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)
    return points, normals


def collect_candidates(points, normals, rng) -> list:
    """Run sequential RANSAC N_RUNS times, return all candidate models."""
    candidates = []
    use_normal_coherence = (ALGORITHM == "gair-ransac")
    for run in range(N_RUNS):
        seed = int(rng.integers(0, 2**31))

        # subsample for perturbation
        if SUBSAMPLE < 1.0:
            n = len(points)
            idx = rng.choice(n, size=int(n * SUBSAMPLE), replace=False)
            pts_run = points[idx]
            nor_run = normals[idx]
        else:
            pts_run, nor_run = points, normals

        t0 = time.perf_counter()
        models, _, _ = gair_ransac(
            pts_run, nor_run,
            threshold=THRESHOLD,
            max_models=MAX_MODELS,
            max_iterations=MAX_ITER,
            inner_iterations=INNER_ITER,
            radius=GRAPH_RADIUS,
            use_normal_coherence=use_normal_coherence,
            min_coverage=0.4,
            random_seed=seed,
        )
        elapsed = time.perf_counter() - t0
        print(f"  run {run:2d} | {len(models)} models  ({elapsed:.1f}s)")
        candidates.extend(models)

    print(f"Total candidates: {len(candidates)}")
    return candidates


def compute_all_inliers(candidates, points, normals):
    """Re-evaluate each candidate on the full point cloud."""
    inlier_masks = []
    for model in candidates:
        mask = compute_consensus(model, points, THRESHOLD, error_metric="radial")
        inlier_masks.append(mask)
    return inlier_masks


def formula_select(inlier_masks, n_points, k_max):
    """
    For each k in 1..k_max, greedily pick k models by unique coverage,
    then compute:

        score(k) = SUM_i( unique_i ) - k * (N / K_max)
                 = total_unique(k)   - k * cost_per_model

    where cost_per_model = N / k_max is fixed (the fair-share if all
    k_max slots were equally used).  A model is worth adding only if
    it contributes more than cost_per_model unique points; otherwise
    it drags the score down.

    K* = argmax_k score(k).
    """
    cost_per_model = n_points / k_max

    # --- greedy pass: record unique coverage at each step ---
    covered        = np.zeros(n_points, dtype=bool)
    remaining      = list(range(len(inlier_masks)))
    greedy_order   = []
    unique_per_step = []

    for _ in range(k_max):
        if not remaining:
            break
        best_idx = max(remaining, key=lambda i: (inlier_masks[i] & ~covered).sum())
        new_pts  = int((inlier_masks[best_idx] & ~covered).sum())
        if new_pts == 0:
            break
        covered |= inlier_masks[best_idx]
        greedy_order.append(best_idx)
        unique_per_step.append(new_pts)
        remaining.remove(best_idx)

    # --- evaluate score(k) for each k ---
    print(f"\n  cost_per_model = N/K_max = {n_points}/{k_max} = {cost_per_model:.0f}")
    print(f"\n  {'k':>3}  {'unique_covered':>14}  {'marginal':>9}  {'score':>8}")
    best_score = -np.inf
    best_k     = 1
    cumulative = 0
    for k, u in enumerate(unique_per_step, start=1):
        cumulative += u
        score = cumulative - k * cost_per_model
        print(f"  {k:>3}  {cumulative:>14}  {u:>+9.0f}  {score:>+8.0f}  {'<-- best' if score > best_score else ''}")
        if score > best_score:
            best_score = score
            best_k     = k

    print(f"\n  => best K={best_k}  score={best_score:+.0f}")
    return greedy_order[:best_k]


def main():
    print(f"Loading {PC_FILE.name} ...")
    points, normals = load_point_cloud(PC_FILE)
    print(f"  {len(points)} points\n")

    rng = np.random.default_rng()

    # --- step 1: collect candidates ---
    print(f"=== collecting candidates ({N_RUNS} runs x up to {MAX_MODELS} models) ===")
    candidates = collect_candidates(points, normals, rng)
    if not candidates:
        print("No candidates found.")
        return

    # --- step 2: re-evaluate inliers on full cloud ---
    print("\n=== re-evaluating inliers on full point cloud ===")
    inlier_masks = compute_all_inliers(candidates, points, normals)
    inlier_counts = [m.sum() for m in inlier_masks]
    print(f"  inlier counts: min={min(inlier_counts)}  max={max(inlier_counts)}  mean={np.mean(inlier_counts):.0f}")

    # --- step 3: formula selection ---
    print(f"\n=== formula selection (k_max={K}) ===")
    selected_idx = formula_select(inlier_masks, len(points), k_max=3*K)

    selected_models = [candidates[i] for i in selected_idx]
    selected_masks  = [inlier_masks[i] for i in selected_idx]

    # combined inlier mask
    combined_mask = np.zeros(len(points), dtype=bool)
    for m in selected_masks:
        combined_mask |= m
    print(f"\nSelected {len(selected_models)} models covering {combined_mask.sum()}/{len(points)} points ({combined_mask.mean():.1%})")

    # --- step 4: metrics ---
    meshes = [supmesh.superquadric_mesh(m) for m in selected_models]
    sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    est_pts = np.vstack(sampled_est)

    cd = chamfer_distance(points, est_pts)
    tree_est = cKDTree(est_pts)
    tree_inp = cKDTree(points)
    hd = max(tree_est.query(points, k=1)[0].max(),
             tree_inp.query(est_pts, k=1)[0].max())
    print(f"CD={cd:.4f}  HD={hd:.4f}")

    # --- step 5: visualize ---
    if VISUALIZE:
        palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
        colors = [palette[i % len(palette)] for i in range(len(meshes))]
        vis.show_mesh_and_points(
            meshes, pts=points, point_size=5,
            colors=colors, inlier_mask=combined_mask, models=selected_models,
        )


if __name__ == "__main__":
    main()
