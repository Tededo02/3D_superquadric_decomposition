import csv
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
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.visualizations import visualization as vis

# ── config ────────────────────────────────────────────────────────────────────
VISUALIZE  = False   # set to False to skip visualization and run all trials headlessly
PC_DIR     = ROOT / "src" / "point_clouds"
THRESHOLD  = 0.3
GRAPH_RADIUS = 0.06
MAX_MODELS = 3
MAX_ITER   = 10
INNER_ITER = 100
N_TRIALS   = 10
K = 3
EVAL_SEED  = 42          # fixed so chamfer is comparable across trials
OUT_DIR    = ROOT / "experiments" / "artifacts" / "final_experiment"
N_OUTLIERS = 0           # number of random uniform outliers to inject (0 = none)
NOISE      = 0        # gaussian noise std applied to positions and normals (0.0 = none)
# ──────────────────────────────────────────────────────────────────────────────


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


def run_one(points, normals, clean_points, n_clean, algorithm: str, seed: int):
    t0 = time.perf_counter()
    models, inliers_masks, _ = gair_ransac(
        points,
        normals if algorithm == "gair-ransac" else None,
        threshold=THRESHOLD,
        max_models=MAX_MODELS,
        max_iterations=MAX_ITER,
        inner_iterations=INNER_ITER,
        radius=GRAPH_RADIUS,
        min_coverage=0.4,
        random_seed=seed,
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
    if VISUALIZE:
        vis.show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, inlier_mask=inlier_mask, models=models)

    return {"chamfer": cd, "cd_coverage": cd_coverage, "cd_accuracy": cd_accuracy, "hausdorff": hd, "classification_rate": classification_rate, "runtime_s": runtime, "n_models": len(models), "models": models}


def build_chamfer_cache(candidates, points, n_surface=1200):
    """
    Precompute, once per candidate:
      - point -> model distances  (n_candidates x n_points)
      - model -> point mean distance (n_candidates,)
    """
    tree_points = cKDTree(points)
    n_cand = len(candidates)
    n_pts = len(points)

    d_p2m = np.empty((n_cand, n_pts), dtype=np.float32)
    d_m2p = np.empty(n_cand, dtype=np.float32)

    for j, model in enumerate(candidates):
        mesh = supmesh.superquadric_mesh(model)

        # Sample a fixed number of surface points for each candidate.
        sampled, _ = samp.sampling_sq_random([mesh], n_points=n_surface, seed=EVAL_SEED + j)
        surf = np.asarray(sampled[0], dtype=np.float64)

        tree_model = cKDTree(surf)
        d_p2m[j] = tree_model.query(points, k=1)[0].astype(np.float32)
        d_m2p[j] = float(tree_points.query(surf, k=1)[0].mean())

    return d_p2m, d_m2p


def prune_dominated_candidates(d_p2m, d_m2p, tol=1e-8):
    """
    Soft analogue of RansaCov pruning:
    if candidate A is <= B on all points and has accuracy cost <=,
    then B is dominated and can be discarded.
    """
    n = d_p2m.shape[0]
    keep = np.ones(n, dtype=bool)
    order = np.argsort(d_m2p)

    for ia, a in enumerate(order):
        if not keep[a]:
            continue
        for b in order[ia + 1:]:
            if not keep[b]:
                continue
            if d_m2p[a] <= d_m2p[b] + tol and np.all(d_p2m[a] <= d_p2m[b] + tol):
                keep[b] = False

    return np.flatnonzero(keep)


def chamfer_subset_score(selected, d_p2m, d_m2p, clip=None, lambda_acc=1.0):
    if len(selected) == 0:
        return np.inf

    cover = np.min(d_p2m[selected], axis=0)
    if clip is not None:
        cover = np.minimum(cover, clip)

    cov = float(cover.mean())
    acc = float(d_m2p[selected].mean())
    return cov + lambda_acc * acc


def greedy_ransacov_chamfer(candidates, points, k_max, clip=None, lambda_acc=1.0, max_swap_passes=2):
    """
    Greedy RansaCov-like selection on a soft coverage objective based on Chamfer.
    Complexity: O(precompute) + O(k_max * n_candidates * n_points)
    """
    d_p2m, d_m2p = build_chamfer_cache(candidates, points)
    active = prune_dominated_candidates(d_p2m, d_m2p)

    selected = []
    current_best = np.full(points.shape[0], np.inf, dtype=np.float32)
    current_score = np.inf
    acc_sum = 0.0

    # Forward greedy.
    for _ in range(k_max):
        best_j = None
        best_score = current_score
        best_best = None

        for j in active:
            j = int(j)
            if j in selected:
                continue

            trial_best = np.minimum(current_best, d_p2m[j])
            cover = np.minimum(trial_best, clip) if clip is not None else trial_best
            cov = float(cover.mean())
            acc = (acc_sum + float(d_m2p[j])) / (len(selected) + 1)

            score = cov + lambda_acc * acc
            if score < best_score - 1e-12:
                best_j = j
                best_score = score
                best_best = trial_best

        if best_j is None:
            break

        selected.append(best_j)
        current_best = best_best
        current_score = best_score
        acc_sum += float(d_m2p[best_j])

    # 1-swap local refinement.
    improved = True
    passes = 0
    while improved and passes < max_swap_passes and selected:
        improved = False
        passes += 1
        remaining = [int(j) for j in active if int(j) not in selected]

        for pos, _old in enumerate(selected):
            for new in remaining:
                trial = selected.copy()
                trial[pos] = new
                trial_score = chamfer_subset_score(
                    trial, d_p2m, d_m2p, clip=clip, lambda_acc=lambda_acc
                )
                if trial_score < current_score - 1e-12:
                    selected = trial
                    current_score = trial_score
                    improved = True
                    break
            if improved:
                break

    return selected, current_score


def run_for_pc(pc_file: Path, rng):
    print(f"\n{'='*60}")
    print(f"Point cloud: {pc_file.name}")
    print(f"{'='*60}")
    points, normals = load_point_cloud(pc_file)
    print(f"  {points.shape[0]} points loaded")
    clean_points = points.copy()
    n_clean = len(points)
    points, normals = corrupt_point_cloud(points, normals, rng)
    if N_OUTLIERS > 0 or NOISE > 0.0:
        print(f"  after corruption: {points.shape[0]} points (noise={NOISE}, outliers={N_OUTLIERS})")

    pc_out_dir = OUT_DIR / pc_file.stem
    pc_out_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["gair-ransac", "gc-ransac"]
    all_results = []
    all_candidates = {algo: [] for algo in algorithms}

    for algo in algorithms:
        print(f"\n=== {algo} ===")
        for trial in range(N_TRIALS):
            seed = int(rng.integers(0, 2**31))
            result = run_one(points, normals, clean_points, n_clean, algo, seed)
            if result is None:
                print(f"  trial {trial}: no model found, skipping")
                continue
            print(f"  trial {trial} | CD={result['chamfer']:.4f}  CD_COV={result['cd_coverage']:.4f}  CD_ACC={result['cd_accuracy']:.4f}  HD={result['hausdorff']:.4f}  CLF={result['classification_rate']:.3f}  RT={result['runtime_s']:.1f}s  models={result['n_models']}")
            all_results.append({"algo": algo, "trial": trial, "seed": seed, **{k: v for k, v in result.items() if k != "models"}})
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
    csv_path = pc_out_dir / "results_sequential.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "trial", "seed", "chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "classification_rate", "runtime_s", "n_models"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # ── greedy RansaCov-like subset selection on Chamfer surrogate ───────────
    print(f"\n=== RansaCov-Chamfer (k_max={K}) ===")
    cover_results = []
    for algo in algorithms:
        candidates = all_candidates[algo]
        if not candidates:
            print(f"  {algo}: no candidates, skipping")
            continue
        print(f"\n  {algo} — {len(candidates)} candidates")
        selected_idx, surrogate_score = greedy_ransacov_chamfer(
            candidates,
            clean_points,   # evaluation-only; for production use, prefer points
            k_max=K,
            clip=None,      # use THRESHOLD or 1.5 * THRESHOLD with noisy points
            lambda_acc=1.0, # 1.0 ~ symmetric Chamfer; 0.2-0.5 favors coverage more
            max_swap_passes=2,
        )
        if not selected_idx:
            print(f"  {algo}: no subset selected, skipping")
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

        print(
            f"  {algo} | k={len(selected_models)}  surrogate={surrogate_score:.4f}"
            f"  CD={cd:.4f}  COV={cd_coverage:.4f}  ACC={cd_accuracy:.4f}  HD={hd:.4f}"
        )
        cover_results.append({
            "algo": algo,
            "k": len(selected_models),
            "surrogate_score": round(surrogate_score, 6),
            "chamfer": round(cd, 6),
            "cd_coverage": round(cd_coverage, 6),
            "cd_accuracy": round(cd_accuracy, 6),
            "hausdorff": round(hd, 6),
            "n_candidates": len(candidates),
        })

        if VISUALIZE:
            palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
            colors = [palette[i % len(palette)] for i in range(len(meshes))]
            vis.show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, models=selected_models)

    csv_cover = pc_out_dir / "results_setcover.csv"
    with open(csv_cover, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algo", "k", "surrogate_score", "chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "n_candidates"],
        )
        writer.writeheader()
        writer.writerows(cover_results)
    print(f"RansaCov-Chamfer results saved to {csv_cover}")


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


if __name__ == "__main__":
    main()
