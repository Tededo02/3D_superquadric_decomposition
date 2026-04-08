import csv
import hashlib
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
from src.superquadrics import superquadric_residual as supres
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
NOISE      = 0           # gaussian noise std applied to positions and normals (0.0 = none)
# New selection config block
CANDIDATE_SURFACE_POINTS = 1200
EVAL_SURFACE_POINTS = 4000
SELECTION_POINT_SOURCE = "observed"   # use observed points for model selection; "clean" only for benchmarking
RANSACOV_REFIT_CANDIDATES = True       # refit every tentative model on its consensus set before selection
RANSACOV_USE_NORMALS = True            # include normal agreement in the consensus test when normals are available
RANSACOV_NORMAL_ANGLE_DEG = 35.0       # max angle between point normal and model normal
RANSACOV_MIN_INLIERS = 30              # discard tiny consensus sets
RANSACOV_GREEDY = True                 # greedy maximum-coverage approximation (RansaCov-style)
# ──────────────────────────────────────────────────────────────────────────────


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


def _model_seed(model, purpose: str) -> int:
    params = np.asarray(
        [model.a1, model.a2, model.a3, model.e1, model.e2, *model.rot, *model.t],
        dtype=np.float64,
    )
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(np.asarray([EVAL_SEED], dtype=np.int64).tobytes())
    hasher.update(purpose.encode("ascii"))
    hasher.update(params.tobytes())
    return int.from_bytes(hasher.digest(), "little") % np.iinfo(np.int32).max


def sample_model_surface(model, n_points: int, purpose: str):
    mesh = supmesh.superquadric_mesh(model)
    sampled, sampled_normals = samp.sampling_sq_random(
        [mesh],
        n_points=n_points,
        seed=_model_seed(model, purpose),
    )
    return (
        np.asarray(sampled[0], dtype=np.float64),
        np.asarray(sampled_normals[0], dtype=np.float64),
    )


def sample_models_surface(models, n_points: int, purpose: str):
    points_list = []
    normals_list = []
    for model in models:
        surf, surf_normals = sample_model_surface(model, n_points=n_points, purpose=purpose)
        points_list.append(surf)
        normals_list.append(surf_normals)
    return points_list, normals_list


def evaluate_models(reference_points: np.ndarray, models, purpose: str):
    sampled_est, _ = sample_models_surface(models, n_points=EVAL_SURFACE_POINTS, purpose=purpose)
    est_pts = np.vstack(sampled_est)

    cd = chamfer_distance(reference_points, est_pts)
    tree_est = cKDTree(est_pts)
    tree_ref = cKDTree(reference_points)
    d_ref_to_est = tree_est.query(reference_points, k=1)[0]
    d_est_to_ref = tree_ref.query(est_pts, k=1)[0]

    return est_pts, {
        "chamfer": float(cd),
        "cd_coverage": float(d_ref_to_est.mean()),
        "cd_accuracy": float(d_est_to_ref.mean()),
        "hausdorff": float(max(d_ref_to_est.max(), d_est_to_ref.max())),
    }


def resolve_selection_data(points, normals, clean_points, clean_normals):
    if SELECTION_POINT_SOURCE == "observed":
        return points, normals, "points"
    if SELECTION_POINT_SOURCE == "clean":
        return clean_points, clean_normals, "clean_points"
    raise ValueError(f"Unsupported SELECTION_POINT_SOURCE: {SELECTION_POINT_SOURCE}")


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

    if points.shape != normals.shape:
        raise ValueError(
            f"Points/normals shape mismatch for {path.name}: "
            f"points={points.shape}, normals={normals.shape}"
        )

    return points, normalize_vectors(normals)


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
    _, metrics = evaluate_models(clean_points, models, purpose="trial-evaluation")

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

    return {
        **metrics,
        "classification_rate": classification_rate,
        "runtime_s": runtime,
        "n_models": len(models),
        "models": models,
    }




# --- RansaCov maximum-coverage selection implementation ---

def compute_model_inliers(model, points, normals=None, threshold=THRESHOLD, use_normals=False, normal_angle_deg=35.0):
    """
    Binary consensus set for a superquadric on the observed point cloud.
    This follows the RansaCov formulation: each candidate induces a set of inliers.
    """
    points = np.asarray(points, dtype=np.float64)
    residuals = np.asarray(supres.radial_euclidean_distance(model, points), dtype=np.float64)
    mask = residuals < threshold

    if use_normals and normals is not None:
        normals_unit = normalize_vectors(normals)
        model_normals = supres.superquadric_normal_world(model, points)
        alignment = np.clip(
            np.einsum("ij,ij->i", model_normals, normals_unit, optimize=True),
            -1.0,
            1.0,
        )
        cos_thr = float(np.cos(np.deg2rad(normal_angle_deg)))
        mask &= alignment >= cos_thr

    return mask, residuals


def refit_candidate_from_inliers(model, points, normals, threshold=THRESHOLD, use_normals=False, normal_angle_deg=35.0):
    """
    RansaCov preprocessing step: refit each tentative structure on its consensus set,
    and keep the refined model if consensus increases.
    """
    mask, residuals = compute_model_inliers(
        model,
        points,
        normals=normals,
        threshold=threshold,
        use_normals=use_normals,
        normal_angle_deg=normal_angle_deg,
    )

    if mask.sum() < 9:
        return model, mask, residuals

    try:
        refit = supres.fit_radial_euclidean(points[mask], model)
    except Exception:
        return model, mask, residuals

    refit_mask, refit_residuals = compute_model_inliers(
        refit,
        points,
        normals=normals,
        threshold=threshold,
        use_normals=use_normals,
        normal_angle_deg=normal_angle_deg,
    )

    if refit_mask.sum() >= mask.sum():
        return refit, refit_mask, refit_residuals
    return model, mask, residuals


def build_ransacov_candidates(
    candidates,
    points,
    normals=None,
    threshold=THRESHOLD,
    use_normals=False,
    normal_angle_deg=35.0,
    min_inliers=1,
    refit=True,
):
    """
    Build consensus sets for all tentative models, optionally refining each model
    on its current consensus set as suggested in RansaCov preprocessing.
    """
    prepared = []
    for model in candidates:
        if refit:
            model, mask, residuals = refit_candidate_from_inliers(
                model,
                points,
                normals,
                threshold=threshold,
                use_normals=use_normals,
                normal_angle_deg=normal_angle_deg,
            )
        else:
            mask, residuals = compute_model_inliers(
                model,
                points,
                normals=normals,
                threshold=threshold,
                use_normals=use_normals,
                normal_angle_deg=normal_angle_deg,
            )

        support = int(mask.sum())
        if support < min_inliers:
            continue

        prepared.append({
            "model": model,
            "mask": mask.copy(),
            "residuals": np.asarray(residuals, dtype=np.float64),
            "support": support,
        })

    prepared.sort(key=lambda item: item["support"], reverse=True)
    return prepared


def prune_ransacov_candidates(prepared):
    """
    RansaCov preprocessing: discard any consensus set fully contained in the union
    of larger consensus sets processed before it.
    """
    kept = []
    if not prepared:
        return kept

    covered = np.zeros_like(prepared[0]["mask"], dtype=bool)
    for item in prepared:
        mask = item["mask"]
        if np.all(mask <= covered):
            continue
        kept.append(item)
        covered |= mask

    return kept


def greedy_maximum_coverage(prepared, k_max):
    """
    Greedy approximation for RansaCov maximum coverage.
    At each step, pick the model that covers the largest number of still-uncovered points.
    """
    if not prepared or k_max <= 0:
        return [], np.zeros(0, dtype=bool)

    covered = np.zeros_like(prepared[0]["mask"], dtype=bool)
    selected = []
    available = list(range(len(prepared)))

    for _ in range(k_max):
        best_idx = None
        best_gain = 0
        for idx in available:
            gain = int(np.count_nonzero(prepared[idx]["mask"] & ~covered))
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None or best_gain == 0:
            break

        selected.append(best_idx)
        covered |= prepared[best_idx]["mask"]
        available.remove(best_idx)

    return selected, covered


def exact_maximum_coverage(prepared, k_max):
    """
    Small-scale exact search for maximum coverage. This is exponential, therefore it is
    intended only as an optional diagnostic alternative to the greedy RansaCov selection.
    """
    from itertools import combinations

    if not prepared or k_max <= 0:
        return [], np.zeros(0, dtype=bool)

    n = len(prepared)
    k_eff = min(k_max, n)
    best_combo = []
    best_covered = np.zeros_like(prepared[0]["mask"], dtype=bool)
    best_score = -1

    for r in range(1, k_eff + 1):
        for combo in combinations(range(n), r):
            union = np.zeros_like(prepared[0]["mask"], dtype=bool)
            for idx in combo:
                union |= prepared[idx]["mask"]
            score = int(union.sum())
            if score > best_score:
                best_score = score
                best_combo = list(combo)
                best_covered = union

    return best_combo, best_covered


def select_ransacov_subset(
    candidates,
    points,
    k_max,
    normals=None,
    threshold=THRESHOLD,
    use_normals=False,
    normal_angle_deg=35.0,
    min_inliers=1,
    refit=True,
    greedy=True,
):
    prepared = build_ransacov_candidates(
        candidates,
        points,
        normals=normals,
        threshold=threshold,
        use_normals=use_normals,
        normal_angle_deg=normal_angle_deg,
        min_inliers=min_inliers,
        refit=refit,
    )
    prepared = prune_ransacov_candidates(prepared)

    if greedy:
        selected_idx, covered = greedy_maximum_coverage(prepared, k_max)
    else:
        selected_idx, covered = exact_maximum_coverage(prepared, k_max)

    selected_models = [prepared[i]["model"] for i in selected_idx]
    selected_masks = [prepared[i]["mask"] for i in selected_idx]

    return selected_models, selected_masks, prepared, covered


def run_for_pc(pc_file: Path, rng):
    print(f"\n{'='*60}")
    print(f"Point cloud: {pc_file.name}")
    print(f"{'='*60}")
    points, normals = load_point_cloud(pc_file)
    print(f"  {points.shape[0]} points loaded")
    clean_points = points.copy()
    clean_normals = normals.copy()
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

    # ── RansaCov maximum-coverage selection on observed consensus sets ───────
    print(f"\n=== RansaCov (maximum coverage, k_max={K}) ===")
    selection_points, selection_normals, selection_label = resolve_selection_data(
        points,
        normals,
        clean_points,
        clean_normals,
    )
    print(
        "  selection config:"
        f" source={selection_label}"
        f" threshold={THRESHOLD}"
        f" refit={RANSACOV_REFIT_CANDIDATES}"
        f" use_normals={RANSACOV_USE_NORMALS}"
        f" normal_angle_deg={RANSACOV_NORMAL_ANGLE_DEG}"
        f" min_inliers={RANSACOV_MIN_INLIERS}"
        f" greedy={RANSACOV_GREEDY}"
    )
    cover_results = []
    for algo in algorithms:
        candidates = all_candidates[algo]
        if not candidates:
            print(f"  {algo}: no candidates, skipping")
            continue
        print(f"\n  {algo} — {len(candidates)} raw candidates")
        selected_models, selected_masks, prepared, covered = select_ransacov_subset(
            candidates,
            selection_points,
            k_max=K,
            normals=selection_normals,
            threshold=THRESHOLD,
            use_normals=RANSACOV_USE_NORMALS,
            normal_angle_deg=RANSACOV_NORMAL_ANGLE_DEG,
            min_inliers=RANSACOV_MIN_INLIERS,
            refit=RANSACOV_REFIT_CANDIDATES,
            greedy=RANSACOV_GREEDY,
        )
        if not selected_models:
            print(f"  {algo}: no subset selected, skipping")
            continue

        meshes = [supmesh.superquadric_mesh(m) for m in selected_models]
        _, cover_metrics = evaluate_models(clean_points, selected_models, purpose="setcover-evaluation")
        covered_points = int(covered.sum()) if covered.size else 0
        coverage_ratio = covered_points / len(selection_points) if len(selection_points) > 0 else 0.0

        print(
            f"  {algo} | k={len(selected_models)}"
            f"  prepared={len(prepared)}"
            f"  covered={covered_points}/{len(selection_points)} ({coverage_ratio*100:.1f}%)"
            f"  CD={cover_metrics['chamfer']:.4f}"
            f"  COV={cover_metrics['cd_coverage']:.4f}"
            f"  ACC={cover_metrics['cd_accuracy']:.4f}"
            f"  HD={cover_metrics['hausdorff']:.4f}"
        )
        cover_results.append({
            "algo": algo,
            "k": len(selected_models),
            "prepared_candidates": len(prepared),
            "selection_source": selection_label,
            "selection_threshold": round(float(THRESHOLD), 6),
            "selection_refit": int(RANSACOV_REFIT_CANDIDATES),
            "selection_use_normals": int(RANSACOV_USE_NORMALS),
            "selection_normal_angle_deg": round(float(RANSACOV_NORMAL_ANGLE_DEG), 6),
            "covered_points": covered_points,
            "coverage_ratio": round(coverage_ratio, 6),
            "chamfer": round(cover_metrics["chamfer"], 6),
            "cd_coverage": round(cover_metrics["cd_coverage"], 6),
            "cd_accuracy": round(cover_metrics["cd_accuracy"], 6),
            "hausdorff": round(cover_metrics["hausdorff"], 6),
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
            fieldnames=[
                "algo",
                "k",
                "prepared_candidates",
                "selection_source",
                "selection_threshold",
                "selection_refit",
                "selection_use_normals",
                "selection_normal_angle_deg",
                "covered_points",
                "coverage_ratio",
                "chamfer",
                "cd_coverage",
                "cd_accuracy",
                "hausdorff",
                "n_candidates",
            ],
        )
        writer.writeheader()
        writer.writerows(cover_results)
    print(f"RansaCov results saved to {csv_cover}")


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
