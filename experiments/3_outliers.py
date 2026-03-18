import sys
import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.visualizations import visualization as vis
import numpy as np
from src.superquadrics import superquadric_sampling as samp
from src.gair_ransac.inner_ransac import InnerRansacResult, inner_ransac, fit_superquadric_ls
from src.gair_ransac.ransac import ransac
from point_cloud_utils import k_nearest_neighbors, chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.gair_ransac import gair_ransac

NOISE_STD = 0.4
THRESHOLD = 2.5 * NOISE_STD
ITERATIONS_STEP = 1

def create_and_estimate_supq():
    gt_params = [
    SuperQuadricParams(9.0, 9.0, 9.0, 3.5, 2.09, [2.0, 2.0, 1.0], [5.0, 5.0, 5.0]),
    SuperQuadricParams(3.0, 3.0, 3.0, 0.5, 0.9, [2.0, 2.0, 1.0], [-5.0, -5.0, -5.0]),
    SuperQuadricParams(4.0, 4.0, 4.0, 0.8, 1.1, [1.7, 2.1, 0.9], [13.0, 5.0, 5.0]),
    SuperQuadricParams(2.5, 2.5, 2.5, 0.7, 1.2, [2.1, 1.9, 0.8], [-10.5, -5.0, -5.0])
    ]
    outlier_values = list(range(0, 4001, 500))  # [0, 1000, 2000, 3000, 4000]
    gt_meshes = [supmesh.superquadric_mesh(p) for p in gt_params]
    algorithms = ["gair-ransac", "ransac"]
    res_chamfers          = {a: {s: [] for s in outlier_values} for a in algorithms}
    res_hausdorff         = {a: {s: [] for s in outlier_values} for a in algorithms}
    res_misclassification = {a: {s: [] for s in outlier_values} for a in algorithms}
    res_runtime           = {a: {s: [] for s in outlier_values} for a in algorithms}

    for n_out in outlier_values:
        for algo in algorithms:
            for trial in range(ITERATIONS_STEP):
                print(f"Outliers: {n_out}; Algo: {algo}; Trial: {trial}")
                seed = trial * 1000 + n_out
                sampled_points_noisy, normals_sp_noisy = samp.sampling_sq_noisy(gt_meshes, n_points=30000, noise_std=NOISE_STD, clip_k=3.0, seed=seed)
                sampled_points_random, _ = samp.sampling_sq_random(gt_meshes, n_points=4000, seed=seed)
                sampled_points_random = np.vstack(sampled_points_random)
                sampled_points_outliers, normals_sp_outliers = samp.sampling_outliers(gt_meshes, n_out=n_out, margin=0.10, mode="uniform", seed=seed + 10000)
                total_best_mss_used = None
                sampled_points = np.vstack([*sampled_points_noisy, sampled_points_outliers])
                normals = np.vstack([*normals_sp_noisy, normals_sp_outliers])
                list_mesh = [supmesh.superquadric_mesh(p) for p in gt_params]
                colors = ["lightblue"] * len(list_mesh)
                n_gt = len(list_mesh)
                mesh_estimated = None

                t_start = time.time()
                if algo == "ransac":
                    graph_radius = 0.08
                    models, inliers_masks = ransac(sampled_points, threshold=THRESHOLD, max_models=len(list_mesh), max_iterations=10, inner_iterations=40, radius=graph_radius, graphcut=True)
                    if not models:
                        raise RuntimeError("ransac did not return any model")
                    for model in models:
                        mesh_estimated = supmesh.superquadric_mesh(model)
                        list_mesh.append(mesh_estimated)
                        colors.append("lightgreen")
                elif algo == "gair-ransac":
                    graph_radius = 0.08
                    models, inliers_masks, total_best_mss_used = gair_ransac(sampled_points, normals, threshold=THRESHOLD, max_models=len(list_mesh), max_iterations=10, inner_iterations=40, radius=graph_radius)
                    if not models:
                        raise RuntimeError("gair_ransac did not return any model")
                    for model in models:
                        mesh_estimated = supmesh.superquadric_mesh(model)
                        list_mesh.append(mesh_estimated)
                        colors.append("lightgreen")

                elapsed = time.time() - t_start
                res_runtime[algo][n_out].append(elapsed)
                print(f"runtime = {elapsed:.2f}s")

                estimated_meshes = list_mesh[n_gt:]
                sampled_estimated, _ = samp.sampling_sq_random(estimated_meshes, n_points=4000, seed=50)
                sample_from_supq_estimated = np.vstack(sampled_estimated)
                _, index_sample_to_estimate = k_nearest_neighbors(sampled_points, sample_from_supq_estimated, k=1)
                index_sample_to_estimate = np.asarray(index_sample_to_estimate).reshape(-1)

                cd_one_side = np.linalg.norm(sample_from_supq_estimated[index_sample_to_estimate] - sampled_points, axis=1, ord=2).mean()
                print(f"one-side chamfer distance = {cd_one_side:.4f}")

                cd = chamfer_distance(sampled_points, sample_from_supq_estimated)
                print(f"symmetric chamfer distance = {cd:.4f}")
                res_chamfers[algo][n_out].append(chamfer_distance(sampled_points_random, sample_from_supq_estimated))
                print(f"symmetric chamfer distance from exact mesh = {res_chamfers[algo][n_out][-1]:.4f}")

                tree_est = cKDTree(sample_from_supq_estimated)
                tree_gt  = cKDTree(sampled_points_random)
                d_gt_to_est = tree_est.query(sampled_points_random, k=1)[0]
                d_est_to_gt = tree_gt.query(sample_from_supq_estimated, k=1)[0]
                res_hausdorff[algo][n_out].append(max(d_gt_to_est.max(), d_est_to_gt.max()))
                print(f"Hausdorff distance = {res_hausdorff[algo][n_out][-1]:.4f}")

                n_surface = sum(len(pts) for pts in sampled_points_noisy)
                gt_inlier_mask = np.zeros(sampled_points.shape[0], dtype=bool)
                gt_inlier_mask[:n_surface] = True
                if inliers_masks:
                    combined = inliers_masks[0].copy()
                    for m in inliers_masks[1:]:
                        combined |= m
                else:
                    combined = np.zeros(sampled_points.shape[0], dtype=bool)
                accuracy = float(np.mean(combined == gt_inlier_mask))
                res_misclassification[algo][n_out].append(accuracy)
                print(f"accuracy = {accuracy:.4f}")

                tau = 0.05
                precision = (d_est_to_gt < tau).mean()
                recall    = (d_gt_to_est < tau).mean()
                fs = 2 * precision * recall / (precision + recall + 1e-8)
                print(f"F1-score (tau={tau}) = {fs:.4f}")

                inlier_mask = None
                if inliers_masks:
                    inlier_mask = inliers_masks[0].copy()
                    for mask in inliers_masks[1:]:
                        inlier_mask |= mask
                if inlier_mask is not None:
                    n_inliers = inlier_mask.sum()
                    n_outliers_found = len(inlier_mask) - n_inliers
                    print(f"Inliers: {n_inliers} | Outliers: {n_outliers_found} | Total: {len(inlier_mask)} | Outlier ratio: {n_outliers_found/len(inlier_mask):.2%}")
                    print(f"---------------------------------------")
                vis.show_mesh_and_points(
                    list_mesh,
                    pts=sampled_points,
                    point_size=5,
                    show_bounds=True,
                    colors=colors,
                    inlier_mask=inlier_mask,
                    mss_used=total_best_mss_used if algo == "gair-ransac" else None,
                )
            hausdorff = np.mean(res_hausdorff[algo][n_out])
            chamfer = np.mean(res_chamfers[algo][n_out])
            runtime = np.mean(res_runtime[algo][n_out])
            print(f"[avg] n_out={n_out} algo={algo}  chamfer={chamfer:.4f}  hausdorff={hausdorff:.4f}  runtime={runtime:.2f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(__file__).resolve().parent / "artifacts" / "3_outliers"
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_arr = np.array(outlier_values)
    colors_algo = {"gair-ransac": "steelblue", "ransac": "darkorange"}
    for metric_name, res in [("chamfer", res_chamfers), ("hausdorff", res_hausdorff), ("accuracy", res_misclassification), ("runtime_s", res_runtime)]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for algo in algorithms:
            means = np.array([np.nanmean(res[algo][s]) for s in outlier_values])
            stds  = np.array([np.nanstd(res[algo][s])  for s in outlier_values])
            ax.plot(outlier_arr, means, marker="o", linewidth=2, label=algo, color=colors_algo[algo])
            ax.fill_between(outlier_arr, means - stds, means + stds, alpha=0.15, color=colors_algo[algo])
        ax.set_xlabel("number of outliers")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Outliers vs {metric_name}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"outliers_vs_{metric_name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved → {path}")

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "n_outliers", "trial", "chamfer", "hausdorff", "accuracy", "runtime_s"])
        writer.writeheader()
        for algo in algorithms:
            for s in outlier_values:
                for i in range(len(res_chamfers[algo][s])):
                    writer.writerow({
                        "algo": algo, "n_outliers": s, "trial": i,
                        "chamfer":    res_chamfers[algo][s][i],
                        "hausdorff":  res_hausdorff[algo][s][i],
                        "accuracy":   res_misclassification[algo][s][i],
                        "runtime_s":  res_runtime[algo][s][i],
                    })
    print(f"Saved CSV → {csv_path}")

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    create_and_estimate_supq()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
