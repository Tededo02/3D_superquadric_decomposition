import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import os
import sys
from pathlib import Path

import numpy as np
from point_cloud_utils import chamfer_distance
from scipy.spatial import cKDTree

import main_pc_import as main_pc
from src.gair_ransac.ransacov import ransacov
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.visualizations import visualization as vis


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_OBJECTS_DIR = PROJECT_ROOT / "test_objects"
TEST_OBJECT_FILE_ENV = "TEST_OBJECT_FILE"
DEFAULT_TEST_OBJECT_FILE = "mush.glb"
ALGORITHM = "gair-ransac"
RANSACOV_MAX_MODELS = 2
MAX_PARALLEL_PROCESSES = 6
SHOW_RANSACOV_RESULT = True
RANSACOV_PALETTE = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]


def resolve_default_input_meshes() -> tuple[Path, ...]:
    test_object_file = os.environ.get(TEST_OBJECT_FILE_ENV, DEFAULT_TEST_OBJECT_FILE).strip()
    if not test_object_file:
        test_object_file = DEFAULT_TEST_OBJECT_FILE

    relative_path = Path(test_object_file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"{TEST_OBJECT_FILE_ENV} must be a file path relative to {TEST_OBJECTS_DIR}, "
            f"got {test_object_file!r}"
        )

    return (TEST_OBJECTS_DIR / relative_path,)


DEFAULT_INPUT_MESHES = resolve_default_input_meshes()
DEFAULT_OUTPUT_CSV = Path("artifacts") / "main_pc_import_batch" / "consensus_std_results.csv"
DEFAULT_RANSACOV_OUTPUT_CSV = Path("artifacts") / "main_pc_import_batch" / "ransacov_results.csv"
RUN_FIELDNAMES = [
    "mesh_idx",
    "mesh_name",
    "run_idx",
    "seed",
    "status",
    "error",
    "input_mesh",
    "algorithm",
    "base_seed",
    "input_sampling_seed",
    "algorithm_seed",
    "evaluation_sampling_seed",
    "sampled_point_count",
    "n_models",
    "runtime_s",
    "threshold",
    "input_noise_std",
    "chamfer",
    "one_sided_chamfer",
    "saved_object_views_dir",
    "saved_object_views_count",
    "n_inliers",
    "n_outliers",
    "outlier_ratio",
    "gt_n_inliers",
    "gt_n_outliers",
    "gt_outlier_ratio",
    "gt_noise_std",
    "gt_classification_rate",
    "gt_misclassification_error",
    "gt_inliers_assumed_from_tail",
]

ARTIFACT_RESULT_KEYS = {"models", "sampled_points", "normals", "inliers_masks"}
TRANSIENT_RANSACOV_KEYS = {
    "_selected_models",
    "_selected_indices",
    "_selected_origins",
    "_selected_inlier_counts",
    "_selected_new_inlier_counts",
}

RANSACOV_FIELDNAMES = [
    "mesh_idx",
    "mesh_name",
    "status",
    "error",
    "algorithm",
    "n_runs",
    "start_seed",
    "seed_step",
    "seeds",
    "ransacov_k",
    "reference_seed",
    "reference_evaluation_sampling_seed",
    "threshold",
    "sampled_point_count",
    "n_candidates",
    "selected_count",
    "selected_candidate_indices",
    "selected_model_origins",
    "selected_inlier_counts",
    "selected_new_inlier_counts",
    "n_covered",
    "coverage_ratio",
    "chamfer",
    "one_sided_chamfer",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("n_runs", type=int, help="Number of seeds to evaluate.")
    parser.add_argument(
        "input_meshes",
        nargs="*",
        default=[str(path) for path in DEFAULT_INPUT_MESHES],
        help=(
            "Optional mesh list. If omitted, uses test_objects/${TEST_OBJECT_FILE}; "
            f"default: {DEFAULT_TEST_OBJECT_FILE}."
        ),
    )
    parser.add_argument("--start-seed", type=int, default=main_pc.DEFAULT_BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--ransacov-k", type=int, default=RANSACOV_MAX_MODELS)
    parser.add_argument("--ransacov-output-csv", default=str(DEFAULT_RANSACOV_OUTPUT_CSV))
    parser.add_argument("--max-parallel-processes", type=int, default=MAX_PARALLEL_PROCESSES)
    return parser.parse_args(argv)


def build_seeds(n_runs: int, start_seed: int, seed_step: int) -> list[int]:
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")
    if seed_step <= 0:
        raise ValueError(f"seed_step must be positive, got {seed_step}")
    return [start_seed + run_idx * seed_step for run_idx in range(n_runs)]


def resolve_output_path(output_csv: str | Path) -> Path:
    output_path = Path(output_csv).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def unsigned_radial_residual(model, points: np.ndarray) -> np.ndarray:
    return np.abs(superquadric_radial_residual(model, points))


def count_selected_model_inliers(
    selected_models: list[object],
    reference_points: np.ndarray,
    threshold: float,
) -> tuple[list[int], list[int]]:
    inlier_counts: list[int] = []
    new_inlier_counts: list[int] = []
    covered = np.zeros(reference_points.shape[0], dtype=bool)

    for model in selected_models:
        mask = unsigned_radial_residual(model, reference_points) < threshold
        inlier_counts.append(int(np.count_nonzero(mask)))
        new_inlier_counts.append(int(np.count_nonzero(mask & ~covered)))
        covered |= mask

    return inlier_counts, new_inlier_counts


def format_model_origin(origin: dict[str, int]) -> str:
    return (
        f"run={origin['run_idx']},"
        f"seed={origin['seed']},"
        f"model={origin['model_idx']}"
    )


def evaluate_selected_models(
    selected_models: list[object],
    reference_points: np.ndarray,
    evaluation_seed: int | None,
) -> tuple[float | None, float | None]:
    if not selected_models:
        return None, None

    selected_meshes = [supmesh.superquadric_mesh(model) for model in selected_models]
    sampled_estimated, _ = samp.sampling_sq(
        selected_meshes,
        n_points=main_pc.SAMPLED_POINT_COUNT,
        seed=evaluation_seed,
    )
    selected_points = np.vstack(sampled_estimated)
    chamfer = chamfer_distance(reference_points, selected_points)
    one_sided = cKDTree(selected_points).query(reference_points, k=1)[0].mean()
    return float(chamfer), float(one_sided)


def show_ransacov_result(
    selected_models: list[object],
    reference_points: np.ndarray,
    threshold: float,
) -> None:
    if not selected_models:
        return

    selected_meshes = [supmesh.superquadric_mesh(model) for model in selected_models]
    colors = [
        RANSACOV_PALETTE[i % len(RANSACOV_PALETTE)]
        for i in range(len(selected_meshes))
    ]
    vis.show_mesh_and_points(
        selected_meshes,
        pts=reference_points,
        point_size=5,
        show_bounds=True,
        colors=colors,
        models=selected_models,
        treshold=threshold,
    )


def build_empty_run_row(
    mesh_idx: int,
    mesh_path: Path,
    run_idx: int,
    seed: int,
    error: str,
) -> dict[str, object]:
    return {
        "mesh_idx": int(mesh_idx),
        "mesh_name": mesh_path.name,
        "run_idx": int(run_idx),
        "seed": int(seed),
        "status": "error",
        "error": error,
        "input_mesh": str(mesh_path),
        "algorithm": ALGORITHM,
        "base_seed": int(seed),
        "input_sampling_seed": None,
        "algorithm_seed": None,
        "evaluation_sampling_seed": None,
        "sampled_point_count": None,
        "n_models": None,
        "runtime_s": None,
        "threshold": None,
        "input_noise_std": None,
        "chamfer": None,
        "one_sided_chamfer": None,
        "saved_object_views_dir": None,
        "saved_object_views_count": None,
        "n_inliers": None,
        "n_outliers": None,
        "outlier_ratio": None,
        "gt_n_inliers": None,
        "gt_n_outliers": None,
        "gt_outlier_ratio": None,
        "gt_noise_std": None,
        "gt_classification_rate": None,
        "gt_misclassification_error": None,
        "gt_inliers_assumed_from_tail": False,
    }


def build_reference_input(
    mesh_path: Path,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    run_seeds = main_pc.build_run_seeds(seed)
    input_path = main_pc.resolve_input_path(mesh_path)
    loaded_input = main_pc.load_input_points_and_normals(
        input_path,
        sample_seed=run_seeds.input_sampling,
    )
    cloud_stats = main_pc.summarize_point_cloud(loaded_input.points)
    threshold = main_pc.get_threshold(
        float(loaded_input.noise_std),
        median_nn_distance=cloud_stats.median_nn_distance,
    )
    return (
        np.asarray(loaded_input.points, dtype=np.float64),
        float(threshold),
        int(run_seeds.evaluation_sampling),
    )


def strip_artifact_result(result: dict[str, object]) -> tuple[dict[str, object], list[object]]:
    models = list(result.get("models") or [])
    csv_result = {
        key: value
        for key, value in result.items()
        if key not in ARTIFACT_RESULT_KEYS
    }
    return csv_result, models


def strip_transient_ransacov_fields(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in TRANSIENT_RANSACOV_KEYS
    }


def run_seed_worker(mesh_path: str, run_idx: int, seed: int) -> dict[str, object]:
    stderr_buffer = io.StringIO()
    with open(os.devnull, "w", encoding="utf-8") as stdout_sink:
        with redirect_stdout(stdout_sink), redirect_stderr(stderr_buffer):
            try:
                # Parallel workers would otherwise all write the same GAIR debug filenames.
                main_pc.DEBUG_V = False
                result = main_pc.create_and_estimate_supq(
                    mesh_path,
                    algorithm=ALGORITHM,
                    base_seed=seed,
                    visualize=False,
                    save_debug_views=False,
                    save_object_views=False,
                    return_artifacts=True,
                )
                csv_result, models = strip_artifact_result(result)
            except Exception as exc:
                return {
                    "ok": False,
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "error": str(exc),
                    "stderr": stderr_buffer.getvalue(),
                }

    return {
        "ok": True,
        "run_idx": int(run_idx),
        "seed": int(seed),
        "result": csv_result,
        "models": models,
        "stderr": stderr_buffer.getvalue(),
    }


def run_seed_jobs_for_mesh(
    mesh_path: Path,
    seeds: list[int],
    max_parallel_processes: int,
) -> list[dict[str, object]]:
    max_workers = min(max_parallel_processes, len(seeds))
    if max_workers <= 1:
        return [
            run_seed_worker(str(mesh_path), run_idx, seed)
            for run_idx, seed in enumerate(seeds)
        ]

    jobs: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {
            executor.submit(run_seed_worker, str(mesh_path), run_idx, seed): (run_idx, seed)
            for run_idx, seed in enumerate(seeds)
        }
        for future in as_completed(future_to_run):
            run_idx, seed = future_to_run[future]
            try:
                job = future.result()
            except Exception as exc:
                job = {
                    "ok": False,
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "error": str(exc),
                    "stderr": "",
                }
            jobs.append(job)
            print(f"  completed seed {seed} ({len(jobs)}/{len(seeds)})")

    return sorted(jobs, key=lambda item: int(item["run_idx"]))


def run_ransacov_for_mesh(
    mesh_idx: int,
    mesh_path: Path,
    seeds: list[int],
    start_seed: int,
    seed_step: int,
    ransacov_k: int,
    candidates: list[object],
    candidate_origins: list[dict[str, int]],
    reference_points: np.ndarray | None,
    reference_threshold: float | None,
    reference_seed: int | None,
    reference_evaluation_seed: int | None,
) -> dict[str, object]:
    base_row = {
        "mesh_idx": int(mesh_idx),
        "mesh_name": mesh_path.name,
        "algorithm": ALGORITHM,
        "n_runs": int(len(seeds)),
        "start_seed": int(start_seed),
        "seed_step": int(seed_step),
        "seeds": ",".join(str(seed) for seed in seeds),
        "ransacov_k": int(ransacov_k),
        "reference_seed": reference_seed,
        "reference_evaluation_sampling_seed": reference_evaluation_seed,
        "threshold": reference_threshold,
        "sampled_point_count": None if reference_points is None else int(reference_points.shape[0]),
        "n_candidates": int(len(candidates)),
        "selected_count": 0,
        "selected_candidate_indices": "",
        "selected_model_origins": "",
        "selected_inlier_counts": "",
        "selected_new_inlier_counts": "",
        "n_covered": 0,
        "coverage_ratio": None,
        "chamfer": None,
        "one_sided_chamfer": None,
    }

    if reference_points is None or reference_threshold is None:
        return {
            **base_row,
            "status": "error",
            "error": "no successful run produced a reference point cloud",
        }
    if not candidates:
        return {
            **base_row,
            "status": "no-candidates",
            "error": "no models were produced by the K GAIR-RANSAC runs",
        }

    try:
        selected_indices, n_covered = ransacov(
            candidates,
            reference_points,
            k=min(int(ransacov_k), len(candidates)),
            threshold=float(reference_threshold),
            residual_fn=unsigned_radial_residual,
        )
        selected_models = [candidates[idx] for idx in selected_indices]
        chamfer, one_sided_chamfer = evaluate_selected_models(
            selected_models,
            reference_points,
            reference_evaluation_seed,
        )
        selected_inlier_counts, selected_new_inlier_counts = count_selected_model_inliers(
            selected_models,
            reference_points,
            float(reference_threshold),
        )
    except Exception as exc:
        return {
            **base_row,
            "status": "error",
            "error": str(exc),
        }

    selected_origins = [format_model_origin(candidate_origins[idx]) for idx in selected_indices]
    return {
        **base_row,
        "status": "ok",
        "error": "",
        "selected_count": int(len(selected_indices)),
        "selected_candidate_indices": ",".join(str(idx) for idx in selected_indices),
        "selected_model_origins": ";".join(selected_origins),
        "selected_inlier_counts": ",".join(str(count) for count in selected_inlier_counts),
        "selected_new_inlier_counts": ",".join(str(count) for count in selected_new_inlier_counts),
        "n_covered": int(n_covered),
        "coverage_ratio": float(n_covered / reference_points.shape[0]),
        "chamfer": chamfer,
        "one_sided_chamfer": one_sided_chamfer,
        "_selected_models": selected_models,
        "_selected_indices": selected_indices,
        "_selected_origins": selected_origins,
        "_selected_inlier_counts": selected_inlier_counts,
        "_selected_new_inlier_counts": selected_new_inlier_counts,
    }


def run_batch(
    n_runs: int = 10,
    input_meshes: list[str | Path] | tuple[str | Path, ...] = DEFAULT_INPUT_MESHES,
    start_seed: int = main_pc.DEFAULT_BASE_SEED,
    seed_step: int = 1,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    ransacov_k: int = RANSACOV_MAX_MODELS,
    ransacov_output_csv: str | Path = DEFAULT_RANSACOV_OUTPUT_CSV,
    max_parallel_processes: int = MAX_PARALLEL_PROCESSES,
) -> tuple[list[dict[str, object]], Path, list[dict[str, object]], Path]:
    if ransacov_k <= 0:
        raise ValueError(f"ransacov_k must be positive, got {ransacov_k}")
    if max_parallel_processes <= 0:
        raise ValueError(
            f"max_parallel_processes must be positive, got {max_parallel_processes}"
        )

    output_path = resolve_output_path(output_csv)
    ransacov_output_path = resolve_output_path(ransacov_output_csv)

    rows: list[dict[str, object]] = []
    ransacov_rows: list[dict[str, object]] = []
    seeds = build_seeds(n_runs, start_seed, seed_step)
    mesh_paths = [Path(mesh) for mesh in input_meshes]

    for mesh_idx, mesh_path in enumerate(mesh_paths):
        print(f"\n=== mesh {mesh_path.name} ({mesh_idx + 1}/{len(mesh_paths)}) ===")
        candidates: list[object] = []
        candidate_origins: list[dict[str, int]] = []
        reference_points: np.ndarray | None = None
        reference_threshold: float | None = None
        reference_seed: int | None = None
        reference_evaluation_seed: int | None = None

        try:
            reference_points, reference_threshold, reference_evaluation_seed = (
                build_reference_input(mesh_path, seeds[0])
            )
            reference_seed = int(seeds[0])
        except Exception as exc:
            print(f"Could not build reference point cloud for RansaCov: {exc}")

        max_workers = min(max_parallel_processes, len(seeds))
        print(
            f"Running {len(seeds)} seed runs with up to "
            f"{max_workers} parallel process(es) ..."
        )
        run_jobs = run_seed_jobs_for_mesh(mesh_path, seeds, max_parallel_processes)

        for job in run_jobs:
            run_idx = int(job["run_idx"])
            seed = int(job["seed"])
            print(f"\n--- seed {seed} ({run_idx + 1}/{len(seeds)}) ---")
            print(f"Finished {ALGORITHM} on {mesh_path.name}.")
            try:
                if not job["ok"]:
                    raise RuntimeError(str(job["error"]))

                result = job["result"]
                models = list(job.get("models") or [])
                for model_idx, model in enumerate(models):
                    candidates.append(model)
                    candidate_origins.append(
                        {
                            "run_idx": int(run_idx),
                            "seed": int(seed),
                            "model_idx": int(model_idx),
                        }
                    )

                row = {
                    "mesh_idx": int(mesh_idx),
                    "mesh_name": mesh_path.name,
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "status": "ok",
                    "error": "",
                    **result,
                }
                chamfer_text = (
                    "n/a"
                    if row["chamfer"] is None
                    else f"{row['chamfer']:.4f}"
                )
                print(
                    f"  ok | models={row['n_models']} "
                    f"runtime_s={row['runtime_s']:.3f} "
                    f"chamfer={chamfer_text}"
                    + (
                        f" mis={row['gt_misclassification_error']:.4f}"
                        if row["gt_misclassification_error"] is not None
                        else ""
                    )
                )
            except Exception as exc:
                row = build_empty_run_row(mesh_idx, mesh_path, run_idx, seed, str(exc))
                stderr_text = str(job.get("stderr") or "").strip()
                if stderr_text:
                    print(stderr_text)
                print(f"  error | {exc}")
            rows.append(row)

        ransacov_row = run_ransacov_for_mesh(
            mesh_idx=mesh_idx,
            mesh_path=mesh_path,
            seeds=seeds,
            start_seed=start_seed,
            seed_step=seed_step,
            ransacov_k=ransacov_k,
            candidates=candidates,
            candidate_origins=candidate_origins,
            reference_points=reference_points,
            reference_threshold=reference_threshold,
            reference_seed=reference_seed,
            reference_evaluation_seed=reference_evaluation_seed,
        )
        print(
            f"\nRansaCov | status={ransacov_row['status']} "
            f"candidates={ransacov_row['n_candidates']} "
            f"selected={ransacov_row['selected_count']} "
            f"covered={ransacov_row['n_covered']}"
        )
        if ransacov_row["status"] == "ok":
            selected_indices = list(ransacov_row.get("_selected_indices") or [])
            selected_origins = list(ransacov_row.get("_selected_origins") or [])
            selected_inlier_counts = list(
                ransacov_row.get("_selected_inlier_counts") or []
            )
            selected_new_inlier_counts = list(
                ransacov_row.get("_selected_new_inlier_counts") or []
            )
            print("RansaCov selected model inliers:")
            for rank, (candidate_idx, origin, inlier_count, new_inlier_count) in enumerate(
                zip(
                    selected_indices,
                    selected_origins,
                    selected_inlier_counts,
                    selected_new_inlier_counts,
                ),
                start=1,
            ):
                print(
                    f"  #{rank} candidate={candidate_idx} {origin} | "
                    f"inliers={inlier_count} new_inliers={new_inlier_count}"
                )
        if (
            SHOW_RANSACOV_RESULT
            and ransacov_row["status"] == "ok"
            and reference_points is not None
            and reference_threshold is not None
        ):
            show_ransacov_result(
                list(ransacov_row.get("_selected_models") or []),
                reference_points,
                float(reference_threshold),
            )
        ransacov_rows.append(strip_transient_ransacov_fields(ransacov_row))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with ransacov_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANSACOV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ransacov_rows)

    return rows, output_path, ransacov_rows, ransacov_output_path


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    rows, output_path, ransacov_rows, ransacov_output_path = run_batch(
        n_runs=args.n_runs,
        input_meshes=args.input_meshes,
        start_seed=args.start_seed,
        seed_step=args.seed_step,
        output_csv=args.output_csv,
        ransacov_k=args.ransacov_k,
        ransacov_output_csv=args.ransacov_output_csv,
        max_parallel_processes=args.max_parallel_processes,
    )
    failures = sum(row["status"] != "ok" for row in rows)
    ransacov_failures = sum(row["status"] != "ok" for row in ransacov_rows)
    print(f"\nSaved CSV -> {output_path}")
    print(f"Saved RansaCov CSV -> {ransacov_output_path}")
    if failures or ransacov_failures:
        print(
            f"Completed with {failures} failed GAIR-RANSAC runs "
            f"and {ransacov_failures} failed RansaCov selections."
        )
        return 1
    print(f"Completed {len(rows)} GAIR-RANSAC runs successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
