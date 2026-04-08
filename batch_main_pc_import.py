import argparse
import csv
import sys
from pathlib import Path

from main_pc_import import DEFAULT_BASE_SEED, create_and_estimate_supq


PROJECT_ROOT = Path(__file__).resolve().parent
ALGORITHMS = (
    "gair-ransac",
    "gc-ransac",
    "gair-mss-no-normals",
    "gc-mss-no-normals",
)
DEFAULT_INPUT_MESHES = (
    PROJECT_ROOT / "test_objects" / "131969.stl",
    PROJECT_ROOT / "test_objects" / "cartoon_character.glb",
    PROJECT_ROOT / "test_objects" / "anthropomorphic_mushroom_character.glb",
)
DEFAULT_OUTPUT_CSV = Path("artifacts") / "main_pc_import_batch" / "consensus_std_results.csv"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("n_runs", type=int, help="Number of seeds to evaluate.")
    parser.add_argument(
        "input_meshes",
        nargs="*",
        default=[str(path) for path in DEFAULT_INPUT_MESHES],
        help="Optional mesh list. Default: 131969.stl, cartoon_character.glb, anthropomorphic_mushroom_character.glb",
    )
    parser.add_argument("--start-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args(argv)


def build_seeds(n_runs: int, start_seed: int, seed_step: int) -> list[int]:
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")
    if seed_step <= 0:
        raise ValueError(f"seed_step must be positive, got {seed_step}")
    return [start_seed + run_idx * seed_step for run_idx in range(n_runs)]


def run_batch(
    n_runs: int = 10,
    input_meshes: list[str | Path] | tuple[str | Path, ...] = DEFAULT_INPUT_MESHES,
    start_seed: int = DEFAULT_BASE_SEED,
    seed_step: int = 1,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
) -> tuple[list[dict[str, object]], Path]:
    output_path = Path(output_csv).expanduser()
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    seeds = build_seeds(n_runs, start_seed, seed_step)
    mesh_paths = [Path(mesh) for mesh in input_meshes]

    for mesh_idx, mesh_path in enumerate(mesh_paths):
        print(f"\n=== mesh {mesh_path.name} ({mesh_idx + 1}/{len(mesh_paths)}) ===")
        for run_idx, seed in enumerate(seeds):
            print(f"\n--- seed {seed} ({run_idx + 1}/{len(seeds)}) ---")
            for algorithm in ALGORITHMS:
                print(f"Running {algorithm} on {mesh_path.name} ...")
                try:
                    result = create_and_estimate_supq(
                        mesh_path,
                        algorithm=algorithm,
                        base_seed=seed,
                        visualize=False,
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
                    print(
                        f"  ok | models={row['n_models']} "
                        f"runtime_s={row['runtime_s']:.3f} "
                        f"chamfer={row['chamfer']:.4f}"
                        + (
                            f" mis={row['gt_misclassification_error']:.4f}"
                            if row["gt_misclassification_error"] is not None
                            else ""
                        )
                    )
                except Exception as exc:
                    row = {
                        "mesh_idx": int(mesh_idx),
                        "mesh_name": mesh_path.name,
                        "run_idx": int(run_idx),
                        "seed": int(seed),
                        "status": "error",
                        "error": str(exc),
                        "input_mesh": str(mesh_path),
                        "algorithm": algorithm,
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
                    print(f"  error | {exc}")
                rows.append(row)

    fieldnames = [
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
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, output_path


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    rows, output_path = run_batch(
        n_runs=args.n_runs,
        input_meshes=args.input_meshes,
        start_seed=args.start_seed,
        seed_step=args.seed_step,
        output_csv=args.output_csv,
    )
    failures = sum(row["status"] != "ok" for row in rows)
    print(f"\nSaved CSV -> {output_path}")
    if failures:
        print(f"Completed with {failures} failed runs.")
        return 1
    print(f"Completed {len(rows)} runs successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
