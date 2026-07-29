import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

import main_scan_pc as scan
from src.gair_ransac.energy_strategies import FullGairEnergy, GcRansacEnergy
from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.normals_estimation import estimate_normals_open3d_consistent
from src.gair_ransac.ransacov import ransacov
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.visualizations import plot as vis


PALETTE = (
    "lightgreen",
    "orange",
    "violet",
    "cyan",
    "yellow",
    "red",
    "lime",
    "pink",
    "gold",
    "turquoise",
)


def positive_integer(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("the value must be positive")
    return parsed_value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a candidate pool with repeated GAIR/GC runs and inspect "
            "interactive RansaCov selections."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=scan.PC_DIR / scan.PC_NAME,
    )
    parser.add_argument(
        "--runs",
        type=positive_integer,
        required=True,
        help="Number of independent candidate-generation runs.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("gair", "gc"),
        required=True,
        help="Candidate-generation algorithm.",
    )
    parser.add_argument(
        "--mesh-samples",
        type=positive_integer,
        default=scan.MESH_SAMPLE_COUNT,
    )
    parser.add_argument("--seed", type=int, default=scan.RANDOM_SEED)
    return parser.parse_args(argv)


def radial_residuals(model, points: np.ndarray) -> np.ndarray:
    return np.abs(superquadric_radial_residual(model, points))


def build_candidate_pool(
    points: np.ndarray,
    normals: np.ndarray,
    threshold: float,
    algorithm: str,
    run_count: int,
    base_seed: int,
) -> tuple[list, list[tuple[int, int, int]], int]:
    candidates: list = []
    candidate_origins: list[tuple[int, int, int]] = []
    total_local_optimizations = 0
    seed_sequences = np.random.SeedSequence(base_seed).spawn(run_count)

    for run_index, seed_sequence in enumerate(seed_sequences, start=1):
        run_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        strategy = FullGairEnergy() if algorithm == "gair" else GcRansacEnergy()
        print(
            f"Run {run_index}/{run_count} | algorithm={algorithm.upper()} "
            f"| seed={run_seed}"
        )
        models, _, _, local_optimizations = gair_ransac(
            threshold=threshold,
            point_cloud=points,
            max_models=scan.MAX_MODELS,
            m_neighbors=scan.M_NEIGHBORS,
            max_iterations=scan.MAX_ITERATIONS,
            sample_size=scan.SAMPLE_SIZE,
            min_inliers=scan.MIN_INLIERS,
            inner_iterations=scan.INNER_ITERATIONS,
            use_normal_coherence=True,
            normals=normals,
            random_seed=run_seed,
            min_coverage=scan.MIN_COVERAGE,
            energy_strategy=strategy,
        )
        total_local_optimizations += local_optimizations
        candidates.extend(models)
        candidate_origins.extend(
            (run_index, model_index, run_seed)
            for model_index in range(1, len(models) + 1)
        )
        print(
            f"  models found={len(models)} | "
            f"candidate pool={len(candidates)} | "
            f"local optimizations={local_optimizations}"
        )

    return candidates, candidate_origins, total_local_optimizations


def selection_mask(
    candidates: list,
    selected_indices: list[int],
    points: np.ndarray,
    threshold: float,
) -> np.ndarray:
    mask = np.zeros(points.shape[0], dtype=bool)
    for candidate_index in selected_indices:
        mask |= radial_residuals(candidates[candidate_index], points) < threshold
    return mask


def show_selection(
    models: list,
    points: np.ndarray,
    inlier_mask: np.ndarray,
    camera_position=None,
):
    meshes = [supmesh.superquadric_mesh(model) for model in models]
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(meshes))]

    plotter = pv.Plotter()
    plotter.set_background("white")
    vis._add_meshes_to_plotter(plotter, meshes, colors=colors, opacity=1.0)

    display_point_size = 12
    if (~inlier_mask).any():
        plotter.add_points(
            points[~inlier_mask],
            render_points_as_spheres=True,
            point_size=display_point_size,
            color="#ff1744",
            opacity=0.55,
        )
    if inlier_mask.any():
        plotter.add_points(
            points[inlier_mask],
            render_points_as_spheres=True,
            point_size=display_point_size,
            color="#00e676",
            opacity=0.55,
        )

    plotter.enable_eye_dome_lighting()
    if camera_position is None:
        vis._focus_camera_on_points(plotter, points)
    else:
        plotter.camera_position = camera_position
    plotter.show()
    return plotter.camera_position


def interactive_ransacov(
    candidates: list,
    candidate_origins: list[tuple[int, int, int]],
    points: np.ndarray,
    threshold: float,
) -> None:
    camera_position = None
    prompt = f"Enter k [1-{len(candidates)}], or 'quit': "

    while True:
        try:
            raw_value = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTerminating.")
            return

        if raw_value.lower() == "quit":
            print("Terminating.")
            return

        try:
            k = int(raw_value)
        except ValueError:
            print("Invalid value: enter a positive integer or 'quit'.")
            continue
        if not 1 <= k <= len(candidates):
            print(f"Invalid k: expected a value between 1 and {len(candidates)}.")
            continue

        try:
            selected_indices, covered_count = ransacov(
                candidates,
                points,
                k=k,
                threshold=threshold,
                residual_fn=radial_residuals,
            )
        except RuntimeError as error:
            print(error)
            continue
        if not selected_indices:
            print("RansaCov did not select any model.")
            continue

        selected_models = [candidates[index] for index in selected_indices]
        inlier_mask = selection_mask(
            candidates,
            selected_indices,
            points,
            threshold,
        )
        print(
            f"RansaCov | requested k={k} | selected={len(selected_indices)} "
            f"| covered={covered_count}/{len(points)} "
            f"({covered_count / len(points):.2%})"
        )
        for selected_position, candidate_index in enumerate(selected_indices, start=1):
            run_index, model_index, run_seed = candidate_origins[candidate_index]
            print(
                f"  selection {selected_position}: candidate={candidate_index} "
                f"run={run_index} model={model_index} seed={run_seed}"
            )

        camera_position = show_selection(
            selected_models,
            points,
            inlier_mask,
            camera_position=camera_position,
        )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    input_path = scan.resolve_input_path(args.input_file)
    points, point_colors, input_normals = scan.load_point_cloud(
        input_path,
        mesh_sample_count=args.mesh_samples,
        seed=args.seed,
    )
    print(f"Loaded {input_path.name}: {len(points)} points")
    if point_colors is not None:
        print("Loaded vertex colors; RansaCov visualization uses coverage colors.")

    threshold, point_spacing = scan.compute_effective_threshold(points)
    print(
        "Consensus | "
        f"min_threshold={scan.THRESHOLD:.4f} "
        f"point_spacing={point_spacing:.4f} "
        f"effective_threshold={threshold:.4f}"
    )

    if input_normals is None:
        print(f"Estimating normals with Open3D | k_neighbors={scan.K_NEIGHBORS}")
        normals = estimate_normals_open3d_consistent(points, scan.K_NEIGHBORS)
    else:
        print("Using normals sampled from input mesh faces")
        normals = input_normals

    candidates, candidate_origins, total_local_optimizations = build_candidate_pool(
        points,
        normals,
        threshold,
        args.algorithm,
        args.runs,
        args.seed,
    )
    if not candidates:
        raise RuntimeError("No candidate model was found in any run")

    print(
        f"Candidate generation completed | candidates={len(candidates)} "
        f"| runs={args.runs} | "
        f"local optimizations={total_local_optimizations}"
    )
    interactive_ransacov(
        candidates,
        candidate_origins,
        points,
        threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
