from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

from src.point_cloud_normals import estimate_normals_consistent


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POINT_CLOUD = PROJECT_ROOT / "test_objects" / "real_pc" / "etp_no_floor_r.ply"


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Point cloud not found: {resolved}")
    return resolved


def _load_points(path: Path) -> np.ndarray:
    raw = trimesh.load(str(path))
    points = np.asarray(raw.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{path} does not contain a valid Nx3 point cloud")
    return points


def _sample_indices(
    point_count: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if max_points <= 0 or point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(point_count, size=max_points, replace=False)).astype(np.int64)


def _median_nn_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    return float(np.median(distances))


def _normal_line_mesh(points: np.ndarray, normals: np.ndarray, length: float) -> pv.PolyData:
    line_points = np.empty((2 * points.shape[0], 3), dtype=np.float64)
    line_points[0::2] = points
    line_points[1::2] = points + normals * float(length)
    lines = np.column_stack(
        [
            np.full(points.shape[0], 2, dtype=np.int64),
            np.arange(0, 2 * points.shape[0], 2, dtype=np.int64),
            np.arange(1, 2 * points.shape[0], 2, dtype=np.int64),
        ]
    ).ravel()
    return pv.PolyData(line_points, lines=lines)


def _normal_subset(
    points: np.ndarray,
    normals: np.ndarray,
    max_normals: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = _sample_indices(points.shape[0], max_normals, seed)
    return points[indices], normals[indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a point cloud with consistently oriented normals."
    )
    parser.add_argument(
        "point_cloud",
        nargs="?",
        default=str(DEFAULT_POINT_CLOUD),
        help="PLY/OBJ point cloud path.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=20,
        help="k used for local normal estimation and Open3D orientation propagation.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=60000,
        help="Maximum points used for visualization and normal estimation. Use 0 for all points.",
    )
    parser.add_argument(
        "--max-normals",
        type=int,
        default=3000,
        help="Maximum normal vectors drawn. Use 0 to draw all visible points.",
    )
    parser.add_argument(
        "--normal-length",
        type=float,
        default=0.0,
        help="Normal segment length. If 0, uses 8x median nearest-neighbor distance.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--pca-fallback",
        action="store_true",
        help="Disable Open3D and use the legacy PCA center-outward fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = _resolve_path(args.point_cloud)
    points = _load_points(path)
    indices = _sample_indices(points.shape[0], args.max_points, args.seed)
    visible_points = points[indices]

    result = estimate_normals_consistent(
        visible_points,
        n_neighbors=args.neighbors,
        orient_neighbors=args.neighbors,
        prefer_open3d=not args.pca_fallback,
    )
    normals = result.normals

    normal_length = float(args.normal_length)
    if normal_length <= 0.0:
        normal_length = 8.0 * _median_nn_distance(visible_points)

    normal_points, normal_vectors = _normal_subset(
        visible_points,
        normals,
        args.max_normals,
        args.seed + 1,
    )

    center = np.median(visible_points, axis=0)
    outward_score = np.einsum(
        "ij,ij->i",
        normals,
        visible_points - center,
        optimize=True,
    )

    cloud = pv.PolyData(visible_points)
    cloud["normal_outward_score"] = outward_score
    normal_lines = _normal_line_mesh(normal_points, normal_vectors, normal_length)

    print(f"Loaded {points.shape[0]} points from {path}")
    print(f"Visualizing {visible_points.shape[0]} points")
    print(f"Normal estimation: {result.method}")
    print(f"Normal length: {normal_length:.6f}")

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_mesh(
        cloud,
        scalars="normal_outward_score",
        cmap="coolwarm",
        point_size=3.0,
        render_points_as_spheres=True,
        scalar_bar_args={
            "title": "dot(n, p - median)",
            "color": "black",
            "fmt": "%.3f",
        },
    )
    plotter.add_mesh(normal_lines, color="black", line_width=1)
    plotter.add_text(
        f"{path.name}\n{result.method}",
        font_size=9,
        color="black",
    )
    plotter.show_axes()
    plotter.show_grid()
    plotter.reset_camera()
    plotter.reset_camera_clipping_range()
    plotter.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
