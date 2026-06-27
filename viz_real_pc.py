from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


PC_DIR = Path(__file__).resolve().parent / "test_objects" / "real_pc"
SUPPORTED_POINT_CLOUD_EXTENSIONS = {".obj", ".ply"}

# Change this value to select the point cloud to visualize.
# Available examples:
# - "pointcloud_gnome.ply"
# - "pointcloud_kettle_and_forklift.ply"
# - "pointcloud_kettle_and_forklift_masked.ply"
# - "Blue_Sea_Star.obj"
POINT_CLOUD_FILE = "etp_no_floor.ply"
POINT_SIZE = 4.0
PCA_CENTER_AND_ALIGN = True
ROBUST_QUANTILE = 0.95
SHOW_ONLY_ROBUST_QUANTILE_POINTS = False
CAMERA_FIT_MARGIN = 0.9


def _load_point_cloud(path: Path) -> pv.PolyData:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_POINT_CLOUD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_POINT_CLOUD_EXTENSIONS))
        raise ValueError(f"Unsupported point-cloud format '{extension}'. Supported: {supported}")

    loaded = pv.read(path)
    if not isinstance(loaded, pv.DataSet):
        raise ValueError(f"{path} could not be loaded as a point cloud")
    if loaded.n_points == 0:
        raise ValueError(f"{path} does not contain any points")

    # OBJ files are commonly triangle meshes. Visualize their vertices as a
    # point cloud while preserving any per-vertex data loaded by PyVista.
    cloud = pv.PolyData(np.asarray(loaded.points))
    for name, values in loaded.point_data.items():
        if len(values) == loaded.n_points:
            cloud.point_data[name] = values

    return cloud


def _available_point_clouds() -> list[Path]:
    return sorted(
        path
        for path in PC_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_EXTENSIONS
    )


def _robust_pca_view_cloud(cloud: pv.DataSet) -> pv.PolyData:
    points = np.asarray(cloud.points)
    median = np.median(points, axis=0)
    centered_points = points - median

    covariance = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    pca_axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    pca_points = centered_points @ pca_axes

    tail = (1.0 - ROBUST_QUANTILE) / 2.0
    low = np.quantile(pca_points, tail, axis=0)
    high = np.quantile(pca_points, 1.0 - tail, axis=0)
    robust_center = (low + high) / 2.0
    robust_points = pca_points - robust_center

    if SHOW_ONLY_ROBUST_QUANTILE_POINTS:
        mask = np.all((pca_points >= low) & (pca_points <= high), axis=1)
    else:
        mask = np.ones(cloud.n_points, dtype=bool)

    viewed_cloud = pv.PolyData(robust_points[mask])
    for name, values in cloud.point_data.items():
        if len(values) == cloud.n_points:
            viewed_cloud.point_data[name] = values[mask]

    return viewed_cloud


def _add_cloud(plotter: pv.Plotter, cloud: pv.DataSet, path: Path, point_size: float) -> None:
    scalars = None
    rgb = False

    for name in ("RGB", "rgb", "RGBA", "rgba", "colors", "Colors"):
        if name in cloud.point_data:
            scalars = name
            rgb = cloud.point_data[name].shape[-1] in (3, 4)
            break

    plotter.add_points(
        cloud,
        scalars=scalars,
        rgb=rgb,
        point_size=point_size,
        render_points_as_spheres=True,
    )
    plotter.add_text(path.name, font_size=10)


def main() -> None:
    path = PC_DIR / POINT_CLOUD_FILE
    if not path.exists():
        available = "\n".join(f"- {file.name}" for file in _available_point_clouds())
        raise SystemExit(f"Point cloud not found: {path}\nAvailable point clouds:\n{available}")

    cloud = _load_point_cloud(path)
    if PCA_CENTER_AND_ALIGN:
        cloud = _robust_pca_view_cloud(cloud)

    plotter = pv.Plotter()
    _add_cloud(plotter, cloud, path, POINT_SIZE)
    plotter.show_axes()
    plotter.show_grid()
    plotter.reset_camera()
    plotter.camera.zoom(CAMERA_FIT_MARGIN)
    plotter.reset_camera_clipping_range()
    plotter.show()


if __name__ == "__main__":
    main()
