from dataclasses import replace
from pathlib import Path

import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.superquadrics import superquadric_mesh as supmesh
from src.gair_ransac.consensus import expanded_removal_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_POSITION = [
    (-4.5307183938878675, 1.7429659977931096, -0.2853180548569162),
    (0.0, 0.0, 0.0),
    (-0.1126909338726822, -0.13171346961495714, 0.9848615716662379),
]


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize_vector(axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    vector = np.asarray(vector, dtype=np.float64)
    cos_theta = float(np.cos(angle_rad))
    sin_theta = float(np.sin(angle_rad))
    return (
        vector * cos_theta
        + np.cross(axis, vector) * sin_theta
        + axis * np.dot(axis, vector) * (1.0 - cos_theta)
    )


def _axis_angle_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ax = _normalize_vector(axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    x, y, z = ax
    cos_theta = float(np.cos(angle_rad))
    sin_theta = float(np.sin(angle_rad))
    one_minus_cos = 1.0 - cos_theta
    return np.array(
        [
            [
                cos_theta + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_theta,
                x * z * one_minus_cos + y * sin_theta,
            ],
            [
                y * x * one_minus_cos + z * sin_theta,
                cos_theta + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_theta,
            ],
            [
                z * x * one_minus_cos - y * sin_theta,
                z * y * one_minus_cos + x * sin_theta,
                cos_theta + z * z * one_minus_cos,
            ],
        ],
        dtype=np.float64,
    )


def _collect_frame_points(meshes: list, points: np.ndarray | None = None) -> np.ndarray:
    frame_chunks: list[np.ndarray] = []
    if points is not None:
        point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if point_array.size > 0:
            frame_chunks.append(point_array)

    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        if vertices.size > 0:
            frame_chunks.append(vertices)

    if not frame_chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(frame_chunks)


def _camera_frame_points(meshes: list, points: np.ndarray | None = None) -> np.ndarray:
    """Use the point cloud as camera frame whenever one is being displayed."""
    if points is not None:
        point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if point_array.size > 0:
            return point_array
    return _collect_frame_points(meshes)


def _camera_axes(camera_position: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_pos = np.asarray(camera_position[0], dtype=np.float64)
    focal_point = np.asarray(camera_position[1], dtype=np.float64)
    view_up = _normalize_vector(
        np.asarray(camera_position[2], dtype=np.float64),
        fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    forward = _normalize_vector(
        focal_point - camera_pos,
        fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    right = _normalize_vector(
        np.cross(forward, view_up),
        fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    corrected_up = _normalize_vector(
        np.cross(right, forward),
        fallback=view_up,
    )
    return forward, right, corrected_up


def _transform_points(
    points: np.ndarray | None,
    rotation_matrix: np.ndarray,
    center: np.ndarray,
) -> np.ndarray | None:
    if points is None:
        return None

    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if point_array.size == 0:
        return point_array.copy()
    return (point_array - center) @ rotation_matrix.T + center


def _transform_meshes(
    meshes: list,
    rotation_matrix: np.ndarray,
    center: np.ndarray,
) -> list:
    transformed_meshes = []
    for mesh in meshes:
        mesh_copy = mesh.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        mesh_copy.vertices = _transform_points(vertices, rotation_matrix, center)
        transformed_meshes.append(mesh_copy)
    return transformed_meshes


def _transform_models(
    models,
    rotation_matrix: np.ndarray,
    center: np.ndarray,
):
    if not models:
        return models

    transformed_models = []
    for model in models:
        transformed_rotation = rotation_matrix @ model.rotation_matrix()
        transformed_translation = _transform_points(model.t, rotation_matrix, center)
        transformed_euler = Rotation.from_matrix(transformed_rotation).as_euler("zyx")
        transformed_models.append(
            replace(
                model,
                rot=np.asarray(transformed_euler, dtype=np.float64),
                t=np.asarray(transformed_translation, dtype=np.float64).reshape(3),
            )
        )
    return transformed_models


def _build_camera_positions(frame_points: np.ndarray, n_views: int = 3) -> list[list[tuple[float, float, float]]]:
    if n_views <= 0:
        return []

    default_position = np.asarray(DEFAULT_CAMERA_POSITION[0], dtype=np.float64)
    default_focal_point = np.asarray(DEFAULT_CAMERA_POSITION[1], dtype=np.float64)
    default_view_up = _normalize_vector(
        np.asarray(DEFAULT_CAMERA_POSITION[2], dtype=np.float64),
        fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    if frame_points.size == 0:
        center = default_focal_point
        scene_extent = np.linalg.norm(default_position - default_focal_point)
    else:
        mins = frame_points.min(axis=0)
        maxs = frame_points.max(axis=0)
        center = (mins + maxs) / 2.0
        scene_extent = float(np.linalg.norm(maxs - mins))

    base_direction = default_position - default_focal_point
    base_distance = max(float(np.linalg.norm(base_direction)), max(scene_extent, 1.0) * 1.35)
    base_direction = _normalize_vector(base_direction, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    base_offset = base_direction * base_distance

    camera_positions: list[list[tuple[float, float, float]]] = []
    for view_idx in range(n_views):
        angle_rad = (2.0 * np.pi * view_idx) / float(n_views)
        rotated_offset = _rotate_vector_around_axis(base_offset, default_view_up, angle_rad)
        camera_positions.append(
            [
                tuple(center + rotated_offset),
                tuple(center),
                tuple(default_view_up),
            ]
        )
    return camera_positions


def _focus_camera_on_points(plotter: pv.Plotter, points: np.ndarray) -> None:
    if points.size == 0:
        plotter.camera_position = DEFAULT_CAMERA_POSITION
        return

    plotter.camera_position = _build_camera_positions(points, n_views=1)[0]
    plotter.reset_camera_clipping_range()


def _print_camera_position(label: str, camera_position) -> None:
    if camera_position is None:
        print(f"{label}: unavailable")
        return

    try:
        camera_rows = [
            tuple(float(component) for component in row)
            for row in camera_position
        ]
    except (TypeError, ValueError):
        print(f"{label}: {camera_position}")
        return

    print(f"{label}: [")
    for row in camera_rows:
        print(f"    {row},")
    print("]")


def _resolve_camera_position(camera_position, frame_points: np.ndarray):
    if camera_position is not None:
        return camera_position

    camera_positions = _build_camera_positions(frame_points, n_views=3)
    return camera_positions[0] if camera_positions else DEFAULT_CAMERA_POSITION


def _add_meshes_to_plotter(
    plotter: pv.Plotter,
    meshes: list,
    colors: np.ndarray | None = None,
    opacity: float = 0.65,
) -> None:
    for i, mesh in enumerate(meshes):
        faces = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        ]).ravel()
        poly = pv.PolyData(mesh.vertices, faces)
        mesh_color = None if colors is None else colors[i]
        plotter.add_mesh(poly, smooth_shading=True, opacity=opacity, color=mesh_color)


def _populate_mesh_and_points_plotter(
    plotter: pv.Plotter,
    meshes: list,
    pts: np.ndarray | None = None,
    point_size: int = 8,
    colors: np.ndarray | None = None,
    inlier_mask: np.ndarray | None = None,
    mss_used: np.ndarray | None = None,
    models=None,
    treshold: float = 0.1,
    include_points: bool = True,
) -> np.ndarray:
    plotter.set_background("white")
    points = None if pts is None else np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    error_inlier = None
    remaining_mask = None
    cloud_point_size = max(point_size * 2, 12)

    if models and points is not None:
        error_inlier = np.full(points.shape[0], np.inf, dtype=np.float64)
        remaining_mask = np.ones(points.shape[0], dtype=bool)

        for model in models:
            temp_error_inlier = np.abs(superquadric_radial_residual(model, points))
            error_inlier = np.minimum(error_inlier, temp_error_inlier)

            if not remaining_mask.any():
                break

            current_indices = np.flatnonzero(remaining_mask)
            remove_mask = expanded_removal_mask(
                model,
                points[current_indices],
                treshold,
                factor=1.3,
            )
            remaining_mask[current_indices[remove_mask]] = False

    _add_meshes_to_plotter(plotter, meshes, colors=colors, opacity=0.65)

    if include_points and points is not None:
        n_points_total = points.shape[0]
        has_residual_colormap = error_inlier is not None and np.isfinite(error_inlier).any()
        if has_residual_colormap:
            finite_error = error_inlier[np.isfinite(error_inlier)]
            color_max = float(np.percentile(finite_error, 90))
            if color_max <= 0.0:
                color_max = float(finite_error.max())
            if color_max <= 0.0:
                color_max = 1.0

            point_cloud = pv.PolyData(points)
            point_cloud["min_residual"] = np.nan_to_num(
                error_inlier,
                nan=color_max,
                posinf=color_max,
                neginf=0.0,
            )
            plotter.add_mesh(
                point_cloud,
                scalars="min_residual",
                cmap="turbo",
                clim=(0.0, color_max),
                render_points_as_spheres=True,
                point_size=cloud_point_size,
                opacity=1.0,
                ambient=0.25,
                specular=0.15,
                nan_color="black",
                above_color="#7f0000",
                scalar_bar_args={
                    "title": "Min radial residual",
                    "color": "black",
                    "fmt": "%.3f",
                    "vertical": True,
                    "position_x": 0.82,
                    "position_y": 0.1,
                    "width": 0.08,
                    "height": 0.8,
                },
            )
        else:
            mask_is_valid = inlier_mask is not None and len(inlier_mask) == n_points_total
            if mask_is_valid:
                mask = np.asarray(inlier_mask, dtype=bool)
                if mask.any():
                    plotter.add_points(
                        points[mask],
                        render_points_as_spheres=True,
                        point_size=cloud_point_size,
                        color="#00e676",
                    )
                if (~mask).any():
                    plotter.add_points(
                        points[~mask],
                        render_points_as_spheres=True,
                        point_size=cloud_point_size,
                        color="#ff1744",
                        opacity=0.8,
                    )
            else:
                plotter.add_points(
                    points,
                    render_points_as_spheres=True,
                    point_size=cloud_point_size,
                    color="#1565c0",
                )

    plotter.enable_eye_dome_lighting()
    return _collect_frame_points(meshes, points)


def show_point_cloud(
    pts: np.ndarray,
    point_size: int = 8,
    camera_position=None,
    print_camera_on_close: bool = False,
) -> None:
    points = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    pl = pv.Plotter()
    pl.set_background("white")
    pl.add_points(
        points,
        render_points_as_spheres=True,
        point_size=max(point_size * 2, 12),
        color="#1565c0",
    )
    pl.enable_eye_dome_lighting()
    pl.camera_position = _resolve_camera_position(camera_position, points)

    returned_camera_position = pl.show(return_cpos=print_camera_on_close)
    if print_camera_on_close:
        _print_camera_position("PyVista camera position (point cloud)", returned_camera_position)


def save_point_cloud_inlier_view(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: str | Path,
    point_size: int = 8,
) -> Path:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != points.shape[0]:
        raise ValueError(f"inlier_mask must have length {points.shape[0]}, got {mask.shape[0]}")

    image_path = Path(output_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    image_path.parent.mkdir(parents=True, exist_ok=True)

    display_point_size = max(point_size * 2, 12)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")

    if (~mask).any():
        pl.add_points(
            points[~mask],
            render_points_as_spheres=True,
            point_size=display_point_size/3,
            color="black",
            opacity=0.95,
        )
    if mask.any():
        pl.add_points(
            points[mask],
            render_points_as_spheres=True,
            point_size=display_point_size,
            color="red",
            opacity=1.0,
        )

    pl.enable_eye_dome_lighting()
    _focus_camera_on_points(pl, points)
    pl.screenshot(str(image_path))
    pl.close()
    return image_path


def save_point_cloud_inlier_model_view(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    model,
    output_path: str | Path,
    point_size: int = 8,
    model_color: str = "#64b5f6",
    model_opacity: float = 0.35,
) -> Path:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != points.shape[0]:
        raise ValueError(f"inlier_mask must have length {points.shape[0]}, got {mask.shape[0]}")

    image_path = Path(output_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    image_path.parent.mkdir(parents=True, exist_ok=True)

    fitted_mesh = supmesh.superquadric_mesh(model)
    display_point_size = max(point_size * 2, 12)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")

    _add_meshes_to_plotter(pl, [fitted_mesh], colors=[model_color], opacity=model_opacity)

    if (~mask).any():
        pl.add_points(
            points[~mask],
            render_points_as_spheres=True,
            point_size=display_point_size / 3,
            color="black",
            opacity=0.95,
        )
    if mask.any():
        pl.add_points(
            points[mask],
            render_points_as_spheres=True,
            point_size=display_point_size,
            color="red",
            opacity=1.0,
        )

    pl.enable_eye_dome_lighting()
    _focus_camera_on_points(pl, points)
    pl.screenshot(str(image_path))
    pl.close()
    return image_path


def save_mesh_view_triplet(
    meshes: list,
    output_dir: str | Path,
    output_stem: str,
    pts: np.ndarray | None = None,
    point_size: int = 8,
    colors: np.ndarray | None = None,
    inlier_mask: np.ndarray | None = None,
    mss_used: np.ndarray | None = None,
    models=None,
    treshold: float = 0.1,
    window_size: tuple[int, int] = (1600, 1200),
) -> list[Path]:
    image_dir = Path(output_dir)
    if not image_dir.is_absolute():
        image_dir = PROJECT_ROOT / image_dir
    image_dir.mkdir(parents=True, exist_ok=True)

    frame_points = _camera_frame_points(meshes, pts)
    base_camera_position = _build_camera_positions(frame_points, n_views=1)[0]
    _, right_axis, up_axis = _camera_axes(base_camera_position)
    center = np.asarray(base_camera_position[1], dtype=np.float64)
    view_specs = (
        ("01_right90", _axis_angle_rotation_matrix(up_axis, -0.5 * np.pi)),
        ("02_perpendicular90", _axis_angle_rotation_matrix(right_axis, 0.5 * np.pi)),
        (
            "03_oblique",
            _axis_angle_rotation_matrix(right_axis, np.deg2rad(30.0))
            @ _axis_angle_rotation_matrix(up_axis, np.deg2rad(-35.0)),
        ),
    )
    saved_paths: list[Path] = []

    for view_name, rotation_matrix in view_specs:
        rotated_meshes = _transform_meshes(meshes, rotation_matrix, center)
        rotated_points = _transform_points(pts, rotation_matrix, center)
        rotated_models = _transform_models(models, rotation_matrix, center)

        for variant_name, include_points in (("with_pc", True), ("mesh_only", False)):
            pl = pv.Plotter(off_screen=True, window_size=window_size)
            _populate_mesh_and_points_plotter(
                pl,
                rotated_meshes,
                pts=rotated_points if include_points else None,
                point_size=point_size,
                colors=colors,
                inlier_mask=inlier_mask if include_points else None,
                mss_used=None,
                models=rotated_models if include_points else None,
                treshold=treshold,
                include_points=include_points,
            )
            pl.camera_position = base_camera_position
            pl.reset_camera_clipping_range()
            image_path = image_dir / f"{output_stem}_{view_name}_{variant_name}.png"
            pl.screenshot(str(image_path))
            saved_paths.append(image_path)
            pl.close()

    return saved_paths


def show_mesh_and_points(meshes: list, pts: list =None, point_size=8, show_bounds=True,colors: np.ndarray = None, inlier_mask: np.ndarray = None,mss_used: np.ndarray = None,
        models=None,
        treshold=0.1,
        print_camera_on_close: bool = False,
        camera_position=None,
    ) -> None:

    # --- plotter ---
    pl = pv.Plotter()
    _populate_mesh_and_points_plotter(
        pl,
        meshes,
        pts=pts,
        point_size=point_size,
        colors=colors,
        inlier_mask=inlier_mask,
        mss_used=mss_used,
        models=models,
        treshold=treshold,
        include_points=True,
    )
    camera_frame_points = _camera_frame_points(meshes, pts)
    initial_camera_position = _resolve_camera_position(camera_position, camera_frame_points)
    pl.camera_position = initial_camera_position

    returned_camera_position = pl.show(return_cpos=print_camera_on_close)
    if print_camera_on_close:
        _print_camera_position("PyVista camera position (points + meshes)", returned_camera_position)

    mesh_only_plotter = pv.Plotter()
    _populate_mesh_and_points_plotter(
        mesh_only_plotter,
        meshes,
        pts=pts,
        point_size=point_size,
        colors=colors,
        inlier_mask=inlier_mask,
        mss_used=mss_used,
        models=models,
        treshold=treshold,
        include_points=False,
    )
    mesh_only_plotter.camera_position = initial_camera_position
    returned_camera_position = mesh_only_plotter.show(return_cpos=print_camera_on_close)
    if print_camera_on_close:
        _print_camera_position("PyVista camera position (meshes only)", returned_camera_position)
