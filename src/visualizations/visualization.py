from pathlib import Path

import pyvista as pv
import numpy as np
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.gair_ransac.consensus import expanded_removal_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_POSITION = [
    (-4.5307183938878675, 1.7429659977931096, -0.2853180548569162),
    (0.0, 0.0, 0.0),
    (-0.1126909338726822, -0.13171346961495714, 0.9848615716662379),
]


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
    pl.camera_position = DEFAULT_CAMERA_POSITION
    pl.screenshot(str(image_path))
    pl.close()
    return image_path


def show_mesh_and_points(meshes: list, pts: list =None, point_size=8, show_bounds=True,colors: np.ndarray = None, inlier_mask: np.ndarray = None,mss_used: np.ndarray = None,
        models=None,
        treshold=0.1
    ) -> None:

    # --- plotter ---
    pl = pv.Plotter()
    pl.set_background("white")  # sfondo chiaro


    # --- aggiungi tutte le mesh ---
    all_vertices = []
    total_faces = 0
    i:int=0
    points = None if pts is None else np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    error_inlier = None
    remaining_mask = None
    cloud_point_size = max(point_size * 2, 12)
    mss_point_size = max(point_size * 5, 18)

    if models and points is not None:
        error_inlier = np.full(points.shape[0], np.inf, dtype=np.float64)
        remaining_mask = np.ones(points.shape[0], dtype=bool)

        for model in models:
            temp_error_inlier = np.abs(superquadric_radial_residual(model, points))
            error_inlier = np.minimum(error_inlier, temp_error_inlier)

            if not remaining_mask.any():
                break

            current_indices = np.flatnonzero(remaining_mask)
            # Keep the original colormap semantics from commit `colormap`,
            # independent from newer consensus defaults used elsewhere.
            remove_mask = expanded_removal_mask(
                model,
                points[current_indices],
                treshold,
                factor=1.3,
                error_metric="radial",
            )
            remaining_mask[current_indices[remove_mask]] = False

    for mesh in meshes:

        faces = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64)
        ]).ravel() 
        poly = pv.PolyData(mesh.vertices, faces)
        #lightblue
        pl.add_mesh(poly, smooth_shading=True, opacity=0.65,color=colors[i])

        all_vertices.append(np.asarray(mesh.vertices))
        total_faces += len(mesh.faces)
        i+=1

    # --- punti (se presenti) ---
    n_points_total = 0
    if points is not None:
        n_points_total = points.shape[0] # total number of points across all meshes
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
            pl.add_mesh(
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
                    pl.add_points(points[mask], render_points_as_spheres=True, point_size=cloud_point_size, color="#00e676")
                if (~mask).any():
                    pl.add_points(points[~mask], render_points_as_spheres=True, point_size=cloud_point_size, color="#ff1744", opacity=0.8)
            else:
                pl.add_points(points, render_points_as_spheres=True, point_size=cloud_point_size, color="#1565c0")
                """
        if mss_used is not None:
            mss_points = np.asarray(mss_used, dtype=np.float64).reshape(-1, 3)
            pl.add_points(mss_points, render_points_as_spheres=True, point_size=mss_point_size, color="violet")
        """
    pl.enable_eye_dome_lighting()

    # fixed camera angle (same every run)
    pl.camera_position = DEFAULT_CAMERA_POSITION

    pl.show()
    #print(pl.camera_position)
