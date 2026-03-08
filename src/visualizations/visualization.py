import pyvista as pv
import numpy as np

def show_mesh_and_points(meshes: list, pts: list =None, point_size=6, show_bounds=True,colors: np.ndarray = None, inlier_mask: np.ndarray = None):

    # --- plotter ---
    pl = pv.Plotter()
    pl.set_background("white")  # sfondo chiaro


    # --- aggiungi tutte le mesh ---
    all_vertices = []
    total_faces = 0
    i:int=0
    for mesh in meshes:
        faces = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64)
        ]).ravel()  # VTK gradisce 1D: [3,i,j,k, 3,i,j,k, ...]
        poly = pv.PolyData(mesh.vertices, faces)
        #lightblue
        pl.add_mesh(poly, smooth_shading=True, opacity=0.65,color=colors[i])

        all_vertices.append(np.asarray(mesh.vertices))
        total_faces += len(mesh.faces)
        i+=1

    # --- punti (se presenti) ---
    n_points_total = 0
    if pts is not None:
        points = np.asarray(pts, dtype=np.float32).reshape(-1, 3) # -1 because numpy can infer the number of points, 3 for XYZ
        n_points_total = points.shape[0] # total number of points across all meshes
        mask_is_valid = inlier_mask is not None and len(inlier_mask) == n_points_total
        if mask_is_valid:
            mask = np.asarray(inlier_mask, dtype=bool)
            if mask.any():
                pl.add_points(points[mask], render_points_as_spheres=True, point_size=point_size, color="green")
            if (~mask).any():
                pl.add_points(points[~mask], render_points_as_spheres=True, point_size=point_size, color="red", opacity=0.4)
        else:
            pl.add_points(points, render_points_as_spheres=True, point_size=point_size)


    # Assi + griglia “in scena”
    pl.show_axes()            # triade assi in basso (widget)
    pl.add_axes()             # assi 3D nell’oggetto (freccioni XYZ)
    pl.show_grid(             # griglia sul piano (tipo “pavimento”)
        location="outer",
        ticks="outside",
        grid="front",
        all_edges=True
    )

    # Bounding box + tick con valori (molto utile)
    if show_bounds:
        pl.show_bounds(
            grid="back",
            location="outer",
            all_edges=True,
            ticks="outside",
            xtitle="X", ytitle="Y", ztitle="Z",
            font_size=10
        )


    # --- testo info (range su tutte le mesh) ---
    v = np.vstack(all_vertices) if len(all_vertices) else np.zeros((0, 3))
    if v.shape[0] > 0:
        vmin = v.min(axis=0)
        vmax = v.max(axis=0)
        info = [
            f"Meshes: {len(meshes)}",
            f"Vertices (tot): {v.shape[0]}",
            f"Faces (tot): {total_faces}",
            f"X range: [{vmin[0]:.3f}, {vmax[0]:.3f}]",
            f"Y range: [{vmin[1]:.3f}, {vmax[1]:.3f}]",
            f"Z range: [{vmin[2]:.3f}, {vmax[2]:.3f}]",
        ]
    else:
        info = [f"Meshes: {len(meshes)}"]

    if pts is not None:
        info.append(f"Sample points: {n_points_total}")

    pl.add_text("\n".join(info), position="upper_left", font_size=10)

    # Camera: inquadra l’oggetto bene
    pl.reset_camera()
    pl.enable_eye_dome_lighting()  # migliora la percezione 3D

    pl.show()
