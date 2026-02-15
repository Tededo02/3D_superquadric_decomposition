import pyvista as pv
import numpy as np

def show_mesh_and_points(mesh, pts=None, point_size=6, show_bounds=True):
    # --- trimesh -> pyvista PolyData ---
    faces = np.hstack([
        np.full((len(mesh.faces), 1), 3, dtype=np.int64),
        mesh.faces.astype(np.int64)
    ]).ravel()  # VTK spesso gradisce 1D
    poly = pv.PolyData(mesh.vertices, faces)

    # --- plotter ---
    pl = pv.Plotter()
    pl.set_background("white")  # sfondo chiaro

    # Mesh
    pl.add_mesh(poly, smooth_shading=True, opacity=0.65)

    # Punti (se presenti)
    if pts is not None:
        pl.add_points(
            np.asarray(pts),
            render_points_as_spheres=True,
            point_size=point_size
        )

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
            xlabel="X", ylabel="Y", zlabel="Z",
            font_size=10
        )

    # Testo con “qualche valore”
    v = np.asarray(mesh.vertices)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)

    info = [
        f"Vertices: {len(mesh.vertices)}",
        f"Faces: {len(mesh.faces)}",
        f"X range: [{vmin[0]:.3f}, {vmax[0]:.3f}]",
        f"Y range: [{vmin[1]:.3f}, {vmax[1]:.3f}]",
        f"Z range: [{vmin[2]:.3f}, {vmax[2]:.3f}]",
    ]
    if pts is not None:
        info.append(f"Sample points: {len(pts)}")

    pl.add_text("\n".join(info), position="upper_left", font_size=10)

    # Camera: inquadra l’oggetto bene
    pl.reset_camera()
    pl.enable_eye_dome_lighting()  # migliora la percezione 3D

    pl.show()
