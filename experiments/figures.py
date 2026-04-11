"""
figures.py
==========
Generates PNG figures for the paper.

Figure 1 — Input point cloud (mushroom), 5 angles.
Figure 2 — Before vs after inlier refinement, 5 angles each.
            "Before": vanilla RANSAC (no LO) with a fixed seed.
            "After":  GAIR-RANSAC (full LO + normal coherence) with the same seed.
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.vanilla_ransac import vanilla_ransac
from src.superquadrics import superquadric_mesh as supmesh

# ── config ─────────────────────────────────────────────────────────────────────
PC_FILE      = ROOT / "src" / "point_clouds" / "mushroom.glb"
OUT_DIR      = ROOT / "experiments" / "artifacts" / "figures"
THRESHOLD    = 0.1
GRAPH_RADIUS = 0.06
MAX_MODELS   = 2
MAX_ITER     = 5
INNER_ITER   = 100
SEED         = 42
# ───────────────────────────────────────────────────────────────────────────────

CAMERA_POSITIONS = [
    ( 6.0,  0.0,  0.5),
    (-6.0,  0.0,  0.5),
    ( 0.0,  6.0,  0.5),
    ( 0.0, -6.0,  0.5),
    ( 4.0,  4.0,  6.0),
]
ANGLE_NAMES = ["front", "back", "left", "right", "top"]

PALETTE = ["lightgreen", "orange", "violet", "cyan", "yellow"]


def load_point_cloud(path: Path):
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene
    points  = np.asarray(mesh_raw.vertices, dtype=np.float64)
    normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)
    return points, normals


def save_screenshots(tag: str, meshes: list, points: np.ndarray, colors: list, out_dir: Path):
    """Save 5-angle screenshots for a given set of meshes + point cloud."""
    for cam_pos, angle in zip(CAMERA_POSITIONS, ANGLE_NAMES):
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
        pl.set_background("white")
        for mesh, color in zip(meshes, colors):
            faces = np.hstack([
                np.full((len(mesh.faces), 1), 3, dtype=np.int64),
                mesh.faces.astype(np.int64),
            ]).ravel()
            pl.add_mesh(pv.PolyData(mesh.vertices, faces), smooth_shading=True, opacity=0.7, color=color)
        pl.add_points(points, render_points_as_spheres=True, point_size=4, color="black", opacity=0.3)
        pl.enable_eye_dome_lighting()
        pl.camera_position = [cam_pos, (0, 0, 0), (0, 0, 1)]
        img_path = out_dir / f"{tag}_{angle}.png"
        pl.screenshot(str(img_path))
        pl.close()
        print(f"  saved {img_path.name}")


def save_pointcloud_screenshots(tag: str, points: np.ndarray, out_dir: Path):
    """Save 5-angle screenshots of the point cloud only."""
    for cam_pos, angle in zip(CAMERA_POSITIONS, ANGLE_NAMES):
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
        pl.set_background("white")
        pl.add_points(points, render_points_as_spheres=True, point_size=4, color="black", opacity=0.6)
        pl.enable_eye_dome_lighting()
        pl.camera_position = [cam_pos, (0, 0, 0), (0, 0, 1)]
        img_path = out_dir / f"{tag}_{angle}.png"
        pl.screenshot(str(img_path))
        pl.close()
        print(f"  saved {img_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PC_FILE.name} ...")
    points, normals = load_point_cloud(PC_FILE)
    print(f"  {len(points)} points\n")

    # ── Figure 1: input point cloud ────────────────────────────────────────────
    print("=== Figure 1: input point cloud ===")
    save_pointcloud_screenshots("fig1_pointcloud", points, OUT_DIR)

    # ── Figure 2: before vs after inlier refinement ───────────────────────────
    print("\n=== Figure 2: before (vanilla RANSAC) ===")
    models_before, _ = vanilla_ransac(
        points, normals,
        threshold=THRESHOLD,
        max_models=MAX_MODELS,
        max_iterations=MAX_ITER,
        random_seed=SEED,
    )
    if not models_before:
        print("  vanilla RANSAC found no models — try adjusting SEED or MAX_ITER")
    else:
        meshes_before = [supmesh.superquadric_mesh(m) for m in models_before]
        colors_before = [PALETTE[i % len(PALETTE)] for i in range(len(meshes_before))]
        save_screenshots("fig2_before", meshes_before, points, colors_before, OUT_DIR)

    print("\n=== Figure 2: after (GAIR-RANSAC) ===")
    models_after, _, _ = gair_ransac(
        points, normals,
        threshold=THRESHOLD,
        max_models=MAX_MODELS,
        max_iterations=MAX_ITER,
        inner_iterations=INNER_ITER,
        radius=GRAPH_RADIUS,
        use_normal_coherence=True,
        min_coverage=0.4,
        random_seed=SEED,
    )
    if not models_after:
        print("  GAIR-RANSAC found no models — try adjusting SEED or MAX_ITER")
    else:
        meshes_after = [supmesh.superquadric_mesh(m) for m in models_after]
        colors_after = [PALETTE[i % len(PALETTE)] for i in range(len(meshes_after))]
        save_screenshots("fig2_after", meshes_after, points, colors_after, OUT_DIR)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
