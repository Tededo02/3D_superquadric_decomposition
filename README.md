# Superquadric Decomposition

Decompose a 3D point cloud into superquadric primitives with multi-model RANSAC
(vanilla, LO, GC and GAIR-RANSAC) followed by a RANSACov set-cover selection.

> **Note:** this is a temporary, notebook-based setup. It will be cleaned up and improved
> in the foreseeable future.

## Quick start

Requires Python 3.11+ and the dependencies in `pyproject.toml`
([uv](https://docs.astral.sh/uv/) suggested: `uv sync`).

Open `explore_pointcloud.ipynb`, set `PC_PATH` in the configuration cell, and run the cells top to bottom.

- Any format [trimesh](https://trimesh.org) reads works (`.ply`, `.stl`, `.obj`, ...); three
  sample clouds are in `data/point_clouds/`.
- Normals are read from a sibling `normals_<filename>` file if present, otherwise taken from
  the mesh or estimated.
- Per-cloud parameters (most importantly the inlier `threshold`) are in `data/pc_configs.json`.
