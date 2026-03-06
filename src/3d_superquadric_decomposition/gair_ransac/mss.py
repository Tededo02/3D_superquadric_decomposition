import numpy as np
import pyvista as pv
from scipy.spatial import KDTree


# Constants
_CLOSED_FORM_SIZES = {
    'plane':  3,
    'sphere': 4,
    'circle': 3,
}

# How many points the algorithm picks for each patch
_PTS_PER_PATCH = 2

# Voxel side = median NN-spacing × this factor (spatial_walk_mss only).
# Really important, you can choose how big the single voxel cell is
_VOXEL_SPACING_FACTOR = 15


# Both functions return M_j
def spatial_walk_mss(
    D: np.ndarray,
    normals: np.ndarray | None = None,
    primitive_type: str = 'superquadric',
    sample_size: int = 30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    MSS via density-weighted spatial walk on a voxel grid.

    Picks a random anchor voxel (weighted by number of points), then expands
    to neighbouring voxels, again choosing randomly weighting by number of points until
    sample_size is reached.

    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(D)

    # closed-form primitives: just pick the minimal sample (for instance pick 3 points for circles)
    if primitive_type in _CLOSED_FORM_SIZES:
        k = _CLOSED_FORM_SIZES[primitive_type]
        return D[rng.choice(N, size=min(k, N), replace=False)]

    if N <= sample_size:
        return D.copy()


    # k=2: index 0 is self (dist=0), index 1 is true nearest neighbour
    nn_dists, _ = KDTree(D).query(D, k=2)
    voxel_size = max(float(np.median(nn_dists[:, 1])) * _VOXEL_SPACING_FACTOR, 1e-9)

    # assign each point to a voxel (integer grid key)
    voxel_coords = np.floor(D / voxel_size).astype(int)
    voxel_map: dict[tuple, list[int]] = {}
    for i in range(N):
        key = tuple(voxel_coords[i])
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(i)

    non_empty = list(voxel_map.keys())

    # anchor: pick seed voxel weighted by density
    counts  = np.array([len(voxel_map[v]) for v in non_empty], dtype=float)
    seed    = non_empty[rng.choice(len(non_empty), p=counts / counts.sum())]
    print(len(voxel_map))
    selected_indices: set[int]   = set()
    visited:          set[tuple] = set()
    frontier:         set[tuple] = {seed}

    while len(selected_indices) < sample_size and frontier:
        # pick next voxel from frontier, weighted by density
        # in the first loop frontier is just the seed, so it forces the algorithm to pick it.
        fl      = list(frontier)
        fc      = np.array([len(voxel_map[v]) for v in fl], dtype=float)
        voxel   = fl[rng.choice(len(fl), p=fc / fc.sum())]
        frontier.discard(voxel)

        if voxel in visited:
            continue
        visited.add(voxel)

        # sample up to _PTS_PER_PATCH new points from this voxel
        cands = [i for i in voxel_map[voxel] if i not in selected_indices]
        n_add = min(_PTS_PER_PATCH, len(cands), sample_size - len(selected_indices))
        if n_add > 0:
            selected_indices.update(int(i) for i in rng.choice(cands, size=n_add, replace=False))

        # expand frontier to 26-connected non-empty, unvisited neighbours (9 + 9 + 8, the 3D neighbors)
        vi, vj, vk = voxel
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == dj == dk == 0:
                        continue
                    nbr = (vi + di, vj + dj, vk + dk)
                    if nbr in voxel_map and nbr not in visited:
                        frontier.add(nbr)

    # pad if the frontier ran out before reaching sample_size
    if len(selected_indices) < sample_size:
        remaining = [i for i in range(N) if i not in selected_indices]
        n_pad = min(sample_size - len(selected_indices), len(remaining))
        if n_pad > 0:
            selected_indices.update(int(i) for i in rng.choice(remaining, size=n_pad, replace=False))

    return D[np.array(list(selected_indices)[:sample_size], dtype=int)]


def uniform_partition_mss(
    D: np.ndarray,
    pts_per_patch: int = 240,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Partitions the point cloud into equal-count patches, sample _PTS_PER_PATCH from each.

    Steps: Pick a random unassigned point, then collect its
    pts_per_patch nearest unassigned neighbours to generate a patch. 
    
    In short: Denser regions produce spatially smaller patches, 
    so coverage is more uniform over the surface.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(D)
    if N < _PTS_PER_PATCH:
        return D.copy()

    tree      = KDTree(D)
    unassigned = np.ones(N, dtype=bool)
    patches: list[list[int]] = []

    while unassigned.sum() >= pts_per_patch:
        seed_idx = int(rng.choice(np.where(unassigned)[0]))

        _, nn = tree.query(D[seed_idx], k=N)
        patch = [int(i) for i in nn if unassigned[i]][:pts_per_patch]

        if len(patch) < pts_per_patch:
            break

        patches.append(patch)
        unassigned[patch] = False

    selected: list[int] = []
    for patch in patches:
        chosen = rng.choice(patch, size=min(_PTS_PER_PATCH, len(patch)), replace=False)
        selected.extend(int(i) for i in chosen)

    return D[np.array(selected, dtype=int)]


def visualize_mss(D: np.ndarray, Mj: np.ndarray) -> None:
    """
    Show the point cloud with MSS points highlighted.

    Green : selected MSS points
    Red   : rest of the cloud, dimmed
    """
    pl = pv.Plotter()
    pl.set_background("white")

    mss_mask = np.zeros(len(D), dtype=bool)
    _, idx = KDTree(D).query(Mj, k=1)
    mss_mask[idx] = True

    point_size = 6

    if (~mss_mask).any():
        pl.add_points(
            D[~mss_mask].astype(np.float32),
            render_points_as_spheres=True,
            point_size=point_size,
            color="red",
            opacity=0.4,
        )

    if mss_mask.any():
        pl.add_points(
            D[mss_mask].astype(np.float32),
            render_points_as_spheres=True,
            point_size=point_size,
            color="green",
        )

    pl.show_axes()
    pl.add_axes()
    pl.show_grid(location="outer", ticks="outside", grid="front", all_edges=True)
    pl.show_bounds(
        grid="back", location="outer", all_edges=True, ticks="outside",
        xtitle="X", ytitle="Y", ztitle="Z", font_size=10,
    )
    pl.add_text(f"MSS: {mss_mask.sum()} / {len(D)} points", position="upper_left", font_size=10)
    pl.reset_camera()
    pl.enable_eye_dome_lighting()
    pl.show()
