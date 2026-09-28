"""Snapshot what explore_pointcloud.ipynb computes, as CSVs you can diff after a refactor.

The notebook's own code cells are executed (not a reimplementation of them), once per point
cloud, and the results are written as:

    <out_dir>/<cloud>/config.csv           effective pipeline configuration
    <out_dir>/<cloud>/runs.csv             per (trial, algo) metrics
    <out_dir>/<cloud>/models.csv           every candidate superquadric's 11 parameters
    <out_dir>/<cloud>/setcover.csv         the ransacov selection table
    <out_dir>/<cloud>/setcover_models.csv  parameters of the selected superquadrics

Usage
-----
    python tools/notebook_snapshot.py data/nb_baseline/golden
    # ... refactor ...
    python tools/notebook_snapshot.py data/nb_baseline/after
    python tools/baseline_compare.py data/nb_baseline/golden data/nb_baseline/after

baseline_compare.py diffs every CSV cell-exactly and skips runtime_s, which is wall clock.
Comparing models.csv matters: two decompositions can land on similar Chamfer numbers while
being different shapes, and only the parameters catch that.

Visualisation is stubbed out, so sections 4 and 7 are exercised but no window opens.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root, this file lives in tools/
NOTEBOOK = ROOT / "explore_pointcloud.ipynb"

# small, fast clouds covering both loader paths: bare point clouds (.ply, with sibling
# normals_ files) and meshes (.stl, vertex normals)
DEFAULT_CLOUDS = ["supersingle0_0_0", "cat1_0_0", "bird"]

# pipeline settings worth recording; a refactor that changes a default shows up as a diff
CONFIG_KEYS = [
    "THRESHOLD", "GRAPH_RADIUS", "MAX_MODELS", "MAX_ITER", "INNER_ITER", "K",
    "EVAL_SEED", "MASTER_SEED", "N_OUTLIERS", "NOISE", "MAX_LOAD_PTS",
    "MAX_COVER_ITER", "COVER_METHODS",
]

MODEL_COLS = ["a1", "a2", "a3", "e1", "e2",
              "rot0", "rot1", "rot2", "t0", "t1", "t2"]


def model_row(m):
    return {
        "a1": m.a1, "a2": m.a2, "a3": m.a3, "e1": m.e1, "e2": m.e2,
        "rot0": m.rot[0], "rot1": m.rot[1], "rot2": m.rot[2],
        "t0": m.t[0], "t1": m.t[1], "t2": m.t[2],
    }


def run_notebook(cloud: str, n_trials: int, seed: int, outlier_frac: float) -> dict:
    """Execute the notebook's code cells for one cloud and return its namespace."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    cfg_i = next(i for i, c in enumerate(cells) if "PC_PATH = ROOT" in "".join(c["source"]))

    matches = [p for p in (ROOT / "data" / "point_clouds").iterdir()
               if p.stem == cloud and not p.name.startswith("normals_")]
    if not matches:
        available = sorted(p.stem for p in (ROOT / "data" / "point_clouds").iterdir()
                           if not p.name.startswith("normals_"))
        raise SystemExit(f"point cloud {cloud!r} not found in data/point_clouds; "
                         f"available: {available}")
    suffix = matches[0].suffix

    override = (
        f'PC_PATH      = ROOT / "data" / "point_clouds" / "{cloud}{suffix}"\n'
        f"N_TRIALS     = {n_trials}\n"
        f"SEED         = {seed}\n"
        f"OUTLIER_FRAC = {outlier_frac}\n"
    )
    # keep the blocking 3D window shut; the cells themselves still run
    stub = "vis.show_mesh_and_points = lambda meshes, *a, **kw: []\n"

    g = {"__name__": "__main__"}
    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
        if i == cfg_i:
            src += "\n" + override
        exec(compile(src, f"<cell {i}>", "exec"), g)
        if i == cfg_i:
            exec(compile(stub, "<stub>", "exec"), g)
    return g


def snapshot(out_root: Path, cloud: str, n_trials: int, seed: int, outlier_frac: float):
    t0 = time.perf_counter()
    print(f"\n{'=' * 70}\n{cloud}\n{'=' * 70}", flush=True)
    g = run_notebook(cloud, n_trials, seed, outlier_frac)

    out = out_root / cloud
    out.mkdir(parents=True, exist_ok=True)
    run_mod = g["pipeline"]

    # 1. effective configuration
    cfg = {"cloud": cloud, "n_trials": n_trials, "seed": seed,
           "outlier_frac": outlier_frac,
           "algorithms": "|".join(g["ALGORITHMS"]),
           "n_clean": g["n_clean"], "n_points": len(g["points"])}
    for key in CONFIG_KEYS:
        val = getattr(run_mod, key, None)
        cfg[key] = "|".join(map(str, val)) if isinstance(val, tuple) else val
    pd.DataFrame([cfg]).to_csv(out / "config.csv", index=False)

    # 2. per-run metrics, sorted so row order never depends on dict iteration
    runs = g["df"].sort_values(["trial", "algo"], kind="stable")
    runs.to_csv(out / "runs.csv", index=False)

    # 3. every candidate superquadric
    rows = []
    for (trial, algo), models in sorted(g["models_by"].items()):
        for idx, m in enumerate(models):
            rows.append({"trial": trial, "algo": algo, "idx": idx, **model_row(m)})
    pd.DataFrame(rows, columns=["trial", "algo", "idx", *MODEL_COLS]).to_csv(
        out / "models.csv", index=False)

    # 4. set-cover table and the models it picked
    g["cover_df"].to_csv(out / "setcover.csv", index=False)
    sel = []
    for algo, models in sorted(g["selected_by"].items()):
        for idx, m in enumerate(models):
            sel.append({"algo": algo, "idx": idx, **model_row(m)})
    pd.DataFrame(sel, columns=["algo", "idx", *MODEL_COLS]).to_csv(
        out / "setcover_models.csv", index=False)

    print(f"  -> {out}  ({len(runs)} runs, {len(rows)} candidates, "
          f"{len(sel)} selected)  {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="directory to write the snapshot into")
    ap.add_argument("--clouds", default=",".join(DEFAULT_CLOUDS),
                    help="comma-separated point cloud stems")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outlier-frac", type=float, default=0.01)
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    clouds = [c.strip() for c in args.clouds.split(",") if c.strip()]

    print(f"notebook  : {NOTEBOOK.name}")
    print(f"out_dir   : {out_root}")
    print(f"clouds    : {clouds}")
    print(f"trials    : {args.trials}   seed: {args.seed}   "
          f"outlier_frac: {args.outlier_frac}")

    t0 = time.perf_counter()
    for cloud in clouds:
        snapshot(out_root, cloud, args.trials, args.seed, args.outlier_frac)

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")
    print("compare a later snapshot with:")
    print(f"  python tools/baseline_compare.py {args.out_dir} <new_dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
