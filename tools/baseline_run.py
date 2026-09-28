"""Sweep the pipeline over every point cloud, into a chosen output dir.

Usage:  python tools/baseline_run.py <out_dir> [pc_name ...]

With no pc_name, every cloud in data/point_clouds/ is processed, sharing one RNG across
them - so a cloud's trial seeds depend on which clouds preceded it alphabetically.
"""
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np

from src import pipeline as R


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out_dir = Path(sys.argv[1]).resolve()
    names = sys.argv[2:]

    R.OUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pc_files = sorted(
        f for f in R.PC_DIR.iterdir()
        if f.is_file() and not f.name.startswith("normals_") and not f.name.startswith(".")
    )
    if names:
        pc_files = [f for f in pc_files if f.name in names or f.stem in names]
    if not pc_files:
        print("no matching point clouds")
        return 1

    print(f"out_dir   = {out_dir}")
    print(f"seed      = {R.MASTER_SEED}")
    print(f"clouds    = {[f.name for f in pc_files]}")

    # one master rng for the whole sweep, the sweep semantics described above
    rng = np.random.default_rng(R.MASTER_SEED)
    for pc_file in pc_files:
        t0 = time.perf_counter()
        R.run_for_pc(pc_file, rng)
        print(f"\n### {pc_file.name} took {time.perf_counter() - t0:.1f}s\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
