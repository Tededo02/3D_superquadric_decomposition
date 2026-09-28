"""Compare two baseline output trees cell by cell.

Usage:  python baseline_compare.py <baseline_dir> <new_dir> [--tol 1e-12] [--with-runtime]

Exit code 0 means every value matched. runtime_s is wall-clock and is skipped
unless --with-runtime is passed.
"""
import csv
import sys
from pathlib import Path

IGNORE = {"runtime_s"}


def csv_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)).replace("\\", "/"): p for p in sorted(root.rglob("*.csv"))}


def read_rows(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def cells_equal(a: str, b: str, tol: float) -> bool:
    if a == b:
        return True
    if tol <= 0:
        return False
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    if fa == fb:
        return True
    scale = max(abs(fa), abs(fb), 1.0)
    return abs(fa - fb) <= tol * scale


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2

    tol = 0.0
    for f in flags:
        if f.startswith("--tol"):
            tol = float(f.split("=", 1)[1]) if "=" in f else 1e-12
    ignore = set() if "--with-runtime" in flags else set(IGNORE)

    base, new = Path(args[0]), Path(args[1])
    a_files, b_files = csv_files(base), csv_files(new)

    problems = 0
    only_a = sorted(set(a_files) - set(b_files))
    only_b = sorted(set(b_files) - set(a_files))
    for name in only_a:
        print(f"MISSING in new:      {name}")
        problems += 1
    for name in only_b:
        print(f"UNEXPECTED in new:   {name}")
        problems += 1

    checked_files = checked_cells = 0
    for name in sorted(set(a_files) & set(b_files)):
        fa, ra = read_rows(a_files[name])
        fb, rb = read_rows(b_files[name])
        checked_files += 1

        if fa != fb:
            print(f"HEADER differs:      {name}\n    baseline: {fa}\n    new:      {fb}")
            problems += 1
            continue
        if len(ra) != len(rb):
            print(f"ROW COUNT differs:   {name}  baseline={len(ra)} new={len(rb)}")
            problems += 1
            continue

        cols = [c for c in fa if c not in ignore]
        for i, (row_a, row_b) in enumerate(zip(ra, rb), start=2):
            for col in cols:
                va, vb = row_a[col], row_b[col]
                checked_cells += 1
                if not cells_equal(va, vb, tol):
                    print(f"VALUE differs:        {name}  line {i}  col '{col}'"
                          f"\n    baseline: {va}\n    new:      {vb}")
                    problems += 1

    skipped = ", ".join(sorted(ignore)) or "none"
    print(f"\n{checked_files} file(s), {checked_cells} cell(s) compared"
          f"  |  tol={tol}  |  skipped columns: {skipped}")
    if problems:
        print(f"RESULT: {problems} difference(s) -- NOT a match")
        return 1
    print("RESULT: identical -- refactor preserved all results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
