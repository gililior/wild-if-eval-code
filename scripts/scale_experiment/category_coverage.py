"""Compute per-constraint-type coverage: how many tasks contain >=1 constraint of each type.

Also project to subset sizes [100, 250, 500, 1000, 2000, 3000, 5000, 7523] assuming
uniform random sampling — i.e. expected count = (category_task_count / 7523) * N.
Flag categories whose expected count drops below 500 at smaller N.
"""
from pathlib import Path
from collections import Counter

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "scripts" / "scale_experiment" / "data"
TOTAL_N = 7523
SUBSET_SIZES = [100, 250, 500, 1000, 2000, 3000, 5000, 7523]
THRESHOLD = 500


def main():
    meta = pd.read_parquet(DATA_DIR / "task_meta.parquet")
    assert len(meta) == TOTAL_N

    # For each task, "categories" is a "|"-joined sorted unique list
    counts = Counter()
    for cats in meta["categories"]:
        for c in cats.split("|"):
            counts[c] += 1

    # Order by frequency descending; pull "Other" to the end
    cats_sorted = sorted(counts, key=lambda x: (x == "Other", -counts[x]))

    rows = []
    for cat in cats_sorted:
        c = counts[cat]
        row = {"category": cat, "task_count_full": c, "pct_of_tasks": round(100 * c / TOTAL_N, 1)}
        for N in SUBSET_SIZES:
            row[f"exp_at_N{N}"] = int(round(c * N / TOTAL_N))
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "category_coverage.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df.to_string(index=False))

    # Below-threshold flags
    print(f"\nCategories with expected count < {THRESHOLD} at each N (assuming uniform sampling):")
    for N in SUBSET_SIZES:
        below = [c for c in cats_sorted if c != "Other" and counts[c] * N / TOTAL_N < THRESHOLD]
        print(f"  N={N:>5}: {len(below)} categories below {THRESHOLD} -> {below}")


if __name__ == "__main__":
    main()
