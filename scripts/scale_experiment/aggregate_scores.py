"""Aggregate per-task strict/soft scores for all 14 models into a tidy table.

Output: scripts/scale_experiment/data/per_task_scores.parquet
Columns: task, model, strict_score, soft_score, num_constraints, categories
(`categories` is a "|"-joined sorted unique list of category labels in the task.)

Uses the deepseek-v3 LLM-as-a-judge scores.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "llm_aaj_scores"
CATEGORIES_CSV = REPO_ROOT / "data" / "constraint_categories.csv"
OUT_DIR = REPO_ROOT / "scripts" / "scale_experiment" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DS_NAME = "gililior/wild-if-eval-decomposed-gpt4"
MIN_NUM_CONSTRAINTS = 2
JUDGE_PREFIX = "llm-aaj-deepseek-v3"

MODELS = [
    "Llama-3.1-8B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
    "deepseek-v3",
    "gemma-2-2b",
    "gemma-2-9b",
    "llama3.1-405b",
    "llama3.3-70b",
    "mistral-large",
    "qwen2.5-72b",
]


def load_categories():
    df = pd.read_csv(CATEGORIES_CSV)
    cat = {}
    for _, row in df.iterrows():
        category = row["categories"]
        if row.get("Other", 0) == 1 or pd.isna(category):
            category = "Other"
        cat[row["constraint"]] = category
    return cat


def task_score(scores_dict):
    """Returns (strict, soft, n) for one task's `scores` dict."""
    bin_scores = []
    for v in scores_dict.values():
        if v == "ERR":
            v = 0.0
        bin_scores.append(1 if float(v) >= 0.5 else 0)
    n = len(bin_scores)
    if n == 0:
        return 0, 0.0, 0
    soft = float(np.mean(bin_scores))
    strict = 1 if all(b == 1 for b in bin_scores) else 0
    return strict, soft, n


def main():
    print("Loading dataset...")
    ds = load_dataset(DS_NAME, split="test")
    ds = ds.filter(lambda x: len(x["decomposition"]) >= MIN_NUM_CONSTRAINTS)
    print(f"  tasks after filter (>={MIN_NUM_CONSTRAINTS} constraints): {len(ds)}")

    cat_map = load_categories()
    print(f"  category map entries: {len(cat_map)}")

    task_to_cats = {}
    task_to_n = {}
    for row in ds:
        decomp = row["decomposition"]
        cats = sorted({cat_map.get(c, "Other") for c in decomp})
        task_to_cats[row["task"]] = "|".join(cats)
        task_to_n[row["task"]] = len(decomp)

    rows = []
    for model in MODELS:
        fname = SCORES_DIR / f"{JUDGE_PREFIX}.{model}-0shot-wild-if-eval.json"
        with open(fname) as f:
            preds = json.load(f)
        n_found = 0
        for task in task_to_cats:
            if task not in preds:
                continue
            strict, soft, n = task_score(preds[task]["scores"])
            rows.append({
                "task": task,
                "model": model,
                "strict_score": strict,
                "soft_score": soft,
                "num_constraints": n,
                "categories": task_to_cats[task],
            })
            n_found += 1
        print(f"  {model}: {n_found} tasks")

    df = pd.DataFrame(rows)
    out_parquet = OUT_DIR / "per_task_scores.parquet"
    out_csv = OUT_DIR / "per_task_scores.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_parquet}")
    print(f"Wrote {out_csv}")
    print(f"Rows: {len(df)}; models: {df['model'].nunique()}; unique tasks: {df['task'].nunique()}")

    # Save task-level metadata too
    meta = pd.DataFrame([
        {"task": t, "num_constraints": task_to_n[t], "categories": task_to_cats[t]}
        for t in task_to_cats
    ])
    meta.to_parquet(OUT_DIR / "task_meta.parquet", index=False)
    print(f"Wrote {OUT_DIR / 'task_meta.parquet'} ({len(meta)} tasks)")


if __name__ == "__main__":
    main()
