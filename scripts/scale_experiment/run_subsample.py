"""Subsampling stability experiment for benchmark scale justification.

For each N in [100,250,500,1000,2000,3000,5000,7523]:
    Repeat 50 times:
        - sample N tasks without replacement
        - compute each model's mean strict and mean soft on the subset
        - record rankings
Metrics per N:
    (a) std across 50 replicates of each model's mean score, averaged across models
    (b) mean Kendall's tau between subsample ranking and full-data ranking
    (c) 95% CI half-width for the gap between rank-1 and rank-2 model

Outputs:
    scripts/scale_experiment/data/subsample_stability.csv
    scripts/scale_experiment/data/per_replicate.csv  (raw replicate-level data)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "scripts" / "scale_experiment" / "data"
SUBSET_SIZES = [100, 250, 500, 1000, 2000, 3000, 5000, 7523]
N_REPLICATES = 50
SEED = 42


def main():
    df = pd.read_parquet(DATA_DIR / "per_task_scores.parquet")
    models = sorted(df["model"].unique())
    tasks = df["task"].unique()
    n_tasks_total = len(tasks)
    assert n_tasks_total == 7523, f"expected 7523 tasks, got {n_tasks_total}"

    # Build per-model task→score matrices for fast subsetting
    # Use task index (0..7522). Ensure consistent ordering.
    task_idx = {t: i for i, t in enumerate(tasks)}
    strict_mat = np.zeros((len(models), n_tasks_total), dtype=np.float32)
    soft_mat = np.zeros((len(models), n_tasks_total), dtype=np.float32)
    for mi, model in enumerate(models):
        sub = df[df["model"] == model]
        idx = sub["task"].map(task_idx).to_numpy()
        strict_mat[mi, idx] = sub["strict_score"].to_numpy()
        soft_mat[mi, idx] = sub["soft_score"].to_numpy()

    # Full-data rankings (lower rank index = higher score)
    full_strict_means = strict_mat.mean(axis=1)
    full_soft_means = soft_mat.mean(axis=1)
    full_strict_rank = (-full_strict_means).argsort().argsort()
    full_soft_rank = (-full_soft_means).argsort().argsort()

    rng = np.random.default_rng(SEED)

    rows_replicate = []
    rows_summary = []

    for N in SUBSET_SIZES:
        strict_means = np.zeros((N_REPLICATES, len(models)))
        soft_means = np.zeros((N_REPLICATES, len(models)))
        kt_strict = np.zeros(N_REPLICATES)
        kt_soft = np.zeros(N_REPLICATES)
        top_gap_strict = np.zeros(N_REPLICATES)
        top_gap_soft = np.zeros(N_REPLICATES)

        for r in range(N_REPLICATES):
            if N == n_tasks_total:
                sel = np.arange(n_tasks_total)
            else:
                sel = rng.choice(n_tasks_total, size=N, replace=False)

            s_means = strict_mat[:, sel].mean(axis=1)
            f_means = soft_mat[:, sel].mean(axis=1)
            strict_means[r] = s_means
            soft_means[r] = f_means

            sub_strict_rank = (-s_means).argsort().argsort()
            sub_soft_rank = (-f_means).argsort().argsort()
            kt_strict[r], _ = kendalltau(sub_strict_rank, full_strict_rank)
            kt_soft[r], _ = kendalltau(sub_soft_rank, full_soft_rank)

            # gap between best and 2nd-best model in this replicate
            sorted_s = np.sort(s_means)[::-1]
            sorted_f = np.sort(f_means)[::-1]
            top_gap_strict[r] = sorted_s[0] - sorted_s[1]
            top_gap_soft[r] = sorted_f[0] - sorted_f[1]

            for mi, model in enumerate(models):
                rows_replicate.append({
                    "N": N, "replicate": r, "model": model,
                    "strict_mean": s_means[mi], "soft_mean": f_means[mi],
                })

        # (a) per-model std across replicates, averaged over models
        strict_std_per_model = strict_means.std(axis=0, ddof=1) if N_REPLICATES > 1 and N < n_tasks_total else np.zeros(len(models))
        soft_std_per_model = soft_means.std(axis=0, ddof=1) if N_REPLICATES > 1 and N < n_tasks_total else np.zeros(len(models))
        mean_strict_std = float(strict_std_per_model.mean())
        mean_soft_std = float(soft_std_per_model.mean())

        # (b) mean Kendall's tau vs full ranking
        if N == n_tasks_total:
            mean_kt_strict = 1.0
            mean_kt_soft = 1.0
        else:
            mean_kt_strict = float(np.mean(kt_strict))
            mean_kt_soft = float(np.mean(kt_soft))

        # (c) 95% CI half-width for top-vs-second gap (percentile-based)
        if N == n_tasks_total:
            gap_ci_strict = 0.0
            gap_ci_soft = 0.0
        else:
            lo_s, hi_s = np.percentile(top_gap_strict, [2.5, 97.5])
            lo_f, hi_f = np.percentile(top_gap_soft, [2.5, 97.5])
            gap_ci_strict = float(hi_s - lo_s)
            gap_ci_soft = float(hi_f - lo_f)

        rows_summary.append({
            "N": N,
            "mean_strict_std": mean_strict_std,
            "mean_soft_std": mean_soft_std,
            "mean_kt_strict": mean_kt_strict,
            "mean_kt_soft": mean_kt_soft,
            "gap_ci95_width_strict": gap_ci_strict,
            "gap_ci95_width_soft": gap_ci_soft,
            "mean_top_gap_strict": float(top_gap_strict.mean()),
            "mean_top_gap_soft": float(top_gap_soft.mean()),
        })
        print(f"N={N:>5}: std_strict={mean_strict_std:.4f} std_soft={mean_soft_std:.4f} "
              f"tau_strict={mean_kt_strict:.3f} tau_soft={mean_kt_soft:.3f} "
              f"gap_ci_strict={gap_ci_strict:.4f} gap_ci_soft={gap_ci_soft:.4f}")

    summary_df = pd.DataFrame(rows_summary)
    rep_df = pd.DataFrame(rows_replicate)
    summary_path = DATA_DIR / "subsample_stability.csv"
    rep_path = DATA_DIR / "per_replicate.csv"
    summary_df.to_csv(summary_path, index=False)
    rep_df.to_csv(rep_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {rep_path}")

    # Save full-data per-model means for downstream reference
    full_means = pd.DataFrame({
        "model": models,
        "strict_mean_full": full_strict_means,
        "soft_mean_full": full_soft_means,
    }).sort_values("strict_mean_full", ascending=False)
    full_means.to_csv(DATA_DIR / "full_dataset_means.csv", index=False)
    print(f"Wrote {DATA_DIR / 'full_dataset_means.csv'}")


if __name__ == "__main__":
    main()
