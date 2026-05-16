"""Render Figure A (score std vs N) and Figure B (Kendall tau vs N) as PDF."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "scripts" / "scale_experiment" / "data"
FIG_DIR = REPO_ROOT / "figures" / "scale_experiment"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FULL_N = 7523


def main():
    df = pd.read_csv(DATA_DIR / "subsample_stability.csv")
    rep = pd.read_csv(DATA_DIR / "per_replicate.csv")

    # use only subset sizes < full for plotting variance trend; mark the full point as anchor
    df_plot = df[df["N"] < FULL_N].copy()

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})

    # --- Figure A: std of model mean score vs N ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(df_plot["N"], df_plot["mean_strict_std"], marker="o",
            label="Strict score", color="#1f77b4", linewidth=2)
    ax.plot(df_plot["N"], df_plot["mean_soft_std"], marker="s",
            label="Soft score", color="#d62728", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Subset size $N$ (log scale)")
    ax.set_ylabel("Std of model mean score across 50 resamples\n(averaged over models)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks(df_plot["N"])
    ax.set_xticklabels([str(n) for n in df_plot["N"]], rotation=0)
    fig.tight_layout()
    out_a = FIG_DIR / "figA_score_std_vs_N.pdf"
    fig.savefig(out_a, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figA_score_std_vs_N.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_a}")

    # --- Figure B: Kendall tau vs N ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(df["N"], df["mean_kt_strict"], marker="o",
            label="Strict ranking", color="#1f77b4", linewidth=2)
    ax.plot(df["N"], df["mean_kt_soft"], marker="s",
            label="Soft ranking", color="#d62728", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Subset size $N$ (log scale)")
    ax.set_ylabel("Mean Kendall's $\\tau$ vs. full-data ranking\n(over 50 resamples)")
    ax.set_ylim(0.85, 1.005)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xticks(df["N"])
    ax.set_xticklabels([str(n) for n in df["N"]], rotation=0)
    fig.tight_layout()
    out_b = FIG_DIR / "figB_kendall_tau_vs_N.pdf"
    fig.savefig(out_b, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figB_kendall_tau_vs_N.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_b}")

    # --- Bonus: top-gap CI width vs N (saved too — useful for appendix) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(df_plot["N"], df_plot["gap_ci95_width_strict"], marker="o",
            label="Strict", color="#1f77b4", linewidth=2)
    ax.plot(df_plot["N"], df_plot["gap_ci95_width_soft"], marker="s",
            label="Soft", color="#d62728", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Subset size $N$ (log scale)")
    ax.set_ylabel("95% CI width: top-vs-second model gap")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks(df_plot["N"])
    ax.set_xticklabels([str(n) for n in df_plot["N"]], rotation=0)
    fig.tight_layout()
    out_c = FIG_DIR / "figC_top_gap_ci_vs_N.pdf"
    fig.savefig(out_c, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figC_top_gap_ci_vs_N.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
