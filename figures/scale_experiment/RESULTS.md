# WildIFEval — Dataset Scale / Diminishing-Returns Analysis

**Setup.** 14 models scored by a DeepSeek-V3 LLM-as-a-judge on the 7,523-task
WildIFEval test set (only tasks with ≥2 constraints retained, matching the paper's
existing pipeline). For each subset size *N* we draw 50 task subsamples without
replacement, recompute each model's mean **strict** (all constraints satisfied)
and mean **soft** (fraction satisfied) scores, and compare the resulting ranking
to the full-data ranking via Kendall's τ. The full-data ranking (deepseek-v3 >
llama3.3-70b > mistral-large > llama3.1-405b > qwen2.5-72b > Qwen2.5-7B > … >
Qwen2.5-0.5B) is used as the reference.

## Stability at key subset sizes

| *N* | std(strict) | std(soft) | mean τ (strict) | mean τ (soft) | 95% CI width — top-vs-2nd gap (strict) | 95% CI width — top-vs-2nd gap (soft) |
|---:|---:|---:|---:|---:|---:|---:|
| 100   | 0.0446 | 0.0286 | 0.892 | 0.888 | 0.0900 | 0.0430 |
| 250   | 0.0279 | 0.0174 | 0.916 | 0.916 | 0.0613 | 0.0311 |
| 500   | 0.0200 | 0.0130 | 0.953 | 0.938 | 0.0602 | 0.0220 |
| 1000  | 0.0127 | 0.0082 | 0.967 | 0.957 | 0.0360 | 0.0188 |
| 2000  | 0.0090 | 0.0058 | 0.984 | 0.974 | 0.0199 | 0.0120 |
| 3000  | 0.0063 | 0.0041 | 0.990 | 0.982 | 0.0230 | 0.0100 |
| 5000  | 0.0039 | 0.0024 | 0.996 | 0.993 | 0.0157 | 0.0060 |
| 7523  | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.0000 | 0.0000 |

`std(*)` is the per-model standard deviation of the mean score across 50 resamples,
averaged across the 14 models. The 95% CI width is the percentile-based 2.5–97.5
range of the (rank-1 minus rank-2) score gap across the 50 resamples; smaller is
more reliable.

## Interpretation (paper appendix)

> Benchmark signal stabilises rapidly with dataset size. At *N* = 500 the mean
> Kendall's τ between the subsample-induced ranking and the full-data ranking is
> already 0.95 (strict) / 0.94 (soft), and the across-model score standard
> deviation has dropped from 0.045 (strict) at *N* = 100 to 0.020. However, at
> these smaller sizes the 95% confidence interval for the top-vs-second-best
> model gap is wider than the gap itself for several pairs (e.g. for *N* ≤ 1000
> the strict CI width is ≥ 0.036, comparable in size to the gap between the 2nd
> and 3rd ranked models — 0.006 in the full data), meaning small subsets cannot
> reliably resolve close model pairs. Beyond *N* ≈ 2000 returns diminish: τ
> exceeds 0.97, top-gap CI width shrinks below 0.02, and the full 7,523-task
> scale primarily improves discrimination between nearby models and ensures
> adequate per-category coverage (next section).

## Constraint-type coverage

Number of tasks containing at least one constraint of each category (8 paper
categories + `Other`). Expected counts at smaller *N* assume uniform random
subsampling.

| Category | # tasks (full) | % of tasks | exp. @ N=100 | @250 | @500 | @1000 | @2000 | @3000 | @5000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Include / Avoid      | 5094 | 67.7% | 68 | 169 | 339 | 677  | 1354 | 2031 | 3386 |
| Format and Structure | 2566 | 34.1% | 34 |  85 | 171 | 341  |  682 | 1023 | 1705 |
| Focus / Emphasis     | 2079 | 27.6% | 28 |  69 | 138 | 276  |  553 |  829 | 1382 |
| Length               | 1987 | 26.4% | 26 |  66 | 132 | 264  |  528 |  792 | 1321 |
| Style and Tone       | 1811 | 24.1% | 24 |  60 | 120 | 241  |  481 |  722 | 1204 |
| Persona and Role     | 1392 | 18.5% | 19 |  46 |  93 | 185  |  370 |  555 |  925 |
| Ensure Quality       | 1175 | 15.6% | 16 |  39 |  78 | 156  |  312 |  469 |  781 |
| Editing              |  642 |  8.5% |  9 |  21 |  43 |  85  |  171 |  256 |  427 |
| Other                | 1267 | 16.8% | 17 |  42 |  84 | 168  |  337 |  505 |  842 |

**Categories below 500 tasks at each subset size** (excluding `Other`; 500-task
threshold chosen as a rough minimum for stable per-category mean estimation):

- *N* ≤ 1000: 7–8 of 8 categories under 500 tasks.
- *N* = 2000: still **4** under 500 (Style and Tone, Persona and Role, Ensure Quality, Editing).
- *N* = 3000: still **2** under 500 (Ensure Quality, Editing).
- *N* = 5000: only **1** under 500 (Editing — 427 tasks).
- *N* = 7523 (full): **0** under 500.

Editing is the rarest category (642 tasks, 8.5% of the data); reliably measuring
per-category performance on Editing — and Ensure Quality (1,175 tasks) — is the
main scale-driven motivation, since these would otherwise have <500 tasks in any
subset and the rarer pairwise category combinations used in the co-occurrence
analysis would have far fewer.

## Reproducing

```bash
python3 analysis/scale_experiment/aggregate_scores.py   # build per_task_scores
python3 analysis/scale_experiment/run_subsample.py      # stability metrics
python3 analysis/scale_experiment/category_coverage.py  # per-category counts
python3 analysis/scale_experiment/make_figures.py       # Figure A, B (and bonus C)
```

Outputs land in `analysis/scale_experiment/data/` and
`analysis/scale_experiment/figures/`.
