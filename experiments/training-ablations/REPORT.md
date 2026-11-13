# Local experiment results

These are measured local runs, not the paper's published scores.
A missing correlation is undefined or unavailable; it is never replaced with zero.

| Run | Rows / source | Valid | Pearson vs GPT-4 | Spearman | Kendall τ-b | Ranking accuracy (all) |
|---|---:|---:|---:|---:|---:|---:|
| full | 10 / 1000 | 10 | 0.8191 | 0.8608 | 0.7695 | — |
| no_reference | 10 / 1000 | 10 | 0.5080 | 0.4743 | 0.4067 | — |
| no_rubric | 10 / 1000 | 8 | 0.9007 | 0.9103 | 0.8292 | — |
| score_only | 10 / 1000 | 8 | 0.7171 | 0.8170 | 0.7426 | — |

Correlations use the mean of all requested successful draws per row. Incomplete rows are excluded
from correlation, but are retained in coverage and the ranking all-row denominator. Confidence intervals
resample original instruction groups, keeping related responses together. Small samples are exploratory.

`summary.json`, each run's `metrics.json`, `scores.csv`, `manifest.json`, and `predictions.jsonl`
contain the complete machine-readable evidence, including generation failures and retry costs.
