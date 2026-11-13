# Local experiment results

These are measured local runs, not the paper's published scores.
A missing correlation is undefined or unavailable; it is never replaced with zero.

| Run | Rows / source | Valid | Pearson vs GPT-4 | Spearman | Kendall τ-b | Ranking accuracy (all) |
|---|---:|---:|---:|---:|---:|---:|
| feedback_seen | 10 / 1000 | 10 | 0.7517 | 0.7329 | 0.6099 | — |
| feedback_unseen | 10 / 1000 | 10 | 0.8866 | 0.8545 | 0.7716 | — |
| flask | 10 / 2001 | 10 | 0.6348 | 0.6421 | 0.4714 | — |
| hhh | 8 / 221 | 8 | — | — | — | 0.6250 |
| mt_bench | 10 / 320 | 10 | — | — | — | — |
| mt_human | 8 / 1431 | 8 | — | — | — | 0.5000 |
| vicuna | 10 / 320 | 10 | 0.3032 | 0.3016 | 0.2095 | — |

Correlations use the mean of all requested successful draws per row. Incomplete rows are excluded
from correlation, but are retained in coverage and the ranking all-row denominator. Confidence intervals
resample original instruction groups, keeping related responses together. Small samples are exploratory.

`summary.json`, each run's `metrics.json`, `scores.csv`, `manifest.json`, and `predictions.jsonl`
contain the complete machine-readable evidence, including generation failures and retry costs.
