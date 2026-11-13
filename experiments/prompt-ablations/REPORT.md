# Local experiment results

These are measured local runs, not the paper's published scores.
A missing correlation is undefined or unavailable; it is never replaced with zero.

| Run | Rows / source | Valid | Pearson vs GPT-4 | Spearman | Kendall τ-b | Ranking accuracy (all) |
|---|---:|---:|---:|---:|---:|---:|
| full | 10 / 1000 | 10 | 0.8411 | 0.8255 | 0.7278 | — |
| no_reference | 10 / 1000 | 10 | 0.5941 | 0.5715 | 0.4679 | — |
| no_rubric | 10 / 1000 | 9 | 0.5344 | 0.5913 | 0.4763 | — |
| score_only | 10 / 1000 | 5 | 0.7826 | 0.7826 | 0.5976 | — |

Correlations use the mean of all requested successful draws per row. Incomplete rows are excluded
from correlation, but are retained in coverage and the ranking all-row denominator. Confidence intervals
resample original instruction groups, keeping related responses together. Small samples are exploratory.

`summary.json`, each run's `metrics.json`, `scores.csv`, `manifest.json`, and `predictions.jsonl`
contain the complete machine-readable evidence, including generation failures and retry costs.
