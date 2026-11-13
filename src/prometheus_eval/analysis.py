"""Metrics with explicit missingness, denominator accounting, and saved plots."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .io import read_records, write_json


def numeric_label(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not math.isfinite(x)
        or not 1 <= x <= 5
        for x in values
    ):
        return None
    return sum(values) / len(values)


def correlations(x, y):
    if len(x) < 3:
        return {
            "n": len(x),
            "pearson": None,
            "spearman": None,
            "kendall_tau_b": None,
            "reason": "fewer than three labelled complete rows",
        }
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "n": len(x),
            "pearson": None,
            "spearman": None,
            "kendall_tau_b": None,
            "reason": "constant predictions or labels",
        }
    return {
        "n": len(x),
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "kendall_tau_b": float(stats.kendalltau(x, y, variant="b").statistic),
        "reason": None,
    }


def cluster_pearson_ci(rows, label, seed=42, draws=1000):
    groups = defaultdict(list)
    for row in rows:
        gold = numeric_label(row["labels"].get(label))
        if row["status"] == "ok" and gold is not None:
            groups[row.get("instruction_group", row["row_id"])].append((row["mean_score"], gold))
    if len(groups) < 3:
        return {"interval": None, "reason": "fewer than three independent instruction groups"}
    values, keys, rng = [], list(groups), np.random.default_rng(seed)
    for _ in range(draws):
        pairs = [pair for i in rng.integers(0, len(keys), len(keys)) for pair in groups[keys[i]]]
        x, y = np.array(pairs).T
        if np.ptp(x) and np.ptp(y):
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "interval": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
        "valid_bootstraps": len(values),
        "requested_bootstraps": draws,
        "instruction_groups": len(groups),
        "method": "percentile cluster bootstrap",
        "seed": seed,
    }


def all_attempts(row):
    if row["mode"] == "absolute":
        return [a for g in row["grades"] for a in g["attempts"]]
    return [a for pair in row["rounds"] for grade in pair.values() for a in grade["attempts"]]


def summarize(rows):
    if not rows:
        return {"rows": 0, "complete": 0}
    complete = [r for r in rows if r["status"] == "ok"]
    attempts = [a for row in rows for a in all_attempts(row)]
    report = {
        "rows": len(rows),
        "complete": len(complete),
        "coverage": len(complete) / len(rows),
        "attempts": len(attempts),
        "attempt_status": dict(Counter(a["status"] for a in attempts)),
        "total_generation_seconds": sum(a.get("elapsed_seconds", 0) for a in attempts),
        "generation_tokens": sum(a.get("generation_tokens", 0) for a in attempts),
        "peak_memory_gb": max((a.get("peak_memory_gb", 0) for a in attempts), default=0),
    }
    latency = [a["elapsed_seconds"] for a in attempts if "elapsed_seconds" in a]
    report["latency_seconds"] = {
        "median": float(np.median(latency)) if latency else None,
        "p95": float(np.quantile(latency, 0.95)) if latency else None,
    }
    if rows[0]["mode"] == "absolute":
        for label in ("gpt4_score", "human_score"):
            labelled = [(r["mean_score"], numeric_label(r["labels"].get(label))) for r in complete]
            pairs = [(pred, gold) for pred, gold in labelled if gold is not None]
            x, y = list(zip(*pairs)) if pairs else ([], [])
            report[label] = correlations(x, y)
            report[label]["missing_labels"] = sum(
                numeric_label(r["labels"].get(label)) is None for r in rows
            )
            report[label]["pearson_ci95"] = cluster_pearson_ci(rows, label)
            report[label]["mae"] = float(np.mean(np.abs(np.array(x) - y))) if pairs else None
        report["score_histogram"] = dict(
            Counter(str(g["score"]) for r in rows for g in r["grades"] if g["score"] is not None)
        )
        report["length_bias"] = correlations(
            [r["response_words"] for r in complete], [r["mean_score"] for r in complete]
        )
        report["length_bias"]["interpretation"] = (
            "Observational association, not a causal/adversarial bias test"
        )
    else:
        correct = sum(r["correct"] is True for r in complete)
        report.update(
            correct=correct,
            accuracy_valid=correct / len(complete) if complete else None,
            accuracy_all=correct / len(rows),
            predicted_ties=sum(r["prediction"] == "tie" for r in complete),
            exhausted_ties=sum(r.get("tie_exhausted", False) for r in rows),
        )
        categories = defaultdict(list)
        for row in rows:
            categories[row["metadata"].get("source", "all")].append(row)
        report["categories"] = {
            k: {"rows": len(v), "accuracy_all": sum(r["correct"] is True for r in v) / len(v)}
            for k, v in categories.items()
        }
        report["macro_accuracy_all"] = float(
            np.mean([v["accuracy_all"] for v in report["categories"].values()])
        )
    return report


def analyze_run(run_dir):
    run_dir = Path(run_dir)
    rows = read_records(run_dir / "predictions.jsonl")
    report = summarize(rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report["planned_rows"] = len(manifest["spec"]["indices"])
    report["source_rows"] = manifest["spec"]["source_rows"]
    report["full_dataset"] = report["rows"] == report["source_rows"]
    report["protocol"] = manifest["spec"]
    write_json(run_dir / "metrics.json", report)
    with (run_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_index",
                "status",
                "mean_score",
                "gpt4_score",
                "human_score",
                "prediction",
                "correct",
                "response_words",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_index": row["source_index"],
                    "status": row["status"],
                    "mean_score": row.get("mean_score"),
                    "gpt4_score": numeric_label(row["labels"].get("gpt4_score")),
                    "human_score": numeric_label(row["labels"].get("human_score")),
                    "prediction": row.get("prediction"),
                    "correct": row.get("correct"),
                    "response_words": row.get("response_words"),
                }
            )
    return report


def report_suite(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    summaries = {}
    for path in sorted(root.glob("*/predictions.jsonl")):
        summaries[path.parent.name] = analyze_run(path.parent)
    write_json(root / "summary.json", summaries)
    lines = [
        "# Local experiment results",
        "",
        "These are measured local runs, not the paper's published scores.",
        "A missing correlation is undefined or unavailable; it is never replaced with zero.",
        "",
        "| Run | Rows / source | Valid | Pearson vs GPT-4 | Spearman | Kendall τ-b | Ranking accuracy (all) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "—" if x is None else f"{x:.4f}"

    for name, result in summaries.items():
        c = result.get("gpt4_score", {})
        lines.append(
            f"| {name} | {result['rows']} / {result['source_rows']} | {result['complete']} | "
            f"{fmt(c.get('pearson'))} | {fmt(c.get('spearman'))} | {fmt(c.get('kendall_tau_b'))} | "
            f"{fmt(result.get('accuracy_all'))} |"
        )
    lines += [
        "",
        "Correlations use the mean of all requested successful draws per row. Incomplete rows are excluded",
        "from correlation, but are retained in coverage and the ranking all-row denominator. Confidence intervals",
        "resample original instruction groups, keeping related responses together. Small samples are exploratory.",
        "",
        "`summary.json`, each run's `metrics.json`, `scores.csv`, `manifest.json`, and `predictions.jsonl`",
        "contain the complete machine-readable evidence, including generation failures and retry costs.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    absolute = [(name, r) for name, r in summaries.items() if "gpt4_score" in r]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if absolute:
        names = [x[0] for x in absolute]
        values = [x[1]["gpt4_score"]["pearson"] for x in absolute]
        axes[0].barh(names, [v if v is not None else 0 for v in values], color="#d1603d")
        for i, value in enumerate(values):
            if value is None:
                axes[0].text(0.02, i, "unavailable", va="center")
        axes[0].set(xlim=(-1, 1), title="Local Pearson correlation with GPT-4 labels")
        axes[0].axvline(0, color="#777", linewidth=0.5)
    axes[1].barh(list(summaries), [v["coverage"] for v in summaries.values()], color="#266b72")
    axes[1].set(
        xlim=(0, 1.05), title="Complete output coverage", xlabel="Fraction of selected rows"
    )
    fig.savefig(root / "overview.png", dpi=160)
    plt.close(fig)
    return summaries
