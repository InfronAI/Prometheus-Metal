#!/usr/bin/env python3
"""Dataset length/score/rubric diagnostics for paper Appendices A, B, and F."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prometheus_eval.io import file_hash, read_records, write_json
from prometheus_eval.prompts import section


def words(text):
    return re.findall(r"\w+", text.lower())


def rouge_l(first, second):
    """Word-level ROUGE-L F1, no stemming; exact LCS dynamic program."""
    a, b = words(first), words(second)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="experiments/dataset-analysis")
    args = p.parse_args()
    rows = read_records(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    score_counts = Counter(int(r["orig_score"]) for r in rows)
    lengths = {
        score: [len(words(r["orig_response"])) for r in rows if int(r["orig_score"]) == score]
        for score in range(1, 6)
    }
    rubrics = sorted({r["orig_criteria"] for r in rows})
    unseen = read_records("evaluation/benchmark/data/feedback_collection_ood_test.json")
    unseen_rubrics = sorted(
        {section(r["instruction"], "Score Rubrics:").split("\n")[0].strip("[]") for r in unseen}
    )
    rng = np.random.default_rng(42)
    random_pairs = [
        {"train": rubrics[int(rng.integers(len(rubrics)))], "test": r} for r in unseen_rubrics
    ]
    random_sim = [rouge_l(pair["train"], pair["test"]) for pair in random_pairs]
    # Max overlap is stricter than the paper's random-pair overlap diagnostic.
    nearest = []
    for rubric in unseen_rubrics:
        scores = [rouge_l(rubric, train) for train in rubrics]
        idx = int(np.argmax(scores))
        nearest.append(
            {"test_rubric": rubric, "closest_train_rubric": rubrics[idx], "rouge_l_f1": scores[idx]}
        )
    exact_criteria = sorted(set(rubrics) & set(unseen_rubrics))
    exact_rows = sum(
        section(r["instruction"], "Score Rubrics:").split("\n")[0].strip("[]") in exact_criteria
        for r in unseen
    )
    report = {
        "source_sha256": file_hash(args.input),
        "rows": len(rows),
        "unique_criteria": len(rubrics),
        "unique_instructions": len({r["orig_instruction"] for r in rows}),
        "score_counts": dict(score_counts),
        "response_word_lengths": {
            str(k): {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p10": float(np.quantile(v, 0.1)),
                "p90": float(np.quantile(v, 0.9)),
            }
            for k, v in lengths.items()
        },
        "unseen_criteria": len(unseen_rubrics),
        "random_pair_rouge_l": random_sim,
        "random_pair_seed": 42,
        "nearest_train_overlap": nearest,
        "exact_criterion_overlap": exact_criteria,
        "unseen_rows_with_exact_criterion_overlap": exact_rows,
        "tokenization": "Unicode word regex, lowercase, no stemming",
        "limits": "Lexical overlap and observed length distributions do not establish semantic independence or absence of adversarial bias.",
    }
    write_json(out / "statistics.json", report)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(list(score_counts), list(score_counts.values()), color="#266b72")
    axes[0].set(title="Released corpus score distribution", xlabel="Target score", ylabel="Rows")
    axes[1].boxplot(list(lengths.values()), tick_labels=list(lengths), showfliers=False)
    axes[1].set(title="Response lengths by target score", xlabel="Target score", ylabel="Words")
    axes[2].hist(random_sim, bins=12, color="#d1603d", alpha=0.8, label="Random pair")
    axes[2].hist(
        [r["rouge_l_f1"] for r in nearest],
        bins=12,
        color="#266b72",
        alpha=0.6,
        label="Nearest train criterion",
    )
    axes[2].set(title="Unseen versus training criteria", xlabel="ROUGE-L F1")
    axes[2].legend(fontsize=8)
    fig.savefig(out / "dataset-diagnostics.png", dpi=160)
    plt.close(fig)
    text = [
        "# Dataset diagnostics",
        "",
        f"Audited {len(rows):,} released rows, {len(rubrics):,} unique criterion texts, and {report['unique_instructions']:,} instruction texts.",
        "",
        "The release is smaller than the paper's nominal 100,000 examples. Exact counts and source hash are in `statistics.json`.",
        "",
        "![Dataset diagnostics](dataset-diagnostics.png)",
        "",
        "Score balance and response lengths are measured over the full release. Random-pair ROUGE-L follows the lexical overlap idea in Appendix B; nearest-training overlap is an additional stricter check. Neither demonstrates semantic non-overlap.",
        "",
        f"**Observed overlap:** {len(exact_criteria)} of {len(unseen_rubrics)} unseen criterion texts exactly match a training criterion, affecting {exact_rows} of {len(unseen)} unseen benchmark rows. This is criterion-text overlap, not proof that the whole five-level rubric or a response was duplicated. Preserve the original split, but interpret unseen-rubric generalization with this caveat.",
        *[f"- {criterion}" for criterion in exact_criteria],
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(text))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("nearest_train_overlap", "random_pair_rouge_l")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
