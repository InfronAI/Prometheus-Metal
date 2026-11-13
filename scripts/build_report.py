#!/usr/bin/env python3
"""Build the final research report exclusively from persisted measured artifacts."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prometheus_eval.io import read_records, write_json
from prometheus_eval.analysis import all_attempts, correlations, numeric_label
from prometheus_eval.prompts import section


def main():
    root = Path("experiments")
    metrics = json.loads((root / "mac-validation/summary.json").read_text())
    training_eval = json.loads((root / "training-ablations/summary.json").read_text())
    prompt_eval = json.loads((root / "prompt-ablations/summary.json").read_text())
    dataset = json.loads((root / "dataset-analysis/statistics.json").read_text())
    synthesis = json.loads((root / "synthesis/summary.json").read_text())
    hardware = json.loads((root / "environment.json").read_text())
    variants = ["full", "no_reference", "no_rubric", "score_only"]
    unseen_spec = metrics["feedback_unseen"]["protocol"]
    unseen_rows = read_records(unseen_spec["input_path"])
    selected_overlap = [
        index
        for index in unseen_spec["indices"]
        if section(unseen_rows[index]["instruction"], "Score Rubrics:").split("\n")[0].strip("[]")
        in dataset["exact_criterion_overlap"]
    ]
    paired = {}
    for suite in ("training-ablations", "prompt-ablations"):
        valid_rows = {
            variant: {
                r["row_id"]: r
                for r in read_records(root / suite / variant / "predictions.jsonl")
                if r["status"] == "ok" and numeric_label(r["labels"].get("gpt4_score")) is not None
            }
            for variant in variants
        }
        common = sorted(set.intersection(*(set(rows) for rows in valid_rows.values())))
        paired[suite] = {"common_complete_rows": len(common), "row_ids": common, "variants": {}}
        for variant in variants:
            paired[suite]["variants"][variant] = correlations(
                [valid_rows[variant][key]["mean_score"] for key in common],
                [numeric_label(valid_rows[variant][key]["labels"]["gpt4_score"]) for key in common],
            )
    write_json(root / "paired-ablation-analysis.json", paired)
    training = {
        v: json.loads((root / "adapters" / v / "summary.json").read_text()) for v in variants
    }
    manifests = {
        v: json.loads((root / "adapters" / v / "manifest.json").read_text()) for v in variants
    }
    assert len({tuple(m["training_order_source_indices"]) for m in manifests.values()}) == 1
    assert len(metrics) == 7 and len(training_eval) == len(prompt_eval) == 4
    all_metrics = [*metrics.values(), *training_eval.values(), *prompt_eval.values()]
    rows = sum(m["rows"] for m in all_metrics)
    attempts = sum(m["attempts"] for m in all_metrics)
    complete = sum(m["complete"] for m in all_metrics)
    wall = sum(m["total_generation_seconds"] for m in all_metrics)
    peak = max(m["peak_memory_gb"] for m in all_metrics)
    overview = {
        "evaluated_rows_across_protocols": rows,
        "complete_rows": complete,
        "generation_attempts": attempts,
        "generation_wall_seconds": wall,
        "max_inference_mlx_memory_gb": peak,
        "max_training_mlx_memory_gb": max(v["peak_memory_gb"] for v in training.values()),
        "training_steps_per_variant": {k: v["steps"] for k, v in training.items()},
        "full_paper_reproduction": False,
        "hardware": hardware["metal"]["device_name"],
        "selected_unseen_rows_with_exact_criterion_overlap": selected_overlap,
    }
    write_json(root / "result-overview.json", overview)
    failures = {}
    for suite in ("mac-validation", "training-ablations", "prompt-ablations"):
        for path in sorted((root / suite).glob("*/predictions.jsonl")):
            saved_rows = read_records(path)
            saved_attempts = [a for row in saved_rows for a in all_attempts(row)]
            invalid = [a for a in saved_attempts if a["status"] != "ok"]
            score_only_grades = [
                grade
                for row in saved_rows
                if row["variant"] == "score_only" and row["mode"] == "absolute"
                for grade in row["grades"]
                if grade["status"] == "ok"
            ]
            failures[f"{suite}/{path.parent.name}"] = {
                "attempts": len(saved_attempts),
                "invalid_attempts": len(invalid),
                "invalid_at_output_limit": sum(a.get("finish_reason") == "length" for a in invalid),
                "failure_reasons": dict(Counter(a.get("error", a["status"]) for a in invalid)),
                "incomplete_source_indices": [
                    r["source_index"] for r in saved_rows if r["status"] != "ok"
                ],
                "score_only_valid_grades": len(score_only_grades),
                "score_only_valid_grades_with_feedback": sum(
                    bool(g["feedback"]) for g in score_only_grades
                ),
            }
    write_json(root / "failure-analysis.json", failures)
    invalid_total = sum(r["invalid_attempts"] for r in failures.values())
    truncated_total = sum(r["invalid_at_output_limit"] for r in failures.values())
    score_only_valid = sum(r["score_only_valid_grades"] for r in failures.values())
    score_only_prose = sum(r["score_only_valid_grades_with_feedback"] for r in failures.values())

    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [
        "# M1 experiment report",
        "",
        "## What was actually run",
        "",
        f"The measured host is **{hardware['metal']['device_name']}**, 16 GB unified memory, {hardware['platform']}, Python {hardware['python']}. The evaluator is Prometheus v1 7B, using a pinned community FP16 export converted to MLX 4-bit affine weights. See [model provenance](model/conversion.json).",
        "",
        f"Across the seven benchmark samples, four trained-adapter evaluations and four prompt ablations, **{rows} rows** and **{attempts} generation attempts** were processed; **{complete} rows** produced every required valid grade. Total recorded inference time was **{wall / 60:.2f} minutes**, excluding model loading and conversion. Peak MLX inference memory was **{peak:.3f} GB**; peak training memory was **{overview['max_training_mlx_memory_gb']:.3f} GB**. These are process-lifetime allocator high-water marks, not total system RAM. Tasks within one suite process can carry forward an earlier task's memory peak.",
        "",
        "**This is a bounded engineering reproduction, not full reproduction of the paper's tables.** Balanced label-stratified sampling improves test coverage but is not an unbiased estimate of the full benchmark distribution. Small-sample correlations and confidence intervals must not be used to claim parity with the paper, humans, or GPT-4.",
        "",
        "## 1. Data construction and audit",
        "",
        f"The full pinned release contains **{dataset['rows']:,} rows**, **{dataset['unique_instructions']:,} unique instruction texts** and **{dataset['unique_criteria']} unique criterion texts**. Criterion-text counts are not necessarily counts of distinct five-level rubrics. Scores 1–5 occur {', '.join(str(dataset['score_counts'][str(i)]) for i in range(1, 6))} times respectively.",
        "",
        "The 200-row prepared subset contains 160 training, 20 validation and 20 test examples, split by original instruction group. Each split has equal target-score counts. Exact benchmark-instruction overlap was excluded. The full 200 selected examples fit the 2048-token training limit on the converted evaluator tokenizer.",
        "",
        f"The local data-construction experiment completed **{synthesis['calls']} real teacher calls**, accepting **{synthesis['accepted_rows']} examples** after format, target-score and sentence-count checks. This verifies the staged construction pipeline using Qwen3-4B and new seeds. It does not regenerate the GPT-4 training corpus or guarantee factual correctness. Earlier format failures are preserved in the `local-synthesis*` directories. [Teacher snapshot hashes](synthesis/teacher-source.json) and [full corpus source](training-data/source.json) are saved separately.",
        "",
        f"**Unseen-split caveat:** {len(dataset['exact_criterion_overlap'])} of 50 unseen criterion texts exactly overlaps the released training corpus, covering {dataset['unseen_rows_with_exact_criterion_overlap']} test rows. This finding concerns criterion text, not proof of full-rubric or answer duplication. Random-pair lexical overlap alone would miss this nearest-neighbor match.",
        f"The fixed unseen sample used here contains **{len(selected_overlap)}** rows with that exact criterion-text overlap. This check does not rule out paraphrases or semantic overlap.",
        "",
        "![Full-corpus diagnostics](dataset-analysis/dataset-diagnostics.png)",
        "",
        "The released MT-Bench file lacks GPT-4 and human scores on all 320 rows. MT human judgments includes 454 exact duplicate rows, retained as released. HHH has no tie field and is interpreted as a binary preference dataset. See [audit](data_audit.json) and [dataset statistics](dataset-analysis/statistics.json).",
        "",
        "## 2. Completion-only training and training ablations",
        "",
        "Each variant continues the same 7B checkpoint for eight rank-8 QLoRA updates on the last four query/value layers. All use the same eight source examples in the same order; the manifests record that order. Prompt and padding targets are masked, including in the gradient-level test. The adapters are saved and reloaded for the evaluations below.",
        "",
        "| Variant | Initial validation loss | Final validation loss | Test loss | Train time (s) | Peak MLX GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, variant in zip(axes.flat, variants):
        losses = read_records(root / "adapters" / variant / "losses.jsonl")
        initial = next(r["val_loss"] for r in losses if r["kind"] == "validation")
        s = training[variant]
        lines.append(
            f"| {variant} | {initial:.4f} | {s['final_validation_loss']:.4f} | {s['test_loss']:.4f} | {s['elapsed_seconds']:.1f} | {s['peak_memory_gb']:.3f} |"
        )
        values = [r for r in losses if r["kind"] == "train"]
        ax.plot(
            [r["iteration"] for r in values],
            [r["train_loss"] for r in values],
            "o-",
            color="#266b72",
        )
        ax.set(title=variant, xlabel="Optimizer update", ylabel="Completion cross entropy")
    fig.savefig(root / "training-losses.png", dpi=160)
    plt.close(fig)
    lines += [
        "",
        "![Training loss traces](training-losses.png)",
        "",
        "Validation and test loss each use two fixed held-out examples, as recorded in the configuration. Eight updates and tiny holdouts establish that gradients, masking, saving and reloading work; they do not establish convergence or generalization. Score-only loss concerns a much shorter target and is not directly comparable to feedback-plus-score loss.",
        "",
        "## 3. Absolute grading and preference ranking",
        "",
        "| Dataset | Rows / source | Complete | Pearson vs GPT-4 | Spearman | Kendall τ-b | Ranking accuracy (all rows) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in metrics.items():
        c = r.get("gpt4_score", {})
        lines.append(
            f"| {name} | {r['rows']} / {r['source_rows']} | {r['complete']} | {fmt(c.get('pearson'))} | {fmt(c.get('spearman'))} | {fmt(c.get('kendall_tau_b'))} | {fmt(r.get('accuracy_all'))} |"
        )
    lines += [
        "",
        "![Evaluation overview](mac-validation/overview.png)",
        "",
        "Absolute evaluation averages three successfully parsed draws per row. Incomplete rows are excluded from correlations but remain in coverage. Ranking accuracy includes failed rows in its denominator. HHH uses at most three tie-breaking rounds; unresolved ties are errors. MT human ties are evaluated directly. This finite HHH procedure differs from the original unbounded resampling and must be disclosed in comparisons.",
        "",
        f"FLASK correlation with the existing human labels: Pearson **{fmt(metrics['flask']['human_score']['pearson'])}**, Spearman **{fmt(metrics['flask']['human_score']['spearman'])}**, Kendall τ-b **{fmt(metrics['flask']['human_score']['kendall_tau_b'])}**, on **{metrics['flask']['human_score']['n']}** complete labelled rows. This is not the original nine-annotator human study.",
        "",
        "## 4. Ablations: separate training from inference",
        "",
        "| Variant | Trained adapter: Pearson / complete | Fixed model, altered input: Pearson / complete |",
        "|---|---:|---:|",
    ]
    for variant in variants:
        a, b = training_eval[variant], prompt_eval[variant]
        lines.append(
            f"| {variant} | {fmt(a['gpt4_score']['pearson'])} / {a['complete']} | {fmt(b['gpt4_score']['pearson'])} / {b['complete']} |"
        )
    lines += [
        "",
        "Both columns use the same ten selected Feedback Seen rows. The left column updates a separate adapter for each objective; the right changes only inference inputs. These comparisons are paired engineering checks, not substitutes for Table 6's full from-base training. Invalid/constant outputs are represented by missing correlations rather than fabricated zeros.",
        "",
        f"**Score-only compliance:** {score_only_prose} of {score_only_valid} successfully parsed score-only generations still contained feedback prose. In this protocol, the parser permits optional prose when feedback is not required. The variant removes feedback from the training target and asks for a score-only response; eight updates or an instruction change do not guarantee that the inherited evaluator stops generating feedback. These results therefore do not establish the paper's no-feedback mechanism.",
        "",
        "Available-case correlations can be misleading when variants fail on different rows. In particular, a higher correlation on fewer surviving examples is not evidence that removing a component helped. The following comparison restricts each experiment family to rows valid for all four variants:",
        "",
        "| Variant | Training: common-row Pearson | Inference: common-row Pearson |",
        "|---|---:|---:|",
        *[
            f"| {variant} | {fmt(paired['training-ablations']['variants'][variant]['pearson'])} | {fmt(paired['prompt-ablations']['variants'][variant]['pearson'])} |"
            for variant in variants
        ],
        "",
        f"Common complete rows: **{paired['training-ablations']['common_complete_rows']}** for training ablations and **{paired['prompt-ablations']['common_complete_rows']}** for inference ablations. This controls the compared row set, but still conditions on successful formatting and does not remove the small-sample limitation. Exact row IDs are in [paired-ablation-analysis.json](paired-ablation-analysis.json).",
        "",
        "## 5. Length association and qualitative review",
        "",
        f"There were **{invalid_total} unsuccessful generation attempts** across the 15 evaluation runs; **{truncated_total}** ended at the 256-token output limit. These counts include attempts recovered by a subsequent retry, not only failed rows. The exact parse errors and affected source indices are saved in [failure-analysis.json](failure-analysis.json). The paper-sized output budget can cut off feedback before its final score. A larger output budget is available for practical use (the live demo uses 384), but changing it requires a separately labelled experiment.",
        "",
        "Each absolute run's `metrics.json` includes word-length/score Pearson, Spearman and Kendall association, plus instruction-cluster bootstrap confidence intervals for evaluator/label correlation. The small sampled evaluation is observational; it cannot establish the absence of verbosity bias. The full corpus plot also shows that score-5 answers tend to contain more words than score-2 answers, despite approximately balanced score frequencies.",
        "",
        "The live demo evaluates an explanation of green leaves against a child-readability rubric. Inspect both the prose and the numerical judgment in [the saved real example](first-prometheus-evaluation.json): a plausible critique can still merit disagreement about its exact score. Model outputs remain judgments, not ground truth.",
        "The [continuous browser video](../docs/media/prometheus-m1-demo.mp4) and [its exact live result](../docs/media/demo-result.json) document the interactive path.",
        "",
        "## 6. Reproducibility and remaining work",
        "",
        "- [Workflow commands and exit codes](workflow.json), [tested environment](environment.json), [test output](test-results.txt), [model conversion provenance](model/conversion.json).",
        "- [Real-model resume check](resume-check.json), [artifact verification](artifact-verification.json), and [final implementation hashes](implementation-manifest.json).",
        "- Each run retains its selection/configuration manifest, all raw generation attempts, completion hash, CSV and JSON metrics. Rerunning with an identical manifest resumes; changed settings require a separate directory.",
        "- The native 7B model, adapter training/reload, original data formats and numerical analysis were exercised locally. Optional CPU/MPS Transformers and remote TGI integrations are not claimed as measured 7B performance here.",
        "- Full 6,293-row evaluation, original 7B/13B full-parameter training, original base-model/coarse-data ablations, historical proprietary baselines and new human annotations remain unexecuted. The all-row runner and blinded annotation tools are supplied; missing source materials and human work are not simulated.",
        "",
        "See the complete [paper correspondence and limits](../docs/REPRODUCTION.md).",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
