"""Audit original files and prepare instruction-grouped, balanced training splits."""

import random
from collections import Counter, defaultdict
from pathlib import Path

from .io import digest, file_hash, read_records, write_json, write_jsonl
from .prompts import parse_grade, section

BENCHMARKS = {
    "feedback_seen": ("feedback_collection_test.json", "absolute"),
    "feedback_unseen": ("feedback_collection_ood_test.json", "absolute"),
    "vicuna": ("vicuna_eval.json", "absolute"),
    "mt_bench": ("mt_bench_eval.json", "absolute"),
    "flask": ("flask_eval.json", "absolute"),
    "hhh": ("hhh_alignment_eval.json", "relative"),
    "mt_human": ("mt_bench_human_judgement_eval.json", "relative"),
}


def group_id(row):
    prompt = row["instruction"]
    instruction = section(prompt, "The instruction to evaluate:")
    if not instruction:
        raise ValueError("Cannot split unstructured training prompt without an instruction group")
    # All five responses to the same instruction must stay together, even if rubrics differ.
    return digest(instruction)


def training_score(row):
    return parse_grade(row["output"]).score


def select_indices(rows, limit, seed, stratify=True):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive; omit it for all rows")
    if limit is None or limit >= len(rows):
        return list(range(len(rows)))
    rng = random.Random(seed)
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        label = row.get("gpt4_score", row.get("tie", row.get("source", "all")))
        if isinstance(label, list):
            label = sum(label) / len(label) if label else "missing"
        strata[str(label) if stratify else "all"].append(i)
    for indices in strata.values():
        rng.shuffle(indices)
    keys = sorted(strata)
    rng.shuffle(keys)
    selected = []
    while len(selected) < limit:
        for key in keys:
            if strata[key] and len(selected) < limit:
                selected.append(strata[key].pop())
    return sorted(selected)


def audit_benchmarks(data_dir, output):
    report = {}
    for name, (filename, mode) in BENCHMARKS.items():
        path = Path(data_dir) / filename
        rows = read_records(path)
        fields = Counter(k for row in rows for k in row)
        item = {
            "path": str(path),
            "sha256": file_hash(path),
            "rows": len(rows),
            "mode": mode,
            "field_coverage": dict(fields),
            "exact_duplicate_rows": len(rows) - len({digest(x) for x in rows}),
        }
        if mode == "absolute":
            item["instruction_groups"] = len({group_id(x) for x in rows})
            item["gpt4_labels_missing"] = sum("gpt4_score" not in x for x in rows)
            item["human_labels_missing"] = sum("human_score" not in x for x in rows)
        else:
            item["ties"] = sum(x.get("tie", 0) == 1 for x in rows)
            item["missing_tie_defaults_to_non_tie"] = sum("tie" not in x for x in rows)
        report[name] = item
    write_json(output, report)
    return report


def prepare_data(source, output, limit=100, seed=42, benchmark_dir=None):
    """Select complete instruction groups, split by group, never use benchmark labels to train."""
    rows = read_records(source)
    groups, rejected = defaultdict(list), []
    for i, row in enumerate(rows):
        try:
            score, group = training_score(row), group_id(row)
            groups[group].append(
                {
                    "instruction": row["instruction"],
                    "output": row["output"],
                    "score": score,
                    "source_index": i,
                    "group_id": group,
                }
            )
        except (ValueError, KeyError) as e:
            rejected.append({"source_index": i, "reason": str(e)})
    blocked = set()
    if benchmark_dir:
        for filename, mode in BENCHMARKS.values():
            if mode == "absolute":
                blocked.update(group_id(r) for r in read_records(Path(benchmark_dir) / filename))
    excluded_overlap = sum(len(v) for k, v in groups.items() if k in blocked)
    eligible = [
        k
        for k, v in groups.items()
        if k not in blocked
        and Counter(r["score"] for r in v) == Counter({i: 1 for i in range(1, 6)})
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if limit is not None:
        if limit < 15 or limit % 5:
            raise ValueError("Training limit must be a multiple of 5 and at least 15")
        eligible = eligible[: limit // 5]
    if len(eligible) < 3:
        raise ValueError("Need at least three complete, non-overlapping instruction groups")
    n_valid = max(1, len(eligible) // 10)
    n_test = max(1, len(eligible) // 10)
    split_groups = {
        "valid": eligible[:n_valid],
        "test": eligible[n_valid : n_valid + n_test],
        "train": eligible[n_valid + n_test :],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError(f"Prepared data exists at {output}; choose a new directory")
    split_stats = {}
    for split, keys in split_groups.items():
        selected = [r for k in keys for r in groups[k]]
        write_jsonl(output / f"{split}.jsonl", selected)
        split_stats[split] = {
            "rows": len(selected),
            "groups": keys,
            "scores": dict(Counter(r["score"] for r in selected)),
            "sha256": file_hash(output / f"{split}.jsonl"),
        }
    manifest = {
        "source_sha256": file_hash(source),
        "source_rows": len(rows),
        "seed": seed,
        "selected_groups": len(eligible),
        "split_unit": "original instruction text",
        "excluded_benchmark_overlap_rows": excluded_overlap,
        "invalid_rows": rejected,
        "splits": split_stats,
        "incomplete_or_unbalanced_groups": sum(
            Counter(r["score"] for r in v) != Counter({i: 1 for i in range(1, 6)})
            for v in groups.values()
        ),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
