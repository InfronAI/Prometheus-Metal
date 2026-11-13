#!/usr/bin/env python3
"""Verify measured artifacts, data integrity, documented links, and recording hashes."""

import json
import re
from pathlib import Path

from prometheus_eval.io import digest, file_hash, read_records, write_json
from prometheus_eval.backends import model_identity
from prometheus_eval.human import study_sources


def verify_run(path, model):
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["spec"]
    assert spec["backend"]["conversion_sha256"] == model["conversion_sha256"], path
    assert digest(spec) == manifest["spec_sha256"], path
    for name, expected in spec["implementation_sha256"].items():
        assert file_hash(Path("src/prometheus_eval", name)) == expected, (
            f"Evaluator source changed: {name}"
        )
    assert file_hash(spec["input_path"]) == spec["input_sha256"], path
    source = read_records(spec["input_path"])
    rows = read_records(path / "predictions.jsonl")
    assert [r["source_index"] for r in rows] == spec["indices"], path
    assert len({r["row_id"] for r in rows}) == len(rows), path
    for row in rows:
        assert digest(source[row["source_index"]]) == row["input_sha256"], path
        assert spec["backend"]["backend"] == "mlx", path
        if row["mode"] == "absolute":
            assert len(row["grades"]) == spec["repeats"]
            grades = row["grades"]
        else:
            assert len(row["rounds"]) <= spec["tie_rounds"]
            grades = [g for pair in row["rounds"] for g in pair.values()]
        for grade in grades:
            assert len(grade["attempts"]) <= spec["attempts"]
            if grade["status"] == "ok":
                assert 1 <= grade["score"] <= 5
                assert grade["attempts"][-1]["generation_tokens"] > 0
                assert grade["attempts"][-1]["elapsed_seconds"] > 0
    completion = json.loads((path / "completion.json").read_text())
    assert completion["rows"] == len(rows)
    assert completion["predictions_sha256"] == file_hash(path / "predictions.jsonl")
    return len(rows)


def main():
    model = model_identity("artifacts/models/prometheus-7b-v1-4bit")
    runs = []
    for name, expected in (
        ("mac-validation", 7),
        ("training-ablations", 4),
        ("prompt-ablations", 4),
    ):
        paths = sorted(Path("experiments", name).glob("*/predictions.jsonl"))
        assert len(paths) == expected, name
        for path in paths:
            runs.append({"run": str(path.parent), "verified_rows": verify_run(path.parent, model)})
    manifests = [
        json.loads(p.read_text()) for p in Path("experiments/adapters").glob("*/manifest.json")
    ]
    assert len(manifests) == 4
    assert len({tuple(m["training_order_source_indices"]) for m in manifests}) == 1
    for path in Path("experiments/adapters").glob("*/summary.json"):
        summary = json.loads(path.read_text())
        assert summary["steps"] == 8
        assert file_hash(path.parent / "adapters.safetensors") == summary["adapter_sha256"]
    data = json.loads(Path("experiments/training-data/manifest.json").read_text())
    groups = [set(s["groups"]) for s in data["splits"].values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    for name, split in data["splits"].items():
        assert file_hash(Path("experiments/training-data", name + ".jsonl")) == split["sha256"]
    documents = [
        Path("README.md"),
        Path("README.zh-CN.md"),
        Path("docs/REPRODUCTION.md"),
        Path("docs/ARCHITECTURE.md"),
        Path("experiments/REPORT.md"),
        Path("docs/media/README.md"),
    ]
    checked_links = 0
    generated_here = {
        Path("experiments/artifact-verification.json").resolve(),
        Path("experiments/implementation-manifest.json").resolve(),
    }
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if link.startswith(("http:", "https:", "#")):
                continue
            target = document.parent / link.split("#", 1)[0]
            # The report links to this verification receipt and source manifest.
            # They are written below only after the other checks have succeeded.
            assert target.exists() or target.resolve() in generated_here, (
                f"Broken link in {document}: {link}"
            )
            checked_links += 1
    recording = json.loads(Path("docs/media/recording.json").read_text())
    for name, artifact in recording["artifacts"].items():
        assert file_hash(Path("docs/media", name)) == artifact["sha256"]
    assert recording["browser_console_errors"] == []
    study = json.loads(Path("experiments/human-study/study_manifest.json").read_text())
    assert study["sources"] == study_sources(
        "experiments/training-ablations/full", "experiments/training-ablations/no_reference"
    )
    source_paths = [
        *Path("src").rglob("*.py"),
        *Path("src").rglob("*.html"),
        *Path("scripts").glob("*.py"),
        *Path("configs").glob("*.json"),
        Path("pyproject.toml"),
        Path("requirements-macos.lock"),
    ]
    write_json(
        "experiments/implementation-manifest.json",
        {
            "scope": "Final delivery working tree. Exact evaluator hashes at execution are retained in each run manifest.",
            "sha256": {str(p): file_hash(p) for p in sorted(source_paths)},
        },
    )
    result = {
        "status": "passed",
        "runs": runs,
        "verified_prediction_rows": sum(r["verified_rows"] for r in runs),
        "verified_training_adapters": 4,
        "checked_local_document_links": checked_links,
        "video_hashes_verified": True,
        "converted_model_weights_verified": True,
        "human_study_sources_verified": True,
        "training_groups_disjoint": True,
        "training_ablation_order_identical": True,
    }
    write_json("experiments/artifact-verification.json", result)
    assert all(path.exists() for path in generated_here)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
