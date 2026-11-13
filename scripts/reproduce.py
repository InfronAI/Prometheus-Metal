#!/usr/bin/env python3
"""Run the local experiment workflow in separate processes to release GPU memory.

All steps have durable logs. No paid API or human label is synthesized by this runner.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from prometheus_eval.io import file_hash, write_json
from prometheus_eval.backends import model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "validation"], default="validation")
    parser.add_argument("--model", default="artifacts/models/prometheus-7b-v1-4bit")
    parser.add_argument("--training-data", default="experiments/training-data")
    parser.add_argument("--output", default="experiments/reproduction")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    statuses = []

    def run(name, command):
        print(f"[{name}] {' '.join(command)}", flush=True)
        start = time.perf_counter()
        with (root / f"{name}.log").open("a") as f:
            result = subprocess.run(
                [sys.executable, "-m", "prometheus_eval", *command],
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        statuses.append(
            {
                "stage": name,
                "command": command,
                "exit_code": result.returncode,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        write_json(root / "workflow.json", statuses)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")

    run("doctor", ["doctor", "--output", str(root / "environment.json")])
    run("audit", ["audit-data", "--output", str(root / "data_audit.json")])
    if not Path(args.model, "conversion.json").exists():
        run("model", ["prepare-model", "--output", args.model])
    if not Path(args.training_data, "manifest.json").exists():
        run("fetch-data", ["fetch-data"])
        run(
            "prepare-data",
            [
                "prepare-data",
                "--input",
                "artifacts/data/feedback_collection.json",
                "--output",
                args.training_data,
                "--limit",
                "200",
            ],
        )
    if not args.skip_training:
        for variant in ("full", "no_reference", "no_rubric", "score_only"):
            adapter = root / "adapters" / variant
            if (adapter / "summary.json").exists():
                saved = json.loads((adapter / "manifest.json").read_text())
                if saved["base_model"] != model_identity(args.model):
                    raise ValueError("Existing adapter belongs to a different base checkpoint")
                expected_steps = 2 if args.profile == "smoke" else 8
                if (
                    saved["config"]["steps"] != expected_steps
                    or saved["config"]["variant"] != variant
                    or saved["config"]["model"] != args.model
                ):
                    raise ValueError("Existing adapter configuration differs; use another output")
                for split in ("train", "valid", "test"):
                    if saved["selection"][split]["source_sha256"] != file_hash(
                        Path(args.training_data) / f"{split}.jsonl"
                    ):
                        raise ValueError("Training data changed; use another output")
                summary = json.loads((adapter / "summary.json").read_text())
                if summary["adapter_sha256"] != file_hash(adapter / "adapters.safetensors"):
                    raise ValueError("Adapter integrity check failed")
                print(f"Existing training artifact: {adapter}; choose a new --output to retrain")
            else:
                run(
                    f"train-{variant}",
                    [
                        "train",
                        "--model",
                        args.model,
                        "--data",
                        args.training_data,
                        "--output",
                        str(adapter),
                        "--variant",
                        variant,
                        "--steps",
                        "2" if args.profile == "smoke" else "8",
                    ],
                )
            run(
                f"eval-trained-{variant}",
                [
                    "benchmark",
                    "--model",
                    args.model,
                    "--adapter",
                    str(adapter),
                    "--input",
                    "evaluation/benchmark/data/feedback_collection_test.json",
                    "--output",
                    str(root / "training-ablations" / variant),
                    "--variant",
                    variant,
                    "--limit",
                    "2" if args.profile == "smoke" else "10",
                    "--repeats",
                    "1",
                ],
            )
        run("training-report", ["analyze", "--suite", str(root / "training-ablations")])
    config = json.loads(Path(f"configs/mac-{args.profile}.json").read_text())
    config["backend"]["model"] = args.model
    write_json(root / "evaluation-config.json", config)
    run(
        "benchmarks",
        [
            "suite",
            "--config",
            str(root / "evaluation-config.json"),
            "--output",
            str(root / f"mac-{args.profile}"),
        ],
    )
    prompt_config = json.loads(Path("configs/prompt-ablations.json").read_text())
    prompt_config["backend"]["model"] = args.model
    write_json(root / "prompt-config.json", prompt_config)
    run(
        "prompt-ablations",
        [
            "suite",
            "--config",
            str(root / "prompt-config.json"),
            "--output",
            str(root / "prompt-ablations"),
        ],
    )
    run(
        "demo",
        [
            "evaluate",
            "--model",
            args.model,
            "--max-tokens",
            "384",
            "--output",
            str(root / "demo.json"),
        ],
    )
    print(
        f"Local workflow finished. Reports: {root}. See docs/REPRODUCTION.md for unexecuted paper experiments."
    )


if __name__ == "__main__":
    main()
