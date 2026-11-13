"""Compatibility bridge for the original benchmark command-line arguments."""

import argparse
from pathlib import Path

from .analysis import analyze_run
from .backends import GenerationConfig, MLXBackend, TGIBackend
from .evaluation import benchmark
from .io import read_records, write_jsonl


def legacy_benchmark(mode):
    parser = argparse.ArgumentParser(
        description="Original flags, maintained local evaluation engine"
    )
    parser.add_argument("--input_file_name", required=True)
    parser.add_argument("--output_file_name", required=True)
    parser.add_argument("--server")
    parser.add_argument("--model", default="artifacts/models/prometheus-7b-v1-4bit")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    backend = TGIBackend(args.server, args.model) if args.server else MLXBackend(args.model)
    output = Path(args.output_file_name)
    run = output.with_suffix(output.suffix + ".run")
    benchmark(
        backend,
        args.input_file_name,
        run,
        mode=mode,
        config=GenerationConfig(),
        limit=args.limit,
        repeats=args.repeats,
    )
    original = read_records(args.input_file_name)
    rows = []
    for result in read_records(run / "predictions.jsonl"):
        row = {**original[result["source_index"]], "evaluation_status": result["status"]}
        if mode == "absolute":
            row["prometheus_output"] = [g["feedback"] for g in result["grades"]]
            row["prometheus_score"] = [g["score"] for g in result["grades"]]
        else:
            for side in ("chosen", "rejected"):
                row[f"{side}_output"] = result["rounds"][-1][side]["feedback"]
                row[f"{side}_score"] = result["rounds"][-1][side]["score"]
            row["accuracy"] = int(result["correct"]) if result["correct"] is not None else None
        rows.append(row)
    write_jsonl(output, rows)
    analyze_run(run)
