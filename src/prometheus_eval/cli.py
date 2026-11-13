"""One entry point for the local research workflow."""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

from .backends import GenerationConfig, backend_from_config
from .io import environment, write_json


def add_backend(parser):
    parser.add_argument("--backend", choices=["mlx", "torch", "tgi"], default="mlx")
    parser.add_argument("--model", default="artifacts/models/prometheus-7b-v1-4bit")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chat-format", choices=["llama2", "tokenizer"], default="llama2")
    parser.add_argument("--server", help="TGI /generate endpoint, only for --backend tgi")


def backend_options(args):
    if args.backend == "torch" and args.adapter:
        raise ValueError(
            "Adapter loading is supported by the MLX workflow; use a merged checkpoint with --backend torch"
        )
    if args.backend == "tgi":
        if not args.server:
            raise ValueError("--server is required with --backend tgi")
        return {"backend": "tgi", "server": args.server, "tokenizer_model": args.model}
    options = {
        "backend": args.backend,
        "model": args.model,
        "adapter": args.adapter,
        "chat_format": args.chat_format,
    }
    if args.backend == "torch":
        options["device"] = args.device
    return options


def add_generation(parser):
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--seed", type=int, default=42)


def generation_options(args):
    return GenerationConfig(**{k: getattr(args, k) for k in GenerationConfig.__dataclass_fields__})


def parser():
    root = argparse.ArgumentParser(description="Prometheus v1 on Apple Silicon")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Inspect environment without downloading models")
    p.add_argument("--output")
    p = commands.add_parser("fetch-data", help="Download the pinned released Feedback Collection")
    p.add_argument("--output", default="artifacts/data/feedback_collection.json")
    p = commands.add_parser("prepare-model", help="Download and stream-quantize original 7B v1")
    p.add_argument("--source", help="Optional local original HF .bin checkpoint directory")
    p.add_argument("--output", default="artifacts/models/prometheus-7b-v1-4bit")
    p.add_argument("--bits", type=int, choices=[4, 8], default=4)
    p = commands.add_parser("audit-data")
    p.add_argument("--data-dir", default="evaluation/benchmark/data")
    p.add_argument("--output", default="experiments/data_audit.json")
    p = commands.add_parser("prepare-data")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="artifacts/data/prepared")
    p.add_argument(
        "--limit", type=int, default=100, help="Multiple of 5; --all selects all complete groups"
    )
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark-dir", default="evaluation/benchmark/data")
    p = commands.add_parser("evaluate", help="Evaluate one structured input JSON")
    add_backend(p)
    add_generation(p)
    p.add_argument("--input", default="examples/evaluation.json")
    p.add_argument("--output", default="artifacts/demo/result.json")
    p = commands.add_parser("benchmark")
    add_backend(p)
    add_generation(p)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--limit", type=int)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument(
        "--variant", choices=["full", "no_reference", "no_rubric", "score_only"], default="full"
    )
    p.add_argument("--tie-policy", choices=["keep", "resample"], default="keep")
    p.add_argument("--tie-rounds", type=int, default=3)
    p = commands.add_parser("suite", help="Execute a versioned evaluation matrix, resumably")
    p.add_argument("--config", default="configs/mac-validation.json")
    p.add_argument("--output", default="experiments/mac-validation")
    p = commands.add_parser("train", help="Completion-only MLX QLoRA continuation")
    p.add_argument("--model", default="artifacts/models/prometheus-7b-v1-4bit")
    p.add_argument("--data", default="artifacts/data/prepared")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--variant", choices=["full", "no_reference", "no_rubric", "score_only"], default="full"
    )
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-examples", type=int, default=2)
    p.add_argument("--memory-gb", type=float, default=10)
    p.add_argument("--chat-format", choices=["llama2", "tokenizer"], default="llama2")
    p = commands.add_parser("analyze")
    p.add_argument("--run")
    p.add_argument("--suite")
    p = commands.add_parser("serve", help="Start a loopback-only live evaluation demo")
    add_backend(p)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--results", default="experiments/mac-validation")
    p = commands.add_parser("synthesize", help="Appendix G staged local data construction")
    add_backend(p)
    p.add_argument("--seeds", default="examples/seed_rubrics.json")
    p.add_argument("--output", required=True)
    p.add_argument("--rubrics", type=int, default=1)
    p.add_argument("--instructions-per-rubric", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser(
        "human-study", help="Prepare blinded annotation files or aggregate real labels"
    )
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--annotations")
    p.add_argument("--seed", type=int, default=42)
    return root


def run_suite(config_path, output):
    from .analysis import report_suite
    from .evaluation import benchmark

    config = json.loads(Path(config_path).read_text())
    if config["backend"].get("backend") == "torch" and config["backend"].get("adapter"):
        raise ValueError(
            "The PyTorch workflow requires a merged checkpoint; adapter loading is MLX-only"
        )
    output = Path(output)
    existing = output / "suite_config.json"
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError("Suite config differs from saved run; choose a new output directory")
    write_json(existing, config)
    backend = backend_from_config(config["backend"])
    for task in config["tasks"]:
        name = task["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Task names must be simple directory names")
        options = {k: v for k, v in task.items() if k not in ("name", "input")}
        benchmark(
            backend,
            task["input"],
            output / name,
            config=GenerationConfig(**config.get("generation", {})),
            **options,
        )
        report_suite(output)
    return str(output / "REPORT.md")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = environment()
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                try:
                    import mlx.core as mx

                    result["metal"] = mx.device_info()
                except ImportError:
                    result["metal"] = "Install with pip install -e '.[mac]'"
            if args.output:
                write_json(args.output, result)
        elif args.command == "fetch-data":
            from .models import fetch_data

            result = fetch_data(args.output)
        elif args.command == "prepare-model":
            from .models import convert_model

            result = convert_model(args.output, args.source, args.bits)
        elif args.command == "audit-data":
            from .data import audit_benchmarks

            result = audit_benchmarks(args.data_dir, args.output)
        elif args.command == "prepare-data":
            from .data import prepare_data

            result = prepare_data(
                args.input,
                args.output,
                None if args.all else args.limit,
                args.seed,
                args.benchmark_dir,
            )
        elif args.command == "evaluate":
            from .evaluation import grade_once
            from .prompts import make_prompt

            data = json.loads(Path(args.input).read_text())
            prompt = data.get("prompt") or make_prompt(**data)
            backend = backend_from_config(backend_options(args))
            result = grade_once(backend, prompt, generation_options(args))
            write_json(
                args.output,
                {
                    "input": data,
                    "backend": backend.identity,
                    "generation": asdict(generation_options(args)),
                    **result,
                },
            )
            if result["status"] != "ok":
                print(json.dumps(result, indent=2))
                return 2
        elif args.command == "benchmark":
            from .analysis import analyze_run
            from .evaluation import benchmark

            benchmark(
                backend_from_config(backend_options(args)),
                args.input,
                args.output,
                mode=args.mode,
                config=generation_options(args),
                limit=args.limit,
                repeats=args.repeats,
                attempts=args.attempts,
                variant=args.variant,
                tie_policy=args.tie_policy,
                tie_rounds=args.tie_rounds,
            )
            result = analyze_run(args.output)
        elif args.command == "suite":
            result = run_suite(args.config, args.output)
        elif args.command == "train":
            from .training import TrainingConfig, train_adapter

            result = train_adapter(
                TrainingConfig(**{k: getattr(args, k) for k in TrainingConfig.__dataclass_fields__})
            )
        elif args.command == "analyze":
            from .analysis import analyze_run, report_suite

            if bool(args.run) == bool(args.suite):
                raise ValueError("Specify exactly one of --run and --suite")
            result = analyze_run(args.run) if args.run else report_suite(args.suite)
        elif args.command == "serve":
            from .server import serve

            serve(backend_options(args), args.port, args.results)
            return 0
        elif args.command == "synthesize":
            from .synthesis import synthesize

            result = synthesize(
                backend_from_config(backend_options(args)),
                args.seeds,
                args.output,
                args.rubrics,
                args.instructions_per_rubric,
                args.seed,
            )
        elif args.command == "human-study":
            from .human import human_study

            result = human_study(args.left, args.right, args.output, args.annotations, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
