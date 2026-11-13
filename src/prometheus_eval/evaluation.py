"""Bounded, resumable evaluations with immutable manifests and raw attempts."""

import fcntl
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from .backends import ContextOverflow, GenerationConfig
from .data import select_indices
from .io import append_jsonl, digest, environment, file_hash, read_records, utc_now, write_json
from .prompts import ablate, parse_grade, section


def grade_once(backend, prompt, config, attempts=2, require_feedback=True):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    history = []
    for attempt in range(attempts):
        try:
            result = backend.generate(prompt, replace(config, seed=config.seed + attempt))
            try:
                grade = parse_grade(result["text"], require_feedback=require_feedback)
                history.append({**result, "status": "ok"})
                return {
                    "status": "ok",
                    "score": grade.score,
                    "feedback": grade.feedback,
                    "attempts": history,
                }
            except ValueError as e:
                history.append({**result, "status": "invalid_output", "error": str(e)})
        except ContextOverflow as e:
            history.append({"status": "context_overflow", "error": str(e)})
            break
        except (OSError, RuntimeError, ValueError) as e:
            # No unbounded exception loop, and never silently substitute a guessed score.
            history.append({"status": "backend_error", "error": f"{type(e).__name__}: {e}"})
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 5))
    return {"status": history[-1]["status"], "score": None, "feedback": None, "attempts": history}


def evaluate_record(
    backend,
    row,
    index,
    mode,
    config,
    repeats=3,
    attempts=2,
    variant="full",
    tie_policy="keep",
    tie_rounds=3,
):
    row_id = digest({"index": index, "row": row})
    base_seed = (config.seed + int(row_id[:8], 16)) % (2**31 - 10000)
    result = {
        "row_id": row_id,
        "source_index": index,
        "input_sha256": digest(row),
        "mode": mode,
        "variant": variant,
        "labels": {key: row[key] for key in ("gpt4_score", "human_score", "tie") if key in row},
        "metadata": {
            key: row[key]
            for key in ("idx", "source", "category", "criteria", "response_source")
            if key in row
        },
    }
    if mode == "absolute":
        prompt = ablate(row["instruction"], variant)
        result["response_words"] = len(section(row["instruction"], "Response to evaluate:").split())
        result["instruction_group"] = digest(
            section(row["instruction"], "The instruction to evaluate:")
        )
        result["prompt_sha256"] = digest(prompt)
        result["grades"] = [
            grade_once(
                backend,
                prompt,
                replace(config, seed=base_seed + n * 100),
                attempts,
                variant != "score_only",
            )
            for n in range(repeats)
        ]
        scores = [g["score"] for g in result["grades"] if g["status"] == "ok"]
        result["status"] = "ok" if len(scores) == repeats else "incomplete"
        result["mean_score"] = sum(scores) / len(scores) if scores else None
    elif mode == "relative":
        result["labels"].setdefault("tie", 0)  # HHH has no tie field.
        if result["labels"]["tie"] not in (0, 1):
            raise ValueError("Relative tie labels must be 0 or 1")
        if tie_policy == "resample" and result["labels"]["tie"] == 1:
            raise ValueError("Do not resample a benchmark containing labelled ties; use keep")
        rounds = []
        for round_number in range(tie_rounds if tie_policy == "resample" else 1):
            pair = {}
            for side, offset in (("chosen", 0), ("rejected", 100)):
                # Candidate identity and gold labels are never included in the prompt.
                pair[side] = grade_once(
                    backend,
                    ablate(row[f"{side}_instruction"], variant),
                    replace(config, seed=base_seed + round_number * 1000 + offset),
                    attempts,
                    variant != "score_only",
                )
            rounds.append(pair)
            if any(pair[s]["score"] is None for s in ("chosen", "rejected")):
                break
            if pair["chosen"]["score"] != pair["rejected"]["score"] or tie_policy == "keep":
                break
        result["rounds"] = rounds
        chosen, rejected = (rounds[-1][s]["score"] for s in ("chosen", "rejected"))
        result["status"] = "ok" if chosen is not None and rejected is not None else "incomplete"
        result["prediction"] = (
            None
            if result["status"] != "ok"
            else ("tie" if chosen == rejected else "chosen" if chosen > rejected else "rejected")
        )
        expected = "tie" if result["labels"]["tie"] else "chosen"
        result["correct"] = result["prediction"] == expected if result["prediction"] else None
        result["tie_exhausted"] = tie_policy == "resample" and result["prediction"] == "tie"
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    return result


def recover_results(path):
    """Recover only an interrupted trailing write; retain the bytes in a recovery file."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        tail_start = raw.rfind(b"\n") + 1
        tail = raw[tail_start:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            backup = path.with_name(path.name + ".interrupted-tail")
            backup.write_bytes(tail)
            with path.open("r+b") as f:
                f.truncate(tail_start)
        else:
            with path.open("ab") as f:
                f.write(b"\n")
    return read_records(path)


def benchmark(
    backend,
    input_path,
    output,
    mode="absolute",
    config=None,
    limit=None,
    repeats=3,
    attempts=2,
    variant="full",
    tie_policy="keep",
    tie_rounds=3,
    resume=True,
):
    if repeats < 1 or attempts < 1 or tie_rounds < 1:
        raise ValueError("repeats, attempts and tie_rounds must be positive")
    if tie_policy not in ("keep", "resample"):
        raise ValueError("tie_policy must be keep or resample")
    config = config or GenerationConfig()
    rows = read_records(input_path)
    if not rows:
        raise ValueError("The evaluation dataset is empty")
    fields = (
        ("instruction",) if mode == "absolute" else ("chosen_instruction", "rejected_instruction")
    )
    for index, row in enumerate(rows):
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in fields):
            raise ValueError(f"Row {index} requires nonempty strings: {', '.join(fields)}")
    indices = select_indices(rows, limit, config.seed)
    if mode == "relative" and tie_policy == "resample" and any(r.get("tie") for r in rows):
        raise ValueError("Resampling a benchmark with ground-truth ties would bias the evaluation")
    spec = {
        "input_path": str(input_path),
        "input_sha256": file_hash(input_path),
        "source_rows": len(rows),
        "indices": indices,
        "backend": backend.identity,
        "generation": asdict(config),
        "mode": mode,
        "repeats": repeats if mode == "absolute" else 1,
        "attempts": attempts,
        "variant": variant,
        "tie_policy": tie_policy,
        "tie_rounds": tie_rounds,
        "protocol_version": 1,
        "runtime_versions": environment()["packages"],
        "implementation_sha256": {
            name: file_hash(Path(__file__).parent / name)
            for name in ("evaluation.py", "prompts.py", "backends.py")
        },
        "sampling": "seeded round-robin label strata",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path, result_path = output / "manifest.json", output / "predictions.jsonl"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if not resume or previous["spec_sha256"] != digest(spec):
                raise ValueError(
                    "Run already exists with different configuration/data/model; choose a new output"
                )
        else:
            if result_path.exists():
                raise ValueError("Predictions exist without a manifest; refusing to overwrite")
            write_json(
                manifest_path,
                {"spec": spec, "spec_sha256": digest(spec), "environment": environment()},
            )
        completed = recover_results(result_path)
        done = {row["source_index"] for row in completed}
        if len(done) != len(completed) or not done.issubset(set(indices)):
            raise ValueError("Duplicate or unexpected result indices")
        for row in completed:
            if row["input_sha256"] != digest(rows[row["source_index"]]):
                raise ValueError("Saved result/input mismatch")
        for index in indices:
            if index in done:
                continue
            record = evaluate_record(
                backend,
                rows[index],
                index,
                mode,
                config,
                repeats,
                attempts,
                variant,
                tie_policy,
                tie_rounds,
            )
            append_jsonl(result_path, record)
            done.add(index)
            print(
                f"{output.name}: {len(done)}/{len(indices)} row={index} "
                f"status={record['status']} score={record.get('mean_score', record.get('prediction'))}",
                flush=True,
            )
        write_json(
            output / "completion.json",
            {
                "finished_utc": utc_now(),
                "rows": len(done),
                "predictions_sha256": file_hash(result_path),
            },
        )
    return str(output)
