import json
from pathlib import Path

import pytest

from prometheus_eval.analysis import correlations, numeric_label, summarize
from prometheus_eval.backends import ContextOverflow, GenerationConfig
from prometheus_eval.data import BENCHMARKS, prepare_data, select_indices
from prometheus_eval.evaluation import benchmark, evaluate_record, grade_once, recover_results
from prometheus_eval.io import read_records, write_jsonl
from prometheus_eval.prompts import ablate, llama2_chat, make_prompt, parse_grade, section


def sample(i=0, score=3):
    prompt = make_prompt(
        f"Question {i}",
        f"Response {score}",
        "Good reference",
        {"criteria": "Accuracy", **{str(i): f"Description {i}" for i in range(1, 6)}},
    )
    return {
        "instruction": prompt,
        "output": f"Specific feedback. [RESULT] {score}",
        "gpt4_score": score,
    }


class ScriptedBackend:
    identity = {"backend": "test-only"}

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def generate(self, prompt, config):
        self.calls.append((prompt, config.seed))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return {"text": answer, "elapsed_seconds": 0.01, "generation_tokens": 3}


@pytest.mark.parametrize(
    "text",
    ["Good explanation. [RESULT] 1", "Feedback: Helpful. [RESULT] 5\n", "Helpful.\n[RESULT] 4</s>"],
)
def test_strict_valid_grade(text):
    assert 1 <= parse_grade(text).score <= 5


@pytest.mark.parametrize(
    "text",
    [
        "[RESULT] 3",
        "Fine [RESULT] 0",
        "Fine [RESULT] 6",
        "Fine [RESULT] 3.5",
        "Fine [RESULT] 30",
        "Fine [RESULT] 3 more prose",
        "Score: 3",
        "Fine [RESULT] 3 [RESULT] 4",
        "Fine [RESULT] ٣",
    ],
)
def test_never_guess_a_score(text):
    with pytest.raises(ValueError):
        parse_grade(text)


def test_score_only_requires_explicit_mode():
    assert parse_grade("[RESULT] 3", require_feedback=False).score == 3


def test_chat_and_ablations():
    prompt = sample()["instruction"]
    rendered = llama2_chat(prompt)
    assert rendered.count("<s>") == 1
    assert rendered.startswith("<s>[INST] <<SYS>>") and rendered.endswith("[/INST]")
    assert "Good reference" not in ablate(prompt, "no_reference")
    assert "Description 5" not in ablate(prompt, "no_rubric")
    assert "###Feedback:" not in ablate(prompt, "score_only")
    assert section(prompt, "Response to evaluate:") == "Response 3"
    assert ablate(prompt, "full") == prompt


def test_json_jsonl_lfs_and_errors(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([sample()]))
    assert len(read_records(p)) == 1
    p.write_text(json.dumps(sample()) + "\n" + json.dumps(sample(2)))
    assert len(read_records(p)) == 2
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    with pytest.raises(ValueError, match="LFS"):
        read_records(p)
    p.write_text('{"a":1}\nBROKEN')
    with pytest.raises(ValueError, match=":2:"):
        read_records(p)


def test_all_original_formats_are_readable():
    expected = [1000, 1000, 320, 320, 2001, 221, 1431]
    for (name, (filename, _)), n in zip(BENCHMARKS.items(), expected):
        assert len(read_records(Path("evaluation/benchmark/data") / filename)) == n, name


def test_group_split_and_balance(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = [sample(i, score) for i in range(10) for score in range(1, 6)]
    write_jsonl(source, rows)
    manifest = prepare_data(source, tmp_path / "prepared", limit=50)
    splits = [set(v["groups"]) for v in manifest["splits"].values()]
    assert all(not a & b for i, a in enumerate(splits) for b in splits[i + 1 :])
    assert sum(v["rows"] for v in manifest["splits"].values()) == 50
    for split in manifest["splits"].values():
        assert len(set(split["scores"].values())) == 1


def test_bounded_retries_and_context():
    backend = ScriptedBackend(["invalid", "specific [RESULT] 3"])
    assert grade_once(backend, "prompt", GenerationConfig(), attempts=2)["score"] == 3
    backend = ScriptedBackend([ContextOverflow("too long")])
    assert grade_once(backend, "prompt", GenerationConfig(), attempts=10)["score"] is None
    assert len(backend.calls) == 1
    backend = ScriptedBackend(["invalid", "invalid"])
    assert (
        grade_once(backend, "prompt", GenerationConfig(), attempts=2)["status"] == "invalid_output"
    )


def test_relative_ties_bounded_and_no_gold_leak():
    row = {"chosen_instruction": "candidate one", "rejected_instruction": "candidate two"}
    backend = ScriptedBackend(["fine [RESULT] 3"] * 6)
    result = evaluate_record(
        backend,
        row,
        0,
        "relative",
        GenerationConfig(),
        attempts=1,
        tie_policy="resample",
        tie_rounds=3,
    )
    assert len(backend.calls) == 6
    assert result["tie_exhausted"] and result["correct"] is False
    assert {p for p, _ in backend.calls} == {"candidate one", "candidate two"}
    tied = evaluate_record(
        ScriptedBackend(["fine [RESULT] 3"] * 2),
        {**row, "tie": 1},
        0,
        "relative",
        GenerationConfig(),
        attempts=1,
    )
    assert tied["correct"] is True


def test_resume_exactly_once_and_config_guard(tmp_path):
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, [sample(i) for i in range(2)])
    out = tmp_path / "run"
    backend = ScriptedBackend(["fine [RESULT] 3", "fine [RESULT] 4"])
    benchmark(backend, input_path, out, repeats=1, attempts=1)
    benchmark(ScriptedBackend([]), input_path, out, repeats=1, attempts=1)
    assert len(read_records(out / "predictions.jsonl")) == 2
    with pytest.raises(ValueError, match="different configuration"):
        benchmark(ScriptedBackend([]), input_path, out, repeats=2, attempts=1)
    with (out / "predictions.jsonl").open("ab") as f:
        f.write(b'{"partial":')
    assert len(recover_results(out / "predictions.jsonl")) == 2
    assert (out / "predictions.jsonl.interrupted-tail").read_bytes() == b'{"partial":'


def test_statistics_and_missing_labels():
    assert correlations([1, 2, 3], [1, 2, 3])["pearson"] == pytest.approx(1)
    assert correlations([2, 2, 2], [1, 2, 3])["pearson"] is None
    assert correlations([1, 2], [1, 2])["pearson"] is None
    assert numeric_label([1, 2, 3]) == 2
    assert numeric_label([]) is None and numeric_label(True) is None
    rows = [
        evaluate_record(
            ScriptedBackend(["fine [RESULT] 3"]),
            {"instruction": sample()["instruction"]},
            0,
            "absolute",
            GenerationConfig(),
            repeats=1,
            attempts=1,
        )
    ]
    assert summarize(rows)["gpt4_score"]["missing_labels"] == 1
    assert summarize(rows)["gpt4_score"]["pearson"] is None


def test_sampling_reproducible_and_balanced():
    rows = [sample(i, i % 5 + 1) for i in range(50)]
    indices = select_indices(rows, 10, 42)
    assert indices == select_indices(rows, 10, 42)
    assert indices != select_indices(rows, 10, 43)
    assert {
        score: sum(rows[i]["gpt4_score"] == score for i in indices) for score in range(1, 6)
    } == {i: 2 for i in range(1, 6)}


def test_empty_benchmark_is_rejected_before_inference(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        benchmark(ScriptedBackend([]), path, tmp_path / "output")
