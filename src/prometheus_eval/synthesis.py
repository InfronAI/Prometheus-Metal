"""The four-stage Appendix G construction process, using an explicit local teacher.

The shipped seeds are new examples, not the unavailable original 50 author seeds.
Raw generations and quality checks are retained; failures never become training data.
"""

import json
import random
import re
from pathlib import Path

from .backends import GenerationConfig
from .io import append_jsonl, digest, environment, read_records, write_json, write_jsonl
from .prompts import make_prompt


def parse_json_object(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()
    text = text.removesuffix("[END]").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def validate_rubric(value):
    required = ("criteria", "1", "2", "3", "4", "5")
    if not isinstance(value, dict) or any(
        not isinstance(value.get(k), str) or not value[k].strip() for k in required
    ):
        raise ValueError("Rubric requires criteria and five nonempty descriptions")
    return {k: value[k] for k in required}


def parse_pair(text, first, second):
    text = text.strip().removesuffix("[END]").strip()
    parts = text.split("[NEXT]")
    if len(parts) != 2:
        raise ValueError("Expected two sections separated by [NEXT]")
    result = []
    for part, prefix in zip(parts, (first, second)):
        part = part.strip()
        # The unique [NEXT] separator also unambiguously identifies the second
        # section when a teacher omits its decorative heading. Preserve raw text
        # in generations.jsonl; do not invent or repair missing content.
        if part.startswith(prefix + ":"):
            part = part[len(prefix) + 1 :].strip()
        elif prefix == first:
            raise ValueError(f"Missing {prefix}: section")
        if not part:
            raise ValueError(f"Empty {prefix}: section")
        result.append(part)
    return result


def sentence_count(text):
    # Treat sentence-final punctuation inside quotes as a boundary too.
    return max(1, len([s for s in re.split(r"""[.!?]["'”’)]*\s+""", text.strip()) if s.strip()]))


def synthesize(backend, seeds_path, output, rubrics=1, instructions_per_rubric=1, seed=42):
    seeds = [validate_rubric(r) for r in read_records(seeds_path)]
    if len(seeds) < 4:
        raise ValueError("Appendix G brainstorming requires at least four seed rubrics")
    if rubrics < 1 or instructions_per_rubric < 1:
        raise ValueError("Construction bounds must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Synthesis output is nonempty; use a new directory")
    output.mkdir(parents=True)
    config = GenerationConfig(max_tokens=1024, max_input_tokens=3072, temperature=0.7, seed=seed)
    write_json(
        output / "manifest.json",
        {
            "teacher": backend.identity,
            "seed": seed,
            "rubrics": rubrics,
            "instructions_per_rubric": instructions_per_rubric,
            "seed_origin": "new illustrative rubrics; not the paper's original 50 seeds",
            "environment": environment(),
            "method": "Appendix G sequence, local teacher substitution",
        },
    )
    rng, calls, rows, failures, checks = random.Random(seed), 0, [], [], []

    def generate(stage, prompt, parse):
        nonlocal calls
        for attempt in range(2):
            calls += 1
            from dataclasses import replace

            result = backend.generate(prompt, replace(config, seed=seed + calls))
            record = {"call": calls, "stage": stage, "attempt": attempt, "prompt": prompt, **result}
            try:
                value = parse(result["text"])
                append_jsonl(output / "generations.jsonl", {**record, "valid": True})
                return value
            except (ValueError, json.JSONDecodeError) as e:
                append_jsonl(
                    output / "generations.jsonl", {**record, "valid": False, "error": str(e)}
                )
        raise ValueError(f"{stage}: no valid output within two attempts")

    for ri in range(rubrics):
        try:
            examples = rng.sample(seeds, 4)
            new = generate(
                "brainstorm",
                "We are brainstorming useful, real-world criteria to grade "
                "language model responses in diverse situations. Here are four examples:\n"
                + json.dumps(examples)
                + "\nCreate one novel criterion and score descriptions 1–5. "
                "Return only one JSON object with keys criteria, 1, 2, 3, 4, 5, then [END].",
                lambda x: validate_rubric(parse_json_object(x)),
            )
            paraphrase = generate(
                "paraphrase",
                "Paraphrase the values in this scoring rubric without "
                "changing their meaning. Vary diction and sentence structure. Return only "
                "the same JSON dictionary with keys criteria, 1, 2, 3, 4, 5 and [END].\n"
                + json.dumps(new),
                lambda x: validate_rubric(parse_json_object(x)),
            )
            write_json(output / f"rubric-{ri}.json", {"original": new, "paraphrase": paraphrase})
            for ii in range(instructions_per_rubric):
                instruction, reference = generate(
                    "instruction_reference",
                    "Generate a novel realistic user problem related to this "
                    "score rubric, and a reference response that deserves score 5. The problem must be at "
                    "most 60 words. The reference must be exactly four sentences and at most 140 words. Do not mention "
                    "the rubric in the problem. Use a natural user question, not a role assignment.\n"
                    + json.dumps(paraphrase)
                    + "\nUse exactly this output structure, keeping the literal separator "
                    "[NEXT] between the problem and the response. No Markdown headings or extra sections:\n"
                    "Problem: <the problem>\n[NEXT]\nResponse: <four sentences>\n[END]",
                    lambda x: parse_pair(x, "Problem", "Response"),
                )
                group = []
                for score in range(1, 6):
                    count = sentence_count(reference)
                    response, feedback = generate(
                        "response_feedback",
                        f"Generate a response deserving score {score} and detailed "
                        f"feedback explaining that score under the rubric. Match the reference's {count} "
                        "sentences; quality must differ by substance, not length. Do not mention the rubric "
                        "or score in the response. Feedback must cite specific content and end with "
                        f"'So the overall score is {score}.'\nInstruction:\n{instruction}\nRubric:\n"
                        + json.dumps(paraphrase)
                        + f"\nReference (score 5):\n{reference}\n"
                        + "Use exactly this output structure with the literal [NEXT] separator. "
                        "Keep feedback to at most 120 words. No Markdown headings or extra sections:\n"
                        "Response: <the response>\n[NEXT]\nFeedback: <the feedback>\n[END]",
                        lambda x: parse_pair(x, "Response", "Feedback"),
                    )
                    check = {
                        "rubric": ri,
                        "instruction": ii,
                        "score": score,
                        "reference_sentences": count,
                        "response_sentences": sentence_count(response),
                        "length_matched": sentence_count(response) == count,
                        "feedback_score_matches": bool(
                            re.search(rf"overall score is\s+{score}\b", feedback, re.I)
                        ),
                        "distinct_from_reference": response != reference,
                    }
                    checks.append(check)
                    # Score control is essential. Sentence-count mismatches are also excluded.
                    if not all(
                        check[k]
                        for k in (
                            "length_matched",
                            "feedback_score_matches",
                            "distinct_from_reference",
                        )
                    ):
                        continue
                    group.append(
                        {
                            "instruction": make_prompt(
                                instruction, response, reference, paraphrase
                            ),
                            "output": f"{feedback}\n[RESULT] {score}",
                            "score": score,
                            "group_id": digest(instruction),
                            "source": "local_synthetic",
                        }
                    )
                if len(group) == 5:
                    rows.extend(group)
                else:
                    failures.append(
                        {
                            "rubric": ri,
                            "instruction": ii,
                            "reason": "Incomplete quality-controlled five-score group",
                            "accepted_before_group_filter": len(group),
                        }
                    )
        except ValueError as e:
            failures.append({"rubric": ri, "reason": str(e)})
    write_jsonl(output / "training.jsonl", rows)
    write_json(output / "quality_checks.json", checks)
    result = {
        "calls": calls,
        "accepted_rows": len(rows),
        "failures": failures,
        "complete": len(rows) == rubrics * instructions_per_rubric * 5,
        "note": "Local teacher experiment; not GPT-4-0613 corpus regeneration",
    }
    write_json(output / "summary.json", result)
    return result
