"""Original v1 prompt contract and explicitly labelled experimental ablations."""

import re
from dataclasses import dataclass

SYSTEM = "You are a fair evaluator language model."
VARIANTS = ("full", "no_reference", "no_rubric", "score_only")

TASK = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations."""


def make_prompt(instruction, response, reference_answer, rubric):
    if not all(isinstance(x, str) and x.strip() for x in (instruction, response)):
        raise ValueError("instruction and response must be nonempty strings")
    if isinstance(rubric, dict):
        if not rubric.get("criteria") or any(not rubric.get(str(i)) for i in range(1, 6)):
            raise ValueError("rubric requires criteria and score descriptions '1' through '5'")
        rubric = f"[{rubric['criteria']}]\n" + "\n".join(
            f"Score {i}: {rubric[str(i)]}" for i in range(1, 6)
        )
    if not isinstance(rubric, str) or not rubric.strip():
        raise ValueError("rubric must be a nonempty string or complete rubric object")
    ref = f"\n\n###Reference Answer (Score 5):\n{reference_answer}" if reference_answer else ""
    task = (
        TASK
        if reference_answer
        else TASK.replace("a reference answer that gets a score of 5, and ", "")
    )
    return (
        f"{task}\n\n###The instruction to evaluate:\n{instruction}"
        f"\n\n###Response to evaluate:\n{response}{ref}"
        f"\n\n###Score Rubrics:\n{rubric}\n\n###Feedback: \n"
    )


def llama2_chat(prompt):
    # Include BOS exactly once; backend tokenization must use add_special_tokens=False.
    return f"<s>[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n{prompt.strip()} [/INST]"


def section(prompt, heading):
    match = re.search(
        r"###" + re.escape(heading) + r"[^\n]*\n(.*?)(?=\n###|\Z)", prompt, flags=re.S
    )
    return match.group(1).strip() if match else ""


def ablate(prompt, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == "no_reference":
        prompt = re.sub(r"\n*###Reference Answer[^\n]*\n.*?(?=\n###|\Z)", "", prompt, flags=re.S)
        prompt = prompt.replace("a reference answer that gets a score of 5, and ", "")
    elif variant == "no_rubric":
        prompt = re.sub(r"\n*###Score Rubrics[^\n]*\n.*?(?=\n###|\Z)", "", prompt, flags=re.S)
        prompt = prompt.replace(
            "strictly based on the given score rubric, not evaluating in general",
            "based on the quality of the response",
        )
        prompt = prompt.replace(" You should refer to the score rubric.", "")
        prompt = prompt.replace(
            ", and a score rubric representing a evaluation criteria are given.", " are given."
        )
    elif variant == "score_only":
        start = prompt.find("###The instruction to evaluate:")
        if start < 0:
            raise ValueError("Score-only ablation requires a structured Prometheus prompt")
        prompt = (
            "###Task Description:\nEvaluate the response using the reference answer and score "
            "rubric. Output only [RESULT] followed by an integer from 1 to 5.\n\n" + prompt[start:]
        )
        prompt = prompt.replace("###Feedback:", "###Score:")
    return prompt


@dataclass(frozen=True)
class Grade:
    feedback: str
    score: int


def parse_grade(text, require_feedback=True):
    # Do not extract a number from the prompt, prose, decimal, or multiple conflicting results.
    text = re.sub(r"(?:</s>|\[END\])\s*$", "", text.strip()).strip()
    if text.count("[RESULT]") != 1:
        raise ValueError("Expected exactly one [RESULT] marker")
    match = re.fullmatch(r"(.*?)\[RESULT\]\s*([1-5])\s*", text, re.S)
    if not match:
        raise ValueError("Expected terminal [RESULT] followed by one integer in 1..5")
    feedback = re.sub(r"^(?:###)?Feedback:\s*", "", match.group(1).strip()).strip()
    if require_feedback and not feedback:
        raise ValueError("Missing feedback before score")
    return Grade(feedback, int(match.group(2)))
