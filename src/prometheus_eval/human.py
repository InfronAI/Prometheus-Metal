"""Blind feedback comparisons; no simulated human scores."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .io import digest, file_hash, read_records, write_json


def study_sources(left, right):
    return {
        side: {
            "manifest_sha256": file_hash(Path(run) / "manifest.json"),
            "predictions_sha256": file_hash(Path(run) / "predictions.jsonl"),
        }
        for side, run in (("left", left), ("right", right))
    }


def human_study(left, right, output, annotations=None, seed=42):
    left, right, output = Path(left), Path(right), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if annotations:
        sources = json.loads((output / "study_manifest.json").read_text())["sources"]
        if sources != study_sources(left, right):
            raise ValueError("Study source runs changed or were swapped; use the original runs")
        key = json.loads((output / "private_key.json").read_text())
        counts = {"left": 0, "right": 0, "tie": 0}
        seen, scores, reasons = set(), [], []
        with Path(annotations).open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if not row.get("preference"):
                continue
            pair = (row["item_id"], row["annotator_id"])
            if pair in seen or not row["annotator_id"].strip():
                raise ValueError("Duplicate item/annotator or empty annotator ID")
            seen.add(pair)
            if row["item_id"] not in key or row["preference"] not in ("A", "B", "tie"):
                raise ValueError("Unknown item or preference; use A, B, or tie")
            score = int(row["response_score"])
            if not 1 <= score <= 5:
                raise ValueError("Human response score must be 1..5")
            selected = row["preference"]
            winner = "tie" if selected == "tie" else key[row["item_id"]][selected]
            counts[winner] += 1
            scores.append(
                {"item_id": row["item_id"], "annotator_id": row["annotator_id"], "score": score}
            )
            reasons.append(row.get("rejection_reason", ""))
        total = sum(counts.values())
        from .analysis import correlations

        by_item = defaultdict(list)
        for score_row in scores:
            by_item[score_row["item_id"]].append(score_row["score"])
        correlations_with_humans = {}
        for side, run in (("left", left), ("right", right)):
            predictions = {r["row_id"]: r for r in read_records(run / "predictions.jsonl")}
            predicted, observed = [], []
            for item_id, ratings in by_item.items():
                row = predictions[key[item_id]["row_id"]]
                if row["status"] == "ok" and row["mode"] == "absolute":
                    predicted.append(row["mean_score"])
                    observed.append(sum(ratings) / len(ratings))
            correlations_with_humans[side] = correlations(predicted, observed)
        result = {
            "annotations": total,
            "wins": counts,
            "left_win_rate_excluding_ties": counts["left"] / (counts["left"] + counts["right"])
            if counts["left"] + counts["right"]
            else None,
            "human_scores": scores,
            "rejection_reasons": reasons,
            "unique_annotators": len({r["annotator_id"] for r in scores}),
            "correlations_with_mean_human_scores": correlations_with_humans,
            "status": "measured" if total else "awaiting_human_annotations",
        }
        write_json(output / "summary.json", result)
        return result
    if (output / "private_key.json").exists():
        raise ValueError("Human study already exists; use --annotations to aggregate")
    left_rows = {
        r["row_id"]: r for r in read_records(left / "predictions.jsonl") if r["status"] == "ok"
    }
    right_rows = {
        r["row_id"]: r for r in read_records(right / "predictions.jsonl") if r["status"] == "ok"
    }
    manifest = json.loads((left / "manifest.json").read_text())
    inputs = read_records(manifest["spec"]["input_path"])
    rng, items, key = random.Random(seed), [], {}
    for row_id in sorted(left_rows.keys() & right_rows.keys()):
        if left_rows[row_id]["mode"] != "absolute" or right_rows[row_id]["mode"] != "absolute":
            continue
        order = ["left", "right"]
        rng.shuffle(order)
        item_id = digest({"row": row_id, "seed": seed})[:16]
        texts = {
            "left": left_rows[row_id]["grades"][0]["feedback"],
            "right": right_rows[row_id]["grades"][0]["feedback"],
        }
        # Hide the separately parsed [RESULT] score. Feedback prose is preserved verbatim.
        items.append(
            {
                "id": item_id,
                "data": {
                    "prompt": inputs[left_rows[row_id]["source_index"]]["instruction"],
                    "feedback_A": texts[order[0]],
                    "feedback_B": texts[order[1]],
                    "questions": [
                        "Score the response 1–5 before reading the feedback.",
                        "Which feedback is better: A, B, or tie?",
                        "Why did you reject the other feedback?",
                    ],
                },
            }
        )
        key[item_id] = {"A": order[0], "B": order[1], "row_id": row_id}
    write_json(output / "label_studio_tasks.json", items)
    write_json(output / "private_key.json", key)
    write_json(
        output / "study_manifest.json", {"sources": study_sources(left, right), "seed": seed}
    )
    with (output / "annotations.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["item_id", "annotator_id", "response_score", "preference", "rejection_reason"]
        )
        for item in items:
            writer.writerow([item["id"], "", "", "", ""])
    result = {
        "items": len(items),
        "seed": seed,
        "status": "awaiting_human_annotations",
        "instructions": "Keep private_key.json hidden from annotators. Fill annotations.csv with real judgments.",
    }
    write_json(output / "summary.json", result)
    return result
