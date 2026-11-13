import csv
import json

import pytest

from prometheus_eval.human import human_study
from prometheus_eval.io import write_json, write_jsonl


def test_blind_export_and_only_real_supplied_annotations_are_counted(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"instruction": "A real task would go here"}])
    for side in ("left", "right"):
        run = tmp_path / side
        run.mkdir()
        write_json(run / "manifest.json", {"spec": {"input_path": str(source)}})
        write_jsonl(
            run / "predictions.jsonl",
            [
                {
                    "row_id": "row1",
                    "status": "ok",
                    "mode": "absolute",
                    "source_index": 0,
                    "mean_score": 3,
                    "grades": [{"feedback": f"{side} test feedback"}],
                }
            ],
        )
    study = tmp_path / "study"
    result = human_study(tmp_path / "left", tmp_path / "right", study)
    assert result["status"] == "awaiting_human_annotations" and result["items"] == 1
    annotations = study / "annotations.csv"
    result = human_study(tmp_path / "left", tmp_path / "right", study, annotations)
    assert result["annotations"] == 0 and result["status"] == "awaiting_human_annotations"
    item = json.loads((study / "label_studio_tasks.json").read_text())[0]["id"]
    with annotations.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["item_id", "annotator_id", "response_score", "preference", "rejection_reason"]
        )
        writer.writerow([item, "test-annotator", 4, "tie", "test fixture only"])
    result = human_study(tmp_path / "left", tmp_path / "right", study, annotations)
    assert result["annotations"] == 1 and result["wins"]["tie"] == 1
    assert result["correlations_with_mean_human_scores"]["left"]["pearson"] is None
    with annotations.open("a") as f:
        f.write(f"{item},test-annotator,4,tie,duplicate\n")
    with pytest.raises(ValueError, match="Duplicate"):
        human_study(tmp_path / "left", tmp_path / "right", study, annotations)
    with pytest.raises(ValueError, match="swapped"):
        human_study(tmp_path / "right", tmp_path / "left", study, annotations)
