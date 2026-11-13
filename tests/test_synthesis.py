import pytest

from prometheus_eval.synthesis import parse_json_object, parse_pair, sentence_count, validate_rubric


def test_delimited_pair_is_unambiguous_with_optional_second_heading():
    assert parse_pair("Problem: A [NEXT] Response: B [END]", "Problem", "Response") == ["A", "B"]
    assert parse_pair("Problem: A [NEXT] B [END]", "Problem", "Response") == ["A", "B"]
    with pytest.raises(ValueError):
        parse_pair("Problem: A Response: B", "Problem", "Response")


def test_sentence_count_handles_quotes():
    assert sentence_count('It shows "no signal." Restart it. Check it. Update it.') == 4
    assert sentence_count("One. Two! Three?") == 3


def test_rubric_validation_rejects_missing_levels():
    with pytest.raises(ValueError):
        validate_rubric({"criteria": "Accuracy", "1": "Wrong"})
    assert parse_json_object('```json\n{"a":1}\n```') == {"a": 1}
