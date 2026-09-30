"""M0 exit criterion: the published detector metrics are reproducible from the repo."""

from pathlib import Path

import pytest

from bench.eval_detector import evaluate

CONFIG = Path(__file__).resolve().parents[1] / "Harpocrates" / "ml" / "models" / "model_config.json"


def test_evaluate_counts_confusion_matrix():
    records = [{"label": 1}, {"label": 1}, {"label": 0}, {"label": 0}]
    predictions = iter([True, False, True, False])
    m = evaluate(records, lambda _r: next(predictions))
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5


def test_evaluate_precision_undefined_without_negatives():
    m = evaluate([{"label": 1}], lambda _r: True)
    assert m["precision"] is None  # all-positive set can't measure precision


def test_pipeline_counts_only_the_record_line():
    pytest.importorskip("onnxruntime")
    from bench.eval_detector import _pipeline_predictor

    predict = _pipeline_predictor(0.19)
    flagged = {"token": "PurpleDog197!", "line_content": 'password = "PurpleDog197!"',
               "context_before": ["a = 1"], "context_after": ["b = 2"]}
    # Same secret on a neighbouring line must not count as a hit for this record.
    neighbour = {"token": "PurpleDog197!", "line_content": "b = 2",
                 "context_before": ['password = "PurpleDog197!"'], "context_after": []}
    assert predict(flagged) is True
    assert predict(neighbour) is False


def test_compare_scanners_matching_rules(tmp_path):
    from bench.compare_scanners import materialize, overlaps

    assert overlaps("Wint3r#Sales9", "postgresql://app:Wint3r#Sales9@db/orders")  # span inside token
    assert overlaps("postgresql://app:Wint3r#Sales9@db/orders", "Wint3r#Sales9")   # token inside span
    assert not overlaps("abc", "abcdef")                                           # too short to count
    assert not overlaps("", "anything")
    placed = materialize([{"token": "t", "line_content": "x = 1", "context_before": ["a", "b"],
                           "context_after": ["c"], "file_type": ".py"}], tmp_path)
    assert placed == [("r000000.py", 3)]
    assert (tmp_path / "r000000.py").read_text().splitlines()[2] == "x = 1"
