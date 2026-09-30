"""Pipeline-consistent training rows: features come from the scanner's own candidates."""

from scripts.train_v11 import pipeline_rows

PW = "Wint3r" + "#Sales9"


def test_candidates_on_record_line_are_labeled_by_overlap():
    record = {"token": PW, "label": 1, "line_content": f'conn = "Server=db;User ID=sa;Password={PW};"',
              "context_before": ["import os"], "context_after": ["x = 1"], "file_type": ".py"}
    rows = pipeline_rows([record])
    assert rows, "the ADO password must be offered to ML as a candidate"
    labels = {token: y for _x, y, token in rows}
    assert labels[PW] == 1                      # the true secret
    assert all(y == (PW in t or t in PW) for _x, y, t in rows)
    assert all(len(x) == len(rows[0][0]) for x, _y, _t in rows)


def test_negative_record_candidates_are_all_negative():
    record = {"token": PW, "label": 0, "line_content": f'example = "{PW}"',
              "context_before": [], "context_after": [], "file_type": ".py"}
    assert [y for _x, y, _t in pipeline_rows([record])] == [0]
