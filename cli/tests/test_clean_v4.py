"""Cleaning the 40k synthetic set: dedupe, drop conflicting tokens, relabel non-secret positives."""

from scripts.clean_v4 import clean


def _rec(token, label, line=None, source="llm"):
    return {"token": token, "label": label, "line_content": line or f'x = "{token}"', "source": source}


def test_clean_rules():
    records = [
        _rec("Zq8vK2mN4pL7xR1t", 1),                 # kept as-is
        _rec("Zq8vK2mN4pL7xR1t", 1),                 # exact duplicate -> dropped
        _rec("from_secs", 1),                        # identifier labeled secret -> relabeled 0
        _rec("./deploy.sh", 1, source="script"),     # path labeled secret -> relabeled 0
        _rec("Ab3dE5gH7jK9mN1p", 1),                 # conflicting labels -> both dropped
        _rec("Ab3dE5gH7jK9mN1p", 0, line='y = "Ab3dE5gH7jK9mN1p"'),
    ]
    out, stats = clean(records)
    assert [(r["token"], r["label"]) for r in out] == [
        ("Zq8vK2mN4pL7xR1t", 1), ("from_secs", 0), ("./deploy.sh", 0)]
    relabeled = [r for r in out if r.get("relabeled")]
    assert {r["relabel_reason"] for r in relabeled} == {"identifier/word", "url/path"}
    assert all(r["original_label"] == 1 for r in relabeled)
    assert out[0]["source"] == "llm_synthetic" and out[2]["source"] == "script"
    assert stats == {"input": 6, "duplicates": 1, "conflicting_dropped": 2, "relabeled": 2, "output": 3}
