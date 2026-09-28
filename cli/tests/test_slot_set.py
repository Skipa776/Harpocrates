"""Slot filler: LLM-written files with typed {{SECRET:kind}} / {{NONSECRET:kind}} markers."""

import pytest

from scripts.build_slot_set import fill_file


def test_slots_become_labeled_records(tmp_path):
    src = tmp_path / "billing.py"
    src.write_text(
        "import stripe\n\n"
        'stripe.api_key = "{{SECRET:stripe_key}}"\n'
        'build_id = "{{NONSECRET:git_sha}}"\n'
        "def charge():\n    pass\n"
    )
    records = fill_file(src, tmp_path, seed=1)
    assert [(r["label"], r["secret_type"]) for r in records] == [(1, "stripe_key"), (0, "git_sha")]
    for r in records:
        assert r["token"] in r["line_content"] and "{{" not in r["line_content"]
        assert r["source"] == "llm_slot" and r["file_path"] == "billing.py"
    # Other slots in the context are filled too, never left as markers.
    assert all("{{" not in line for r in records for line in r["context_before"] + r["context_after"])


def test_unknown_kind_is_rejected(tmp_path):
    src = tmp_path / "a.py"
    src.write_text('x = "{{SECRET:not_a_kind}}"\n')
    with pytest.raises(ValueError, match="not_a_kind"):
        fill_file(src, tmp_path, seed=1)


def test_malformed_marker_is_rejected(tmp_path):
    src = tmp_path / "a.py"
    src.write_text('x = "{{SECRET}}"\n')
    with pytest.raises(ValueError, match="malformed"):
        fill_file(src, tmp_path, seed=1)


def test_multiline_value_is_rejected(tmp_path, monkeypatch):
    import scripts.build_slot_set as bss

    monkeypatch.setitem(bss.POSITIVES, "pem_key", lambda: "-----BEGIN KEY-----\nAAAA\n-----END KEY-----")
    src = tmp_path / "a.py"
    src.write_text('k = "{{SECRET:pem_key}}"\n')
    with pytest.raises(ValueError, match="multi-line"):
        fill_file(src, tmp_path, seed=1)
