"""DATA-02/04/06/07: the eval-set builder's split and insertion rules."""

import random

from scripts.build_eval_set import NEUTRAL_NAMES, SECRET_NAMES, _insert, split_of

REQUIRED_FIELDS = {"token", "label", "line_content", "context_before", "context_after", "file_path",
                   "source", "generator", "generator_version", "seed", "repo", "file_type",
                   "language", "insertion_style", "name_style"}


def test_data_07_split_is_deterministic_and_exclusive():
    repos = [f"org/repo{i}" for i in range(500)]
    splits = {r: split_of(r) for r in repos}
    assert splits == {r: split_of(r) for r in repos}
    assert set(splits.values()) == {"train", "val", "test"}


def test_data_06_insert_records_value_and_provenance(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    src = repo_dir / "app.py"
    src.write_text("\n".join(f"x{i} = {i}" for i in range(10)) + "\n")
    rec = _insert(random.Random(0), "FAKEVALUE123", "generic_random", 1, "generated_insert",
                  SECRET_NAMES, "secret", src, repo_dir, "org/repo", 7)
    assert REQUIRED_FIELDS <= rec.keys()
    assert "FAKEVALUE123" in rec["line_content"]
    assert rec["file_path"] == "app.py" and rec["seed"] == 7 and rec["label"] == 1
    assert len(rec["context_before"]) == 3


def test_data_04_neutral_names_carry_no_secret_words(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    src = repo_dir / "app.py"
    src.write_text("\n".join(f"x{i} = {i}" for i in range(10)) + "\n")
    rng = random.Random(1)
    for _ in range(20):
        rec = _insert(rng, "FAKEVALUE123", "generic_random", 1, "generated_insert",
                      NEUTRAL_NAMES, "neutral", src, repo_dir, "org/repo", 1)
        assert not any(w in rec["line_content"].lower() for w in ("key", "secret", "token", "password"))
