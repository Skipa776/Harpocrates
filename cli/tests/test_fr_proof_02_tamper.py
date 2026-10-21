"""FR-PROOF-02 (G3 spec, "Proof"): the proof chain shows tampering. An edited, inserted, deleted or truncated entry, a
missing or stale proof.head, a rotated-away file or a damaged last line each make verify() name a problem, and
append() refuses to go on from a chain it cannot vouch for."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Harpocrates import audit

PROOF = "proof.jsonl"
HEAD = "proof.head"


def build(state: Path, count: int = 5) -> None:
    """Append `count` entries to a proof chain in `state`."""
    chain = audit.ProofChain(state)
    for i in range(1, count + 1):
        chain.append("anthropic", "anthropic.messages", b"body%d" % i, b"body%d" % i)


def load(state: Path) -> list[str]:
    """The proof log's lines, as written."""
    return (state / PROOF).read_text(encoding="utf-8").splitlines()


def store(state: Path, lines: list[str]) -> None:
    """Rewrite the proof log with exactly these lines."""
    (state / PROOF).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def try_append(state: Path) -> dict:
    """Ask the chain in `state` to record one more request."""
    return audit.ProofChain(state).append("anthropic", "anthropic.messages", b"next", b"next")


def test_fr_proof_02_intact_chain_of_five_verifies(tmp_path: Path) -> None:
    """An untouched chain of five entries verifies, numbered 1 to 5, and the first entry follows the genesis hash."""
    build(tmp_path)
    result = audit.verify(tmp_path)
    assert result.problem is None
    assert len(result.entries) == 5
    assert [entry["seq"] for entry in result.entries] == [1, 2, 3, 4, 5]
    assert result.entries[0]["prev"] == audit.GENESIS


def test_fr_proof_02_edited_entry_without_new_hash_is_detected(tmp_path: Path) -> None:
    """Changing a field of entry 3 without recomputing its hash is reported as a changed entry."""
    build(tmp_path)
    lines = load(tmp_path)
    entry = json.loads(lines[2])
    entry["bytes"] += 1
    lines[2] = json.dumps(entry)
    store(tmp_path, lines)
    problem = audit.verify(tmp_path).problem
    assert problem is not None and "was changed" in problem
    assert "proof.jsonl line 3" in problem


def test_fr_proof_02_edited_entry_with_recomputed_hash_breaks_the_next_link(tmp_path: Path) -> None:
    """Editing entry 3 and recomputing its own hash still fails, because entry 4 no longer follows it."""
    build(tmp_path)
    lines = load(tmp_path)
    entry = json.loads(lines[2])
    entry["bytes"] += 1
    entry["hash"] = audit.entry_hash(entry)
    lines[2] = json.dumps(entry)
    store(tmp_path, lines)
    problem = audit.verify(tmp_path).problem
    assert problem is not None and "does not follow" in problem
    assert "proof.jsonl line 4" in problem


def test_fr_proof_02_deleted_middle_entry_is_detected(tmp_path: Path) -> None:
    """Deleting entry 3 leaves entry 4 pointing at a hash that is no longer the entry before it."""
    build(tmp_path)
    lines = load(tmp_path)
    del lines[2]
    store(tmp_path, lines)
    problem = audit.verify(tmp_path).problem
    assert problem is not None and "does not follow" in problem


def test_fr_proof_02_deleted_last_entry_is_detected(tmp_path: Path) -> None:
    """Deleting the newest entry leaves the log ending before proof.head, so the removal shows."""
    build(tmp_path)
    store(tmp_path, load(tmp_path)[:-1])
    problem = audit.verify(tmp_path).problem
    assert problem is not None and "newest entries were removed" in problem


def test_fr_proof_02_deleted_first_entry_is_detected(tmp_path: Path) -> None:
    """Deleting entry 1 leaves entry 2 whose prev is not the chain's base, so the removal shows."""
    build(tmp_path)
    store(tmp_path, load(tmp_path)[1:])
    problem = audit.verify(tmp_path).problem
    assert problem is not None and "entries before it were removed" in problem


def test_fr_proof_02_deleted_head_is_detected(tmp_path: Path) -> None:
    """With the entries present, a missing proof.head is a problem that names proof.head."""
    build(tmp_path)
    (tmp_path / HEAD).unlink()
    problem = audit.verify(tmp_path).problem
    assert problem is not None and "proof.head" in problem


def test_fr_proof_02_deleted_log_with_head_kept_is_gone(tmp_path: Path) -> None:
    """With proof.head kept but proof.jsonl deleted, verify reports that the proof log is gone."""
    build(tmp_path)
    (tmp_path / PROOF).unlink()
    result = audit.verify(tmp_path)
    assert result.entries == []
    assert result.problem is not None and "gone" in result.problem


def test_fr_proof_02_rotation_keeps_the_chain_and_deleting_the_rotated_file_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across a rotation the chain still verifies over both files; deleting the rotated file is detected."""
    monkeypatch.setattr(audit, "ROTATE_BYTES", 1500)
    build(tmp_path, 20)
    assert (tmp_path / "proof.1.jsonl").exists()
    result = audit.verify(tmp_path)
    assert result.problem is None
    seqs = [entry["seq"] for entry in result.entries]
    assert seqs == list(range(seqs[0], 21))
    assert seqs[0] > 1  # the oldest entries were rotated away
    (tmp_path / "proof.1.jsonl").unlink()
    problem = audit.verify(tmp_path).problem
    assert problem is not None and "entries before it were removed" in problem


def test_fr_proof_02_append_refuses_when_proof_head_is_deleted(tmp_path: Path) -> None:
    """append() raises ValueError rather than go on from entries whose proof.head is gone."""
    build(tmp_path)
    (tmp_path / HEAD).unlink()
    with pytest.raises(ValueError, match="proof.head"):
        try_append(tmp_path)


def test_fr_proof_02_append_refuses_when_last_entry_is_deleted(tmp_path: Path) -> None:
    """append() raises ValueError when the log no longer ends where proof.head says."""
    build(tmp_path)
    store(tmp_path, load(tmp_path)[:-1])
    with pytest.raises(ValueError, match="does not end"):
        try_append(tmp_path)


def test_fr_proof_02_append_refuses_when_last_line_is_not_json(tmp_path: Path) -> None:
    """append() raises ValueError when the log's last line is not JSON."""
    build(tmp_path)
    store(tmp_path, load(tmp_path) + ["not json"])
    with pytest.raises(ValueError, match="damaged"):
        try_append(tmp_path)


def test_fr_proof_02_head_one_behind_the_log_is_recovered_by_the_next_append(tmp_path: Path) -> None:
    """After a crash between writing an entry and moving proof.head, the next append continues the chain."""
    build(tmp_path, 3)
    old_head = (tmp_path / HEAD).read_text(encoding="utf-8")
    chain = audit.ProofChain(tmp_path)
    chain.append("anthropic", "anthropic.messages", b"body4", b"body4")
    (tmp_path / HEAD).write_text(old_head, encoding="utf-8")
    entry = chain.append("anthropic", "anthropic.messages", b"body5", b"body5")
    assert entry["seq"] == 5
    result = audit.verify(tmp_path)
    assert result.problem is None
    assert [item["seq"] for item in result.entries] == [1, 2, 3, 4, 5]


def test_fr_proof_02_empty_folder_verifies_then_the_first_append_starts_at_genesis(tmp_path: Path) -> None:
    """An empty state folder verifies with no entries, and the first append is seq 1 after the genesis hash."""
    result = audit.verify(tmp_path)
    assert result.problem is None
    assert result.entries == []
    entry = try_append(tmp_path)
    assert entry["seq"] == 1
    assert entry["prev"] == audit.GENESIS
    assert audit.verify(tmp_path).problem is None
