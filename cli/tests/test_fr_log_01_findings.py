"""FR-LOG-01: the findings log holds one JSON line per finding, and never a secret value.

Every test uses a temp state folder, or the made-up $HOME that the suite sets, so nothing touches the real ~/.harpo.
The canaries come from the generated known-format set, never a real credential.
"""
from __future__ import annotations

import json
import os
import stat
from datetime import datetime
from pathlib import Path

import pytest
from gate_canaries import known_format

from Harpocrates import audit

SID = "sess-1"
KEYS = ["time", "event_type", "layer", "type", "hash", "source", "action", "reason", "session_id"]


def _entries(path: Path) -> list[dict]:
    """The JSON objects in a log file, in order."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fr_log_01_record_writes_one_line_with_the_nine_keys_in_order(tmp_path: Path) -> None:
    """A finding is one JSON line with exactly the documented keys in order; layer and session come from the constructor."""
    log = audit.FindingsLog("gate", SID, state=tmp_path)
    assert log.record("secret_redacted", "api_key", "ab12", "anthropic.messages", "redact") is True
    raw = (tmp_path / "findings.jsonl").read_text(encoding="utf-8")
    assert raw.endswith("\n") and raw.count("\n") == 1
    entry = json.loads(raw)
    assert list(entry) == KEYS
    assert entry["layer"] == "gate" and entry["session_id"] == SID
    assert (entry["event_type"], entry["type"], entry["hash"]) == ("secret_redacted", "api_key", "ab12")
    assert (entry["source"], entry["action"], entry["reason"]) == ("anthropic.messages", "redact", None)
    assert datetime.fromisoformat(entry["time"]).tzinfo is not None


def test_fr_log_01_log_is_mode_0600_and_a_missing_folder_is_created(tmp_path: Path) -> None:
    """The log and its lock file are mode 0600, and a write creates the state folder when it is missing."""
    state = tmp_path / "missing" / ".harpo"
    assert audit.FindingsLog("read", SID, state=state).record("secret_redacted", "api_key", "ab12", "route-a", "redact") is True
    assert state.is_dir()
    assert stat.S_IMODE((state / "findings.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((state / "findings.jsonl.lock").stat().st_mode) == 0o600


def test_fr_log_01_a_repeated_pair_from_one_source_logs_once(tmp_path: Path) -> None:
    """Two identical (type, hash) pairs in one call, from the same source, log one finding and count 1."""
    log = audit.FindingsLog("gate", SID, state=tmp_path)
    assert log.redactions([("api_key", "ab12"), ("api_key", "ab12")], "route-a") == 1
    assert [e["hash"] for e in _entries(tmp_path / "findings.jsonl")] == ["ab12"]


def test_fr_log_01_same_pair_from_another_source_logs_again_and_a_repeat_call_logs_nothing(tmp_path: Path) -> None:
    """Return values 1, 1, 0: a new source logs again, and the same pairs from a source already seen log nothing."""
    log = audit.FindingsLog("gate", SID, state=tmp_path)
    pair = [("api_key", "ab12")]
    assert log.redactions(pair, "route-a") == 1
    assert log.redactions(pair, "route-b") == 1
    assert log.redactions(pair, "route-a") == 0
    assert [e["source"] for e in _entries(tmp_path / "findings.jsonl")] == ["route-a", "route-b"]


def test_fr_log_01_a_new_findings_log_object_logs_the_pair_again(tmp_path: Path) -> None:
    """Dedupe is per FindingsLog object (per session): a new object logs the same pair from the same source again."""
    pair = [("api_key", "ab12")]
    assert audit.FindingsLog("gate", SID, state=tmp_path).redactions(pair, "route-a") == 1
    assert audit.FindingsLog("gate", SID, state=tmp_path).redactions(pair, "route-a") == 1
    assert len(_entries(tmp_path / "findings.jsonl")) == 2


def test_fr_log_01_no_canary_value_reaches_the_findings_log(tmp_path: Path) -> None:
    """Redacted text is logged as placeholders only: no known-format canary value appears in findings.jsonl."""
    canaries = known_format()
    log = audit.FindingsLog("gate", SID, state=tmp_path)
    for kind, value in canaries.items():
        redacted = f"request body {value} end".replace(value, f"<<HARPO:{kind}:c0de>>")
        found = audit.placeholders(redacted)
        assert found == [(kind, "c0de")]
        assert log.redactions(found, "route-a") == 1
    content = (tmp_path / "findings.jsonl").read_text(encoding="utf-8")
    leaked = [kind for kind, value in canaries.items() if value in content]  # kinds only, never a value
    assert leaked == []
    assert len(_entries(tmp_path / "findings.jsonl")) == len(canaries)


def test_fr_log_01_state_none_writes_under_the_made_up_home() -> None:
    """With no state folder given, the log is $HOME/.harpo/findings.jsonl, and ~/.harpocrates is not created."""
    home = Path(os.environ["HOME"])
    assert audit.FindingsLog("hook", SID).record("secret_redacted", "api_key", "ab12", "route-a", "redact") is True
    entry = _entries(home / ".harpo" / "findings.jsonl")[0]
    assert (entry["layer"], entry["action"]) == ("hook", "redact")
    assert not (home / ".harpocrates").exists()


def test_fr_log_01_a_symlinked_state_folder_is_refused_and_its_target_stays_empty(tmp_path: Path) -> None:
    """A state folder that is a symlink makes record() return False, and the folder it points to is never written."""
    target = tmp_path / "target"
    target.mkdir()
    state = tmp_path / "state"
    state.symlink_to(target, target_is_directory=True)
    assert audit.FindingsLog("gate", SID, state=state).record("secret_redacted", "api_key", "ab12", "route-a", "redact") is False
    assert list(target.iterdir()) == []


def test_fr_log_01_a_state_folder_that_is_a_file_is_refused(tmp_path: Path) -> None:
    """A state folder that is a regular file makes record() return False, and the file is left as it was."""
    state = tmp_path / "state"
    state.write_text("not a folder", encoding="utf-8")
    assert audit.FindingsLog("gate", SID, state=state).record("secret_redacted", "api_key", "ab12", "route-a", "redact") is False
    assert state.read_text(encoding="utf-8") == "not a folder"


def test_fr_log_01_rotation_keeps_every_line_in_one_of_two_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At ROTATE_BYTES the log moves to findings.1.jsonl without losing a line, and only one previous file is kept."""
    monkeypatch.setattr(audit, "ROTATE_BYTES", 600)
    log = audit.FindingsLog("gate", SID, state=tmp_path)
    current, previous = tmp_path / "findings.jsonl", tmp_path / "findings.1.jsonl"
    written = 0
    while not previous.exists() and written < 20:
        assert log.record("secret_redacted", "api_key", "ab12", f"route-{written}", "redact") is True
        written += 1
    assert previous.exists()
    seen = [e["source"] for e in _entries(previous)] + [e["source"] for e in _entries(current)]
    assert seen == [f"route-{i}" for i in range(written)]
    assert current.stat().st_size < 600
    for i in range(written, written + 10):
        assert log.record("secret_redacted", "api_key", "ab12", f"route-{i}", "redact") is True
    names = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".jsonl")
    assert names == ["findings.1.jsonl", "findings.jsonl"]
    assert _entries(current)[-1]["source"] == f"route-{written + 9}"


def test_fr_log_01_placeholders_are_the_same_for_bytes_and_str_in_order() -> None:
    """bytes and str give the same (type, hash) pairs, in the order they appear, repeats included."""
    text = "a <<HARPO:github_token:ab12>> b <<HARPO:jwt:0f0f>> c <<HARPO:github_token:ab12>>"
    expected = [("github_token", "ab12"), ("jwt", "0f0f"), ("github_token", "ab12")]
    assert audit.placeholders(text) == expected
    assert audit.placeholders(text.encode("utf-8")) == expected


def test_fr_log_01_text_without_a_placeholder_gives_an_empty_list() -> None:
    """Text with no placeholder gives []; a malformed one (non-hex hash) is not a placeholder either."""
    assert audit.placeholders("no placeholder here, only <<HARPO:github_token:XYZ9>>") == []
    assert audit.placeholders(b"plain bytes") == []
