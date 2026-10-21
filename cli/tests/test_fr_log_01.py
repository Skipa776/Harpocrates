"""FR-LOG-01: the findings log holds one JSON line per finding, and never a secret value.

Covers the log's line format, file mode, rotation and refusal of symlinked or file-shaped state folders, and the
read tool and the hook writing what they redact or refuse to it. Every test uses a temp state folder, or the made-up
$HOME that the suite sets, so nothing touches the real ~/.harpo. The canaries come from the generated known-format set.
"""
from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path

import pytest
from gate_canaries import known_format

from Harpocrates import audit, hooks
from Harpocrates.read import Redactor

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

PLACEHOLDER = re.compile(r"<<HARPO:(\w+):([0-9a-f]{4})>>")


def _canaries(count: int) -> list[tuple[str, str]]:
    """The first `count` single-line generated canaries, as (kind, value) pairs; never printed."""
    pairs = [(kind, value) for kind, value in sorted(known_format().items()) if "\n" not in value]
    assert len(pairs) >= count
    return pairs[:count]


def _lines(state: Path) -> list[dict]:
    """The parsed lines of the state folder's findings log, or [] when nothing was written."""
    path = state / audit.FINDINGS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path: Path, text: str) -> str:
    """Write a text file and return its path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _snapshot(root: Path) -> set[Path]:
    """Every path under a folder, files and folders alike."""
    return set(root.rglob("*"))


def test_fr_log_01_safe_read_logs_one_redaction(tmp_path: Path) -> None:
    """A safe_read of a file with one canary logs one secret_redacted line for that file, with no value."""
    (kind, value), = _canaries(1)
    state = tmp_path / "state"
    path = _write(tmp_path / "creds.env", f"TOKEN={value}\n")
    redactor = Redactor(audit.FindingsLog("read", "s1", state))
    out = redactor.safe_read(path)
    lines = _lines(state)
    assert len(lines) == 1
    entry = lines[0]
    assert entry["event_type"] == "secret_redacted" and entry["layer"] == "read"
    assert entry["source"] == str(path)
    assert entry["type"] is not None
    placeholder = PLACEHOLDER.search(out)
    assert placeholder is not None
    assert entry["type"] == placeholder.group(1) and entry["hash"] == placeholder.group(2)
    assert re.fullmatch(r"[0-9a-f]{4}", entry["hash"])
    assert value not in (state / audit.FINDINGS).read_text(encoding="utf-8")
    assert value not in out, "the read output itself must carry only placeholders"


def test_fr_log_01_repeat_read_adds_no_line_new_file_adds_one(tmp_path: Path) -> None:
    """A repeat safe_read of the same file adds no line; the same canary in a second file adds one for that file."""
    (_, value), = _canaries(1)
    state = tmp_path / "state"
    first = _write(tmp_path / "a.env", f"TOKEN={value}\n")
    second = _write(tmp_path / "b.env", f"TOKEN={value}\n")
    redactor = Redactor(audit.FindingsLog("read", "s1", state))
    redactor.safe_read(first)
    assert len(_lines(state)) == 1
    redactor.safe_read(first)
    assert len(_lines(state)) == 1
    redactor.safe_read(second)
    lines = _lines(state)
    assert len(lines) == 2
    assert [entry["source"] for entry in lines] == [first, second]


def test_fr_log_01_literal_placeholder_is_not_logged(tmp_path: Path) -> None:
    """A file that holds only a literal placeholder and no secret writes no finding."""
    state = tmp_path / "state"
    path = _write(tmp_path / "notes.txt", "see <<HARPO:api_token:abcd>> for the token\n")
    redactor = Redactor(audit.FindingsLog("read", "s1", state))
    redactor.safe_read(path)
    assert _lines(state) == []


def test_fr_log_01_safe_grep_logs_each_file(tmp_path: Path) -> None:
    """A safe_grep over a folder with two canary files logs one line per file, each with its own source."""
    (kind_a, value_a), (kind_b, value_b) = _canaries(2)
    state = tmp_path / "state"
    folder = tmp_path / "repo"
    file_a = _write(folder / "a.env", f"TOKEN={value_a}\n")
    file_b = _write(folder / "sub" / "b.env", f"TOKEN={value_b}\n")
    redactor = Redactor(audit.FindingsLog("read", "s1", state))
    out = redactor.safe_grep("TOKEN", str(folder))
    assert "TOKEN=<<HARPO:" in out
    lines = _lines(state)
    assert len(lines) == 2
    assert sorted(entry["source"] for entry in lines) == sorted([file_a, file_b])
    assert all(entry["event_type"] == "secret_redacted" and entry["layer"] == "read" for entry in lines)
    text = (state / audit.FINDINGS).read_text(encoding="utf-8")
    assert value_a not in text and value_b not in text


def test_fr_log_01_redactor_without_log_writes_nothing(tmp_path: Path) -> None:
    """A Redactor with no log reads and redacts without writing anything under tmp_path or the made-up HOME."""
    (_, value), = _canaries(1)
    path = _write(tmp_path / "creds.env", f"TOKEN={value}\n")
    before = _snapshot(tmp_path)
    home_state = Path(os.environ["HOME"]) / ".harpo"
    redactor = Redactor()
    assert "<<HARPO:" in redactor.safe_read(path)
    assert "<<HARPO:" in redactor.safe_grep("TOKEN", str(tmp_path))
    assert _snapshot(tmp_path) == before
    assert not home_state.exists()


def _hook_event(path: Path, cwd: Path) -> dict:
    """A PreToolUse event for a built-in Read of one file, shaped like the events in test_hooks.py."""
    return {"tool_name": "Read", "tool_input": {"file_path": str(path)}, "cwd": str(cwd),
            "session_id": "sess-1", "hook_event_name": "PreToolUse"}


def test_fr_log_01_hook_refused_read_logs_each_type(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A refused built-in Read prints the deny and logs one read_refused hook line per secret type, with no value."""
    (kind, value), = _canaries(1)
    path = Path(_write(tmp_path / "creds.env", f"TOKEN={value}\n"))
    assert hooks.decide(json.dumps(_hook_event(path, tmp_path))) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    state = Path(os.environ["HOME"]) / ".harpo"
    lines = _lines(state)
    assert lines, "the refusal must be logged"
    assert len(lines) == len({entry["type"] for entry in lines}), "one line per secret type"
    assert {entry["type"] for entry in lines} == {kind for kind, _ in audit.placeholders(Redactor().safe_read(str(path)))}
    for entry in lines:
        assert entry["event_type"] == "read_refused" and entry["layer"] == "hook"
        assert entry["hash"] is None and entry["action"] == "refuse"
        assert entry["session_id"] == "sess-1"
        assert "creds.env" in entry["source"] and "(Read)" in entry["source"]
    assert value not in (state / audit.FINDINGS).read_text(encoding="utf-8")


def test_fr_log_01_hook_allowed_read_writes_no_finding(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """An allowed built-in Read of a file with no secret writes no findings log."""
    path = Path(_write(tmp_path / "README.md", "plain text, nothing to hide\n"))
    assert hooks.decide(json.dumps(_hook_event(path, tmp_path))) == 0
    assert capsys.readouterr().out == ""
    assert not (Path(os.environ["HOME"]) / ".harpo" / audit.FINDINGS).exists()


def test_fr_log_01_hook_symlinked_state_folder_refuses_and_writes_nothing(tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture) -> None:
    """With $HOME/.harpo a symlink, the Read is still denied and the link's target gets no findings log."""
    (_, value), = _canaries(1)
    path = Path(_write(tmp_path / "creds.env", f"TOKEN={value}\n"))
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = Path(os.environ["HOME"]) / ".harpo"
    link.symlink_to(target, target_is_directory=True)
    hooks.decide(json.dumps(_hook_event(path, tmp_path)))
    out = capsys.readouterr().out
    assert out and json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert not (target / audit.FINDINGS).exists()
