"""FR-LOG-01: the read tool and the hook write what they redact or refuse to the findings log, without any secret value."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from gate_canaries import known_format

from Harpocrates import audit, hooks
from Harpocrates.read import Redactor

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
