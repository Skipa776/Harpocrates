"""FR-CORE-05: the read tool and the hook redact a file over MAX_BYTES in chunks, never cutting a secret in two.

A large file needs a line range; its chunks overlap by the context lines, a private-key block is never split, and a line
or block too long for a chunk is refused with TooLargeToScanError, which is not a ValueError.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest
from gate_canaries import canaries, known_format

from Harpocrates import hooks, read
from Harpocrates.read import Redactor

ORIGINAL = read.file_chunks
FILLER = "lorem ipsum dolor sit amet"


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat files over 4096 bytes as large, and make chunks about 128 KiB."""
    monkeypatch.setattr(read, "MAX_BYTES", 4096)
    monkeypatch.setattr(read, "file_chunks", functools.partial(ORIGINAL, memory_cap=1))


def _write(path: Path, lines: list[str]) -> int:
    """Write the lines as a text file and return how many there are."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def test_fr_core_05_large_file_without_range_is_refused(tmp_path: Path, small_chunks: None) -> None:
    """A file over MAX_BYTES read without start_line and end_line raises ValueError naming start_line."""
    path = tmp_path / "big.txt"
    _write(path, [f"{FILLER} {i:06d}" for i in range(6000)])
    with pytest.raises(ValueError, match="start_line"):
        Redactor().safe_read(str(path))


def test_fr_core_05_secret_on_first_and_last_lines_is_redacted(tmp_path: Path, small_chunks: None) -> None:
    """A secret on the first line and on the last line of a large file is redacted in the ranges that show them."""
    v = known_format(11)["github_token"]
    path = tmp_path / "big.env"
    lines = [f"token: {v} plain"] + [f"{FILLER} {i:06d}" for i in range(1, 6000)] + [f"GITHUB_TOKEN={v}"]
    n = _write(path, lines)
    redactor = Redactor()
    head = redactor.safe_read(str(path), 1, 3)
    tail = redactor.safe_read(str(path), n - 2, n)
    for out in (head, tail):
        assert "<<HARPO:" in out
        assert v not in out


def test_fr_core_05_private_key_across_a_chunk_boundary_is_not_shown(tmp_path: Path, small_chunks: None) -> None:
    """A private-key block placed across a chunk boundary stays in one chunk and is redacted: no line of its body is read
    back."""
    block = canaries(11)["private_key"].strip("\n")
    body = [line for line in block.splitlines()[1:-1] if line.strip()]
    assert body
    limit = 2 * read.PIECE_OVERLAP  # the chunk size memory_cap=1 gives
    line = f"{FILLER} 000000\n"
    filler = [f"{FILLER} {i:06d}" for i in range((limit - len(block) // 2) // len(line))]  # the boundary falls mid-block
    path = tmp_path / "keys.txt"
    n = _write(path, filler + block.splitlines() + [f"{FILLER} end"])
    starts = [first for first, text in read.file_chunks(path) if "PRIVATE KEY" in text]
    assert len(starts) == 1, "the block was split across chunks"
    assert sum(len(text) for first, text in read.file_chunks(path) if first == starts[0]) > limit  # it was extended
    out = Redactor().safe_read(str(path), len(filler) + 1, n)
    assert not any(line in out for line in body)


def test_fr_core_05_single_line_far_into_large_file_keeps_its_number(tmp_path: Path, small_chunks: None) -> None:
    """A one-line range deep in a large file returns exactly one numbered line, with its own text."""
    path = tmp_path / "big.txt"
    lines = [f"{FILLER} {i:06d}" for i in range(1, 20001)]
    _write(path, lines)
    n = 15000
    out = Redactor().safe_read(str(path), n, n)
    matched = [line for line in out.splitlines() if line.startswith(f"{n:>6}\t")]
    assert len(matched) == 1
    assert matched[0].endswith(lines[n - 1])


def test_fr_core_05_line_longer_than_a_chunk_raises_too_large(tmp_path: Path, small_chunks: None) -> None:
    """A line of 300000 characters in a 128 KiB chunk limit raises TooLargeToScanError, which is not a ValueError."""
    path = tmp_path / "long.txt"
    path.write_text("x" * 300_000 + "\n", encoding="utf-8")
    with pytest.raises(read.TooLargeToScanError):
        Redactor().safe_read(str(path), 1, 1)
    assert not issubclass(read.TooLargeToScanError, ValueError)


def test_fr_core_05_grep_finds_secret_on_last_line_of_large_file(tmp_path: Path, small_chunks: None) -> None:
    """safe_grep over a folder finds the secret on the last line of a large file once, redacted, with its number."""
    v = known_format(11)["github_token"]
    folder = tmp_path / "repo"
    folder.mkdir()
    lines = [f"{FILLER} {i:06d}" for i in range(1, 6000)] + [f"GITHUB_TOKEN={v}"]
    _write(folder / "big.env", lines)
    out = Redactor().safe_grep("GITHUB_TOKEN", str(folder))
    matches = [line for line in out.splitlines() if line.startswith("big.env:")]
    assert len(matches) == 1
    assert matches[0].startswith(f"big.env:{len(lines)}:")
    assert "<<HARPO:" in matches[0]
    assert v not in out


def test_fr_core_05_whole_large_file_output_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for all lines of a 200 KB file, past the 4096-byte output cap, raises ValueError about fewer lines."""
    monkeypatch.setattr(read, "MAX_BYTES", 4096)
    path = tmp_path / "big.txt"
    _write(path, [f"{FILLER} {i:06d}" for i in range(6000)])
    with pytest.raises(ValueError, match="fewer lines"):
        Redactor().safe_read(str(path), 1, 10**6)


def test_fr_core_05_hook_large_file_clean_passes_and_secret_is_denied(
    tmp_path: Path, small_chunks: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Read hook exits 0 silently for a large clean file, and denies a large file with a secret on its last line."""
    v = known_format(11)["github_token"]
    clean = tmp_path / "clean.log"
    _write(clean, [f"{FILLER} {i:06d}" for i in range(6000)])
    dirty = tmp_path / "dirty.env"
    _write(dirty, [f"{FILLER} {i:06d}" for i in range(6000)] + [f"GITHUB_TOKEN={v}"])

    def event(path: Path) -> str:
        """The PreToolUse Read event for a file."""
        return json.dumps({"tool_name": "Read", "tool_input": {"file_path": str(path)}, "cwd": str(tmp_path),
                           "session_id": "c05", "hook_event_name": "PreToolUse"})

    assert hooks.decide(event(clean)) == 0
    assert capsys.readouterr().out == ""
    assert hooks.decide(event(dirty)) == 0
    out = capsys.readouterr().out
    decision = json.loads(out)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "api_token" in decision["permissionDecisionReason"]
    assert v not in out
