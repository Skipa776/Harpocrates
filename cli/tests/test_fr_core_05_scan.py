"""FR-CORE-05: every file is scanned whole, and the size cap bounds only memory.

Before Oct 21, 2026 a single file over the cap was scanned only up to it, silently, so a secret after the cap was
missed with exit 0 (also through the pre-commit hook, which passes file names). Scanning runs line by line in bounded
pieces (a 1-byte cap forces the smallest pieces), for both the python and rust engines.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from cli_entry import CommandNameRunner
from gate_canaries import known_format

from Harpocrates.cli import app
from Harpocrates.core import scanner
from Harpocrates.core.detector import detect_file, detect_text
from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.core.rust_backend import RustScannerBackend
from Harpocrates.utils.file_utils import PIECE_OVERLAP, count_lines, iter_line_pieces, piece_chars

MIB = 1024 * 1024
FILLER = "lorem ipsum dolor sit amet\n"
ENGINES = ("python", "rust")
runner = CommandNameRunner()


def _big_file(tmp_path: Path) -> tuple[Path, str]:
    """A file of about 1.5 MiB of filler with a generated fake GitHub token (never printed) on its last line."""
    value = known_format(5)["github_token"]
    path = tmp_path / "big.env"
    path.write_text(FILLER * (int(1.5 * MIB) // len(FILLER)) + f"GITHUB_TOKEN={value}\n", encoding="utf-8")
    return path, value


def _skip_without_rust(engine: str) -> None:
    """Skip the rust engine case when no scanner binary is built."""
    if engine == "rust" and RustScannerBackend.discover(required=False) is None:
        pytest.skip("rust scanner binary is not built")


@pytest.mark.parametrize("engine", ENGINES)
def test_fr_core_05_a_file_over_the_cap_is_scanned_to_its_end(tmp_path: Path, engine: str) -> None:
    """With a 1 MiB cap, the secret on the last line of a 1.5 MiB file is found, with its line number."""
    _skip_without_rust(engine)
    path, _ = _big_file(tmp_path)
    result = scanner.scan_file(path, memory_cap=MIB, engine=engine)
    assert result.errors == []
    assert [(f.type, f.line) for f in result.findings] == [("GITHUB_PAT", result.total_lines)]


@pytest.mark.parametrize("engine", ENGINES)
def test_fr_core_05_a_folder_scan_also_scans_it_to_its_end(tmp_path: Path, engine: str) -> None:
    """A folder scan with the same cap finds it too: the file is no longer skipped as large."""
    _skip_without_rust(engine)
    _big_file(tmp_path)
    result = scanner.scan_directory(tmp_path, memory_cap=MIB, engine=engine)
    assert result.errors == []
    assert [f.type for f in result.findings] == ["GITHUB_PAT"]


def test_fr_core_05_cli_reports_the_secret_past_the_cap(tmp_path: Path) -> None:
    """`harpo scan FILE --max-size 1` exits 1 with the finding, and never prints the value."""
    path, value = _big_file(tmp_path)
    result = runner.invoke(app, ["scan", str(path), "--max-size", "1"])
    assert result.exit_code == 1
    assert "Skipped large file" not in result.output and value not in result.output


LONG_LINE = 700_000
# The token starts at 10, at 10 characters before the first piece boundary (it crosses it), and mid-line.
TOKEN_STARTS = (10, piece_chars(1) - 10, 400_000)
FILLER_LINE = "lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor.\n"
MEMORY_SCRIPT = """
import json, resource, sys
cli_path, file_path, engine = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, cli_path)
from Harpocrates.core import scanner
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = scanner.scan_file(file_path, engine=engine)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(json.dumps({"findings": len(result.findings), "lines": [f.line for f in result.findings],
                  "errors": len(result.errors), "self_growth_kib": after - before, "children_peak_kib": children}))
"""


def _regex_keys(findings: list[Finding]) -> list[tuple[str, int, int, str]]:
    """Return the REGEX findings as (type, line, token_start, token) sorted by place; a text scan's None line is 1."""
    keys = [(str(f.type), f.line or 1, f.token_start, f.token) for f in findings if f.evidence == EvidenceType.REGEX]
    return sorted(keys, key=lambda key: (key[1], key[2]))


def _long_line_file(tmp_path: Path) -> tuple[Path, str]:
    """Return a file of one 700_000-char line with the fake GitHub token at TOKEN_STARTS, and that line."""
    token = known_format(9)["github_token"]
    chars = list(("lorem ipsum dolor sit amet " * 26_000)[:LONG_LINE])
    for start in TOKEN_STARTS:
        chars[start - 1 : start + len(token) + 1] = f" {token} "  # spaces keep the token's word boundaries
    line = "".join(chars)
    path = tmp_path / "long_line.txt"
    path.write_text(line + "\n", encoding="utf-8")
    return path, line


def test_fr_core_05_pieces_cover_each_line_with_overlap(tmp_path: Path) -> None:
    """Pieces of each line cover it exactly, stay within piece_chars(1), overlap by PIECE_OVERLAP; count_lines agrees."""
    lines = ["short one", "y" * 600_000, "", "another short line", "z" * 1_000]
    path = tmp_path / "pieces.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pieces: dict[int, list[tuple[int, str]]] = {}
    for lineno, offset, text in iter_line_pieces(path, memory_cap=1):
        pieces.setdefault(lineno, []).append((offset, text))
    assert sorted(pieces) == list(range(1, len(lines) + 1))
    for lineno, line in enumerate(lines, start=1):
        found = pieces[lineno]
        assert found[0][0] == 0
        assert found[-1][0] + len(found[-1][1]) == len(line)
        for offset, text in found:
            assert line[offset : offset + len(text)] == text
            assert len(text) <= piece_chars(1)
        for (offset_a, text_a), (offset_b, _) in zip(found, found[1:]):
            assert offset_b + PIECE_OVERLAP == offset_a + len(text_a)
    assert len(pieces[2]) > 1  # the 600_000-char line really was cut
    assert count_lines(path, memory_cap=1) == len(lines)


def test_fr_core_05_detect_file_finds_token_across_boundary_once(tmp_path: Path) -> None:
    """A token across a piece boundary is found exactly once, at the same offset as a whole-text detection."""
    path, line = _long_line_file(tmp_path)
    want = _regex_keys(detect_text(line))
    got = _regex_keys(detect_file(path, memory_cap=1))
    assert [key[2] for key in want] == list(TOKEN_STARTS)
    assert got == want
    assert len(got) == len(TOKEN_STARTS)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_fr_core_05_scan_file_in_pieces_matches_whole_text(tmp_path: Path, engine: str) -> None:
    """scan_file with a 1-byte cap reports the same regex findings as a whole-text detection, for each engine."""
    _skip_without_rust(engine)
    path, line = _long_line_file(tmp_path)
    want = [key[:3] for key in _regex_keys(detect_text(line))]
    result = scanner.scan_file(path, memory_cap=1, engine=engine)
    assert result.errors == []
    got = [key[:3] for key in _regex_keys(result.findings)]
    assert got == want
    assert len(got) == len(TOKEN_STARTS)


@pytest.mark.integration
@pytest.mark.parametrize("engine", ["rust", "python"])
def test_fr_core_05_memory_scan_of_100mb_stays_under_64mib(tmp_path: Path, engine: str) -> None:
    """A 100 MB file with the token on its last line gives one finding on that line and under 64 MiB of growth."""
    _skip_without_rust(engine)
    cli_path = str(Path(__file__).resolve().parents[1])
    path = tmp_path / "big.txt"
    token = known_format(9)["github_token"]
    filler_lines = 100 * MIB // len(FILLER_LINE)
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(filler_lines):
            handle.write(FILLER_LINE)
        handle.write(f"token = {token}\n")
    proc = subprocess.run(
        [sys.executable, "-I", "-c", MEMORY_SCRIPT, cli_path, str(path), engine],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"child exited {proc.returncode}: {proc.stderr[-300:]}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["errors"] == 0
    assert payload["findings"] == 1
    assert payload["lines"] == [filler_lines + 1]
    growth_kib = payload["self_growth_kib"] if engine == "python" else payload["children_peak_kib"]
    assert growth_kib * 1024 < 64 * MIB, f"{engine} grew {growth_kib} KiB"
