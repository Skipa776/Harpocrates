"""FR-CORE-05: the size cap bounds memory, never how much of a file is scanned.

Before Oct 21, 2026 a single file over the cap was scanned only up to it, silently, so a secret after the cap was
missed with exit 0 (also through the pre-commit hook, which passes file names). Every file is now scanned whole.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from cli_entry import CommandNameRunner
from gate_canaries import known_format

from Harpocrates.cli import app
from Harpocrates.core import scanner
from Harpocrates.core.rust_backend import RustScannerBackend

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
