"""FR-CORE-05: a file over max_file_size is skipped with an error, never scanned only up to the cap."""
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


def _secret() -> str:
    """The generated fake GitHub token used as the canary (never printed)."""
    return known_format(5)["github_token"]


def _big_file(tmp_path: Path) -> tuple[Path, str]:
    """A file of about 1.5 MiB of filler with the canary on its last line."""
    value = _secret()
    path = tmp_path / "big.env"
    body = FILLER * (int(1.5 * MIB) // len(FILLER))
    path.write_text(body + f"GITHUB_TOKEN={value}\n", encoding="utf-8")
    return path, value


def _skip_without_rust(engine: str) -> None:
    """Skip the rust engine case when no scanner binary is built."""
    if engine == "rust" and RustScannerBackend.discover(required=False) is None:
        pytest.skip("rust scanner binary is not built")


@pytest.mark.parametrize("engine", ENGINES)
def test_fr_core_05_over_cap_skipped(tmp_path: Path, engine: str) -> None:
    """A file over the 1 MiB cap yields no findings and one Skipped large file error."""
    _skip_without_rust(engine)
    path, _ = _big_file(tmp_path)

    result = scanner.scan_file(path, max_file_size=MIB, engine=engine)

    assert result.findings == []
    assert result.errors == [f"Skipped large file: {path}"]


@pytest.mark.parametrize("engine", ENGINES)
def test_fr_core_05_fits_under_larger_cap(tmp_path: Path, engine: str) -> None:
    """The same file under a 4 MiB cap is scanned and the trailing secret is found."""
    _skip_without_rust(engine)
    path, _ = _big_file(tmp_path)

    result = scanner.scan_file(path, max_file_size=4 * MIB, engine=engine)

    assert len(result.findings) == 1
    assert result.errors == []


@pytest.mark.parametrize("engine", ENGINES)
def test_fr_core_05_small_file_not_rejected(tmp_path: Path, engine: str) -> None:
    """A file under the cap with the secret at its start is scanned with no error."""
    _skip_without_rust(engine)
    value = _secret()
    path = tmp_path / "small.env"
    path.write_text(f"GITHUB_TOKEN={value}\n" + FILLER * 10, encoding="utf-8")

    result = scanner.scan_file(path, max_file_size=MIB, engine=engine)

    assert len(result.findings) == 1
    assert result.errors == []


def test_fr_core_05_cli_exits_2_on_skip(tmp_path: Path) -> None:
    """harpo scan with --max-size 1 exits 2 on the skipped file and does not print the secret."""
    path, value = _big_file(tmp_path)

    result = runner.invoke(app, ["scan", str(path), "--max-size", "1"])

    assert result.exit_code == 2
    assert "Skipped large file" in result.output
    assert value not in result.output


def test_fr_core_05_cli_scans_when_cap_fits(tmp_path: Path) -> None:
    """harpo scan with --max-size 4 reports the secret (exit 1) and does not print it."""
    path, value = _big_file(tmp_path)

    result = runner.invoke(app, ["scan", str(path), "--max-size", "4"])

    assert result.exit_code == 1
    assert "Skipped large file" not in result.output
    assert value not in result.output


def test_fr_core_05_folder_scan_skips_same_file(tmp_path: Path) -> None:
    """A folder scan with the same 1 MiB cap also reports the file as an error (each engine words it its own way)."""
    path, _ = _big_file(tmp_path)

    result = scanner.scan_directory(tmp_path, max_file_size=MIB)

    assert result.findings == []
    assert len(result.errors) == 1 and str(path) in result.errors[0] and "large file" in result.errors[0].lower()
