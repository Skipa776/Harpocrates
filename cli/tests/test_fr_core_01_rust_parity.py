"""FR-CORE-01: the Rust core gives the same regex + entropy findings as the Python engine.

Skips when the binary isn't built. CI sets HARPOCRATES_REQUIRE_RUST=1 so a
missing binary fails instead of silently testing Python twice.
"""

import os
from pathlib import Path

import pytest

from Harpocrates.core.rust_backend import RustScannerBackend
from Harpocrates.core.scanner import scan_directory

SAMPLES = Path(__file__).resolve().parents[2] / "docs" / "samples" / "samples"


@pytest.fixture(scope="module")
def rust_available():
    if RustScannerBackend.discover() is None:
        if os.environ.get("HARPOCRATES_REQUIRE_RUST") == "1":
            pytest.fail("HARPOCRATES_REQUIRE_RUST=1 but the Rust scanner binary was not found")
        pytest.skip("Rust scanner binary not built")


def _keys(result):
    return sorted((Path(f.file).name, f.line, f.type, f.token) for f in result.findings)


def test_fr_core_01_rust_matches_python(rust_available):
    rust = scan_directory(SAMPLES, engine="rust")
    python = scan_directory(SAMPLES, engine="python")
    assert rust.errors == []  # Rust scan failure or per-file read errors
    assert rust.findings, "samples should contain findings"
    assert _keys(rust) == _keys(python)
