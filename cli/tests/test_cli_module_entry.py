"""`python -m Harpocrates.cli` runs the CLI, so a plugin can name the interpreter instead of trusting PATH."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_fr_install_01_module_entry_runs_the_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    done = subprocess.run(
        [sys.executable, "-m", "Harpocrates.cli", "allow", "--list", "--json"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"entries": []}
