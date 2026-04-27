"""Packaging smoke test: build wheel, install in fresh venv, assert ML pipeline loads."""
from __future__ import annotations

import glob
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "test_tokens.py"


@pytest.mark.integration
def test_wheel_installs_with_ml_pipeline() -> None:
    """
    Build the wheel, install in a clean venv, and confirm the ML
    pipeline loads from bundled model files (not from source tree).

    Fails if:
    - package-data is missing from pyproject.toml (models not bundled)
    - model paths resolve relative to source tree instead of installed package
    - ML dependencies (xgboost, lightgbm) are absent from wheel metadata
    """
    # Step 1: Build wheel
    build_result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert build_result.returncode == 0, (
        f"Wheel build failed:\n{build_result.stdout}\n{build_result.stderr}"
    )

    # Step 2: Locate most-recent wheel
    wheels = sorted(
        glob.glob(str(PROJECT_ROOT / "dist" / "harpocrates-*.whl")),
        key=lambda p: Path(p).stat().st_mtime,
    )
    assert wheels, "No wheel found in dist/ after build"
    wheel = wheels[-1]

    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"

        # Step 3: Create isolated venv
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            check=True,
            capture_output=True,
            text=True,
        )

        pip = venv / "bin" / "pip"
        harpo = venv / "bin" / "harpocrates"

        # Step 4: Install from wheel — NOT from source tree
        install_result = subprocess.run(
            [str(pip), "install", wheel],
            capture_output=True,
            text=True,
        )
        assert install_result.returncode == 0, (
            f"pip install failed:\n{install_result.stdout}\n{install_result.stderr}"
        )

        # Step 5: Run CLI with ML enabled at the shipped default threshold
        result = subprocess.run(
            [
                str(harpo), "scan",
                str(FIXTURE),
                "--ml",
                "--ml-threshold", "0.19",
            ],
            capture_output=True,
            text=True,
        )

        # Step 6: ML pipeline must be active (bundled models loaded)
        assert "ML verification enabled" in result.stdout, (
            f"ML pipeline not active after wheel install. "
            f"Models may be missing from wheel.\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )

        # Step 7: At least one secret must be detected in the fixture
        assert result.returncode != 0, (
            f"Expected secrets detected (non-zero exit) but got 0.\n"
            f"stdout: {result.stdout!r}"
        )
