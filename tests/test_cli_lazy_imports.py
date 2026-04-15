"""
Guardrail tests for Harpocrates.cli import discipline (PRD-01 Task 6).

The CLI ``--help`` / ``version`` paths must not pay for heavy optional
dependencies (ML stacks, training pipeline, FastAPI server). These tests
run the CLI module in a fresh subprocess so they see the real, cold
import graph — ``sys.modules`` in this test process is already polluted
by earlier tests.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Modules that MUST NOT be imported as a side-effect of importing
# ``Harpocrates.cli``. Keep this list tight: the whole point is to fail
# loudly the moment someone adds a heavy top-level import.
_FORBIDDEN_COLD_IMPORTS = (
    "xgboost",
    "lightgbm",
    "sklearn",
    "numpy",
    "pandas",
    "torch",
    "tensorflow",
    "uvicorn",
    "fastapi",
    "Harpocrates.ml",
    "Harpocrates.ml.ensemble",
    "Harpocrates.ml.features",
    "Harpocrates.ml.verifier",
    "Harpocrates.training",
    "Harpocrates.training.train",
    "Harpocrates.training.dataset",
    "Harpocrates.training.generators.generate_data",
    "Harpocrates.api.main",
)


_SUBPROCESS_TIMEOUT_S = 30


def _run_in_subprocess(snippet: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter and capture stdout/stderr.

    A timeout guards against a hung child silently stalling the whole suite
    (e.g. if a leaked import triggers a network call or device probe).
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
        timeout=_SUBPROCESS_TIMEOUT_S,
    )


def _assert_none_loaded(loaded: set[str]) -> None:
    leaked = sorted(m for m in _FORBIDDEN_COLD_IMPORTS if m in loaded)
    assert not leaked, (
        "The following modules were imported as a side-effect of loading "
        "Harpocrates.cli. Move them inside the command function that "
        f"actually needs them: {leaked}"
    )


def test_importing_cli_does_not_load_ml_or_api_modules() -> None:
    """Plain ``import Harpocrates.cli`` must stay lightweight."""
    result = _run_in_subprocess(
        """
        import sys
        import Harpocrates.cli  # noqa: F401
        print("\\n".join(sorted(sys.modules.keys())))
        """
    )

    assert result.returncode == 0, result.stderr
    loaded = set(result.stdout.splitlines())
    _assert_none_loaded(loaded)


def test_top_level_help_does_not_load_ml_or_api_modules() -> None:
    """``harpocrates --help`` (top-level) must not drag in ML deps."""
    result = _run_in_subprocess(
        """
        import sys
        from typer.testing import CliRunner
        from Harpocrates.cli import app

        runner = CliRunner()
        res = runner.invoke(app, ["--help"])
        assert res.exit_code == 0, res.output
        print("\\n".join(sorted(sys.modules.keys())))
        """
    )

    assert result.returncode == 0, result.stderr
    loaded = set(result.stdout.splitlines())
    _assert_none_loaded(loaded)


def test_scan_help_does_not_load_ml_or_api_modules() -> None:
    """``harpocrates scan --help`` must not drag in ML deps.

    Run in its own subprocess (separate from the top-level --help test) so
    that a leak in either path is attributed to the correct command.
    """
    result = _run_in_subprocess(
        """
        import sys
        from typer.testing import CliRunner
        from Harpocrates.cli import app

        runner = CliRunner()
        res = runner.invoke(app, ["scan", "--help"])
        assert res.exit_code == 0, res.output
        print("\\n".join(sorted(sys.modules.keys())))
        """
    )

    assert result.returncode == 0, result.stderr
    loaded = set(result.stdout.splitlines())
    _assert_none_loaded(loaded)


def test_scan_command_body_without_ml_does_not_load_ml_modules() -> None:
    """``scan <path>`` (no --ml) must not import ML/training/api code.

    --help only exercises typer's argument parser; this test exercises the
    actual ``scan`` function body via a nonexistent path that exits early
    at the path-not-found check. A top-level ML import accidentally added
    above the ``if use_ml:`` guard would slip past the --help-only test.
    """
    result = _run_in_subprocess(
        """
        import sys
        from typer.testing import CliRunner
        from Harpocrates.cli import app

        runner = CliRunner()
        res = runner.invoke(app, ["scan", "/nonexistent/path/to/file"])
        # Path not found exits 2 (CLI argument error), but we don't care
        # about the exit code — only that no forbidden module was loaded.
        print("\\n".join(sorted(sys.modules.keys())))
        """
    )

    assert result.returncode == 0, result.stderr
    loaded = set(result.stdout.splitlines())
    _assert_none_loaded(loaded)


def test_version_command_does_not_load_ml_or_api_modules() -> None:
    """``harpocrates version`` must not trigger optional-dep loading."""
    result = _run_in_subprocess(
        """
        import sys
        from typer.testing import CliRunner
        from Harpocrates.cli import app

        runner = CliRunner()
        res = runner.invoke(app, ["version"])
        assert res.exit_code == 0, res.output
        print("\\n".join(sorted(sys.modules.keys())))
        """
    )

    assert result.returncode == 0, result.stderr
    loaded = set(result.stdout.splitlines())
    _assert_none_loaded(loaded)


@pytest.mark.performance
def test_cli_help_is_fast() -> None:
    """
    ``--help`` should complete well under one second on a cold start.

    The threshold is deliberately generous (2s) to avoid flaky CI, but
    anything above that indicates a heavy import has slipped back in.
    """
    import time

    start = time.perf_counter()
    result = _run_in_subprocess(
        """
        from typer.testing import CliRunner
        from Harpocrates.cli import app

        runner = CliRunner()
        res = runner.invoke(app, ["--help"])
        assert res.exit_code == 0, res.output
        """
    )
    elapsed = time.perf_counter() - start

    assert result.returncode == 0, result.stderr
    assert elapsed < 2.0, f"--help took {elapsed:.2f}s (limit: 2.0s)"
