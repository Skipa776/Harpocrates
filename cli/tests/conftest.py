"""Fixtures shared by every test.

SEC-08: paths.home() reads the password database, never $HOME, so a test that moves HOME to a temp folder would
reach the real home folder. The autouse fixture below makes paths.home() follow $HOME again, which keeps every
existing test inside the fixture home it sets. The SEC-08 tests opt out (pytest.mark.passwd_home) and fake the
password database instead. A child process cannot be patched from here: child_python() builds a command that
makes the same patch before it runs a module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from Harpocrates import paths

_RUN_MODULE = (
    "import pathlib, os, runpy, sys\n"
    "import Harpocrates.paths\n"
    "Harpocrates.paths.home = lambda: pathlib.Path(os.environ['HOME'])\n"
    "runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)\n"
)


@pytest.fixture(autouse=True)
def _home_follows_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make paths.home() return $HOME for this test, unless it is marked passwd_home."""
    if request.node.get_closest_marker("passwd_home") is None:
        monkeypatch.setattr(paths, "home", lambda: Path.home())


@pytest.fixture
def child_python():
    """A function (module, *args) -> argv that runs `python -m module args` with paths.home() following $HOME."""
    def build(module: str, *args: str) -> list[str]:
        return [sys.executable, "-c", _RUN_MODULE, module, *args]

    return build
