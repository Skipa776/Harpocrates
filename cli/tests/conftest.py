"""Fixtures shared by every test.

SEC-08: paths.home() reads the password database, never $HOME, so a test that moves HOME to a temp folder would
reach the real home folder. The autouse fixture below makes paths.home() follow $HOME again, which keeps every
existing test inside the fixture home it sets. The SEC-08 tests opt out (pytest.mark.passwd_home) and fake the
password database instead. A child process cannot be patched from here: child_python() builds a command that
makes the same patch before it runs a module.

Two more nets keep the suite off the real home folder: every test starts with a made-up $HOME, and a session fixture
fails the run if the real ~/.harpo, ~/.harpocrates, ~/.claude/settings.json or ~/.codex/config.toml changed
(_real_home.py; test_sec_08_real_home_guard.py).
"""
from __future__ import annotations

import pwd
import sys
from pathlib import Path
from types import SimpleNamespace

import _real_home
import pytest

from Harpocrates import paths

_RUN_MODULE = (
    "import pathlib, os, runpy, sys\n"
    "import Harpocrates.paths\n"
    "Harpocrates.paths.home = lambda: pathlib.Path(os.environ['HOME'])\n"
    "runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)\n"
)


@pytest.fixture(scope="session", autouse=True)
def _real_home_untouched():
    """Fail the run if any test changed the real home's ~/.harpo, ~/.harpocrates, ~/.claude/settings.json or
    ~/.codex/config.toml (a read-only snapshot, mtime, mode, size and SHA-256, before the first test and after the last)."""
    before = _real_home.snapshot()
    yield
    moved = _real_home.changed(before, _real_home.snapshot())
    assert not moved, f"a test touched the real home folder: {moved}"


@pytest.fixture(autouse=True)
def _isolated_home(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
                   tmp_path_factory: pytest.TempPathFactory) -> None:
    """Start every test with a made-up $HOME, so code that follows it (the agent config files that `harpo install`
    edits, a bare Path.home()) cannot reach the real one. A test that needs another sets it itself. The integration
    tests (they build and install a wheel) keep the real $HOME for pip and its caches."""
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))


@pytest.fixture(autouse=True)
def _home_follows_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
                      tmp_path_factory: pytest.TempPathFactory) -> None:
    """Make paths.home() return $HOME for this test.

    A test marked passwd_home keeps the real paths.home() (the password database) and fakes that database instead,
    to a made-up folder; it can fake its own folder on top, but it can never reach the real home."""
    if request.node.get_closest_marker("passwd_home") is None:
        monkeypatch.setattr(paths, "home", lambda: Path.home())
    else:
        made_up = str(tmp_path_factory.mktemp("made-up-passwd-home"))
        monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=made_up))


@pytest.fixture
def child_python():
    """A function (module, *args) -> argv that runs `python -m module args` with paths.home() following $HOME."""
    def build(module: str, *args: str) -> list[str]:
        return [sys.executable, "-c", _RUN_MODULE, module, *args]

    return build
