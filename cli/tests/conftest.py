"""Fixtures shared by every test.

SEC-08: paths.home() reads the password database, never $HOME, so a test that moves HOME to a temp folder would
reach the real home folder. The autouse fixture below makes paths.home() follow $HOME again, which keeps every
existing test inside the fixture home it sets. The SEC-08 tests opt out (pytest.mark.passwd_home) and fake the
password database instead. A child process cannot be patched from here: child_python() builds a command that
makes the same patch before it runs a module.
"""
from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from Harpocrates import paths

_RUN_MODULE = (
    "import pathlib, os, runpy, sys\n"
    "import Harpocrates.paths\n"
    "Harpocrates.paths.home = lambda: pathlib.Path(os.environ['HOME'])\n"
    "runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)\n"
)


def _state_dirs() -> list[Path]:
    """The real home's two state folders, as the password database names them (looked up outside any patch)."""
    base = Path(pwd.getpwuid(os.getuid()).pw_dir)
    return [base / ".harpo", base / ".harpocrates"]


@pytest.fixture(scope="session", autouse=True)
def _real_state_dirs_untouched():
    """Fail the run if any test created, changed or removed ~/.harpo or ~/.harpocrates in the real home."""
    def snapshot() -> list:
        out = []
        for folder in _state_dirs():
            try:
                st = os.lstat(folder)
                out.append((st.st_ino, st.st_mtime_ns, sorted(os.listdir(folder))))
            except OSError:
                out.append(None)
        return out

    before = snapshot()
    yield
    assert snapshot() == before, "a test touched the real ~/.harpo or ~/.harpocrates"


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
