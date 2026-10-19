"""SEC-09: the state-directory guard has no known bypass that does not build the path at run time.

One test per finding of the Oct 18 security review (F3 to F8), each written to fail before its fix. Every test runs
with HOME in a temp dir (the conftest fixture makes paths.home() follow it), so nothing touches the real ~/.harpo or
~/.harpocrates. Fixture data is made up: allowlist paths and session ids, no credentials.
"""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from Harpocrates import paths

SID = "sess-1"
CANARY_PATH = "/proj/canary-0123456789/app.py"
CANARY_FILE_TEXT = "# canary-0123456789\n/etc\n"  # what a foreign folder would add to the allowlist
BOTH = (".harpo", ".harpocrates")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))  # no real claude or codex can be found
    return h


def _make_old(home: Path) -> Path:
    """A ~/.harpocrates as the current release leaves it: allow, allow.log, grants/<session>.json, install.json."""
    old = home / ".harpocrates"
    old.mkdir()
    os.chmod(old, 0o700)
    (old / "grants").mkdir()
    os.chmod(old / "grants", 0o700)
    files = {
        "allow": f"# mine\n{home}/proj/keep\nname:graphify-out\n",
        "allow.log": f"2026-10-01T10:00:00\tRead\t{CANARY_PATH}\n",
        f"grants/{SID}.json": json.dumps({"pending": {}, "granted": {CANARY_PATH: time.time()}}),
        "install.json": json.dumps({"python": "/usr/bin/python3", "python_args": ["-I"]}),
    }
    for rel, text in files.items():
        (old / rel).write_text(text)
        os.chmod(old / rel, 0o600)
    return old


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _foreign(home: Path) -> Path:
    """A folder outside the state directories whose content must never be copied into one."""
    elsewhere = home.parent / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "allow").write_text(CANARY_FILE_TEXT)
    return elsewhere


# ---- F3: the migration walks with directory file descriptors and O_NOFOLLOW ------------------------------------------

class _Listing:
    """What os.scandir returns, from a list taken earlier: an iterator and a context manager (os.walk needs both)."""

    def __init__(self, entries: list) -> None:
        self._it = iter(entries)

    def __enter__(self) -> "_Listing":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def __iter__(self) -> "_Listing":
        return self

    def __next__(self):
        return next(self._it)


def _swap_for_link(victim: Path, away: Path, target: Path) -> None:
    """What an attacker with write access to the old folder does: the real entry moves away, a link takes its place."""
    os.rename(victim, away)
    os.symlink(target, victim)


@pytest.mark.parametrize("when", ["after-listing", "before-open", "top-level"])
def test_sec_09_f3_a_folder_swapped_for_a_link_mid_walk_is_refused(home: Path, monkeypatch, capsys, when: str) -> None:
    old = _make_old(home)
    elsewhere = _foreign(home)
    swapped: list[bool] = []

    def swap_grants() -> None:
        if not swapped:
            swapped.append(True)
            _swap_for_link(old / "grants", home / "grants-away", elsewhere)

    real_scandir, real_open, real_mkdtemp = os.scandir, os.open, paths.tempfile.mkdtemp

    def scandir(arg):
        entries = list(real_scandir(arg))  # the listing is taken, then the folder is swapped
        if when == "after-listing":
            swap_grants()
        return _Listing(entries)

    def open_(path, flags, *args, **kwargs):
        if when == "before-open" and kwargs.get("dir_fd") is not None and path == "grants":
            swap_grants()  # between the check of the entry and the open of it
        return real_open(path, flags, *args, **kwargs)

    def mkdtemp(*args, **kwargs):
        if when == "top-level" and not swapped:
            swapped.append(True)
            _swap_for_link(old, home / "old-away", elsewhere)  # between the check of ~/.harpocrates and the walk
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(paths.tempfile, "mkdtemp", mkdtemp)
    assert paths.state_dir() == old  # refused: still reading ~/.harpocrates
    assert swapped
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "refused" in err and "canary" not in err
    assert not os.path.lexists(home / ".harpo")  # nothing foreign was copied, no half copy, no temp folder
    assert not [n for n in _names(home) if "migrating" in n]
    assert (elsewhere / "allow").read_text() == CANARY_FILE_TEXT


def test_sec_09_f3_a_fifo_in_the_old_folder_is_refused_without_blocking(home: Path, capsys) -> None:
    old = _make_old(home)
    os.mkfifo(old / "pipe")
    assert paths.state_dir() == old
    assert "refused" in capsys.readouterr().err and not os.path.lexists(home / ".harpo")


def test_sec_09_f3_modes_and_times_survive_the_copy(home: Path) -> None:
    old = _make_old(home)
    (old / "grants" / "ro.json").write_text("{}")
    os.chmod(old / "grants" / "ro.json", 0o400)
    os.utime(old / "grants" / "ro.json", ns=(1_700_000_000 * 10**9, 1_700_000_001 * 10**9))
    os.chmod(old / "grants", 0o500)  # a folder that cannot be written is still filled, then closed
    try:
        new = paths.state_dir()
        assert new == home / ".harpo"
        got = (new / "grants" / "ro.json").stat()
        assert stat.S_IMODE(got.st_mode) == 0o400 and got.st_mtime_ns == 1_700_000_001 * 10**9
        assert stat.S_IMODE((new / "grants").stat().st_mode) == 0o500
        assert (new / "grants" / "ro.json").read_text() == "{}"
    finally:
        os.chmod(old / "grants", 0o700)
        if (home / ".harpo" / "grants").exists():
            os.chmod(home / ".harpo" / "grants", 0o700)
