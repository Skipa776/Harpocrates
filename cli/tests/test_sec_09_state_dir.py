"""SEC-09: the state-directory guard has no known bypass that does not build the path at run time.

One test per finding of the Oct 18 security review (F3 to F8), each written to fail before its fix. Every test runs
with HOME in a temp dir (the conftest fixture makes paths.home() follow it), so nothing touches the real ~/.harpo or
~/.harpocrates. Fixture data is made up: allowlist paths and session ids, no credentials.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import time
from pathlib import Path

import pytest

from Harpocrates import allowlist, grants, install, paths

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


# ---- F4: once ~/.harpo exists in any form, ~/.harpocrates is never used again ----------------------------------------

def _migrate_and_edit(home: Path) -> None:
    """Migrate, then do what the user does: allow a folder, take graphify-out off the list."""
    _make_old(home)
    assert paths.state_dir() == home / ".harpo"
    allowlist.remove_name("graphify-out")
    allowlist.add_name("kept-out")
    assert allowlist.names() == ["kept-out"]


def _replace_with(home: Path, kind: str, elsewhere: Path) -> None:
    shutil.rmtree(home / ".harpo")
    if kind == "symlink":
        os.symlink(elsewhere, home / ".harpo")
    elif kind == "dangling":
        os.symlink(home / "nowhere", home / ".harpo")
    else:
        (home / ".harpo").write_text("a file where the folder was")


@pytest.mark.parametrize("kind", ["symlink", "dangling", "file"])
def test_sec_09_f4_an_unusable_harpo_never_falls_back_to_the_old_folder(home: Path, capsys, kind: str) -> None:
    _migrate_and_edit(home)
    old_text = (home / ".harpocrates" / "allow").read_bytes()
    elsewhere = _foreign(home)
    _replace_with(home, kind, elsewhere)
    capsys.readouterr()
    assert paths.state_dir() == home / ".harpo"  # not ~/.harpocrates, whose list still holds what was removed
    assert allowlist.names() == [] and allowlist.entries() == []  # fail closed: an empty allowlist
    assert not allowlist.allowed(home / "proj" / "keep") and not allowlist.allowed(Path("/etc/hosts"))
    assert grants.granted(SID) == frozenset()  # the old folder's grant is not back either
    assert [("path", "/etc")] != allowlist.listing()  # and a planted list behind the link is not read
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("harpocrates: ") and "~/.harpo " in err
    assert (home / ".harpocrates" / "allow").read_bytes() == old_text  # the old folder was not touched


@pytest.mark.parametrize("kind", ["symlink", "dangling", "file"])
def test_sec_09_f4_writers_refuse_instead_of_writing_through(home: Path, kind: str) -> None:
    _migrate_and_edit(home)
    elsewhere = _foreign(home)
    _replace_with(home, kind, elsewhere)
    with pytest.raises(paths.StateDirUnusableError) as exc:
        allowlist.add_name("graphify-out")
    assert "~/.harpo" in str(exc.value) and "not a plain folder" in str(exc.value)
    with pytest.raises(paths.StateDirUnusableError):
        allowlist.add(home)
    with pytest.raises(paths.StateDirUnusableError):
        allowlist.remove_name("kept-out")
    with pytest.raises(paths.StateDirUnusableError):
        allowlist.log_use(home, "Read")
    assert grants.add_pending(SID, "tu-1", home / "x.py", "Edit") is False  # the hook then refuses instead of asking
    grants.clear(SID)  # a SessionEnd does nothing through the link
    reports = install.install_harnesses(["codex"])
    assert reports[0].refused and "~/.harpo" in reports[0].refused
    if kind == "symlink":
        assert _names(elsewhere) == ["allow"] and (elsewhere / "allow").read_text() == CANARY_FILE_TEXT
    else:
        assert not os.path.exists(home / "nowhere")


def test_sec_09_f4_the_cli_names_the_problem(home: Path) -> None:
    from typer.testing import CliRunner

    from Harpocrates.cli import app

    _migrate_and_edit(home)
    _replace_with(home, "file", _foreign(home))
    paths.state_dir()  # the one-line warning goes out here, not into the JSON the CLI prints below
    runner = CliRunner()
    listed = runner.invoke(app, ["allow", "--list", "--json"])
    assert listed.exit_code == 0 and json.loads(listed.output) == {"entries": []}
    for argv in (["allow", str(home)], ["allow", "--name", "out"], ["allow", "--remove", "--name", "out"],
                 ["allow", "--remove", str(home)]):
        done = runner.invoke(app, [*argv, "--json"])
        answer = json.loads(done.output)
        assert done.exit_code == 1 and answer["ok"] is False and "not a plain folder" in answer["error"], argv
    broad = runner.invoke(app, ["allow", "--name", "src", "--yes", "--json"])  # a broad name, confirmed
    assert broad.exit_code == 1 and "not a plain folder" in json.loads(broad.output)["error"]


def test_sec_09_f4_a_symlink_with_no_old_folder_is_not_a_place_to_write(home: Path) -> None:
    elsewhere = _foreign(home)
    os.symlink(elsewhere, home / ".harpo")
    assert paths.state_dir() == home / ".harpo"  # not the (missing) ~/.harpocrates, where a writer would have gone
    assert allowlist.entries() == []
    with pytest.raises(paths.StateDirUnusableError):
        allowlist.add_name("graphify-out")
    assert not os.path.lexists(home / ".harpocrates")


def test_sec_09_f4_a_linked_old_folder_is_read_as_empty(home: Path, capsys) -> None:
    elsewhere = _foreign(home)
    os.symlink(elsewhere, home / ".harpocrates")  # ~/.harpo missing: the migration is refused, the old folder is used
    assert paths.state_dir() == home / ".harpocrates"
    assert allowlist.entries() == [] and allowlist.names() == []  # not the planted list behind the link
    with pytest.raises(paths.StateDirUnusableError):
        allowlist.add_name("graphify-out")
    assert capsys.readouterr().err.count("\n") == 1


def test_sec_09_f4_a_failed_migration_still_falls_back_until_harpo_appears(home: Path, monkeypatch, capsys) -> None:
    old = _make_old(home)

    def boom(*args, **kwargs):
        raise OSError(errno.EIO, "injected failure")

    with monkeypatch.context() as m:
        m.setattr(os, "rename", boom)
        assert paths.state_dir() == old  # before any successful migration the fail-closed fallback stays
    assert allowlist.names() == ["graphify-out"]
    capsys.readouterr()
    os.symlink(_foreign(home), home / ".harpo")  # now ~/.harpo exists, in a form that is unusable
    assert paths.state_dir() == home / ".harpo"
    assert allowlist.names() == [] and allowlist.entries() == []
