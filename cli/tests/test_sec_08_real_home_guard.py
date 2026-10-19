"""SEC-08: no test can reach the real home folder, and the run fails if one does.

The state folder comes from the password database, so moving $HOME no longer moves it (the point of SEC-08). Two
safety nets keep the test suite itself from writing to the real ~/.harpo, ~/.harpocrates, ~/.claude/settings.json or
~/.codex/config.toml: every test starts with a made-up $HOME and a paths.home() that follows it, and a session check
(conftest.py, _real_home.py) compares a read-only snapshot of those four before the first test and after the last.
These tests cover both. They never touch the real home: the snapshot tests build their own.
"""
from __future__ import annotations

import os
import pwd
from pathlib import Path

import pytest
from _real_home import FILES, TREES, changed, real_home, snapshot

from Harpocrates import paths

REAL = real_home()  # taken at import, before any test patches the password database


def _make_home(root: Path) -> Path:
    home = root / "snap-home"
    for tree in TREES:
        (home / tree / "grants").mkdir(parents=True)
        (home / tree / "allow").write_text("# mine\n")
        (home / tree / "grants" / "s1.json").write_text("{}")
    for rel in FILES:
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text("{}\n")
    return home


def test_sec_08_guard_an_untouched_home_compares_equal(tmp_path: Path) -> None:
    home = _make_home(tmp_path)
    before = snapshot(home)
    assert set(before) >= {".harpo", ".harpo/allow", ".harpocrates/grants/s1.json", *FILES}
    assert changed(before, snapshot(home)) == []


@pytest.mark.parametrize("what", [
    "rewrite-same-size-same-mtime", "touch", "chmod", "add-file", "remove-file", "replace-file-by-link",
    "edit-settings", "edit-codex-config", "remove-settings", "create-tree-from-nothing",
])
def test_sec_08_guard_notices_every_kind_of_change(tmp_path: Path, what: str) -> None:
    home = _make_home(tmp_path)
    if what == "create-tree-from-nothing":
        os.remove(home / ".harpo" / "grants" / "s1.json")
        os.rmdir(home / ".harpo" / "grants")
        os.remove(home / ".harpo" / "allow")
        os.rmdir(home / ".harpo")
    before = snapshot(home)
    target = home / ".harpo" / "allow"
    if what == "rewrite-same-size-same-mtime":
        stamp = target.stat()
        target.write_text("# thin\n")
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert target.stat().st_size == stamp.st_size  # only the content differs
        expected = [".harpo/allow"]
    elif what == "touch":
        os.utime(target, ns=(1, 1))
        expected = [".harpo/allow"]
    elif what == "chmod":
        os.chmod(target, 0o600 if target.stat().st_mode & 0o777 != 0o600 else 0o640)
        expected = [".harpo/allow"]
    elif what == "add-file":
        (home / ".harpocrates" / "grants" / "new.json").write_text("{}")
        expected = [".harpocrates/grants", ".harpocrates/grants/new.json"]
    elif what == "remove-file":
        os.remove(home / ".harpocrates" / "grants" / "s1.json")
        expected = [".harpocrates/grants", ".harpocrates/grants/s1.json"]
    elif what == "replace-file-by-link":
        os.remove(target)
        os.symlink(home / "elsewhere", target)
        expected = [".harpo", ".harpo/allow"]
    elif what == "edit-settings":
        (home / ".claude" / "settings.json").write_text('{"hooks": {}}\n')
        expected = [".claude/settings.json"]
    elif what == "edit-codex-config":
        (home / ".codex" / "config.toml").write_text("x = 1\n")
        expected = [".codex/config.toml"]
    elif what == "remove-settings":
        os.remove(home / ".claude" / "settings.json")
        expected = [".claude/settings.json"]
    else:
        (home / ".harpo").mkdir()
        expected = [".harpo"]
    got = changed(before, snapshot(home))
    assert got and set(expected) <= set(got), (what, got)


def test_sec_08_guard_snapshot_holds_no_file_content(tmp_path: Path) -> None:
    home = _make_home(tmp_path)
    (home / ".harpo" / "allow").write_text("canary-0123456789-not-for-reports\n")
    assert "canary-0123456789" not in repr(snapshot(home)) and "canary-0123456789" not in repr(
        changed(snapshot(home), {}))


def test_sec_08_every_test_starts_with_a_made_up_home(tmp_path_factory: pytest.TempPathFactory) -> None:
    """HOME is a fresh folder under pytest's own temp area, paths.home() follows it, and neither is the real home."""
    home = Path(os.environ["HOME"]).resolve()
    assert home.is_relative_to(Path(tmp_path_factory.getbasetemp()).resolve())
    assert paths.home().resolve() == home and home != REAL.resolve()


@pytest.mark.passwd_home
def test_sec_08_a_passwd_home_test_cannot_see_the_real_password_entry(tmp_path_factory) -> None:
    got = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    assert got.is_relative_to(Path(tmp_path_factory.getbasetemp()).resolve()) and got != REAL.resolve()
    assert paths.home().resolve() == got  # the real paths.home() runs, on the made-up entry
