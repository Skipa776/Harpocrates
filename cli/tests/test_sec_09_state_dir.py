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

from Harpocrates import allowlist, grants, hooks, install, paths

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


# ---- F5: any home component that starts with .harpo is guarded ------------------------------------------------------

def _edit(target: str, cwd: Path, capsys, tool: str = "Edit") -> tuple[int, str]:
    key = "notebook_path" if tool == "NotebookEdit" else "file_path"
    event = {"tool_name": tool, "cwd": str(cwd), "permission_mode": "default", "session_id": "s1", "tool_use_id": "t1",
             "tool_input": {key: target, "old_string": "a", "new_string": "b", "content": "b", "new_source": "b"}}
    code = hooks.decide(json.dumps(event), ask=True)
    out = capsys.readouterr().out
    return code, (json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "")


F5_GUARDED = [
    ".harpo.migrating.abc/allow",  # the temp folder the migration renames into place
    ".harpocrates.bak/allow",  # a backup of the old folder
    ".HARPO.x/allow",  # other case
    ".Harpocrates.bak/grants/s1.json",
    "．harpo.bak/allow",  # NFKC: a full-width dot
    ".ｈarpo.old/allow",  # NFKC: a full-width letter
    ".harpoonx/allow",  # accepted over-block: it starts with .harpo
    ".harpo-not/x",
    ".harpocrates-not/x",
    ".harpo.migrating.abc",  # the folder itself
]
F5_FREE = [
    "work/.harpo/x",  # not directly under the home folder: a project folder that happens to carry the name
    "work/.harpo.bak/allow",
    ".harp/x",  # does not start with .harpo
    "harpo/x",  # no dot
    "x.harpo/x",
    ".config/harpo/x",
]


@pytest.mark.parametrize("rel", F5_GUARDED)
def test_sec_09_f5_a_home_component_that_starts_with_harpo_is_state(home: Path, rel: str, capsys) -> None:
    assert allowlist.inside_state_dir(home / rel)
    assert _edit(str(home / rel), home, capsys) == (0, "deny")
    assert _edit(rel, home, capsys) == (0, "deny")  # relative to the home folder
    assert allowlist.touches("echo x >> allow", home / rel)  # and as the working directory of a command


@pytest.mark.parametrize("rel", F5_FREE)
def test_sec_09_f5_other_paths_are_not_state(home: Path, rel: str, capsys) -> None:
    assert not allowlist.inside_state_dir(home / rel)
    assert _edit(str(home / rel), home, capsys) == (0, "")
    assert not allowlist.inside_state_dir(home.parent / ".harpo" / "allow")  # a sibling of the home folder is not under it
    assert not allowlist.inside_state_dir(home)


def test_sec_09_f5_the_temp_folder_of_a_running_migration_is_guarded(home: Path, monkeypatch, capsys) -> None:
    """While the copy runs the agent could edit the tmp folder before it is renamed into place."""
    _make_old(home)
    seen: list[bool] = []
    real = paths._assert_plain_tree

    def spy(root: Path) -> None:
        seen.append(allowlist.inside_state_dir(root / "allow"))
        real(root)

    monkeypatch.setattr(paths, "_assert_plain_tree", spy)
    paths.state_dir()
    assert seen == [True]


# ---- F6: a hard link elsewhere to a state file is state ------------------------------------------------------------

def _state_with_files(home: Path) -> dict[str, Path]:
    """Both state folders (and a migration temp folder) with files in them; returns name -> path."""
    files: dict[str, Path] = {}
    for folder in (".harpo", ".harpocrates", ".harpo.migrating.abc"):
        (home / folder / "grants").mkdir(parents=True)
        for rel in ("allow", "grants/s1.json", "install.json"):
            (home / folder / rel).write_text("# mine\n")
            files[f"{folder}/{rel}"] = home / folder / rel
    (home / "work").mkdir()
    return files


@pytest.mark.parametrize("tool", ["Edit", "MultiEdit", "Write", "NotebookEdit"])
def test_sec_09_f6_a_hard_link_to_a_state_file_is_refused(home: Path, capsys, tool: str) -> None:
    files = _state_with_files(home)
    for i, (name, source) in enumerate(files.items()):
        alias = home / "work" / f"alias{i}"
        os.link(source, alias)
        assert not allowlist.inside_state_dir(alias), name  # by name it is an ordinary file
        assert allowlist.links_to_state_file(alias), name
        assert _edit(str(alias), home, capsys, tool) == (0, "deny"), name
    link = home / "work" / "sym"
    os.symlink(home / "work" / "alias0", link)  # a symlink to the hard link resolves to the same file
    assert _edit(str(link), home, capsys, tool) == (0, "deny")


def test_sec_09_f6_ordinary_hard_links_and_new_files_are_not_over_blocked(home: Path, capsys) -> None:
    _state_with_files(home)
    work = home / "work"
    (work / "a.txt").write_text("x = 1\n")
    os.link(work / "a.txt", work / "b.txt")  # two names of one ordinary file: nothing to do with the state folders
    assert (work / "a.txt").stat().st_nlink == 2
    for target in (work / "a.txt", work / "b.txt", work / "does-not-exist-yet", work / "sub" / "new.py"):
        assert not allowlist.links_to_state_file(target), target
        assert _edit(str(target), home, capsys) == (0, ""), target
    assert not allowlist.links_to_state_file(None)


def test_sec_09_f6_a_directory_or_unreadable_target_does_not_pass_as_a_link(home: Path, capsys, monkeypatch) -> None:
    _state_with_files(home)
    assert not allowlist.links_to_state_file(home / "work")  # a directory has no file identity to compare

    def boom(*args, **kwargs):
        raise PermissionError(13, "denied")

    real_stat = os.stat
    target = home / "work" / "locked"
    target.write_text("x")
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: boom() if str(p) == str(target) else real_stat(p, *a, **k))
    assert _edit(str(target), home, capsys) == (0, "deny")  # the target cannot be examined: fail closed


def test_sec_09_f6_a_state_folder_with_too_many_files_is_refused(home: Path, monkeypatch) -> None:
    files = _state_with_files(home)
    alias = home / "work" / "alias"
    os.link(files[".harpo.migrating.abc/install.json"], alias)  # a match the capped walk would reach late
    monkeypatch.setattr(allowlist, "_MAX_STATE_FILES", 2)
    assert allowlist.links_to_state_file(alias)  # fail closed: it cannot be compared in full, so it is refused
    other = home / "work" / "ordinary"
    other.write_text("x")
    os.link(other, home / "work" / "ordinary2")
    assert allowlist.links_to_state_file(other)  # also for an unrelated link: the walk is over budget, so refuse


# ---- F7: cd, pushd and popd next to either state folder are refused ------------------------------------------------

# `{d}` is the folder name (.harpo or .harpocrates), `{n}` the same without the dot, `{n1}`, `{n2}`, `{n4}` cut at 1, 2, 4.
F7_GUARDED = [
    "cd ~/{d}",
    "cd ~/{d} && ls",
    "cd ~/{d}; cat allow",
    "(cd ~/{d} && ls)",
    "true && cd ~/{d}",
    "ls | cd ~/{d}",
    "cd $HOME/{d}",
    "cd -P ~/{d}",
    "cd '~/{d}'",
    "pushd ~/{d}",
    "pushd ~/{d} && ls",
    "popd ~/{d}",
    "ls ~/{d}; popd",
    "ls ~/{d}; cd /tmp",
    "cd ~ && cd {d}",
    "cd ~; cd .h{n1}",
    "cd ~/.\\{n}",  # obfuscated spellings the guard already undoes
    'cd ~/.ha""{n2}',
    "cd ~/.ha''{n2}",
    "cd ~/.'harp'{n4}",
    "cd ~/{.,}{n}",
    "cd $'\\056{n}'",
    "cd ~/.h*",
    "cd ~/.harp*",
    "cd ~/.?{n1}",
    "c\\d ~/{d}",
    "'cd' ~/{d}",
    "\"cd\" ~/{d}",
    "builtin cd ~/{d}",
    "command cd ~/{d}",
    "D=~/{d}; cd $D",
]
F7_FREE = [
    "cd ~/project",
    "cd ~/project && ls",
    "cd /tmp && cat README.md",
    "pushd ~/project",
    "cd ..; popd",
    "cd ~/work/harpoon-docs && ls",
    "cd ~/.config && ls",
]


def _f7(templates: list[str]) -> list[str]:
    out: list[str] = []
    for d in BOTH:
        n = d[1:]
        for template in templates:
            out.append(template.replace("{d}", d).replace("{n1}", n[1:]).replace("{n2}", n[2:])
                       .replace("{n4}", n[4:]).replace("{n}", n))
    return list(dict.fromkeys(out))


def _bash(cmd: str, cwd: Path, capsys) -> tuple[int, str]:
    event = {"tool_name": "Bash", "cwd": str(cwd), "permission_mode": "default", "tool_input": {"command": cmd}}
    code = hooks.decide(json.dumps(event), ask=True)
    out = capsys.readouterr().out
    return code, (json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "")


@pytest.mark.parametrize("cmd", _f7(F7_GUARDED))
def test_sec_09_f7_changing_directory_next_to_a_state_dir_is_refused(home: Path, capsys, cmd: str) -> None:
    assert allowlist.touches(cmd), cmd
    assert _bash(cmd, home, capsys) == (0, "deny"), cmd


@pytest.mark.parametrize("cmd", _f7(F7_FREE))
def test_sec_09_f7_changing_directory_elsewhere_still_passes(home: Path, capsys, cmd: str) -> None:
    assert not allowlist.touches(cmd), cmd
    assert _bash(cmd, home, capsys)[1] != "deny", cmd


def test_sec_09_f7_a_working_directory_inside_state_is_not_left_by_cd_unnoticed(home: Path) -> None:
    (home / ".harpo").mkdir()
    assert allowlist.touches("cd .. && echo x >> allow", home / ".harpo")


# ---- F8a: every path key of an edit call is checked ----------------------------------------------------------------

def _edit_with(tool: str, extra: dict, cwd: Path, capsys, primary: dict) -> tuple[int, str]:
    event = {"tool_name": tool, "cwd": str(cwd), "permission_mode": "default", "session_id": "s1", "tool_use_id": "t1",
             "tool_input": {**primary, **extra, "old_string": "a", "new_string": "b", "content": "b", "new_source": "b"}}
    code = hooks.decide(json.dumps(event), ask=True)
    out = capsys.readouterr().out
    return code, (json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "")


@pytest.mark.parametrize("d", BOTH)
@pytest.mark.parametrize("tool", ["Edit", "MultiEdit", "Write", "NotebookEdit"])
@pytest.mark.parametrize("keys", [
    ("file_path", "notebook_path"), ("notebook_path", "file_path"),  # the tool's own key is safe, the other is not
    ("file_path", "path"), ("notebook_path", "filePath"), ("file_path", "target_file"), ("file_path", "directory"),
])
def test_sec_09_f8a_a_state_path_in_any_path_key_is_refused(home: Path, capsys, tool: str, keys: tuple, d: str) -> None:
    _state_with_files(home)
    safe, state = str(home / "work" / "notes.txt"), f"~/{d}/allow"
    for first, second in ((keys[0], keys[1]), (keys[1], keys[0])):
        assert _edit_with(tool, {second: state}, home, capsys, {first: safe}) == (0, "deny"), (first, second)
        assert _edit_with(tool, {second: str(home / d / "grants" / "s1.json")}, home, capsys,
                          {first: safe}) == (0, "deny")
    nested = {"edits": [{"old_string": "a", "new_string": "b", "file_path": state}]}  # a path key inside a list
    assert _edit_with(tool, nested, home, capsys, {"file_path": safe}) == (0, "deny")


def test_sec_09_f8a_without_any_path_key_the_call_is_blocked(home: Path, capsys) -> None:
    event = {"tool_name": "Edit", "cwd": str(home), "tool_input": {"old_string": "a", "new_string": "b"}}
    assert hooks.decide(json.dumps(event), ask=True) == 2  # fail closed, as before


def test_sec_09_f8a_one_file_named_twice_passes_and_two_files_are_refused(home: Path, capsys) -> None:
    (home / "work").mkdir()
    notes = str(home / "work" / "notes.txt")
    same = {"notebook_path": "./work/../work/notes.txt"}
    assert _edit_with("NotebookEdit", same, home, capsys, {"file_path": notes}) == (0, "")  # one file, two spellings
    other = {"notebook_path": str(home / "work" / "other.txt")}
    assert _edit_with("NotebookEdit", other, home, capsys, {"file_path": notes}) == (0, "deny")  # which one is written?
    assert _edit_with("Edit", {}, home, capsys, {"file_path": notes}) == (0, "")  # the ordinary call is untouched
    assert _edit_with("NotebookEdit", {}, home, capsys, {"notebook_path": notes}) == (0, "")


def test_sec_09_f8a_a_secret_file_named_by_two_keys_is_still_asked_about(home: Path, capsys) -> None:
    """Naming the same file under both keys changes nothing: it is scanned and asked about like any other."""
    (home / "work").mkdir()
    secret = home / "work" / "n.ipynb"
    secret.write_text('TOKEN = "' + "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8" + '"\n')  # a canary
    same = {"file_path": str(secret)}
    code, decision = _edit_with("NotebookEdit", same, home, capsys, {"notebook_path": str(secret)})
    assert (code, decision) == (0, "ask")  # asked about the notebook, as for any secret-bearing file
