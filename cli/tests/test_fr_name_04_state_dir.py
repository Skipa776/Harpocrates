"""FR-NAME-04: user data lives in ~/.harpo; ~/.harpocrates is copied there once and never touched again.

Security (FR-CLEAN-03): the migration fails closed (a symlink, a failed copy or any other doubt keeps ~/.harpocrates
in use, never an empty allowlist), both directories stay guarded from the agent, and no environment variable can
move the state directory.

Every test runs with HOME in a temp dir, so nothing touches the real ~/.harpo or ~/.harpocrates. Fixture data is
made up: allowlist paths and session ids, no credentials.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from Harpocrates import allowlist, envvars, grants, harness, hooks, install, paths

SID = "sess-1"
CANARY_PATH = "/proj/canary-0123456789/app.py"  # in the fixture files, so a leak into a message is visible
BOTH = (".harpo", ".harpocrates")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))  # no real claude or codex can be found
    return h


def _make_old(home: Path, extra_grants: int = 0) -> Path:
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
        "install.json": json.dumps({"python": "/usr/bin/python3", "python_args": ["-P"]}),
    }
    files.update({f"grants/extra-{i}.json": json.dumps({"pending": {}, "granted": {}}) for i in range(extra_grants)})
    for rel, text in files.items():
        (old / rel).write_text(text)
        os.chmod(old / rel, 0o600)
    return old


def _bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _snap(root: Path) -> dict:
    """Kind, mode, size, mtime and content of everything under root, plus the folder's own mtime."""
    out: dict = {".": root.stat().st_mtime_ns}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        out[str(p.relative_to(root))] = (stat.S_IFMT(st.st_mode), stat.S_IMODE(st.st_mode), st.st_size,
                                         st.st_mtime_ns, p.read_bytes() if p.is_file() else None)
    return out


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _one_line(err: str) -> None:
    assert err.count("\n") == 1 and err.startswith("harpocrates: ") and ".harpocrates" in err, err
    assert "canary" not in err  # file contents never reach the message


# ---- part 1: the state directory and its migration -------------------------------------------------------------

def test_fr_name_04_migration_copies_and_keeps_original(home: Path, capsys) -> None:
    old = _make_old(home)
    before = _snap(old)
    new = home / ".harpo"
    assert paths.state_dir() == new
    copied = _bytes(new)
    for rel, data in _bytes(old).items():
        assert copied[rel] == data, rel  # identical bytes
    assert set(copied) - set(_bytes(old)) == {paths.MIGRATION_NOTE}
    assert _snap(old) == before  # the original is not modified: content, modes, times, and no new file in it
    assert stat.S_IMODE(new.stat().st_mode) == 0o700 and stat.S_IMODE((new / "grants").stat().st_mode) == 0o700
    assert stat.S_IMODE((new / "allow").stat().st_mode) == 0o600
    assert _names(home) == [".harpo", ".harpocrates"]  # no temp folder left behind
    # allowlist and grants survive: both modules now read the new folder
    assert allowlist.names() == ["graphify-out"] and allowlist.entries() == [home / "proj" / "keep"]
    assert grants.granted(SID) == frozenset({CANARY_PATH})
    assert capsys.readouterr().err == ""
    # later writes go to the new folder only
    allowlist.add_name("other-out")
    assert "name:other-out" in (new / "allow").read_text()
    assert "other-out" not in (old / "allow").read_text()
    allowlist.log_use(home / "x", "Read")
    assert (new / "allow.log").read_text().count("\n") == 2 and _snap(old) == before


def test_fr_name_04_fresh_home_uses_the_new_dir_without_creating_it(home: Path) -> None:
    assert paths.state_dir() == home / ".harpo"
    assert _names(home) == []
    assert allowlist.entries() == [] and grants.granted(SID) == frozenset()


def test_fr_name_04_empty_old_dir_migrates(home: Path) -> None:
    (home / ".harpocrates").mkdir()
    assert paths.state_dir() == home / ".harpo"
    assert _names(home / ".harpo") == [paths.MIGRATION_NOTE]  # never an empty folder a racing rename could replace


def test_fr_name_04_no_migration_when_new_dir_exists(home: Path, capsys) -> None:
    old = _make_old(home)
    new = home / ".harpo"
    new.mkdir(mode=0o700)
    (new / "allow").write_text("# already migrated\n")
    old_before, new_before = _snap(old), _snap(new)
    assert paths.state_dir() == new
    assert _snap(new) == new_before and _snap(old) == old_before  # nothing copied over, nothing added
    assert allowlist.entries() == [] and allowlist.names() == []  # the old list is not read once the new folder exists
    assert grants.granted(SID) == frozenset()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("kind", ["old-is-a-link", "link-to-file-inside", "link-to-dir-inside", "dangling-link-inside"])
def test_fr_name_04_symlinked_source_refused(home: Path, capsys, kind: str) -> None:
    elsewhere = home.parent / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "allow").write_text("# canary-0123456789\n/etc\n")
    if kind == "old-is-a-link":
        old = home / ".harpocrates"
        old.symlink_to(elsewhere, target_is_directory=True)
    else:
        old = _make_old(home)
        if kind == "link-to-file-inside":
            (old / "grants" / "s2.json").symlink_to(elsewhere / "allow")
        elif kind == "link-to-dir-inside":
            (old / "more").symlink_to(elsewhere, target_is_directory=True)
        else:
            (old / "gone").symlink_to(home / "nowhere")
    assert paths.state_dir() == old  # still reading ~/.harpocrates
    assert not os.path.lexists(home / ".harpo") and _names(home) == [".harpocrates"]
    _one_line(capsys.readouterr().err)
    assert paths.state_dir() == old and capsys.readouterr().err == ""  # warned once per process, not retried
    if kind != "old-is-a-link":
        assert allowlist.names() == ["graphify-out"]  # nothing lost: the grants and the list are still read
        assert grants.granted(SID) == frozenset({CANARY_PATH})


@pytest.mark.parametrize("old_exists", [True, False])
@pytest.mark.parametrize("target", ["real-dir", "dangling"])
def test_fr_name_04_symlinked_target_refused(home: Path, capsys, target: str, old_exists: bool) -> None:
    """SEC-09 F4: a ~/.harpo that is a link is unusable, and ~/.harpocrates is not used in its place."""
    planted = home.parent / "planted"
    if target == "real-dir":
        planted.mkdir()
        (planted / "allow").write_text(f"{home}/proj/evil\n")  # what an agent would plant to widen the allowlist
    (home / ".harpo").symlink_to(planted, target_is_directory=True)
    old = _make_old(home) if old_exists else home / ".harpocrates"
    before = _snap(old) if old_exists else None
    assert paths.state_dir() == home / ".harpo"  # the link itself: never copied through, never written through
    assert (home / ".harpo").is_symlink()
    if target == "real-dir":
        assert _names(planted) == ["allow"]  # nothing was copied through the link
    else:
        assert not planted.exists()  # and the dangling target was not created
    assert allowlist.entries() == [] and allowlist.names() == []  # unusable: an empty allowlist, old list or not
    assert grants.granted(SID) == frozenset()
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("harpocrates: ") and "~/.harpo " in err
    assert paths.state_dir() == home / ".harpo" and capsys.readouterr().err == ""  # announced once per process
    if old_exists:
        assert _snap(old) == before


@pytest.mark.parametrize("failing", ["shutil.copyfileobj", "os.rename", "os.fsync"])
def test_fr_name_04_failed_copy_falls_back_to_old_dir(home: Path, capsys, monkeypatch, failing: str) -> None:
    old = _make_old(home)
    before = _snap(old)
    module, name = failing.split(".")

    def boom(*args, **kwargs):
        raise OSError(errno.EIO, "injected failure")

    with monkeypatch.context() as m:
        m.setattr({"shutil": shutil, "os": os}[module], name, boom)
        assert paths.state_dir() == old
    assert not os.path.lexists(home / ".harpo") and _names(home) == [".harpocrates"]  # no half copy, no temp folder
    assert _snap(old) == before
    assert allowlist.names() == ["graphify-out"] and grants.granted(SID) == frozenset({CANARY_PATH})
    _one_line(capsys.readouterr().err)
    assert paths.state_dir() == old and capsys.readouterr().err == ""  # not retried in this process, not repeated


_CHILD = """
import os, pathlib, sys, time
from Harpocrates import paths
paths.home = lambda: pathlib.Path(os.environ["HOME"])  # SEC-08: the real home comes from the password database
go, ready = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ready.touch()
while not go.exists():
    time.sleep(0.001)
print(paths.state_dir())
"""


def test_fr_name_04_concurrent_migration_safe(home: Path, tmp_path: Path) -> None:
    old = _make_old(home, extra_grants=400)  # enough files that the copies overlap
    before = _snap(old)
    go = tmp_path / "go"
    env = {**os.environ, "HOME": str(home), "PYTHONPATH": str(Path(paths.__file__).resolve().parents[1])}
    n = 6
    procs = [subprocess.Popen([sys.executable, "-c", _CHILD, str(go), str(tmp_path / f"ready-{i}")], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(n)]
    deadline = time.time() + 90
    while len(list(tmp_path.glob("ready-*"))) < n and time.time() < deadline:
        time.sleep(0.01)
    go.touch()  # all processes start the first state_dir() call together
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0 and out.strip() == str(home / ".harpo") and err == "", err  # the loser is silent
    new = home / ".harpo"
    copied = _bytes(new)
    assert {k: v for k, v in copied.items() if k != paths.MIGRATION_NOTE} == _bytes(old)
    assert _names(home) == [".harpo", ".harpocrates"]  # nobody left a temp folder
    assert _snap(old) == before


def test_fr_name_04_concurrent_rename_race_in_one_process(home: Path) -> None:
    old = _make_old(home, extra_grants=200)
    before = _snap(old)
    new = home / ".harpo"
    barrier, errors = threading.Barrier(8), []

    def run() -> None:
        barrier.wait()
        try:
            paths._migrate(old, new)  # the loser's rename fails: it must discard its copy and report success
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert errors == []
    assert {k: v for k, v in _bytes(new).items() if k != paths.MIGRATION_NOTE} == _bytes(old)
    assert _names(home) == [".harpo", ".harpocrates"] and _snap(old) == before


def test_fr_name_04_no_env_var_moves_the_state_dir(home: Path, tmp_path: Path, monkeypatch) -> None:
    """HARPO_DIR and HARPOCRATES_DIR are not read: a state directory the environment (a project's settings can set
    env vars) could point at a folder the agent writes would replace the allowlist."""
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "allow").write_text(f"{home}/proj/evil\n")
    for name in ("HARPO_DIR", "HARPOCRATES_DIR"):
        monkeypatch.setenv(name, str(planted))
    assert paths.state_dir() == home / ".harpo"
    assert allowlist.entries() == []


def test_fr_name_04_install_record_lives_in_the_state_dir(home: Path) -> None:
    install.install_harnesses(["codex"])
    assert (home / ".harpo" / "install.json").is_file() and not (home / ".harpocrates").exists()
    install.uninstall_harnesses(["codex"])
    assert not (home / ".harpo" / "install.json").exists()


def test_fr_name_04_install_migrates_an_existing_record_and_leaves_the_old_one(home: Path) -> None:
    old = _make_old(home)
    before = (old / "install.json").read_bytes()
    install.install_harnesses(["codex"])
    new_record = json.loads((home / ".harpo" / "install.json").read_text())
    assert "codex" in new_record and (old / "install.json").read_bytes() == before


# ---- part 2: HARPO_* first, then HARPOCRATES_* --------------------------------------------------------------------

@pytest.mark.parametrize("suffix", ["REQUIRE_RUST", "RUST_SCANNER", "RUST_TIMEOUT_SECONDS", "RUST_WORKERS"])
def test_fr_name_04_env_precedence(monkeypatch, suffix: str) -> None:
    new, old = f"HARPO_{suffix}", f"HARPOCRATES_{suffix}"
    monkeypatch.delenv(new, raising=False)
    monkeypatch.delenv(old, raising=False)
    assert envvars.getenv(suffix) is None
    monkeypatch.setenv(old, "old")
    assert envvars.getenv(suffix) == "old"  # the old name still works alone
    monkeypatch.setenv(new, "new")
    assert envvars.getenv(suffix) == "new"  # the new name wins when both are set
    monkeypatch.delenv(old)
    assert envvars.getenv(suffix) == "new"
    monkeypatch.setenv(old, "old")
    monkeypatch.setenv(new, "")
    assert envvars.getenv(suffix) == ""  # set means set: an empty HARPO_ value is not skipped


def test_fr_name_04_env_precedence_in_the_rust_backend(monkeypatch, tmp_path: Path) -> None:
    from Harpocrates.core.rust_backend import RustScannerBackend, RustScannerError

    a, b = tmp_path / "scanner-new", tmp_path / "scanner-old"
    for exe in (a, b):
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    monkeypatch.setattr("Harpocrates.core.rust_backend.shutil.which", lambda name: None)
    monkeypatch.delenv("HARPO_RUST_SCANNER", raising=False)
    monkeypatch.setenv("HARPOCRATES_RUST_SCANNER", str(b))
    assert RustScannerBackend.discover(required=True).executable == b
    monkeypatch.setenv("HARPO_RUST_SCANNER", str(a))
    assert RustScannerBackend.discover(required=True).executable == a

    monkeypatch.delenv("HARPO_RUST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("HARPOCRATES_RUST_TIMEOUT_SECONDS", "42.5")
    assert RustScannerBackend._directory_timeout() == 42.5
    monkeypatch.setenv("HARPO_RUST_TIMEOUT_SECONDS", "7")
    assert RustScannerBackend._directory_timeout() == 7.0
    monkeypatch.setenv("HARPO_RUST_TIMEOUT_SECONDS", "soon")
    with pytest.raises(RustScannerError, match="HARPO_RUST_TIMEOUT_SECONDS"):
        RustScannerBackend._directory_timeout()


# ---- part 3: the agent cannot edit either directory ---------------------------------------------------------------

def _bash(cmd: str, cwd: Path, capsys) -> tuple[int, str]:
    event = {"tool_name": "Bash", "cwd": str(cwd), "permission_mode": "default", "tool_input": {"command": cmd}}
    code = hooks.decide(json.dumps(event), ask=True)
    out = capsys.readouterr().out
    return code, (json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "")


# `{d}` is the folder name (.harpo or .harpocrates), `{n}` the same without the dot. The glob and obfuscated cases are
# built from pieces, as in test_hooks_allowlist.py, so no spelling of the path sits in one literal.
_WRITES = [
    "echo /tmp >> ~/{d}/allow",
    "bash -c 'printf x > $HOME/{d}/allow'",
    "cd ~/{d} && echo /x >> allow",
    "cd ~; cd {d}; echo / >> allow",
    "D=~/{d}; echo / >> $D/allow",
    "rm -rf ~/{d}",
    "mv ~/{d} /tmp/x",
    "ln -s /tmp/evil ~/{d}",
    "echo '{}' > ~/{d}/grants/s9.json",
    "tee -a ~/{d}/allow",
    "cp evil ~/{d}/allow",
    "rm ~/{d}/allow.log",
    "rm ~/{d}/install.json",
    "sed -i '' 1d ~/{d}/allow",
    "sort -o ~/{d}/allow x",
    "uniq x ~/{d}/allow",
    "find ~/{d} -delete",
    "python -c \"open('$HOME/{d}/allow','a')\"",
    "ls ~/{d}; rm ~/{d}/allow",
    "ls $(rm ~/{d}/allow)",
    "ls `rm ~/{d}/allow`",
    "cat <(cp evil ~/{d}/allow)",
    "cat <<EOF ~/{d}/allow\nrm ~/{d}/allow\nEOF",
    "echo 'unbalanced ~/{d}",
    "echo /x | sed -n 'w/Users/me/{d}/allow'",
]
_GLOBS = [
    "cp evil ~/.*/allow",
    "cp evil ~/.h*/allow",
    "cp evil ~/.harp*/allow",
    "cp evil ~/.harpo*/allow",
    "cp evil ~/.harp{o,ocrates}/allow",
    "cp evil ~/.harp{o,}/allow",
    "cp evil ~/.{harpo,x}/allow",
    "cp evil ~/.{harpocrates,x}/allow",
    "cp evil $HOME/.[h]arpo/allow",
    "cp evil $HOME/.[h]arpocrates/allow",
    "cp evil ~/.?arpo/allow",
    "cp evil ~/.?arpocrates/allow",
    "mv x ~/.h????/allow",
    "mv x ~/.h??????????/allow",
]
_NO_DOT = [  # the folder by name with no dot before it: only the `allow` / `grants` alternative of the guard sees it
    "cp evil /srv/{n}/grants/s9.json",
    "rm -rf /srv/{n}/grants",
    "echo '{}' > /srv/{n}/grants/s9.json",
    "tee /Users/u/{n}/grants/s9.json",
    "tee /Users/u/{n}/allow",
    "cp evil $HOME/*{n1}/grants/s9.json",
    "cp evil ~/?{n1}/grants/s9.json",
    "shopt -s dotglob; rm ~/*{n1}/grants/s9.json",
    "echo '{}' > ~/.h\\{n1}/grants/s9.json",
    "echo '{}' > ~/.ha''{n2}/grants/s9.json",
    "echo '{}' > ~/{.,}{n}/grants/s9.json",
    "echo '{}' > $'~/\\056{n}/grants/s9.json'",
]
_SPELLINGS = [  # the obfuscated spellings test_hooks_allowlist.py covers for .harpocrates, for either name
    "echo / >> ~/.\\{n}/allow",
    "cd ~ && echo / >> .h\\{n1}/allow",
    'echo / >> ~/.ha""{n2}/allow',
    "echo / >> ~/.ha''{n2}/allow",
    "echo / >> ~/.'harp'{n4}/allow",
    "echo / >> ~/{.h,x}{n1}/allow",
    "echo / >> ~/{.,}{n}/allow",
    "echo / >> $'\\056{n}'/allow",
    "echo / >> $'\\x2e{n}'/allow",
    "shopt -s dotglob; echo / >> ~/*{n1}/allow",
    "cp x ~/?{n1}/allow",
    "cp x ~/[.]{n}/allow",
    "echo / >> ~/{.,}{n}",
]
_READS = [
    "ls ~/{d}",
    "cat ~/{d}/allow.log",
    "wc -l ~/{d}/allow | head",
    "grep -n tests ~/{d}/allow",
    "cat < ~/{d}/allow",
    "ls ~/{d}/grants",
    "ls ~/.ha''{n2}",
]


def _expand(templates: list[str]) -> list[str]:
    """Each template once per folder name; the `{n..}` pieces are the name without its dot, cut at 1, 2 and 4."""
    out: list[str] = []
    for d in BOTH:
        n = d[1:]
        for t in templates:
            out.append(t.replace("{d}", d).replace("{n1}", n[1:]).replace("{n2}", n[2:])
                       .replace("{n4}", n[4:]).replace("{n}", n))
    return list(dict.fromkeys(out))  # the glob cases do not depend on the name: once each


@pytest.mark.parametrize("cmd", _expand(_WRITES + _GLOBS + _NO_DOT + _SPELLINGS))
def test_fr_name_04_agent_cannot_edit_either_dir(home: Path, capsys, cmd: str) -> None:
    assert allowlist.touches(cmd), cmd
    assert _bash(cmd, home, capsys) == (0, "deny"), cmd


@pytest.mark.parametrize("cmd", _expand(_READS))
def test_fr_name_04_read_only_commands_on_either_dir_pass(home: Path, capsys, cmd: str) -> None:
    assert not allowlist.touches(cmd), cmd
    assert _bash(cmd, home, capsys)[1] != "deny", cmd


@pytest.mark.parametrize("name", BOTH)
def test_fr_name_04_working_directory_inside_either_dir(home: Path, name: str) -> None:
    state = home / name
    (state / "grants").mkdir(parents=True)
    (home / "work").mkdir()
    assert allowlist.touches("echo x >> allow", state) and allowlist.touches("tee s9.json", state / "grants")
    assert not allowlist.touches("ls -la", state) and not allowlist.touches("cat allow", state)
    assert not allowlist.touches("echo x >> allow", home / "work") and not allowlist.touches("echo x >> allow")
    for sibling in (name + "-not", name + "x"):  # SEC-09 F5: a sibling that shares the prefix is guarded too
        (home / sibling).mkdir()
        assert allowlist.touches("echo x >> allow", home / sibling)
    (home / "work" / name).mkdir()  # a project folder that carries the name is not directly under the home folder
    assert not allowlist.touches("echo x >> allow", home / "work" / name)


def test_fr_name_04_working_directory_through_a_link_is_resolved(home: Path) -> None:
    for name in BOTH:
        (home / name).mkdir()
    (home / "shortcut").symlink_to(home / ".harpo", target_is_directory=True)
    (home / "shortcut2").symlink_to(home / ".harpocrates", target_is_directory=True)
    assert allowlist.touches("echo x >> allow", home / "shortcut")
    assert allowlist.touches("echo x >> allow", home / "shortcut2")


@pytest.mark.parametrize("name", [".harpo", ".HARPO", ".Harpo", ".harpocrates", ".HarpoCrates"])
def test_fr_name_04_name_rules_refuse_either_dir(home: Path, name: str) -> None:
    with pytest.raises(ValueError) as exc:
        allowlist.add_name(name, confirmed=True)
    assert not isinstance(exc.value, allowlist.NeedsConfirmationError)
    assert allowlist.names() == [] and _names(home) == []


@pytest.mark.parametrize("name", BOTH)
def test_fr_name_04_name_rules_never_cover_a_path_under_either_dir(home: Path, name: str) -> None:
    inside = home / name / "graphify-out" / "k.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("x = 1\n")
    allowlist.add_name("graphify-out")
    assert not allowlist.allowed(inside)


# ---- part 4: deny rules for both directories --------------------------------------------------------------------

def test_fr_name_04_claude_settings_deny_edits_of_both_dirs() -> None:
    deny = harness.claude_code_settings()["permissions"]["deny"]
    assert "Edit(~/.harpo/**)" in deny and "Edit(~/.harpocrates/**)" in deny
    assert not any(rule.startswith("Write(") for rule in deny)  # Claude Code ignores Write(path) rules


def test_fr_name_04_codex_profile_denies_both_dirs() -> None:
    tomllib = pytest.importorskip("tomllib")
    cfg = tomllib.loads(harness.codex_config())
    filesystem = cfg["permissions"][cfg["default_permissions"]]["filesystem"]
    assert filesystem["~/.harpo"] == "deny" and filesystem["~/.harpocrates"] == "deny"
    for d in harness.SENSITIVE_HOME_DIRS:  # the secret folders are still there
        assert filesystem[d] == "deny"


def test_fr_name_04_setup_output_lists_both_deny_rules() -> None:
    from typer.testing import CliRunner

    from Harpocrates.cli import app

    runner = CliRunner()
    claude = runner.invoke(app, ["setup", "claude-code"])
    assert claude.exit_code == 0 and "Edit(~/.harpo/**)" in claude.output and "Edit(~/.harpocrates/**)" in claude.output
    codex = runner.invoke(app, ["setup", "codex"])
    assert codex.exit_code == 0 and '"~/.harpo" = "deny"' in codex.output and '"~/.harpocrates" = "deny"' in codex.output


def test_fr_name_04_install_writes_the_deny_rules_for_both_dirs(home: Path) -> None:
    tomllib = pytest.importorskip("tomllib")
    install.install_harnesses(["claude-code", "codex"])
    deny = json.loads((home / ".claude" / "settings.json").read_text())["permissions"]["deny"]
    assert "Edit(~/.harpo/**)" in deny and "Edit(~/.harpocrates/**)" in deny
    cfg = tomllib.loads((home / ".codex" / "config.toml").read_text())
    filesystem = cfg["permissions"]["harpocrates"]["filesystem"]
    assert filesystem["~/.harpo"] == "deny" and filesystem["~/.harpocrates"] == "deny"


# ---- part 5: the hook refuses Edit, MultiEdit, Write and NotebookEdit under either state directory ---------------

EDIT_TOOLS = ("Edit", "MultiEdit", "Write", "NotebookEdit")


def _edit_event(tool: str, target: str, cwd: Path, mode: str = "default") -> dict:
    key = "notebook_path" if tool == "NotebookEdit" else "file_path"
    return {"tool_name": tool, "cwd": str(cwd), "permission_mode": mode, "session_id": "s1", "tool_use_id": "t1",
            "tool_input": {key: target, "old_string": "a", "new_string": "b", "content": "b", "new_source": "b"}}


def _pre(event: dict, capsys, ask: bool = True) -> tuple[int, str]:
    code = hooks.decide(json.dumps(event), ask=ask)
    out = capsys.readouterr().out
    return code, (json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "")


def _layout(home: Path, layout: str) -> None:
    """none: neither folder exists (the agent writes before anything exists); old-only: the state before migration;
    both: after it."""
    if layout in ("old-only", "both"):
        (home / ".harpocrates" / "grants").mkdir(parents=True)
        (home / ".harpocrates" / "allow").write_text("# mine\n")
    if layout == "both":
        (home / ".harpo" / "grants").mkdir(parents=True)
        (home / ".harpo" / "allow").write_text("# mine\n")
    (home / "work" / "deep").mkdir(parents=True)


# {home} and {d} (the folder name) are filled in per case. Each of these must land in the state directory.
_EDIT_TARGETS = [
    "~/{d}/allow",
    "~/{d}/grants/s1.json",
    "{home}/{d}/allow",
    "{home}/{d}/allow.log",
    "{home}/{d}/grants/s1.json",
    "{home}/{d}/install.json",
    "{home}/{d}/new-file-that-does-not-exist",
    "{home}/{d}",
    "{home}/{d}/",
    "{home}/{d}/allow/",
    "{home}/{d}/../{d}/allow",
    "{home}/work/../{d}/allow",
    "{home}/./{d}/./allow",
    "{home}//{d}//allow",
    "{home}/{d}/grants/../allow",
    "{home}/{D}/allow",  # other case: the default macOS file system treats .HARPO as .harpo
    "{home}/{T}/allow",
    "{home}/{W}/allow",  # NFKC variants (full-width dot or letter), folded as the name rules are
    "{home}/{F}/allow",
]
_RELATIVE_TARGETS = [  # (target, folder to run in, under the home folder)
    ("{d}/allow", ""),
    ("./{d}/allow", ""),
    ("../{d}/allow", "work"),
    ("../../{d}/allow", "work/deep"),
    ("{d}/grants/s1.json", ""),
]


def _fill(template: str, home: Path, d: str) -> str:
    n = d[1:]
    return (template.replace("{home}", str(home)).replace("{d}", d).replace("{D}", "." + n.upper())
            .replace("{T}", "." + n.title()).replace("{W}", "．" + n).replace("{F}", ".ｈ" + n[1:]))


@pytest.mark.parametrize("layout", ["none", "old-only", "both"])
@pytest.mark.parametrize("d", BOTH)
@pytest.mark.parametrize("tool", EDIT_TOOLS)
def test_fr_name_04_hook_refuses_edits_under_either_dir(home: Path, capsys, tool: str, d: str, layout: str) -> None:
    _layout(home, layout)
    for template in _EDIT_TARGETS:
        target = _fill(template, home, d)
        for ask in (True, False):  # with and without --ask: the refusal does not depend on a human being there
            assert _pre(_edit_event(tool, target, home), capsys, ask) == (0, "deny"), (target, ask)
    for template, where in _RELATIVE_TARGETS:
        target = _fill(template, home, d)
        assert _pre(_edit_event(tool, target, home / where), capsys) == (0, "deny"), (target, where)


@pytest.mark.parametrize("d", BOTH)
@pytest.mark.parametrize("tool", EDIT_TOOLS)
def test_fr_name_04_hook_refuses_edits_through_a_symlink(home: Path, capsys, tool: str, d: str) -> None:
    (home / "work").mkdir()
    (home / "dangling").symlink_to(home / d, target_is_directory=True)  # the state folder does not exist yet
    (home / "homelink").symlink_to(home, target_is_directory=True)
    for target in ("dangling/allow", f"{home}/dangling/grants/s1.json", f"homelink/{d}/allow", f"{home}/homelink/{d}"):
        assert _pre(_edit_event(tool, target, home), capsys) == (0, "deny"), target
    (home / d / "grants").mkdir(parents=True)  # and once it exists
    (home / "shortcut").symlink_to(home / d, target_is_directory=True)
    (home / "work" / "up").symlink_to(home / "work" / ".." / d, target_is_directory=True)
    for target in ("shortcut/allow", f"{home}/work/up/allow", f"{home}/shortcut/grants/s1.json", "shortcut"):
        assert _pre(_edit_event(tool, target, home), capsys) == (0, "deny"), target


@pytest.mark.parametrize("mode", ["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"])
@pytest.mark.parametrize("ask", [True, False])
def test_fr_name_04_hook_refusal_comes_before_any_grant_or_prompt(home: Path, capsys, mode: str, ask: bool) -> None:
    (home / ".harpo").mkdir()
    secret_file = home / ".harpo" / "allow"
    secret_file.write_text('TOKEN = "' + "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8" + '"\n')  # a canary: would be asked about
    event = _edit_event("Edit", str(secret_file), home, mode)
    assert _pre(event, capsys, ask) == (0, "deny")
    # even a grant the user gave earlier in this session does not open the state directory
    assert grants.add_pending("s1", "t0", secret_file, "Edit")
    grants.promote("s1", "t0", "Edit")
    assert os.path.realpath(secret_file) in grants.granted("s1")
    assert _pre(event, capsys, ask) == (0, "deny")


@pytest.mark.parametrize("d", BOTH)
@pytest.mark.parametrize("tool", EDIT_TOOLS)
def test_fr_name_04_hook_leaves_other_edits_alone(home: Path, capsys, tool: str, d: str) -> None:
    _layout(home, "both")
    (home / "work" / d).mkdir()  # a project folder that happens to carry the name is not the state directory
    for target in (f"{home}/work/notes.txt", f"{home}/work/{d}/x",
                   f"{home}/{d}/../work/y", f"{home}/.config/x", "~/notes.txt", f"{home}/work/deep/z"):
        assert _pre(_edit_event(tool, target, home), capsys) == (0, ""), target
    assert _pre(_edit_event(tool, f"{d}/x", home / "work"), capsys) == (0, "")  # relative to a project folder


@pytest.mark.parametrize("tool", EDIT_TOOLS)
def test_fr_name_04_hook_refuses_when_the_target_cannot_be_placed(home: Path, capsys, monkeypatch, tool: str) -> None:
    def boom(*args, **kwargs):
        raise OSError(errno.EIO, "cannot resolve")

    monkeypatch.setattr(allowlist, "inside_state_dir", boom)
    assert _pre(_edit_event(tool, f"{home}/work/notes.txt", home), capsys) == (0, "deny")  # fail closed


def test_fr_name_04_hook_refusal_text_points_to_the_user(home: Path, capsys) -> None:
    hooks.decide(json.dumps(_edit_event("Write", "~/.harpo/allow", home)), ask=True)
    reason = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason.startswith("Only the user can change the Harpocrates allowlist")
