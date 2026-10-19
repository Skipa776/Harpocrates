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

from Harpocrates import allowlist, envvars, grants, install, paths

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
    planted = home.parent / "planted"
    if target == "real-dir":
        planted.mkdir()
        (planted / "allow").write_text(f"{home}/proj/evil\n")  # what an agent would plant to widen the allowlist
    (home / ".harpo").symlink_to(planted, target_is_directory=True)
    old = _make_old(home) if old_exists else home / ".harpocrates"
    before = _snap(old) if old_exists else None
    assert paths.state_dir() == home / ".harpocrates"  # the link is never used, never written through
    assert (home / ".harpo").is_symlink()
    if target == "real-dir":
        assert _names(planted) == ["allow"]  # nothing was copied through the link
    else:
        assert not planted.exists()  # and the dangling target was not created
    assert allowlist.entries() == ([home / "proj" / "keep"] if old_exists else [])
    assert grants.granted(SID) == (frozenset({CANARY_PATH}) if old_exists else frozenset())
    _one_line(capsys.readouterr().err)
    if old_exists:
        assert _snap(old) == before


@pytest.mark.parametrize("failing", ["shutil.copy2", "os.rename", "os.fsync"])
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
import pathlib, sys, time
from Harpocrates import paths
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


