"""SEC-08: the hook trusts no environment variable for anything security-relevant.

The state directory comes from the password database, not $HOME; a planted HOME neither moves it nor disarms the
guard; the hook interpreter runs with flags that ignore PYTHON* variables, so a planted PYTHONPATH package never
runs. Nothing here touches the real home: the password database is faked and every folder is made up in tmp_path.
"""
from __future__ import annotations

import json
import pwd
from pathlib import Path
from types import SimpleNamespace

import pytest

from Harpocrates import allowlist, grants, hooks, paths

pytestmark = pytest.mark.passwd_home  # opt out of the conftest fixture that makes paths.home() follow $HOME

BOTH = (".harpo", ".harpocrates")


@pytest.fixture
def homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(the home the password database names, a planted home $HOME points at, holding an allow-everything list)."""
    real, planted = tmp_path / "real-home", tmp_path / "planted-home"
    real.mkdir()
    (planted / ".harpo").mkdir(parents=True)
    (planted / ".harpo" / "allow").write_text("/\n")
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(real)))
    monkeypatch.setenv("HOME", str(planted))
    return real, planted


def _edit_event(target: str, cwd: Path) -> str:
    return json.dumps({"tool_name": "Edit", "cwd": str(cwd), "permission_mode": "default", "session_id": "s1",
                       "tool_use_id": "t1", "tool_input": {"file_path": target, "old_string": "a",
                                                           "new_string": "b"}})


def _decision(raw: str, capsys) -> tuple[int, str]:
    code = hooks.decide(raw, ask=True)
    out = capsys.readouterr().out
    return code, (json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "")


def test_sec_08_home_is_the_password_database_not_home(homes) -> None:
    real, _ = homes
    assert paths.home() == real


@pytest.mark.parametrize("entry", [KeyError("no such uid"), SimpleNamespace(pw_dir=""), SimpleNamespace(pw_dir="rel")])
def test_sec_08_no_usable_password_entry_fails_closed(homes, monkeypatch, entry) -> None:
    def lookup(uid):
        if isinstance(entry, Exception):
            raise entry
        return entry

    monkeypatch.setattr(pwd, "getpwuid", lookup)
    with pytest.raises(RuntimeError):  # never a fall back to the $HOME an agent could have planted
        paths.home()
    with pytest.raises(RuntimeError):
        paths.state_dir()


def test_sec_08_a_planted_home_does_not_move_the_state_dir(homes) -> None:
    real, planted = homes
    (real / ".harpocrates").mkdir()
    (real / ".harpocrates" / "allow").write_text("name:graphify-out\n")
    (planted / ".harpocrates").mkdir()
    (planted / ".harpocrates" / "allow").write_text("name:planted-out\n")
    assert paths.state_dir() == real / ".harpo"  # migrated from the real home's old folder
    assert allowlist.names() == ["graphify-out"] and allowlist.entries() == []
    assert not allowlist.allowed(Path("/etc/hosts"))  # the planted "/" entry is not the allowlist
    assert sorted(p.name for p in planted.iterdir()) == [".harpo", ".harpocrates"]  # nothing created in the planted home
    assert (planted / ".harpo" / "allow").read_text() == "/\n"


def test_sec_08_grants_live_under_the_password_home(homes) -> None:
    real, planted = homes
    assert grants.add_pending("sess-1", "tu-1", Path("/proj/x.py"), "Edit")
    assert (real / ".harpo" / "grants" / "sess-1.json").is_file()
    assert not (planted / ".harpo" / "grants").exists()


def test_sec_08_name_rules_measure_the_password_home(homes) -> None:
    assert allowlist._refusal("real-home") is not None  # a component of the real home: far too broad for a rule


@pytest.mark.parametrize("tool_target", ["real", "tilde"])
def test_sec_08_a_planted_home_does_not_disarm_the_edit_guard(homes, capsys, tool_target: str) -> None:
    real, planted = homes
    target = str(real / ".harpo" / "allow") if tool_target == "real" else "~/.harpo/allow"
    assert _decision(_edit_event(target, real), capsys) == (0, "deny")


def test_sec_08_the_tilde_is_checked_as_the_password_home_too(homes, capsys, tmp_path: Path, monkeypatch) -> None:
    """The Edit tool may expand ~ differently from the hook; both expansions are guarded."""
    real, _ = homes
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("HOME", str(elsewhere))
    assert _decision(_edit_event("~/.harpocrates/grants/s1.json", real), capsys) == (0, "deny")


@pytest.mark.parametrize("name", BOTH)
def test_sec_08_inside_state_dir_protects_both_homes(homes, name: str) -> None:
    real, planted = homes
    for base in (real, planted):
        assert allowlist.inside_state_dir(base / name / "allow")
        assert allowlist.inside_state_dir(base / name)
    assert not allowlist.inside_state_dir(real / "work" / "allow")
    assert not allowlist.inside_state_dir(planted / "work" / "allow")


def test_sec_08_the_bash_guard_sees_a_working_directory_in_either_home(homes) -> None:
    real, planted = homes
    (real / ".harpo").mkdir()
    for cwd in (real / ".harpo", planted / ".harpo"):
        assert allowlist.touches("echo x >> allow", cwd)
    assert allowlist.touches("echo x >> ~/.harpo/allow", real)


def test_sec_08_the_rust_scanner_path_expands_tilde_with_the_password_home(homes, monkeypatch) -> None:
    from Harpocrates.core.rust_backend import RustScannerBackend

    real, _ = homes
    exe = real / "scanner"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr("Harpocrates.core.rust_backend.shutil.which", lambda name: None)
    monkeypatch.delenv("HARPOCRATES_RUST_SCANNER", raising=False)
    monkeypatch.setenv("HARPO_RUST_SCANNER", "~/scanner")
    assert RustScannerBackend.discover(required=True).executable == exe

