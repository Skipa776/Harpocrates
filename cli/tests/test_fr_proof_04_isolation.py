"""FR-PROOF-04: agent mode runs `claude` and `codex` in a throwaway HOME, and nothing reaches the real one.

These tests build throwaway homes under the system temp folder and run Harpo's installer into them through the test-side
bootstrap (bench/agent_isolation.py), the same pattern as conftest.child_python. They never run an agent. The real home is
only ever read: a snapshot (mode, size, mtime, SHA-256; for ~/.codex/auth.json only existence, mode, size and mtime).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import _real_home
import pytest

from bench import agent_isolation as iso

REAL = _real_home.real_home()


@pytest.fixture
def tw():
    throwaway = iso.make_throwaway()
    yield throwaway
    iso.discard(throwaway)


def test_fr_proof_04_isolation_throwaway_lives_outside_the_real_home(tw) -> None:
    for folder in (tw.root, tw.home, tw.repo):
        assert folder.is_dir() and not folder.resolve().is_relative_to(REAL.resolve()) and folder.resolve() != REAL.resolve()
    assert tw.home.parent == tw.root and tw.repo.parent == tw.root
    iso.assert_outside_real_home(tw.home)
    for bad in (REAL, REAL / "projects" / "x", REAL / ".claude"):
        with pytest.raises(iso.IsolationError):
            iso.assert_outside_real_home(bad)


def test_fr_proof_04_isolation_discard_only_removes_its_own_throwaway(tw, tmp_path: Path) -> None:
    other = iso.Throwaway(tmp_path / "not-a-harness-folder", tmp_path / "h", tmp_path / "r")
    other.root.mkdir()
    with pytest.raises(iso.IsolationError):
        iso.discard(other)
    assert other.root.exists()
    mine = iso.make_throwaway()
    iso.discard(mine)
    assert not mine.root.exists()


def test_fr_proof_04_isolation_snapshot_never_reads_auth_json(tmp_path: Path) -> None:
    home = tmp_path / "h"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "auth.json").write_text('{"token": "SECRET-STAYS-OUT-OF-THE-SNAPSHOT"}')
    (home / ".codex" / "config.toml").write_text("model = 'x'\n")
    before = iso.real_home_snapshot(home)
    row = before[".codex/auth.json"]
    assert row[0] == "f" and row[2] == len('{"token": "SECRET-STAYS-OUT-OF-THE-SNAPSHOT"}') and row[4] is None  # no hash
    assert "SECRET-STAYS" not in repr(before)
    assert before[".codex/config.toml"][4] is not None  # the config file is hashed, as the session guard does
    assert iso.snapshot_diff(before, iso.real_home_snapshot(home)) == []
    os.utime(home / ".codex" / "auth.json", ns=(1, 1))
    assert iso.snapshot_diff(before, iso.real_home_snapshot(home)) == [".codex/auth.json"]
    (home / ".codex" / "auth.json").unlink()
    assert ".codex/auth.json" in iso.snapshot_diff(before, iso.real_home_snapshot(home))


def test_fr_proof_04_isolation_bootstrap_refuses_the_real_home(tw) -> None:
    for bad_home in (str(REAL), str(REAL / "anywhere"), "relative/path"):
        done = subprocess.run([sys.executable, "-I", "-c", iso.BOOT, bad_home, "-", "json.tool"], cwd=tw.root,
                              capture_output=True, text=True, timeout=60, check=False, input="{}")
        assert done.returncode == iso.REFUSED_EXIT, (bad_home, done.stderr)
        assert "json" not in done.stdout  # nothing ran


def test_fr_proof_04_isolation_wrapper_pins_paths_home_to_the_throwaway_home(tw) -> None:
    """The hook's own guard: an Edit of a file inside the state folder is denied only when the state folder is the one the
    process computes. The wrapper makes that the throwaway's; a plain interpreter computes the real one."""
    wrapper = iso.write_wrapper(tw)
    target = tw.home / ".harpo" / "allow"
    event = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Edit", "cwd": "/",
                        "tool_input": {"file_path": str(target), "old_string": "a", "new_string": "b"}})
    env = {"PATH": os.environ["PATH"], "HOME": "/nonexistent-home-for-the-test"}  # not the throwaway: only the pin can matter

    def denied(argv: list[str]) -> bool:
        done = subprocess.run(argv, input=event, cwd=tw.root, env=env, capture_output=True, text=True, timeout=60, check=False)
        try:
            return json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
        except (ValueError, KeyError):
            return False

    assert not denied([sys.executable, "-I", "-m", "Harpocrates.hooks"])  # control: the real home is computed, so no deny
    assert denied([str(wrapper), "-I", "-m", "Harpocrates.hooks"])
    assert denied([str(wrapper), "-E", "-P", "-m", "Harpocrates.hooks"]) or sys.version_info < (3, 11)


def test_fr_proof_04_isolation_wrapper_needs_a_module(tw) -> None:
    wrapper = iso.write_wrapper(tw)
    done = subprocess.run([str(wrapper), "-I"], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 2 and "-m" in done.stderr


@pytest.mark.parametrize("target, marker", [("codex", ".codex/config.toml"), ("claude-code", ".claude/settings.json")])
def test_fr_proof_04_isolation_install_writes_only_into_the_throwaway_home(tw, target: str, marker: str) -> None:
    before = iso.real_home_snapshot()
    result = iso.install_harpo(tw, target, path="/usr/bin:/bin")  # no `claude` on this PATH: the MCP step is skipped
    assert result.returncode == 0, result.stderr
    assert iso.snapshot_diff(before, iso.real_home_snapshot()) == []
    assert (tw.home / marker).is_file() and (tw.home / ".harpo" / "install.json").is_file()
    record = json.loads((tw.home / ".harpo" / "install.json").read_text())
    wrapper = str(tw.root / "bin" / "python-throwaway")
    assert record["python"] == wrapper  # hooks and the MCP server start through the wrapper
    text = (tw.home / marker).read_text()
    assert "Harpocrates.hooks" in text and wrapper in text and sys.executable not in text


def test_fr_proof_04_isolation_env_is_built_from_scratch(tw, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "HTTP_PROXY",
                 "https_proxy", "GITHUB_TOKEN", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.setenv(name, "from-the-parent-process")
    env = iso.clean_env(tw, "http://127.0.0.1:5555", {"ANTHROPIC_API_KEY": iso.FAKE_API_KEY})
    assert env["HOME"] == str(tw.home) and env["TMPDIR"].startswith(str(tw.root))
    assert env["ANTHROPIC_API_KEY"] == iso.FAKE_API_KEY and iso.FAKE_API_KEY.startswith("sk-ant-api03-FAKE")
    assert "from-the-parent-process" not in " ".join(env.values())
    assert not {"OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "HTTP_PROXY", "https_proxy", "GITHUB_TOKEN",
                "CODEX_HOME", "CLAUDE_CONFIG_DIR", "ANTHROPIC_AUTH_TOKEN"} & set(env)
    assert env["NO_PROXY"] == "127.0.0.1,localhost" and env["GIT_CONFIG_GLOBAL"] == os.devnull
    with pytest.raises(iso.IsolationError):
        iso.clean_env(tw, "http://example.com:80", {})
    with pytest.raises(iso.IsolationError):
        iso.clean_env(tw, "http://192.168.1.2:80", {})
