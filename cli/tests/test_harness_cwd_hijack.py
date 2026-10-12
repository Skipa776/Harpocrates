"""`python -m` puts the working directory first on sys.path, and the agent controls the project directory.
A planted `Harpocrates/` package there would run in place of the real one, so every command the installer
writes must drop the cwd entry (harness.python_flag). Each test runs the exact generated command in a project
holding a planted package that writes a marker file; the marker must not appear."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from Harpocrates import harness

PLANT = "from pathlib import Path\nPath(__file__).resolve().parents[{up}].joinpath('PLANTED').write_text('x')\n"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project directory with a planted Harpocrates package; running its hooks or server writes PLANTED."""
    proj = tmp_path / "project"
    (proj / "Harpocrates" / "mcp").mkdir(parents=True)
    (proj / "Harpocrates" / "__init__.py").write_text("")
    (proj / "Harpocrates" / "mcp" / "__init__.py").write_text("")
    (proj / "Harpocrates" / "hooks.py").write_text(PLANT.format(up=1) + "raise SystemExit(0)\n")
    (proj / "Harpocrates" / "mcp" / "server.py").write_text(PLANT.format(up=2))
    return proj


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PYTHONPATH", raising=False)


EVENT = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Read", "tool_input": {"file_path": "x.txt"}})


def run_shell(command: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, shell=True, cwd=cwd, input=EVENT, capture_output=True, text=True,
                          timeout=60, check=False)


def test_sec_hijack_control_a_bare_python_dash_m_runs_the_planted_hook(project: Path) -> None:
    """Without the flag the planted module does run: the tests below would catch nothing otherwise."""
    run_shell(shlex.join([sys.executable, "-m", "Harpocrates.hooks"]), project)
    assert (project / "PLANTED").exists()


def test_sec_hijack_flag_is_p_from_311_and_i_before() -> None:
    assert harness.python_flag() == ("-P" if sys.version_info >= (3, 11) else "-I")


def _commands() -> dict[str, str]:
    cfg = harness.claude_code_settings()["hooks"]
    out = {event: cfg[event][0]["hooks"][0]["command"] for event in cfg}
    out["codex PreToolUse"] = harness.guarded_hook(harness.mcp_server()[0])
    return out


@pytest.mark.parametrize("event", ["PreToolUse", "PostToolUse", "PostToolUseFailure", "SessionEnd", "codex PreToolUse"])
def test_sec_hijack_generated_hook_commands_ignore_a_planted_package(event: str, project: Path) -> None:
    command = _commands()[event]
    done = run_shell(command, project)
    assert not (project / "PLANTED").exists(), done.stderr
    assert "planted" not in done.stdout.lower()


def test_sec_hijack_the_generated_mcp_command_ignores_a_planted_package(project: Path) -> None:
    command, args = harness.mcp_server()
    assert args[0] == harness.python_flag()
    subprocess.run([command, *args], cwd=project, stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
                   check=False)
    assert not (project / "PLANTED").exists()


def test_sec_hijack_every_generated_command_carries_the_flag() -> None:
    flag = harness.python_flag()
    for event, command in _commands().items():
        words = shlex.split(command)
        inner = shlex.split(words[2]) if words[:2] == ["sh", "-c"] else words
        assert flag in inner and inner.index(flag) < inner.index("-m"), event
    root, tables = harness.codex_config_parts()
    assert json.dumps([flag, "-m", "Harpocrates.mcp.server"]) in tables
    text = harness.setup_instructions("claude-code")
    assert f"{flag} -m Harpocrates.mcp.server" in text
    assert f"{flag} -m Harpocrates.hooks --post" in text.replace("\\\"", '"')
