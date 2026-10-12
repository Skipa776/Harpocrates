"""FR-READ-03 (deny built-in reads, route to safe_read) and FR-INSTALL-03 (support tier per harness)."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from Harpocrates import harness as harness_mod
from Harpocrates.harness import (
    HARNESSES,
    SENSITIVE_FILES,
    SENSITIVE_HOME_DIRS,
    claude_code_settings,
    codex_config,
    detect,
    mcp_server,
    tier_report,
)


def _guarded_parts(command: str) -> tuple[str, str]:
    """(python, script) of a PreToolUse hook command, which must be `sh -c '<script>' <python>`."""
    argv = shlex.split(command)
    assert argv[:2] == ["sh", "-c"] and len(argv) == 4, command
    return argv[3], argv[2]


def _detected(*names: str) -> dict[str, bool]:
    return {h.name: h.name in names for h in HARNESSES}


def test_fr_install_03_tier_for_each_detected_harness() -> None:
    out = tier_report(_detected(*(h.name for h in HARNESSES)))
    for h in HARNESSES:
        assert h.name in out and h.tier in out
    assert "egress gate" in out.lower()  # say plainly that the gate isn't there yet


def test_fr_install_03_read_tool_only_says_what_is_unprotected() -> None:
    out = tier_report(_detected("Cursor"))
    assert "Read tool only" in out
    assert "not protected" in out.lower()
    assert "terminal" in out.lower() and "tab completion" in out.lower()


def test_fr_install_03_partial_tier_names_the_gap() -> None:
    out = tier_report(_detected("GitHub Copilot (own API key)"))
    assert "Partial" in out and "inline completions" in out.lower()


def test_fr_install_03_nothing_detected() -> None:
    assert "no supported harness" in tier_report(_detected()).lower()


def test_fr_install_03_detect_uses_path_and_home(tmp_path: Path) -> None:
    (tmp_path / ".codex").mkdir()
    found = detect(home=tmp_path, which=lambda cmd: "/bin/claude" if cmd == "claude" else None,
                   apps=tmp_path / "Applications")
    assert found["Claude Code"] and found["Codex CLI"] and not found["Cursor"]


def test_fr_read_03_claude_code_denies_every_sensitive_path() -> None:
    cfg = json.loads(json.dumps(claude_code_settings()))  # must be plain JSON
    deny = cfg["permissions"]["deny"]
    for name in SENSITIVE_FILES:
        assert f"Read(//**/{name})" in deny, name
    for d in SENSITIVE_HOME_DIRS:
        assert f"Read({d}/**)" in deny, d
    assert set(cfg["permissions"]["allow"]) == {"mcp__harpocrates__safe_read", "mcp__harpocrates__safe_grep"}
    (entry,) = cfg["hooks"]["PreToolUse"]
    assert entry["matcher"] == "Read|Grep|Bash|Edit|MultiEdit|Write|NotebookEdit"
    assert _guarded_parts(entry["hooks"][0]["command"]) == (sys.executable, '"$0" -m Harpocrates.hooks --ask || exit 2')
    assert "timeout" not in entry["hooks"][0]  # a timed-out hook allows the call; keep Claude's 600 s
    command, args = mcp_server()
    assert args == ["-m", "Harpocrates.mcp.server"] and Path(command).exists()


def test_fr_read_06_claude_code_settings_register_the_post_hooks() -> None:
    hooks = json.loads(json.dumps(claude_code_settings()))["hooks"]
    assert set(hooks) == {"PreToolUse", "PostToolUse", "PostToolUseFailure", "SessionEnd"}
    for event in ("PostToolUse", "PostToolUseFailure"):
        (entry,) = hooks[event]
        assert entry["matcher"] == "Edit|MultiEdit|Write|NotebookEdit"
        (hook,) = entry["hooks"]
        assert hook["type"] == "command" and hook["command"].endswith("-m Harpocrates.hooks --post")
    (end,) = hooks["SessionEnd"]
    assert "matcher" not in end and end["hooks"][0]["command"].endswith("-m Harpocrates.hooks --post")
    (pre,) = hooks["PreToolUse"]
    assert _guarded_parts(pre["hooks"][0]["command"])[1].endswith("--ask || exit 2")  # the ask hook is unchanged


def test_fr_read_03_codex_denies_every_sensitive_path() -> None:
    tomllib = pytest.importorskip("tomllib")
    cfg = tomllib.loads(codex_config())
    profile = cfg["permissions"][cfg["default_permissions"]]
    for d in SENSITIVE_HOME_DIRS:
        assert profile["filesystem"][d] == "deny", d
    roots = profile["filesystem"][":workspace_roots"]
    for name in SENSITIVE_FILES:
        assert roots[f"**/{name}"] == "deny", name
    assert cfg["mcp_servers"]["harpocrates"]["args"] == ["-m", "Harpocrates.mcp.server"]
    (entry,) = cfg["hooks"]["PreToolUse"]
    assert entry["matcher"] == "^Bash$"
    assert _guarded_parts(entry["hooks"][0]["command"]) == (sys.executable, '"$0" -m Harpocrates.hooks || exit 2')


def test_fr_read_03_cli_prints_setup_and_tiers() -> None:
    from typer.testing import CliRunner

    from Harpocrates.cli import app

    runner = CliRunner()
    claude = runner.invoke(app, ["setup", "claude-code"])
    assert claude.exit_code == 0 and "claude mcp add --scope user harpocrates" in claude.output
    assert "Read(~/.ssh/**)" in claude.output
    codex = runner.invoke(app, ["setup", "codex"])
    assert codex.exit_code == 0 and "[mcp_servers.harpocrates]" in codex.output
    assert runner.invoke(app, ["setup", "cursor"]).exit_code == 2
    assert runner.invoke(app, ["setup"]).exit_code == 0


def test_sec_06_threat_model_lists_the_three_residual_risks() -> None:
    text = (Path(__file__).resolve().parents[2] / "docs" / "threat-model.md").read_text()
    sections = text.split("\n## ")
    for heading in ("other exfiltration channels", "transcripts saved to local disk with raw tool output",
                    "agent traffic that bypasses the API base URL"):
        body = next((s for s in sections if heading in s.splitlines()[0]), "")
        assert "**Risk.**" in body and "**What helps:**" in body, heading


def test_fr_install_03_harness_without_setup_is_unprotected() -> None:
    out = tier_report(_detected("OpenCode"))
    assert "Protected today: nothing" in out


def test_fr_read_03_odd_interpreter_paths_survive(monkeypatch: pytest.MonkeyPatch) -> None:
    import shlex

    import Harpocrates.harness as harness

    odd = 'C:\\Program Files\\Py "3"\\python.exe'
    monkeypatch.setattr(harness, "mcp_server", lambda: (odd, ["-m", "Harpocrates.mcp.server"]))
    line = next(ln for ln in harness.setup_instructions("Claude-Code").splitlines() if "claude mcp add" in ln)
    assert shlex.split(line)[-3] == odd
    tomllib = pytest.importorskip("tomllib")
    assert tomllib.loads(harness.codex_config())["mcp_servers"]["harpocrates"]["command"] == odd


def test_fr_read_06_setup_text_says_reads_are_refused_and_only_edits_prompt() -> None:
    from Harpocrates.harness import AGENT_NOTE, CLAUDE_NOTE, CODEX_NOTE, setup_instructions

    claude = setup_instructions("claude-code")
    assert CLAUDE_NOTE in claude and "Yes lets Claude read" not in claude
    assert "refused" in claude and "safe_read" in claude and "edit" in claude.lower()
    assert CLAUDE_NOTE.startswith(AGENT_NOTE) and "To edit such a file, use Edit directly" in CLAUDE_NOTE
    assert "once per file and session" in CLAUDE_NOTE
    codex = setup_instructions("codex")
    assert CODEX_NOTE in codex and CODEX_NOTE.endswith("Ask the user to edit such files.")
    assert "Edit directly" not in codex


def test_fr_install_01_codex_config_stays_deny_only() -> None:
    text = codex_config()
    assert "--ask" not in text and "fails open" in text


def test_fr_install_01_pretooluse_hooks_fail_closed_when_the_interpreter_is_gone(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing venv makes a bare hook command exit 127, which a harness treats as allow. Wrapped, it is 2."""
    tomllib = pytest.importorskip("tomllib")
    monkeypatch.setattr(harness_mod, "mcp_server", lambda: ("/nonexistent/venv/bin/python", ["-m", "x"]))
    cfg = json.loads(json.dumps(claude_code_settings()))
    pre = [cfg["hooks"]["PreToolUse"][0]["hooks"][0]["command"],
           tomllib.loads(codex_config())["hooks"]["PreToolUse"][0]["hooks"][0]["command"]]
    for command in pre:
        done = subprocess.run(command, shell=True, input=b"{}", capture_output=True)
        assert done.returncode == 2, command
    post = cfg["hooks"]["SessionEnd"][0]["hooks"][0]["command"]  # --post must never block
    assert subprocess.run(post, shell=True, input=b"{}", capture_output=True).returncode == 127


def test_fr_install_01_wrapped_hook_keeps_the_hooks_own_exit_code(tmp_path: Path) -> None:
    """While the interpreter works the wrapper changes nothing: a benign Read passes (0), garbage is blocked (2)."""
    pre = json.loads(json.dumps(claude_code_settings()))["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    benign = json.dumps({"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "nope.txt")},
                         "session_id": "s", "cwd": str(tmp_path), "hook_event_name": "PreToolUse"}).encode()
    run = {"shell": True, "capture_output": True, "env": {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}}
    assert subprocess.run(pre, input=benign, **run).returncode == 0
    assert subprocess.run(pre, input=b"garbage", **run).returncode == 2
