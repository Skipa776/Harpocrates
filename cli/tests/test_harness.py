"""FR-READ-03 (deny built-in reads, route to safe_read) and FR-INSTALL-03 (support tier per harness)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert entry["matcher"] == "Read|Grep|Bash"
    assert entry["hooks"][0]["command"].endswith("-m Harpocrates.hooks")
    assert "timeout" not in entry["hooks"][0]  # a timed-out hook allows the call; keep Claude's 600 s
    command, args = mcp_server()
    assert args == ["-m", "Harpocrates.mcp.server"] and Path(command).exists()


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
    assert entry["matcher"] == "^Bash$" and entry["hooks"][0]["command"].endswith("-m Harpocrates.hooks")


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
