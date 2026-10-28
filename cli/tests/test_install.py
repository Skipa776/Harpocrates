"""FR-INSTALL-01: `harpo install` / `uninstall` write the read-tool setup into each harness's own files.

Every test runs with HOME in a temp dir and PATH limited to a temp bin dir holding a fake `claude` that logs its
arguments, so nothing touches the real ~/.claude, ~/.codex, ~/.harpo or the real `claude` binary.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from toml_compat import tomllib
from typer.testing import CliRunner

from Harpocrates import harness, install
from Harpocrates.cli import app

PY = sys.executable
ADD_LINE = f"mcp add --scope user harpo -- {PY} {harness.python_flag()} -m Harpocrates.mcp.server"
CLAUDE_START, CLAUDE_END = "<!-- harpo:start -->", "<!-- harpo:end -->"
TOML_START, TOML_END = "# >>> harpo >>>", "# <<< harpo <<<"

FAKE_CLAUDE = """#!/bin/sh
echo "$*" >> "@LOG@"
case "$1 $2" in
  "mcp get") f="@STATE@.$3"; [ -f "$f" ] && { /bin/cat "$f"; exit 0; }; exit 1;;
  "mcp add") echo "$*" > "@STATE@.$5"; exit 0;;
  "mcp remove") /bin/rm -f "@STATE@.$5"; exit 0;;
esac
exit 0
"""


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))  # no real claude or codex can be found
    return h


@pytest.fixture
def fake_claude(home: Path) -> Path:
    """Put a fake `claude` on PATH; returns its call log."""
    bindir, log = home.parent / "bin", home.parent / "claude.log"
    exe = bindir / "claude"
    exe.write_text(FAKE_CLAUDE.replace("@LOG@", str(log)).replace("@STATE@", str(home.parent / "claude.state")))  # one file per server name
    exe.chmod(0o755)
    return log


def calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def snap(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def no_bak(files: dict[str, bytes]) -> dict[str, bytes]:
    return {k: v for k, v in files.items() if not k.endswith(".harpo.bak")}


def write_json(path: Path, obj: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2) + "\n").encode()
    path.write_bytes(data)
    return data


def settings_of(home: Path) -> dict:
    return json.loads((home / ".claude" / "settings.json").read_text())


def ours(report: list) -> bool:
    return not any(r.refused or any(s.state == "failed" for s in r.steps) for r in report)


# --- Claude Code -------------------------------------------------------------------------------------------------

def test_fr_install_01_claude_fresh_install_writes_every_piece(home: Path, fake_claude: Path) -> None:
    report = install.install_harnesses(["claude-code"])
    assert ours(report)
    want = harness.claude_code_settings()
    got = settings_of(home)
    assert got["permissions"] == want["permissions"] and got["hooks"] == want["hooks"]
    assert calls(fake_claude)[-1] == ADD_LINE
    note = (home / ".claude" / "CLAUDE.md").read_text()
    assert note.count(CLAUDE_START) == 1 and note.count(CLAUDE_END) == 1
    assert harness.CLAUDE_NOTE in note
    assert "Edit directly" in harness.CLAUDE_NOTE and "once per file and session" in harness.CLAUDE_NOTE


def test_fr_install_01_claude_second_install_is_a_no_op(home: Path, fake_claude: Path) -> None:
    install.install_harnesses(["claude-code"])
    first = snap(home)
    stamp = (home / ".claude" / "settings.json").stat().st_mtime_ns
    report = install.install_harnesses(["claude-code"])
    assert snap(home) == first
    assert (home / ".claude" / "settings.json").stat().st_mtime_ns == stamp  # not even rewritten
    assert all(s.state == "unchanged" for s in report[0].steps)
    assert sum(1 for c in calls(fake_claude) if c.startswith("mcp add")) == 1  # already current: no re-register


def test_fr_install_01_claude_keeps_unrelated_settings_and_order(home: Path, fake_claude: Path) -> None:
    other_hook = {"matcher": "Bash", "hooks": [{"type": "command", "command": "/usr/bin/true"}]}
    original = {"theme": "dark",
                "permissions": {"allow": ["Bash(ls:*)"], "deny": ["Bash(rm:*)"], "defaultMode": "plan"},
                "hooks": {"PreToolUse": [other_hook], "Stop": [{"hooks": [{"type": "command", "command": "x"}]}]},
                "model": "opus"}
    write_json(home / ".claude" / "settings.json", original)
    install.install_harnesses(["claude-code"])
    got = settings_of(home)
    assert list(got)[:4] == ["theme", "permissions", "hooks", "model"]
    assert got["theme"] == "dark" and got["model"] == "opus" and got["permissions"]["defaultMode"] == "plan"
    assert got["permissions"]["deny"][0] == "Bash(rm:*)" and got["permissions"]["allow"][0] == "Bash(ls:*)"
    assert set(harness.claude_code_settings()["permissions"]["deny"]) <= set(got["permissions"]["deny"])
    assert got["hooks"]["PreToolUse"][0] == other_hook and len(got["hooks"]["PreToolUse"]) == 2
    assert got["hooks"]["Stop"] == original["hooks"]["Stop"]
    assert len(got["hooks"]["PostToolUse"]) == 1


def test_fr_install_01_claude_replaces_a_stale_harpocrates_hook(home: Path, fake_claude: Path) -> None:
    stale = {"matcher": "Read|Grep|Bash", "hooks": [{"type": "command",
                                                      "command": "/old/venv/bin/python -m Harpocrates.hooks"}]}
    stale_post = {"hooks": [{"type": "command", "command": "/old/venv/bin/python -m Harpocrates.hooks --post"}]}
    write_json(home / ".claude" / "settings.json",
               {"hooks": {"PreToolUse": [stale], "SessionEnd": [stale_post]}})
    install.install_harnesses(["claude-code"])
    hooks = settings_of(home)["hooks"]
    assert hooks == harness.claude_code_settings()["hooks"]
    assert "/old/venv" not in json.dumps(hooks)


def test_fr_install_01_claude_invalid_settings_refuse_and_write_nothing(home: Path, fake_claude: Path) -> None:
    for bad in (b"{not json", b'["a", "list"]', b""):
        (home / ".claude").mkdir(exist_ok=True)
        (home / ".claude" / "settings.json").write_bytes(bad)
        before = snap(home)
        report = install.install_harnesses(["claude-code"])
        assert report[0].refused and "settings.json" in report[0].refused
        assert snap(home) == before
        assert calls(fake_claude) == []


def test_fr_install_01_claude_backup_once_and_mode_kept(home: Path, fake_claude: Path) -> None:
    path = home / ".claude" / "settings.json"
    original = write_json(path, {"theme": "dark"})
    path.chmod(0o640)
    install.install_harnesses(["claude-code"])
    bak = path.with_name("settings.json.harpo.bak")
    assert bak.read_bytes() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    write_json(path, {"theme": "light"})
    install.install_harnesses(["claude-code"])
    assert bak.read_bytes() == original  # an existing backup is never overwritten


def test_fr_install_01_claude_reinstall_replaces_the_note_block(home: Path, fake_claude: Path) -> None:
    md = home / ".claude" / "CLAUDE.md"
    md.parent.mkdir()
    md.write_text(f"# mine\n\n{CLAUDE_START}\nold note\n{CLAUDE_END}\n\n## after\n")
    install.install_harnesses(["claude-code"])
    text = md.read_text()
    assert "old note" not in text and harness.CLAUDE_NOTE in text
    assert text.startswith("# mine\n\n") and text.endswith("\n\n## after\n")
    assert text.count(CLAUDE_START) == 1


def test_fr_install_01_claude_re_registers_a_stale_mcp_server(home: Path, fake_claude: Path) -> None:
    (home.parent / "claude.state.harpo").write_text("mcp add --scope user harpo -- /old/python -m x\n")
    install.install_harnesses(["claude-code"])
    seen = calls(fake_claude)
    assert seen == ["mcp get harpocrates", "mcp get harpo", "mcp remove --scope user harpo", ADD_LINE]


def test_fr_install_01_claude_missing_prints_the_command_and_still_writes_settings(home: Path) -> None:
    (home / ".claude").mkdir()  # detected by its directory alone
    result = CliRunner().invoke(app, ["install"])
    assert result.exit_code == 0, result.output
    assert f"claude {ADD_LINE}" in result.output
    assert settings_of(home)["permissions"]["allow"] == list(harness.MCP_TOOLS)


# --- Codex ---------------------------------------------------------------------------------------------------------

def codex_cfg(home: Path) -> dict:
    return tomllib.loads((home / ".codex" / "config.toml").read_text())


def test_fr_install_01_codex_fresh_install(home: Path) -> None:
    report = install.install_harnesses(["codex"])
    assert ours(report)
    text = (home / ".codex" / "config.toml").read_text()
    assert TOML_START in text and TOML_END in text
    cfg = codex_cfg(home)
    want = tomllib.loads(harness.codex_config())
    assert cfg == want and cfg["default_permissions"] == "harpo"
    assert "--ask" not in text and "deny-only" in text.lower()  # Codex fails open on ask
    agents = (home / ".codex" / "AGENTS.md").read_text()
    assert harness.CODEX_NOTE in agents and "Ask the user to edit such files." in agents
    assert "Edit directly" not in agents
    assert any("/hooks" in n for n in report[0].notes)


def test_fr_install_01_codex_second_install_is_a_no_op(home: Path) -> None:
    install.install_harnesses(["codex"])
    first = snap(home)
    install.install_harnesses(["codex"])
    assert snap(home) == first


def test_fr_install_01_codex_keeps_user_config_and_replaces_a_stale_block(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mine = 'model = "o3"\n\n[mcp_servers.other]\ncommand = "x"\n'
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(mine)
    with monkeypatch.context() as m:
        m.setattr(harness, "mcp_server", lambda: ("/old/python", ["-m", "Harpocrates.mcp.server"]))
        install.install_harnesses(["codex"])
    assert codex_cfg(home)["mcp_servers"]["harpo"]["command"] == "/old/python"
    install.install_harnesses(["codex"])
    text = cfg.read_text()
    cfg_now = codex_cfg(home)
    assert "/old/python" not in text
    assert cfg_now["mcp_servers"]["harpo"]["command"] == PY
    assert cfg_now["model"] == "o3" and cfg_now["mcp_servers"]["other"] == {"command": "x"}
    assert len(cfg_now["hooks"]["PreToolUse"]) == 1


@pytest.mark.parametrize("clash", [
    '[mcp_servers.harpo]\ncommand = "mine"\n',
    'default_permissions = ":workspace"\n',
    '[permissions.harpo.filesystem]\n"x" = "deny"\n',
])
def test_fr_install_01_codex_conflict_outside_the_block_refuses(home: Path, clash: str) -> None:
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(clash)
    before = snap(home)
    report = install.install_harnesses(["codex"])
    assert report[0].refused
    assert snap(home) == before  # no AGENTS.md either


def test_fr_install_01_codex_invalid_toml_refuses(home: Path) -> None:
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text("model = = broken\n")
    before = snap(home)
    assert install.install_harnesses(["codex"])[0].refused
    assert snap(home) == before


def test_fr_install_01_codex_user_hook_table_that_breaks_the_result_writes_nothing(home: Path) -> None:
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('[hooks]\nPreToolUse = [{matcher = "x"}]\n')  # a static array our [[hooks.PreToolUse]] can't extend
    before = snap(home)
    assert install.install_harnesses(["codex"])[0].refused
    assert snap(home) == before


# --- uninstall, dry run, detection ---------------------------------------------------------------------------------

def test_fr_install_01_uninstall_restores_the_originals(home: Path, fake_claude: Path) -> None:
    write_json(home / ".claude" / "settings.json",
               {"theme": "dark", "permissions": {"deny": ["Bash(rm:*)"]},
                "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "/bin/true"}]}]}})
    (home / ".claude" / "CLAUDE.md").write_text("# my notes\n\nbe nice\n")
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text('model = "o3"\n\n[mcp_servers.other]\ncommand = "x"\n')
    (home / ".codex" / "AGENTS.md").write_text("# agents\n")
    original = snap(home)
    install.install_harnesses(["claude-code", "codex"])
    assert snap(home) != original
    reports = install.uninstall_harnesses(["claude-code", "codex"])
    assert ours(reports)
    assert no_bak(snap(home)) == original
    assert calls(fake_claude)[-1] == "mcp remove --scope user harpo"
    assert not (home.parent / "claude.state.harpo").exists()


def test_fr_install_01_uninstall_of_a_fresh_install_leaves_no_files(home: Path, fake_claude: Path) -> None:
    install.install_harnesses(["claude-code", "codex"])
    install.uninstall_harnesses(["claude-code", "codex"])
    assert no_bak(snap(home)) == {}


def test_fr_install_01_uninstall_without_install_changes_nothing(home: Path, fake_claude: Path) -> None:
    write_json(home / ".claude" / "settings.json", {"theme": "dark"})
    before = snap(home)
    reports = install.uninstall_harnesses(["claude-code", "codex"])
    assert snap(home) == before
    assert not any(s.state in ("removed", "failed") for r in reports for s in r.steps)


def test_fr_install_01_uninstall_removes_a_stale_path_hook_too(home: Path, fake_claude: Path) -> None:
    write_json(home / ".claude" / "settings.json", {"hooks": {"SessionEnd": [
        {"hooks": [{"type": "command", "command": "/old/python -m Harpocrates.hooks --post"}]}]}})
    install.uninstall_harnesses(["claude-code"])
    assert no_bak(snap(home)) == {}  # nothing of the user's was left in it


def test_fr_install_01_dry_run_writes_nothing(home: Path, fake_claude: Path) -> None:
    (home / ".codex").mkdir()
    write_json(home / ".claude" / "settings.json", {"theme": "dark"})
    before = snap(home)
    result = CliRunner().invoke(app, ["install", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert snap(home) == before and calls(fake_claude) == []
    assert "dry run" in result.output.lower() and "would" in result.output.lower()
    result = CliRunner().invoke(app, ["uninstall", "--dry-run"])
    assert result.exit_code == 0 and snap(home) == before and calls(fake_claude) == []


def test_fr_install_01_detection_picks_only_present_harnesses(home: Path) -> None:
    assert install.detected_targets() == []
    (home / ".codex").mkdir()
    assert install.detected_targets() == ["codex"]
    (home / ".claude").mkdir()
    assert install.detected_targets() == ["claude-code", "codex"]


def test_fr_install_01_cli_without_flags_installs_only_what_is_detected(home: Path) -> None:
    (home / ".codex").mkdir()
    result = CliRunner().invoke(app, ["install"])
    assert result.exit_code == 0, result.output
    assert (home / ".codex" / "config.toml").exists() and not (home / ".claude").exists()
    assert "/hooks" in result.output  # the one manual step


def test_fr_install_01_cli_flags_pick_the_harness_even_when_not_detected(home: Path, fake_claude: Path) -> None:
    result = CliRunner().invoke(app, ["install", "--claude-code"])
    assert result.exit_code == 0, result.output
    assert (home / ".claude" / "settings.json").exists() and not (home / ".codex").exists()
    result = CliRunner().invoke(app, ["uninstall", "--claude-code"])
    assert result.exit_code == 0 and no_bak(snap(home)) == {}


def test_fr_install_01_cli_nothing_detected_is_an_error(home: Path) -> None:
    result = CliRunner().invoke(app, ["install"])
    assert result.exit_code == 1 and "--claude-code" in result.output
    assert snap(home) == {}


def test_fr_install_01_cli_refusal_exits_nonzero(home: Path, fake_claude: Path) -> None:
    (home / ".claude").mkdir()
    (home / ".claude" / "settings.json").write_text("{oops")
    result = CliRunner().invoke(app, ["install"])
    assert result.exit_code == 1 and "settings.json" in result.output


def test_fr_install_01_writes_through_a_symlinked_settings_file(home: Path, fake_claude: Path) -> None:
    real = home.parent / "dotfiles" / "settings.json"
    write_json(real, {"theme": "dark"})
    (home / ".claude").mkdir()
    link = home / ".claude" / "settings.json"
    os.symlink(real, link)
    install.install_harnesses(["claude-code"])
    assert link.is_symlink() and json.loads(real.read_text())["theme"] == "dark"
    assert "hooks" in json.loads(real.read_text())


# --- install record (~/.harpo/install.json): uninstall removes only what install added ------------------------

def record(home: Path) -> dict:
    return json.loads((home / ".harpo" / "install.json").read_text())


def test_fr_install_01_uninstall_keeps_the_users_own_identical_rules_and_hooks(home: Path, fake_claude: Path) -> None:
    want = harness.claude_code_settings()
    mine = {"permissions": {"deny": ["Read(//**/.env)"], "allow": ["mcp__harpo__safe_read"]},
            "hooks": {"SessionEnd": want["hooks"]["SessionEnd"]}}  # the user pasted some of ours by hand earlier
    original = write_json(home / ".claude" / "settings.json", mine)
    install.install_harnesses(["claude-code"])
    got = settings_of(home)
    assert got["permissions"]["deny"].count("Read(//**/.env)") == 1
    assert got["hooks"]["SessionEnd"] == mine["hooks"]["SessionEnd"]
    rec = record(home)["claude-code"]
    assert "Read(//**/.env)" not in rec["permissions"]["deny"]
    assert "mcp__harpo__safe_read" not in rec["permissions"]["allow"]
    assert "SessionEnd" not in rec["hooks"] and "PreToolUse" in rec["hooks"]
    install.uninstall_harnesses(["claude-code"])
    assert (home / ".claude" / "settings.json").read_bytes() == original
    assert not (home / ".harpo" / "install.json").exists()


def test_fr_install_01_record_is_private_and_stays_accurate_on_reinstall(
        home: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_json(home / ".claude" / "settings.json", {"permissions": {"deny": ["Read(//**/.env)"]}})
    install.install_harnesses(["claude-code", "codex"])
    path = home / ".harpo" / "install.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    first = path.read_bytes()
    assert record(home)["claude-code"]["mcp"] is True and "codex" in record(home)
    install.install_harnesses(["claude-code", "codex"])
    assert path.read_bytes() == first  # an unchanged reinstall keeps every record
    with monkeypatch.context() as m:  # a new python: the hooks are replaced, still all ours
        m.setattr(harness, "mcp_server", lambda: ("/new/python", ["-m", "Harpocrates.mcp.server"]))
        install.install_harnesses(["claude-code"])
    rec = record(home)["claude-code"]
    assert all("/new/python" in c for cmds in rec["hooks"].values() for c in cmds)
    assert "Read(//**/.env)" not in rec["permissions"]["deny"] and rec["permissions"]["deny"]
    install.uninstall_harnesses(["claude-code"])
    assert settings_of(home) == {"permissions": {"deny": ["Read(//**/.env)"]}}
    assert "claude-code" not in record(home) and "codex" in record(home)  # the other harness's record stays


def test_fr_install_01_record_notes_the_stale_hooks_install_replaced(home: Path, fake_claude: Path) -> None:
    stale = "/old/venv/bin/python -m Harpocrates.hooks --post"
    write_json(home / ".claude" / "settings.json",
               {"hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command": stale}]}]}})
    install.install_harnesses(["claude-code"])
    assert record(home)["claude-code"]["replaced"] == [stale]
    assert stale not in json.dumps(settings_of(home))


def test_fr_install_01_uninstall_without_a_record_leaves_rules_and_mcp(home: Path, fake_claude: Path) -> None:
    install.install_harnesses(["claude-code"])
    (home / ".harpo" / "install.json").unlink()
    result = CliRunner().invoke(app, ["uninstall", "--claude-code"])
    assert result.exit_code == 0, result.output
    got = settings_of(home)
    assert got["permissions"] == harness.claude_code_settings()["permissions"]  # rules stay
    assert "hooks" not in got  # Harpocrates hook commands go
    assert not (home / ".claude" / "CLAUDE.md").exists()  # the marked note goes
    assert not any(c.startswith("mcp remove") for c in calls(fake_claude))
    assert "install record" in result.output.lower() and "claude mcp remove" in result.output


def test_fr_install_01_uninstall_leaves_an_mcp_server_the_user_registered(home: Path, fake_claude: Path) -> None:
    (home.parent / "claude.state.harpo").write_text(ADD_LINE + "\n")  # theirs, with this very python
    install.install_harnesses(["claude-code"])
    assert record(home)["claude-code"]["mcp"] is False
    install.uninstall_harnesses(["claude-code"])
    assert (home.parent / "claude.state.harpo").exists()
    assert not any(c.startswith("mcp remove") for c in calls(fake_claude))


def test_fr_install_01_record_names_the_interpreter_and_old_records_stay_valid(home: Path, fake_claude: Path) -> None:
    install.install_harnesses(["claude-code", "codex"])
    path = home / ".harpo" / "install.json"
    assert record(home)["python"] == sys.executable and Path(record(home)["python"]).is_absolute()
    assert record(home)["python_args"] == harness.python_flags()
    assert record(home)["version"] == 1
    old = {k: v for k, v in record(home).items() if k not in ("python", "python_args")}  # from before they existed
    path.write_text(json.dumps(old, indent=2) + "\n")
    assert ours(install.install_harnesses(["claude-code", "codex"]))
    assert record(home)["python"] == sys.executable  # a reinstall adds it
    path.write_text(json.dumps(old, indent=2) + "\n")
    assert ours(install.uninstall_harnesses(["claude-code", "codex"]))  # and an old record uninstalls as before
    assert not path.exists()


def test_sec_hijack_reinstall_replaces_hook_commands_and_mcp_registration_that_lack_the_flag(
        home: Path, fake_claude: Path) -> None:
    flag = harness.python_flag()
    old_pre = f"{PY} -m Harpocrates.hooks --ask"
    old_post = f"{PY} -m Harpocrates.hooks --post"
    old_hook = lambda c: {"hooks": [{"type": "command", "command": c}]}  # noqa: E731
    write_json(home / ".claude" / "settings.json", {"hooks": {
        "PreToolUse": [old_hook(old_pre)], "PostToolUse": [old_hook(old_post)],
        "SessionEnd": [old_hook(old_post)]}})
    (home.parent / "claude.state.harpo").write_text(f"mcp add --scope user harpo -- {PY} -m Harpocrates.mcp.server\n")
    assert ours(install.install_harnesses(["claude-code"]))
    text = json.dumps(settings_of(home))
    assert old_pre not in text and old_post not in text
    commands = [h["command"] for entries in settings_of(home)["hooks"].values() for e in entries for h in e["hooks"]]
    assert commands and all(f"{flag} -m Harpocrates.hooks" in c for c in commands)
    assert calls(fake_claude)[-1] == ADD_LINE and f"{PY} {flag} -m Harpocrates.mcp.server" in ADD_LINE
    assert sorted(record(home)["claude-code"]["replaced"]) == sorted({old_pre, old_post})


def test_fr_install_01_unreadable_record_blocks_install_but_not_uninstall(home: Path) -> None:
    install.install_harnesses(["codex"])
    (home / ".harpo" / "install.json").write_text("{nope")
    before = snap(home)
    report = install.install_harnesses(["codex"])
    assert report[0].refused and "install.json" in report[0].refused
    assert snap(home) == before
    assert ours(install.uninstall_harnesses(["codex"]))
    assert not (home / ".codex" / "config.toml").exists()  # install made the file, uninstall emptied and removed it
