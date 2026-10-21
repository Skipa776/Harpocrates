"""FR-NAME-01: the MCP server, the permission rules, the marked blocks and the Codex profile are named `harpo`, and
`harpo install` / `harpo uninstall` migrate an install made before the rename (everything was named `harpocrates`).

Claude Code builds tool names from the name a server is registered under, so the rename reaches the allow rules
(mcp__harpo__safe_read), the hook's refusal text, the CLAUDE.md / AGENTS.md notes and Codex's config. What must hold:
a reinstall over an old install leaves nothing under the old names that it owns, AND keeps every deny rule (the deny
rules are the security boundary: the migration only ever touches allow rules, registrations and marked blocks).
Every test runs with HOME in a temp dir and a fake `claude` on PATH that keeps one state file per server name.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import tomllib

from Harpocrates import harness, hooks, install
from Harpocrates.mcp import server

PY = sys.executable
OLD_MD_START, OLD_MD_END = "<!-- harpocrates:start -->", "<!-- harpocrates:end -->"
NEW_MD_START, NEW_MD_END = "<!-- harpo:start -->", "<!-- harpo:end -->"
OLD_TOML_START, OLD_TOML_END = "# >>> harpocrates >>>", "# <<< harpocrates <<<"
NEW_TOML_START, NEW_TOML_END = "# >>> harpo >>>", "# <<< harpo <<<"
OLD_NOTE = "Read them with the harpocrates MCP tools safe_read and safe_grep."  # what the old install wrote

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
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    return h


@pytest.fixture
def claude(home: Path) -> Path:
    """A fake `claude` on PATH; returns its call log. Registered servers are the files claude.state.<name>."""
    exe = home.parent / "bin" / "claude"
    exe.write_text(FAKE_CLAUDE.replace("@LOG@", str(home.parent / "claude.log"))
                   .replace("@STATE@", str(home.parent / "claude.state")))
    exe.chmod(0o755)
    return home.parent / "claude.log"


def registered(home: Path) -> list[str]:
    return sorted(p.name.removeprefix("claude.state.") for p in home.parent.glob("claude.state.*"))


def calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def snap(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def no_bak(files: dict[str, bytes]) -> dict[str, bytes]:
    return {k: v for k, v in files.items() if not k.endswith((".harpo.bak", ".harpocrates.bak"))}


def settings_of(home: Path) -> dict:
    return json.loads((home / ".claude" / "settings.json").read_text())


def record(home: Path) -> dict:
    return json.loads((home / ".harpo" / "install.json").read_text())


def seed_old_claude_install(home: Path, user_allow: list[str] | None = None, with_record: bool = True) -> dict:
    """Leave what `harpocrates install --claude-code` wrote before the rename: the allow rules, the registration, the
    marked note, and the record that says all of it is the installer's. Returns the settings written."""
    want = harness.claude_code_settings()
    allow = [*(user_allow or []), *harness.LEGACY_MCP_TOOLS]
    settings = {"theme": "dark", "permissions": {"deny": want["permissions"]["deny"], "allow": allow},
                "hooks": want["hooks"]}
    (home / ".claude").mkdir()
    (home / ".claude" / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (home / ".claude" / "CLAUDE.md").write_text(f"# mine\n\n{OLD_MD_START}\n{OLD_NOTE}\n{OLD_MD_END}\n")
    (home.parent / "claude.state.harpocrates").write_text("mcp add --scope user harpocrates -- /py -m x\n")
    if with_record:
        section = {"permissions": {"deny": want["permissions"]["deny"], "allow": list(harness.LEGACY_MCP_TOOLS)},
                   "hooks": {ev: [entries[0]["hooks"][0]["command"]] for ev, entries in want["hooks"].items()},
                   "replaced": [], "blocks": ["CLAUDE.md"], "mcp": True}
        (home / ".harpo").mkdir()
        (home / ".harpo" / "install.json").write_text(json.dumps(
            {"version": 1, "python": PY, "python_args": harness.python_flags(), "claude-code": section}, indent=2) + "\n")
    return settings


def old_codex_block() -> str:
    """The Codex block a pre-rename install wrote: this install's lines with the old names."""
    root, tables = harness.codex_config_parts()
    root = root.replace('default_permissions = "harpo"', 'default_permissions = "harpocrates"')
    tables = (tables.replace("[mcp_servers.harpo]", "[mcp_servers.harpocrates]")
              .replace("[permissions.harpo]", "[permissions.harpocrates]")
              .replace("[permissions.harpo.", "[permissions.harpocrates."))
    return (f"{OLD_TOML_START}\n# default_permissions must stay above the first [table]; the rest of this block is at "
            f"the end.\n{root}{OLD_TOML_END}\n\n{OLD_TOML_START}\n{tables.rstrip(chr(10))}\n{OLD_TOML_END}\n")


# --- the names -----------------------------------------------------------------------------------------------------

def test_fr_name_01_the_server_and_its_tools_are_named_harpo() -> None:
    assert server.mcp.name == "harpo"
    assert harness.MCP_TOOLS == ("mcp__harpo__safe_read", "mcp__harpo__safe_grep")
    assert harness.LEGACY_MCP_TOOLS == ("mcp__harpocrates__safe_read", "mcp__harpocrates__safe_grep")
    assert install.MCP_NAME == "harpo" and install.LEGACY_MCP_NAME == "harpocrates"
    assert "harpo MCP tools safe_read and safe_grep" in harness.AGENT_NOTE
    assert "harpocrates" not in (harness.CLAUDE_NOTE + harness.CODEX_NOTE).lower()


def test_fr_name_01_generated_codex_config_uses_the_harpo_names_and_keeps_every_deny() -> None:
    cfg = tomllib.loads(harness.codex_config())
    assert cfg["default_permissions"] == "harpo"
    assert "harpo" in cfg["mcp_servers"] and "harpocrates" not in cfg["mcp_servers"]
    assert list(cfg["permissions"]) == ["harpo"]
    fs = cfg["permissions"]["harpo"]["filesystem"]
    for d in (*harness.SENSITIVE_HOME_DIRS, "~/.harpo", "~/.harpocrates"):  # both state folders stay denied
        assert fs[d] == "deny", d
    text = harness.codex_config().replace(sys.executable, "<python>")  # the interpreter path can contain "Harpocrates" (a venv inside a clone of the repo)
    assert "Harpocrates" not in text.replace("Harpocrates.", "") and "[mcp_servers.harpocrates" not in text and "[permissions.harpocrates" not in text


def test_fr_name_01_setup_instructions_register_the_harpo_server() -> None:
    claude = harness.setup_instructions("claude-code")
    assert "claude mcp add --scope user harpo -- " in claude
    assert "mcp__harpocrates" not in claude and "mcp__harpo__safe_read" in claude
    assert "shows harpo enabled" in harness.setup_instructions("codex")


def test_fr_name_01_refusals_name_the_tools_as_registered_now_and_before_a_reinstall() -> None:
    ref = hooks._tool_ref("safe_read")
    assert ref.startswith("mcp__harpo__safe_read ") and "mcp__harpocrates__safe_read" in ref and "harpo install" in ref
    assert hooks._tool_ref("safe_grep").startswith("mcp__harpo__safe_grep ")
    assert "mcp__harpo__scan_file" in hooks.SCAN_REFUSAL and hooks.SCAN_REFUSAL.startswith("harpo scan ")
    assert "Harpocrates" not in hooks.STATE_REFUSAL and "Harpo allowlist" in hooks.STATE_REFUSAL


# --- Claude Code: install over an old install ----------------------------------------------------------------------

def test_fr_name_01_reinstall_over_an_old_claude_install_leaves_no_old_names_and_keeps_every_deny_rule(
        home: Path, claude: Path) -> None:
    old = seed_old_claude_install(home)
    deny_before = list(old["permissions"]["deny"])
    (home / ".claude" / "settings.json.harpocrates.bak").write_text("{}")  # the old install's first original

    reports = install.install_harnesses(["claude-code"])
    assert not any(r.refused or any(s.state == "failed" for s in r.steps) for r in reports)

    got = settings_of(home)
    assert got["permissions"]["deny"] == deny_before  # not one deny rule dropped, none added, same order
    assert "Edit(~/.harpo/**)" in got["permissions"]["deny"] and "Edit(~/.harpocrates/**)" in got["permissions"]["deny"]
    assert got["permissions"]["allow"] == list(harness.MCP_TOOLS)
    assert "mcp__harpocrates__" not in json.dumps(got)
    assert got["theme"] == "dark" and got["hooks"] == harness.claude_code_settings()["hooks"]
    note = (home / ".claude" / "CLAUDE.md").read_text()
    assert OLD_MD_START not in note and OLD_MD_END not in note and OLD_NOTE not in note
    assert note == f"# mine\n\n{NEW_MD_START}\n{harness.CLAUDE_NOTE}\n{NEW_MD_END}\n"  # moved in place
    assert registered(home) == ["harpo"]  # the old registration is gone, the new one is there
    assert "mcp remove --scope user harpocrates" in calls(claude)
    rec = record(home)["claude-code"]
    assert rec["mcp"] is True and rec["mcp_name"] == "harpo"  # the record now says which name install owns
    assert rec["permissions"]["allow"] == list(harness.MCP_TOOLS)
    assert set(deny_before) <= set(rec["permissions"]["deny"])
    assert not (home / ".claude" / "settings.json.harpo.bak").exists()  # the old backup already holds the original

    first = snap(home)  # and it settles: a second install changes nothing
    assert all(s.state == "unchanged" for r in install.install_harnesses(["claude-code"]) for s in r.steps)
    assert snap(home) == first


def test_fr_name_01_a_legacy_rule_or_server_the_user_made_is_left_alone(home: Path, claude: Path) -> None:
    seed_old_claude_install(home, with_record=False)  # no record: nothing here is install's to remove
    reports = install.install_harnesses(["claude-code"])
    got = settings_of(home)
    assert set(harness.LEGACY_MCP_TOOLS) <= set(got["permissions"]["allow"])  # theirs, as far as we can tell
    assert set(harness.MCP_TOOLS) <= set(got["permissions"]["allow"])
    assert registered(home) == ["harpo", "harpocrates"]
    mcp_step = next(s for s in reports[0].steps if s.label == "MCP server")
    assert "claude mcp remove --scope user harpocrates" in mcp_step.detail  # told how to drop it, not dropped
    assert record(home)["claude-code"]["mcp_name"] == "harpo"
    install.uninstall_harnesses(["claude-code"])
    assert registered(home) == ["harpocrates"]  # install's own registration goes; the one that was theirs stays


def test_fr_name_01_a_legacy_allow_rule_not_in_the_record_stays_when_the_record_exists(home: Path, claude: Path) -> None:
    seed_old_claude_install(home)
    path = home / ".claude" / "settings.json"
    data = json.loads(path.read_text())
    data["permissions"]["allow"].append("mcp__harpocrates__scan_file")  # theirs: the record does not list it
    path.write_text(json.dumps(data, indent=2) + "\n")
    install.install_harnesses(["claude-code"])
    allow = settings_of(home)["permissions"]["allow"]
    assert "mcp__harpocrates__scan_file" in allow
    assert not set(harness.LEGACY_MCP_TOOLS) & set(allow)


def test_fr_name_01_uninstall_of_an_old_install_removes_the_old_names_and_the_deny_rules_it_added(
        home: Path, claude: Path) -> None:
    seed_old_claude_install(home)
    reports = install.uninstall_harnesses(["claude-code"])
    assert not any(r.refused or any(s.state == "failed" for s in r.steps) for r in reports)
    assert registered(home) == []
    assert "mcp remove --scope user harpocrates" in calls(claude)
    assert settings_of(home) == {"theme": "dark"}  # the allow rules, deny rules and hooks it recorded are gone
    assert not (home / ".claude" / "CLAUDE.md").read_text().count("harpocrates")
    assert (home / ".claude" / "CLAUDE.md").read_text() == "# mine\n"


def test_fr_name_01_uninstall_after_a_migration_removes_the_new_names_only_once(home: Path, claude: Path) -> None:
    seed_old_claude_install(home)
    install.install_harnesses(["claude-code"])
    install.uninstall_harnesses(["claude-code"])
    assert registered(home) == []
    assert settings_of(home) == {"theme": "dark"}
    assert no_bak(snap(home / ".claude")) == {"settings.json": json.dumps({"theme": "dark"}, indent=2).encode() + b"\n",
                                              "CLAUDE.md": b"# mine\n"}


def test_fr_name_01_a_fresh_install_writes_the_new_names_and_the_new_backup_suffix(home: Path, claude: Path) -> None:
    (home / ".claude").mkdir()
    (home / ".claude" / "settings.json").write_text('{"theme": "dark"}\n')
    install.install_harnesses(["claude-code"])
    assert registered(home) == ["harpo"] and calls(claude)[0] == "mcp get harpocrates"
    assert (home / ".claude" / "settings.json.harpo.bak").read_text() == '{"theme": "dark"}\n'
    assert not list((home / ".claude").glob("*.harpocrates.bak"))
    assert NEW_MD_START in (home / ".claude" / "CLAUDE.md").read_text()


# --- Codex ---------------------------------------------------------------------------------------------------------

def test_fr_name_01_reinstall_over_an_old_codex_install_moves_the_profile_and_keeps_every_deny(home: Path) -> None:
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('model = "o3"\n\n' + old_codex_block() + '\n[mcp_servers.other]\ncommand = "x"\n')
    (home / ".codex" / "AGENTS.md").write_text(f"# agents\n\n{OLD_MD_START}\n{OLD_NOTE}\n{OLD_MD_END}\n")
    assert tomllib.loads(cfg.read_text())["default_permissions"] == "harpocrates"  # the fixture really is the old shape

    reports = install.install_harnesses(["codex"])
    assert not any(r.refused for r in reports), reports[0].refused

    text = cfg.read_text()
    assert OLD_TOML_START not in text and OLD_TOML_END not in text and NEW_TOML_START in text and NEW_TOML_END in text
    data = tomllib.loads(text)
    assert data["default_permissions"] == "harpo" and data["model"] == "o3" and data["mcp_servers"]["other"] == {"command": "x"}
    assert "harpocrates" not in data["mcp_servers"] and list(data["permissions"]) == ["harpo"]
    fs = data["permissions"]["harpo"]["filesystem"]
    want = tomllib.loads(harness.codex_config())["permissions"]["harpo"]["filesystem"]
    assert fs == want and all(fs[d] == "deny" for d in (*harness.SENSITIVE_HOME_DIRS, "~/.harpo", "~/.harpocrates"))
    assert len(data["hooks"]["PreToolUse"]) == 1
    agents = (home / ".codex" / "AGENTS.md").read_text()
    assert OLD_MD_START not in agents and OLD_NOTE not in agents and agents.count(harness.CODEX_NOTE) == 1

    first = snap(home)
    install.install_harnesses(["codex"])
    assert snap(home) == first


def test_fr_name_01_uninstall_removes_an_old_codex_block_too(home: Path) -> None:
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('model = "o3"\n\n' + old_codex_block())
    (home / ".codex" / "AGENTS.md").write_text(f"{OLD_MD_START}\n{OLD_NOTE}\n{OLD_MD_END}\n")
    install.uninstall_harnesses(["codex"])
    assert tomllib.loads(cfg.read_text()) == {"model": "o3"}
    assert not (home / ".codex" / "AGENTS.md").exists()


def test_fr_name_01_a_user_profile_named_harpo_outside_the_block_still_refuses(home: Path) -> None:
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('[mcp_servers.harpo]\ncommand = "mine"\n')
    before = snap(home)
    assert install.install_harnesses(["codex"])[0].refused
    assert snap(home) == before
