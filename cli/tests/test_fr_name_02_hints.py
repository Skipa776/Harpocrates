"""FR-NAME-02: text that tells the user which command to run says `harpo`.

Only commands the user must type: the allowlist refusal, the setup lines in the install report and in the setup text, and
the codex merge hint. Other wording (product name, MCP server name, README, docs) is FR-NAME-01's.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Harpocrates import hooks, install
from Harpocrates.harness import HARNESSES, setup_instructions, tier_report


def _no_old_command(text: str, *, what: str) -> None:
    assert "`harpocrates " not in text and " harpocrates allow" not in text and "harpocrates setup" not in text, what


def test_fr_name_02_state_refusal_tells_the_user_to_run_harpo_allow() -> None:
    assert hooks.STATE_REFUSAL == (
        "Only the user can change the Harpocrates allowlist. Ask them to run `harpo allow <path>` in their own terminal.")


def test_fr_name_02_a_refused_allow_command_shows_that_text(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("harpo", "harpocrates"):
        event = {"tool_name": "Bash", "cwd": str(tmp_path), "permission_mode": "default",
                 "tool_input": {"command": f"{name} allow /"}}
        assert hooks.decide(json.dumps(event), ask=True) == 0
        spec = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
        assert spec["permissionDecision"] == "deny" and spec["permissionDecisionReason"] == hooks.STATE_REFUSAL
        assert "`harpo allow <path>`" in spec["permissionDecisionReason"]


def test_fr_name_02_tier_report_points_to_harpo_setup() -> None:
    out = tier_report({h.name: True for h in HARNESSES})
    setups = [h.setup for h in HARNESSES if h.setup]
    assert setups and all(f"Setup: harpo setup {s}" in out for s in setups)
    _no_old_command(out, what=out)


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_fr_name_02_setup_text_says_harpo(harness: str) -> None:
    _no_old_command(setup_instructions(harness), what=harness)


def test_fr_name_02_claude_setup_text_names_harpo_allow() -> None:
    assert "`harpo allow <path>`" in setup_instructions("claude-code")


def test_fr_name_02_codex_merge_hint_says_harpo_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))  # no real claude or codex can be found
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('default_permissions = ":workspace"\n')
    report = install.install_harnesses(["codex"])
    assert report[0].refused and "`harpo setup codex` prints the lines" in report[0].refused
    _no_old_command(report[0].refused, what=report[0].refused)
