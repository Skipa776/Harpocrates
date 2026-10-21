"""G7-T2 / FR-INSTALL-02 (a deliberately broken config is detected), FR-PROOF-05 and SEC-10: `harpo doctor`.

Every test runs with a temporary home (the `home` fixture) and, where the SEC-10 check reads a folder, a clean one.
The gate that `guard.up` starts is stopped at teardown.
"""
from __future__ import annotations

import json
import os

import pytest
import typer.main
from click.testing import CliRunner

import Harpocrates.cli
from Harpocrates import guard


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    (h / ".codex").mkdir()
    monkeypatch.setenv("HOME", str(h))
    yield h
    guard.stop_gate(guard.Files(h / ".harpo"))


def _by_name(checks):
    return {c.name: c for c in checks}


def test_fr_install_02_doctor_passes_after_guard_up(home, tmp_path):
    guard.up(["claude-code", "codex"], port=0)
    checks = guard.doctor(tmp_path, environ={})
    by_name = _by_name(checks)
    expected = {"read hook", "gate entry", "gate process", "gate answers",
                "Claude Code routing", "Codex CLI routing", "bypass (SEC-10)"}
    assert expected <= set(by_name)
    assert by_name["read hook"].status == "warn"  # no `harpo install` in this temporary home
    not_ok = [c.name for c in checks if c.name != "read hook" and c.status != "ok"]
    assert not_ok == []
    assert [c.name for c in checks if c.status == "fail"] == []


def test_fr_install_02_doctor_without_the_gate_fails_and_names_the_fix(home, tmp_path):
    checks = _by_name(guard.doctor(tmp_path, environ={}))
    assert checks["gate entry"].status == "fail"
    assert "harpo guard up" in checks["gate entry"].detail


def test_fr_install_02_doctor_fails_when_the_gate_is_not_running(home, tmp_path):
    guard.up(["claude-code", "codex"], port=0)
    guard.stop_gate(guard.Files(home / ".harpo"))  # the pid file goes, the record stays
    checks = _by_name(guard.doctor(tmp_path, environ={}))
    assert checks["gate entry"].status == "ok"
    assert checks["gate process"].status == "fail"


def test_fr_install_02_doctor_fails_when_a_config_lost_the_token(home, tmp_path):
    guard.up(["claude-code", "codex"], port=0)
    settings = home / ".claude" / "settings.json"
    data = json.loads(settings.read_text())
    data["env"]["ANTHROPIC_CUSTOM_HEADERS"] = "x-harpo-gate-token: wrong"
    settings.write_text(json.dumps(data))
    checks = _by_name(guard.doctor(tmp_path, environ={}))
    assert checks["Claude Code routing"].status == "fail"
    assert checks["Codex CLI routing"].status == "ok"


def test_fr_install_02_doctor_names_a_project_setting_that_bypasses_the_gate(home, tmp_path):
    guard.up(["claude-code", "codex"], port=0)
    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True)
    (proj / ".claude" / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://elsewhere.example"}}))
    fails = [c for c in guard.doctor(proj, environ={})
             if c.name == "bypass (SEC-10)" and c.status == "fail"]
    assert fails
    assert any("settings.json" in c.detail for c in fails)


def test_fr_install_02_doctor_flags_a_provider_switch_in_the_environment(home, tmp_path):
    guard.up(["claude-code", "codex"], port=0)
    checks = guard.doctor(tmp_path, environ={"CLAUDE_CODE_USE_BEDROCK": "1"})
    assert any(c.name == "bypass (SEC-10)" and c.status == "fail" for c in checks)


def test_fr_install_02_cli_doctor_prints_json_and_exits_1_on_a_failure(home, tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    command = typer.main.get_command(Harpocrates.cli.app)  # click CliRunner needs a click command
    result = CliRunner().invoke(command, ["doctor", "--json"])
    assert result.exit_code == 1
    items = json.loads(result.output)
    assert isinstance(items, list)
    assert any(item["status"] == "fail" for item in items)
