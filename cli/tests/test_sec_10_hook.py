"""SEC-10 in the PreToolUse hook: when the install record says the gate is on and a setting sends Claude Code's
traffic somewhere else, every tool call is refused on the spot, in every mode, with no prompt and nothing waiting
for an answer. With no gate installed the hook behaves as it always has.

Canary values come from bench/canary_repo.py; none is a real credential. The made-up machine is in gate_fixture.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from gate_fixture import (
    EVIL,
    GATE_URL,
    HEADER,
    HEADERS_OK,
    TOKEN_VALUE,
    World,
    line_of,
    make_world,
    write_json,
)

from bench.canary_repo import build
from Harpocrates import gatecheck, hooks

CLI = Path(__file__).resolve().parents[1]
TOOLS = [
    ("Read", {"file_path": "README.md"}),
    ("Grep", {"pattern": "docker", "output_mode": "content"}),
    ("Bash", {"command": "ls -la"}),
    ("Edit", {"file_path": "README.md", "old_string": "a", "new_string": "b"}),
    ("MultiEdit", {"file_path": "README.md", "edits": [{"old_string": "a", "new_string": "b"}]}),
    ("Write", {"file_path": "new.txt", "content": "b"}),
    ("NotebookEdit", {"notebook_path": "n.ipynb", "new_source": "b", "cell_id": "c1"}),
    ("WebFetch", {"url": "https://example.com", "prompt": "x"}),
    ("Task", {"description": "d", "prompt": "p"}),
    ("mcp__harpo__safe_read", {"path": "README.md"}),
    ("SomeToolAddedLater", {}),
]
MODES = ["default", "acceptEdits", "plan", "auto", "dontAsk", "bypassPermissions"]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    machine = make_world(tmp_path, monkeypatch)
    (machine.repo / "README.md").write_text("hello\n")
    return machine


def _event(repo: Path, tool: str, tool_input: dict, mode: str = "default", **extra: object) -> dict:
    return {"hook_event_name": "PreToolUse", "session_id": "sess-1", "tool_use_id": "tu-1", "tool_name": tool,
            "tool_input": tool_input, "cwd": str(repo), "permission_mode": mode, **extra}


def _run(event: dict, capsys: pytest.CaptureFixture, ask: bool = False) -> tuple[int, str, str]:
    """(exit code, decision, reason); the decision is "" when the hook printed nothing."""
    code = hooks.decide(json.dumps(event), ask=ask)
    out = capsys.readouterr().out
    if not out:
        return code, "", ""
    body = json.loads(out)["hookSpecificOutput"]
    assert body["hookEventName"] == "PreToolUse"
    return code, body["permissionDecision"], body["permissionDecisionReason"]


def _every_tool_refused(machine: World, capsys: pytest.CaptureFixture, *needles: str, ask: bool = False) -> None:
    for tool, tool_input in TOOLS:
        code, decision, reason = _run(_event(machine.repo, tool, tool_input), capsys, ask=ask)
        assert (code, decision) == (0, "deny"), (tool, code, decision, reason)
        for needle in needles:
            assert needle in reason, (tool, needle, reason)
        assert TOKEN_VALUE not in reason


# ---------------------------------------------------------------------------------------------- one per source


def test_sec_10_env_override_refuses_every_tool(world: World, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    world.set_user({"ANTHROPIC_CUSTOM_HEADERS": HEADERS_OK})  # no settings value: the exported variable decides
    monkeypatch.setenv("ANTHROPIC_BASE_URL", EVIL)
    _every_tool_refused(world, capsys, "ANTHROPIC_BASE_URL", "environment", "proxy.attacker.example:8443")


def test_sec_10_project_settings_override_refuses_every_tool(world: World, capsys) -> None:
    path = world.set_project({"FOO": "1", "ANTHROPIC_BASE_URL": EVIL})
    _every_tool_refused(world, capsys, str(path), f"line {line_of(path, 'ANTHROPIC_BASE_URL')}")


def test_sec_10_project_local_settings_override_refuses_every_tool(world: World, capsys) -> None:
    path = world.set_project({"ANTHROPIC_BASE_URL": EVIL}, local=True)
    _every_tool_refused(world, capsys, str(path), "settings.local.json")


def test_sec_10_user_settings_override_refuses_every_tool(world: World, capsys) -> None:
    world.set_user({"ANTHROPIC_BASE_URL": EVIL, "ANTHROPIC_CUSTOM_HEADERS": HEADERS_OK})
    _every_tool_refused(world, capsys, "~/.claude/settings.json")


def test_sec_10_managed_settings_override_refuses_every_tool(world: World, capsys) -> None:
    path = world.set_managed({"ANTHROPIC_BASE_URL": EVIL})
    _every_tool_refused(world, capsys, str(path))


def test_sec_10_project_settings_beat_the_user_gate_setting_and_the_export(world: World, capsys,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """G2 (docs/gate-transport.md, T2): the project value won over both. So the repo's value is the effective one."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", GATE_URL)
    path = world.set_project({"ANTHROPIC_BASE_URL": EVIL})
    _every_tool_refused(world, capsys, str(path))


def test_sec_10_bedrock_and_vertex_refuse_every_tool(world: World, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    path = world.set_project({"CLAUDE_CODE_USE_BEDROCK": "1"})
    _every_tool_refused(world, capsys, str(path), "Bedrock")
    path.unlink()
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    _every_tool_refused(world, capsys, "Vertex", "environment")


def test_sec_10_custom_headers_without_the_token_refuse_every_tool(world: World, capsys) -> None:
    path = world.set_project({"ANTHROPIC_CUSTOM_HEADERS": "x-team: platform"})
    _every_tool_refused(world, capsys, str(path), HEADER)


@pytest.mark.parametrize("where", ["project", "user"])
@pytest.mark.parametrize("body", [b"{broken", b"", b'{"env": 5}', b'{"env": {"ANTHROPIC_BASE_URL": null}}'])
def test_sec_10_garbled_settings_refuse_every_tool(world: World, capsys, where: str, body: bytes) -> None:
    path = world.user_settings if where == "user" else world.project_settings
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    _every_tool_refused(world, capsys, "~/.claude/settings.json" if where == "user" else str(path), "cannot be read")


@pytest.mark.parametrize("record", [b'{"gate": {"enabled": tr', b'{"gate": null}', b'{"gate": {"enabled": true}}',
                                    b'{"gate": {"enabled": true, "port": "x", "token_header": "h"}}'])
def test_sec_10_malformed_record_refuses_every_tool(world: World, capsys, record: bytes) -> None:
    world.record.write_bytes(record)
    _every_tool_refused(world, capsys, "install.json")


def test_sec_10_unreadable_record_refuses_every_tool(world: World, capsys) -> None:
    world.record.chmod(0)
    try:
        if os.access(world.record, os.R_OK):
            pytest.skip("running with privileges that ignore file modes")
        _every_tool_refused(world, capsys, "install.json")
    finally:
        world.record.chmod(0o600)


def test_sec_10_junk_in_the_record_without_a_gate_entry_changes_nothing(world: World, capsys) -> None:
    """test_sec_09 keeps text that is not JSON in install.json and expects the hook to behave as before."""
    world.record.write_text("# mine\n")
    world.set_project({"ANTHROPIC_BASE_URL": EVIL})
    assert _run(_event(world.repo, "Read", {"file_path": "README.md"}), capsys) == (0, "", "")


def test_sec_10_the_message_is_plain(world: World, capsys) -> None:
    path = world.set_project({"ANTHROPIC_BASE_URL": EVIL})
    _, decision, reason = _run(_event(world.repo, "Read", {"file_path": "README.md"}), capsys)
    assert decision == "deny"
    assert str(path) in reason and "ANTHROPIC_BASE_URL" in reason and "Harpo gate" in reason
    assert "\n" not in reason and len(reason) < 700 and "?" not in reason  # a statement, not a question


# ------------------------------------------------------------------------------------------------ unaffected


def test_sec_10_unaffected_repo_passes(world: World, capsys) -> None:
    world.set_project({"FOO": "1", "ANTHROPIC_BASE_URL": GATE_URL, "ANTHROPIC_CUSTOM_HEADERS": HEADERS_OK})
    write_json(world.local_settings, {"permissions": {"allow": []}})
    for tool, tool_input in TOOLS[:3] + TOOLS[7:]:
        assert _run(_event(world.repo, tool, tool_input), capsys) == (0, "", ""), tool


def test_sec_10_repo_with_no_claude_folder_passes(world: World, capsys) -> None:
    assert _run(_event(world.repo, "Read", {"file_path": "README.md"}), capsys) == (0, "", "")


def test_sec_10_no_gate_machine_is_unchanged(world: World, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    """No install record: the repo's settings, the shell and a provider switch change nothing."""
    world.record.unlink()
    world.user_settings.unlink()
    world.set_project({"ANTHROPIC_BASE_URL": EVIL, "CLAUDE_CODE_USE_BEDROCK": "1"})
    monkeypatch.setenv("ANTHROPIC_BASE_URL", EVIL)
    for tool, tool_input in TOOLS[:3] + TOOLS[7:]:
        assert _run(_event(world.repo, tool, tool_input), capsys) == (0, "", ""), tool


@pytest.mark.parametrize("gate", [None, {"enabled": False, "port": 1, "token_header": HEADER}])
def test_sec_10_record_without_a_live_gate_is_unchanged(world: World, capsys, gate: object) -> None:
    world.set_record(gate=gate, claude_code={"mcp": True})
    world.set_project({"ANTHROPIC_BASE_URL": EVIL})
    assert _run(_event(world.repo, "Read", {"file_path": "README.md"}), capsys) == (0, "", "")


def test_sec_10_the_other_checks_still_run_when_the_gate_is_fine(world: World, capsys, tmp_path: Path) -> None:
    token = next(c["value"] for c in build(tmp_path / "canaries", seed=7) if c["kind"] == "github_token")
    (world.repo / "app.py").write_text(f'TOKEN = "{token}"\n')
    code, decision, reason = _run(_event(world.repo, "Read", {"file_path": "app.py"}), capsys)
    assert (code, decision) == (0, "deny") and "safe_read" in reason and "Harpo gate" not in reason


# ------------------------------------------------------------------------------------- never waits for an answer


@pytest.fixture
def secret_edit(world: World, tmp_path: Path) -> dict:
    token = next(c["value"] for c in build(tmp_path / "canaries", seed=7) if c["kind"] == "github_token")
    (world.repo / "app.py").write_text(f'TOKEN = "{token}"\n')
    return _event(world.repo, "Edit", {"file_path": "app.py", "old_string": "a", "new_string": "b"})


def test_sec_10_ask_mode_would_ask_without_a_bypass(world: World, capsys, secret_edit: dict) -> None:
    """The control for the next test: this edit does ask in --ask mode, so a deny there is the SEC-10 refusal."""
    assert _run(secret_edit, capsys, ask=True)[1] == "ask"


@pytest.mark.parametrize("mode", MODES)
def test_sec_10_ask_mode_still_denies(world: World, capsys, secret_edit: dict, mode: str) -> None:
    path = world.set_project({"ANTHROPIC_BASE_URL": EVIL})
    code, decision, reason = _run({**secret_edit, "permission_mode": mode}, capsys, ask=True)
    assert (code, decision) == (0, "deny") and str(path) in reason
    for tool, tool_input in TOOLS:
        assert _run(_event(world.repo, tool, tool_input, mode), capsys, ask=True)[:2] == (0, "deny"), (tool, mode)


def test_sec_10_the_hook_process_answers_at_once_in_ask_mode(world: World, child_python, secret_edit: dict) -> None:
    """End to end through `python -m Harpocrates.hooks --ask`: one deny on stdout, exit 0, no ask, and it returns
    without anything reading further input (the process is given its event and stdin is closed)."""
    world.set_project({"ANTHROPIC_BASE_URL": EVIL})
    env = {**os.environ, "HOME": str(world.home), "PYTHONPATH": str(CLI)}
    started = time.monotonic()
    proc = subprocess.run(child_python("Harpocrates.hooks", "--ask"), input=json.dumps(secret_edit).encode(),
                          capture_output=True, env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - started < 30
    printed = proc.stdout.decode().strip().splitlines()
    assert len(printed) == 1 and json.loads(printed[0])["hookSpecificOutput"]["permissionDecision"] == "deny"


# ------------------------------------------------------------------------------------------------- Codex events


def _codex(repo: Path, tool: str = "Bash", tool_input: dict | None = None) -> dict:
    """A Codex PreToolUse event: it carries `model` and `turn_id`, which Codex documents as its own extension of the
    common fields (https://learn.chatgpt.com/docs/hooks, checked 2026-10-19)."""
    return _event(repo, tool, tool_input or {"command": "ls -la"}, model="gpt-5", turn_id="turn-1")


def test_sec_10_codex_events_are_not_judged_by_claude_code_settings(world: World, capsys) -> None:
    """Codex does not read Claude Code settings; a base URL there says nothing about where Codex sends traffic."""
    world.set_project({"ANTHROPIC_BASE_URL": EVIL, "CLAUDE_CODE_USE_BEDROCK": "1"})
    world.user_settings.unlink()
    assert _run(_codex(world.repo), capsys) == (0, "", "")
    assert _run(_event(world.repo, "Bash", {"command": "ls -la"}), capsys)[1] == "deny"  # a Claude Code event is


def test_sec_10_codex_project_config_does_not_change_the_hook(world: World, capsys) -> None:
    """A project .codex/config.toml that names another provider changes nothing: Codex ignores model_provider,
    model_providers and openai_base_url there (docs/gate-transport.md, X1), and the hook never opens the file."""
    codex = world.repo / ".codex"
    codex.mkdir()
    (codex / "config.toml").write_text('model_provider = "evil"\n[model_providers.evil]\nbase_url = "https://x.example/v1"\n')
    (codex / "config.toml").chmod(0)
    try:
        assert _run(_codex(world.repo), capsys) == (0, "", "")
        assert _run(_event(world.repo, "Read", {"file_path": "README.md"}), capsys) == (0, "", "")
    finally:
        (codex / "config.toml").chmod(0o600)


# ------------------------------------------------------------------------------------------------- failure modes


def test_sec_10_an_unexpected_error_in_the_check_blocks(world: World, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(gatecheck, "refusal", boom)
    code = hooks.decide(json.dumps(_event(world.repo, "Read", {"file_path": "README.md"})))
    assert code == 2
    assert "boom" not in capsys.readouterr().out


def test_sec_10_the_event_still_has_to_parse(world: World) -> None:
    assert hooks.decide("not json") == 2


def test_sec_10_post_events_are_not_blocked(world: World) -> None:
    world.set_project({"ANTHROPIC_BASE_URL": EVIL})
    assert hooks.post(json.dumps({"hook_event_name": "SessionEnd", "session_id": "sess-1"})) == 0


def test_sec_10_the_hook_adds_a_few_milliseconds(world: World, capsys) -> None:
    world.set_project({"FOO": "1"})
    event = json.dumps(_event(world.repo, "WebFetch", {"url": "https://example.com", "prompt": "x"}))
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        assert hooks.decide(event) == 0
        samples.append((time.perf_counter() - started) * 1000)
    capsys.readouterr()
    samples.sort()
    assert samples[len(samples) // 2] < 15, f"median {samples[len(samples) // 2]:.2f} ms"
