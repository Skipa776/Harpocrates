"""FR-PROOF-04: what agent mode asks the real CLIs to do, and how it checks that they did it.

The unit tests build commands, scripts and environments without starting an agent. The integration test at the end
runs `claude -p` and `codex exec` against the mock in throwaway homes and is skipped when a CLI is not installed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import tomllib

from bench import agent_isolation as iso
from bench import agent_runner as runner
from bench import mock_provider as mock

URL = "http://127.0.0.1:5555"


@pytest.fixture
def tw():
    throwaway = iso.make_throwaway()
    yield throwaway
    iso.discard(throwaway)


def test_fr_proof_04_agent_runner_scripts_read_env_cat_a_file_grep_then_finish(tw, tmp_path: Path) -> None:
    for agent, script in (("claude", runner.claude_script(tw.repo)), ("codex", runner.codex_script(tw.repo))):
        path = tmp_path / f"{agent}.json"
        path.write_text(json.dumps(script))
        assert mock.load_script(path) == script
        calls = [c for turn in script for c in turn.get("tool_calls", [])]
        assert len(calls) == runner.SCRIPTED_CALLS == 3 and script[-1] == {"text": "done"}
    claude = [c for t in runner.claude_script(tw.repo) for c in t.get("tool_calls", [])]
    assert claude[0] == {"name": "Read", "input": {"file_path": str(tw.repo / ".env")}}
    assert claude[1]["name"] == "Bash" and claude[1]["input"]["command"] == "cat config/settings.py"
    assert claude[2]["name"] == "Bash" and claude[2]["input"]["command"] == "grep -rn TOKEN ."
    codex = [c for t in runner.codex_script(tw.repo) for c in t.get("tool_calls", [])]
    assert [c["name"] for c in codex] == ["exec_command"] * 3
    assert [c["input"]["cmd"] for c in codex] == ["cat .env", "cat config/settings.py", "grep -rn TOKEN ."]


def test_fr_proof_04_agent_runner_commands_carry_no_blanket_permission_bypass(tw) -> None:
    claude = runner.claude_argv("prompt")
    codex = runner.codex_argv(tw.repo, "prompt")
    assert claude[:3] == ["claude", "-p", "prompt"]
    allowed = claude[claude.index("--allowedTools") + 1:claude.index("--allowedTools") + 4]
    assert allowed == ["Read", "Bash(cat *)", "Bash(grep *)"]  # an explicit list, not Bash(*)
    joined = " ".join(claude + codex)
    for forbidden in ("--dangerously-skip-permissions", "--allow-dangerously-skip-permissions", "bypass-approvals-and-sandbox",
                      "danger-full-access", "--bare"):
        assert forbidden not in joined, forbidden
    assert codex[:2] == ["codex", "exec"] and "read-only" in codex and str(tw.repo) in codex
    assert "--dangerously-bypass-hook-trust" in codex  # only the hooks of the throwaway CODEX_HOME exist to be trusted


def test_fr_proof_04_agent_runner_environments_point_at_the_mock_with_fake_keys_only(tw) -> None:
    claude = runner.claude_env(tw, URL)
    assert claude["ANTHROPIC_BASE_URL"] == URL and claude["ANTHROPIC_API_KEY"] == iso.FAKE_API_KEY
    assert claude["HOME"] == str(tw.home) and claude["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert not any(name.startswith(("OPENAI", "CODEX")) for name in claude)
    codex = runner.codex_env(tw, URL)
    assert codex["CODEX_HOME"] == str(tw.home / ".codex") and codex["CODEX_API_KEY"] == iso.FAKE_OPENAI_KEY
    assert codex["HOME"] == str(tw.home) and not any(name.startswith("ANTHROPIC") for name in codex)
    assert "OPENAI_BASE_URL" not in codex  # route A names the mock in config.toml, not through a base-URL override


def test_fr_proof_04_agent_runner_codex_config_is_route_a_with_http_only(tw) -> None:
    runner.write_codex_config(tw, URL)
    config = tomllib.loads((tw.home / ".codex" / "config.toml").read_text())
    provider = config["model_providers"][config["model_provider"]]
    assert provider["base_url"] == URL + "/v1" and provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is True and not provider.get("supports_websockets", False)
    assert config["model_provider"].lower() != "openai"
    assert "openai_base_url" not in config
    assert config["features"]["plugins"] is False and config["features"]["remote_plugin"] is False  # no plugin catalog sync
    assert config["features"]["daemon_auto_start"] is False and config["analytics"]["enabled"] is False


def test_fr_proof_04_agent_runner_counts_the_tool_results_the_agent_sent_back() -> None:
    def body(results: int) -> str:
        blocks = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x"} for i in range(results)]
        return json.dumps({"tools": [{}], "messages": [{"role": "user", "content": blocks}]})

    records = [{"route": "anthropic.messages", "body": body(0)}, {"route": "anthropic.messages", "body": body(3)},
               {"route": "unknown", "body": "{}"}, {"route": "anthropic.messages", "body": None}]
    assert runner.results_seen(records) == 3
    codex = [{"route": "openai.responses", "body": json.dumps({"input": [{"type": "function_call_output"}] * 2})}]
    assert runner.results_seen(codex) == 2 and runner.results_seen([]) == 0


def test_fr_proof_04_agent_runner_names_credential_headers_without_values() -> None:
    records = [{"headers": [["x-api-key", "<present>"], ["X-Claude-Code-Session-Id", "<present>"], ["user-agent", "u"]]},
               {"headers": [["Authorization", "<present>"]]}]
    assert runner.credential_header_names(records) == ["authorization", "x-api-key", "x-claude-code-session-id"]
    assert runner.credential_header_names([]) == []


def test_fr_proof_04_agent_runner_turns_the_manifest_into_canaries(tw) -> None:
    manifest = runner.seed_repo(tw, 0, "claude")
    assert len(manifest) == 15 and (tw.repo / ".env").is_file() and (tw.repo / ".git").is_dir()
    canaries = runner.canaries_from_manifest(manifest, "claude")
    assert len(canaries) == 15 and all(c.agent == "claude" and c.ident.startswith("claude:") for c in canaries)
    assert {c.kind for c in canaries} >= {"github_token", "private_key", "db_password"}
    assert len({c.ident for c in canaries}) == 15


def test_fr_proof_04_agent_runner_missing_cli_is_skipped_and_never_a_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    result = runner.run_agents(seed=0, out_root=tmp_path / "out", run_id="unit-missing", which="all")
    assert result.verdict.exit_code == 2 and "inconclusive" in result.verdict.reason
    statuses = {r["agent"]: r["status"] for r in result.report["runs"]}
    assert statuses == {"claude": "skipped", "codex": "skipped"}
    assert all("not on PATH" in r["reason"] for r in result.report["runs"])
    assert result.report["planted"]["total"] == 0 and (tmp_path / "out" / "proof" / "unit-missing.json").exists()


def fake_runner(received_kinds: set[str], control_expected: bool = False):
    """A stand-in for run_one: a finished run whose recording holds the canaries of the given kinds."""
    def run(agent, seed, out_root, run_id, keep_work=False, timeout=0, control=False):
        manifest = [{"file": f"f{i}", "kind": kind, "value": f"{kind}-VALUE-0123456789-{agent}"}
                    for i, kind in enumerate(["github_token", "openai_key", "db_password", "jwt"])]
        canaries = runner.canaries_from_manifest(manifest, agent)
        text = " ".join(m["value"] for m in manifest if m["kind"] in received_kinds)
        body = json.dumps({"tools": [{}], "messages": [{"role": "user", "content": [{"type": "tool_result", "content": text}]}]})
        got = runner.AgentRun(agent, "ok", "", "x 1.0", 0, 1.0, [{"index": 0, "route": "anthropic.messages", "body": body,
                                                                   "headers": []}], canaries, {"scripted_tool_calls": 3})
        return got
    return run


@pytest.mark.parametrize("received, control, code", [
    (set(), False, 0), ({"db_password"}, False, 0), ({"github_token"}, False, 1), ({"jwt", "db_password"}, False, 1),
    ({"github_token"}, True, 0), (set(), True, 1), ({"db_password"}, True, 1),
])
def test_fr_proof_04_agent_runner_exit_status_follows_what_arrived(tmp_path: Path, received: set, control: bool, code: int) -> None:
    result = runner.run_agents(0, tmp_path, "unit-exit", "claude", runner=fake_runner(received), control=control)
    assert result.verdict.exit_code == code, result.verdict
    assert result.report["variant"] == ("control" if control else "read-hook")
    assert result.report["protection"] == ("none" if control else "read-tool")
    text = (tmp_path / "proof" / "unit-exit.json").read_text()
    assert "VALUE-0123456789" not in text


def test_fr_proof_04_agent_runner_an_unfinished_agent_is_inconclusive_even_with_no_leak(tmp_path: Path) -> None:
    def stalled(*args, **kwargs):
        got = fake_runner(set())(*args, **kwargs)
        got.status, got.reason = "unfinished", "only 1 of 3 scripted tool calls returned a result"
        return got

    result = runner.run_agents(0, tmp_path, "unit-stall", "claude", runner=stalled)
    assert result.verdict.exit_code == 2 and result.report["runs"][0]["status"] == "unfinished"


def test_fr_proof_04_agent_runner_a_change_to_the_real_home_fails_the_run(tmp_path: Path, monkeypatch) -> None:
    snapshots = iter([{"a": 1}, {"a": 1}, {"a": 2}, {"a": 2}])
    monkeypatch.setattr(iso, "real_home_snapshot", lambda home=None: next(snapshots))
    result = runner.run_agents(0, tmp_path, "unit-home", "claude", runner=fake_runner(set()))
    assert result.verdict.exit_code == 1 and "real home changed" in result.verdict.reason
    assert result.report["isolation"]["real_home_changed"] == ["a"]


@pytest.mark.integration
def test_fr_proof_04_harness_agent_mode_isolated(tmp_path: Path) -> None:
    """Runs the real CLIs against the mock in throwaway homes. The real home must not change; counts only in the report."""
    present = [name for name in ("claude", "codex") if shutil.which(name)]
    if not present:
        pytest.skip("neither claude nor codex is installed")
    before = iso.real_home_snapshot()
    result = runner.run_agents(seed=0, out_root=tmp_path, run_id="unit-agents", which="all")
    assert iso.snapshot_diff(before, iso.real_home_snapshot()) == []  # the real-home guard stays unchanged
    assert result.report["isolation"]["real_home_changed"] == []
    ran = {r["agent"]: r for r in result.report["runs"]}
    assert set(ran) == {"claude", "codex"}
    for name in ("claude", "codex"):
        if name not in present:
            assert ran[name]["status"] == "skipped" and result.verdict.exit_code == 2
            continue
        assert ran[name]["status"] == "ok", ran[name]
        assert ran[name]["scripted_tool_calls"] == ran[name]["tool_results_seen"] == 3 and ran[name]["requests"] >= 4
        assert ran[name]["install"] == "ok"
    if "claude" in present:  # the API-key path, never a stored login
        assert "x-api-key" in ran["claude"]["credential_header_names"]
        assert "authorization" not in ran["claude"]["credential_header_names"]
    if "codex" in present:
        assert "authorization" in ran["codex"]["credential_header_names"]
    assert result.report["planted"]["total"] == 15 * len(present)
    text = (tmp_path / "proof" / "unit-agents.json").read_text()
    assert not any(c.value in text for c in result.canaries)
    assert (tmp_path / "mock" / "unit-agents").is_dir()
