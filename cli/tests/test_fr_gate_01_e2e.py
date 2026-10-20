"""FR-GATE-01, FR-GATE-03, SEC-10: the end-to-end harness for real agents behind the gate (bench/gate_e2e.py).

The unit tests cover the helpers that build the install record and read the recording. The integration test at the end runs
`claude -p` and `codex exec` through the gate to the mock in throwaway homes (AGENTS.md rule 7) and is skipped when a CLI is
not installed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bench import agent_isolation as iso
from bench import gate_e2e as e2e
from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token


@pytest.fixture
def tw():
    throwaway = iso.make_throwaway()
    yield throwaway
    iso.discard(throwaway)


def test_fr_gate_01_e2e_suites_have_unique_labels_and_known_agents() -> None:
    labels = [sc.label for suite in e2e.SUITES.values() for sc in suite]
    assert len(labels) == len(set(labels)) and {sc.agent for suite in e2e.SUITES.values() for sc in suite} == {"claude", "codex"}
    assert {sc.scan for suite in e2e.SUITES.values() for sc in suite} <= {"default", "passthrough", "stub"}
    assert {sc.record for suite in e2e.SUITES.values() for sc in suite} <= {"match", "mismatch", "none"}


def test_fr_gate_01_e2e_merge_json_file_keeps_other_keys_and_merges_one_level(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"hooks": {"a": 1}, "env": {"KEEP": "1", "ANTHROPIC_BASE_URL": "old"}}))
    e2e.merge_json_file(path, {"env": {"ANTHROPIC_BASE_URL": "new"}, "gate": {"port": 5}})
    assert json.loads(path.read_text()) == {"hooks": {"a": 1}, "env": {"KEEP": "1", "ANTHROPIC_BASE_URL": "new"},
                                            "gate": {"port": 5}}
    e2e.merge_json_file(tmp_path / "new" / "x.json", {"k": {"v": 1}})
    assert json.loads((tmp_path / "new" / "x.json").read_text()) == {"k": {"v": 1}}
    path.write_text("[]")
    with pytest.raises(ValueError):
        e2e.merge_json_file(path, {"k": 1})


@pytest.mark.parametrize("record, same_port", [("match", True), ("mismatch", False), ("none", None)])
def test_fr_gate_01_e2e_install_record_names_the_gate_port_or_another(tw, record: str, same_port) -> None:
    upstream_url = "http://127.0.0.1:9"
    token = new_token()
    with start_gate(GateConfig(anthropic_upstream=upstream_url, openai_upstream=upstream_url), token) as gate:
        e2e.write_gate_install(tw, gate, token, record)
        settings = json.loads((tw.home / ".claude" / "settings.json").read_text())
        assert settings["env"]["ANTHROPIC_BASE_URL"] == gate.url
        assert settings["env"]["ANTHROPIC_CUSTOM_HEADERS"] == f"{TOKEN_HEADER}: {token}"
        path = tw.home / ".harpo" / "install.json"
        if same_port is None:
            assert not path.exists()
            return
        entry = json.loads(path.read_text())["gate"]
        assert entry["enabled"] is True and entry["token_header"] == TOKEN_HEADER
        assert (entry["port"] == gate.port) is same_port and 1 <= entry["port"] <= 65535


def test_fr_gate_01_e2e_counts_sec10_refusals_by_tool_use_id() -> None:
    def request(results: list[tuple[str, str]]) -> dict:
        blocks = [{"type": "tool_result", "tool_use_id": i, "content": text} for i, text in results]
        return {"route": "anthropic.messages", "body": json.dumps({"messages": [{"role": "user", "content": blocks}]})}

    refused = f"PreToolUse:Bash hook error: {e2e.SEC10_PHRASE}: refused"
    records = [request([("a", refused)]), request([("a", refused), ("b", refused), ("c", "fine")]),
               {"route": "openai.responses", "body": refused}, {"route": "anthropic.messages", "body": None}]
    assert e2e.sec10_refusals(records) == 2 and e2e.sec10_refusals([]) == 0


def test_fr_gate_01_e2e_counts_placeholders_in_the_last_model_request_only() -> None:
    first = {"route": "openai.responses", "body": json.dumps({"input": "<<HARPO:github_token:abcd>>"})}
    last = {"route": "openai.responses", "body": json.dumps({"input": ["<<HARPO:github_token:abcd>> <<HARPO:jwt:0123>>",
                                                                       "<<HARPO:github_token:abcd>>"]})}
    other = {"route": "unknown", "body": "<<HARPO:ignored:ffff>>"}
    assert e2e.placeholder_counts([first, last, other]) == {"total": 3, "by_type": {"github_token": 2, "jwt": 1}}
    assert e2e.placeholder_counts([other]) == {"total": 0, "by_type": {}}


def test_fr_gate_01_e2e_gate_log_summary_counts_without_content() -> None:
    lines = [json.dumps({"route": "anthropic.messages", "decision": "blocked", "reason": "HARPO-BLOCK-SCAN-ERROR"}),
             json.dumps({"route": "anthropic.messages", "decision": "forwarded"})]
    assert e2e.gate_log_summary(lines) == {"lines": 2, "by_decision": {"blocked": 1, "forwarded": 1},
                                           "by_reason": {"HARPO-BLOCK-SCAN-ERROR": 1}, "by_route": {"anthropic.messages": 2}}


@pytest.mark.integration
def test_fr_gate_03_e2e_real_agents_behind_a_gate_that_blocks_everything(tmp_path: Path) -> None:
    """Real claude -p and codex exec behind a gate whose scan step blocks everything (the G3 state): both show the gate's
    message and the mock records nothing; SEC-10 stays quiet on a matching record and refuses every tool call on a record for
    another port. The real home must not change."""
    present = [name for name in ("claude", "codex") if shutil.which(name)]
    if not present:
        pytest.skip("neither claude nor codex is installed")
    data, path = e2e.run_suite("g3", 0, tmp_path, "unit-gate-e2e-g3")
    assert data["isolation"]["real_home_changed"] == [] and path.is_file()
    by_label = {s["scenario"]: s for s in data["scenarios"]}
    for label, agent in (("claude-default-scan", "claude"), ("codex-default-scan", "codex")):
        got = by_label[label]
        if agent not in present:
            assert got["status"] == "skipped"
            continue
        assert got["mock_requests"] == 0 and got["exit_code"] != 0
        assert got["block_message_shown"] is True and got["block_reason_shown"] == "HARPO-BLOCK-SCAN-ERROR"
        assert got["gate_log"]["by_decision"] == {"blocked": got["gate_log"]["lines"]}
    if "claude" in present:
        assert by_label["claude-sec10-match"]["sec10_refusals"] == 0 and by_label["claude-sec10-match"]["tool_results_seen"] == 3
        assert by_label["claude-sec10-mismatch"]["sec10_refusals"] == 3 and by_label["claude-sec10-mismatch"]["exit_code"] == 0


def test_fr_gate_04_e2e_counts_the_tool_results_that_carry_a_placeholder() -> None:
    def anthropic(texts: list) -> dict:
        blocks = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": text} for i, text in enumerate(texts)]
        return {"route": "anthropic.messages", "body": json.dumps({"messages": [{"role": "user", "content": blocks}]})}

    def responses(texts: list) -> dict:
        items = [{"type": "function_call_output", "call_id": f"c{i}", "output": text} for i, text in enumerate(texts)]
        return {"route": "openai.responses", "body": json.dumps({"input": [{"type": "message", "role": "user", "content": "x"}] + items})}

    placeholder = "key <<HARPO:api_token:ab12>>"
    assert e2e.tool_results_with_placeholder([anthropic(["a"]), anthropic([placeholder, "plain", [{"type": "text", "text": placeholder}]])]) == (2, 3)
    assert e2e.tool_results_with_placeholder([responses([placeholder, "plain"])]) == (1, 2)
    assert e2e.tool_results_with_placeholder([]) == (0, 0)
    assert e2e.tool_results_with_placeholder([{"route": "unknown", "body": placeholder}]) == (0, 0)


def test_fr_gate_04_e2e_the_g4_suite_has_gate_only_gate_and_hook_and_gate_off_controls() -> None:
    by_label = {sc.label: sc for sc in e2e.SUITES["g4"]}
    for agent in ("claude", "codex"):
        assert by_label[f"{agent}-gate-only"].scan == "default" and not by_label[f"{agent}-gate-only"].hook
        assert by_label[f"{agent}-gate-and-hook"].scan == "default" and by_label[f"{agent}-gate-and-hook"].hook
        assert by_label[f"{agent}-gate-off-control"].scan == "passthrough" and not by_label[f"{agent}-gate-off-control"].hook


@pytest.mark.integration
def test_fr_gate_04_e2e_real_agents_through_the_scanner_zero_known_format_canaries(tmp_path: Path) -> None:
    """AGENTS.md rule 7 for G4: scripted reads of the seeded canary repo (Read .env, cat, grep) by real claude -p and codex exec
    through the gate to the mock. No known-format canary reaches the mock, placeholders stand where they were, every session
    completes its script; with the gate off the same reads do leak (the counter works). The real home must not change."""
    present = [name for name in ("claude", "codex") if shutil.which(name)]
    if not present:
        pytest.skip("neither claude nor codex is installed")
    data, path = e2e.run_suite("g4", 0, tmp_path, "unit-gate-e2e-g4")
    assert data["isolation"]["real_home_changed"] == [] and path.is_file()
    text = path.read_text()
    assert not any(c in text for c in ("github_token=", "ghp_"))  # counts only
    for s in data["scenarios"]:
        if s["agent"] not in present:
            assert s["status"] == "skipped"
            continue
        assert s["status"] == "ok" and s["exit_code"] == 0 and s["tool_results_seen"] == 3 and s["mock_model_requests"] >= 4, s["scenario"]
        assert s["planted"]["total"] == 15
        if s["scan"] == "default":
            assert s["received"]["known_format"] == 0, s["scenario"]
            assert s["gate_log"]["by_decision"] == {"forwarded": s["gate_log"]["lines"]}
            assert s["placeholders"]["total"] >= 1
            if not s["hook_installed"]:  # only the gate stands between the files and the provider
                assert s["tool_results_with_placeholder"] == s["tool_results_seen"] == 3 and s["placeholders"]["total"] >= 8
        else:  # the gate is off: the counter must see the leak
            assert s["received"]["known_format"] >= 5, s["scenario"]
