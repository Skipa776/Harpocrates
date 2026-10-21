"""FR-PROOF-04 (G2): the capture probes (bench/g2_probes.py) and the reading of mock recordings (bench/g2_analysis.py).

The probes run the real CLIs against the mock by hand and write bench/results/proof/g2-probes-<run>.json; these tests
check the fakes they use and that nothing they write can hold a credential-shaped value. The analysis tests answer the
parser-spec and transport-note verify items from names and shapes only, using made-up recordings.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest
import tomllib

from bench import agent_isolation as iso
from bench import agent_runner as runner
from bench import g2_analysis as g2
from bench import g2_probes as probes
from bench import mock_provider as mock

URL = "http://127.0.0.1:5555"
RECOVERY_PHRASES = ("thinking", "signature", "bound to a different conversation", "Extra inputs are not permitted", "prompt is too long")


@pytest.fixture
def tw():
    throwaway = iso.make_throwaway()
    yield throwaway
    iso.discard(throwaway)


def test_fr_proof_04_g2_probes_run_process_can_capture_output(tmp_path: Path) -> None:
    code, timed_out, _, out, err = runner.run_process_captured(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper()); print('e', file=sys.stderr)"],
        {"PATH": os.environ["PATH"]}, tmp_path, 30, b"abc")
    assert (code, timed_out, out.strip(), err.strip()) == (0, False, "ABC", "e")
    assert runner.run_process_captured([sys.executable, "-c", "import time; time.sleep(30)"], {"PATH": os.environ["PATH"]},
                                       tmp_path, 1)[:2] == (None, True)


def test_fr_proof_04_g2_probes_fake_chatgpt_login_is_unmistakably_fake_and_never_expires_soon() -> None:
    auth = probes.fake_chatgpt_auth()
    assert auth["auth_mode"] == "chatgpt" and auth["OPENAI_API_KEY"] is None
    tokens = auth["tokens"]
    for name in ("id_token", "access_token"):
        header, payload, signature = tokens[name].split(".")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        assert claims["exp"] > 4_000_000_000  # no refresh is ever attempted
        assert signature == "fakesignature" and json.loads(base64.urlsafe_b64decode(header + "==")) == {"alg": "none", "typ": "JWT"}
    claims = json.loads(base64.urlsafe_b64decode(tokens["id_token"].split(".")[1] + "=="))
    assert claims["email"].endswith("@example.test") and "fake" in tokens["refresh_token"] and "fake" in tokens["account_id"]
    assert "fake" in json.dumps(auth)


def test_fr_proof_04_g2_probes_codex_configs_point_only_at_the_mock(tw) -> None:
    probes.write_route_b_config(tw, URL, chatgpt=False)
    config = tomllib.loads((tw.home / ".codex" / "config.toml").read_text())
    assert config["openai_base_url"] == URL + "/v1" and "model_provider" not in config and "chatgpt_base_url" not in config
    assert config["features"]["plugins"] is False and config["analytics"]["enabled"] is False
    probes.write_route_b_config(tw, URL, chatgpt=True)
    config = tomllib.loads((tw.home / ".codex" / "config.toml").read_text())
    assert config["chatgpt_base_url"] == URL + "/backend-api/" and config["openai_base_url"] == URL + "/v1"
    assert (tw.home / ".codex" / "auth.json").is_file() and (tw.home / ".codex" / "auth.json").stat().st_mode & 0o077 == 0
    runner.write_codex_config(tw, URL)
    probes.write_chatgpt_auth(tw, URL)
    config = tomllib.loads((tw.home / ".codex" / "config.toml").read_text())
    assert config["chatgpt_base_url"] == URL + "/backend-api/" and config["model_providers"]["mockprov"]["base_url"] == URL + "/v1"
    env = probes.chatgpt_env(tw, URL)
    assert env["CODEX_REFRESH_TOKEN_URL_OVERRIDE"] == URL + "/oauth/token" and "CODEX_API_KEY" not in env
    assert env["CODEX_HOME"] == str(tw.home / ".codex")


def test_fr_proof_04_g2_probes_scripts_are_valid_and_carry_thinking(tw, tmp_path: Path) -> None:
    for name, script in (("claude", probes.claude_corpus_script(tw.repo)), ("codex", probes.codex_corpus_script())):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(script))
        assert mock.load_script(path) == script
        assert any(t.get("thinking") for t in script) and any(t.get("redacted_thinking") for t in script)
        assert script[-1] == {"text": "done"}
    names = [c["name"] for t in probes.claude_corpus_script(tw.repo) for c in t.get("tool_calls", [])]
    assert names == ["Read", "Bash", "Read", "mcp__harpo__safe_read"]
    assert (probes.claude_corpus_script(tw.repo)[2]["tool_calls"][0]["input"]["file_path"]).endswith("logo.png")


def test_fr_proof_04_g2_probes_the_block_message_avoids_every_recovery_phrase() -> None:
    message = probes.BLOCK_MESSAGE.lower()
    assert not any(phrase.lower() in message for phrase in RECOVERY_PHRASES)
    assert "harpo" in message and len(probes.BLOCK_MESSAGE) < 300


def test_fr_proof_04_g2_probes_long_prompts_are_big_and_plain() -> None:
    text = probes.long_prompt(300_000)
    assert 300_000 <= len(text) < 301_000 and text.isascii() and "\0" not in text


def test_fr_proof_04_g2_probes_png_fixture_is_a_real_png(tw) -> None:
    probes.write_png(tw.repo / "logo.png")
    data = (tw.repo / "logo.png").read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and data.endswith(b"IEND\xaeB`\x82")


def test_fr_proof_04_g2_probes_a_report_with_a_fake_credential_is_refused(tmp_path: Path) -> None:
    fakes = [iso.FAKE_API_KEY, iso.FAKE_OPENAI_KEY, probes.fake_chatgpt_auth()["tokens"]["refresh_token"]]
    probes.write_probe_report(tmp_path / "ok.json", {"probe": "x", "n": 1}, fakes)
    assert json.loads((tmp_path / "ok.json").read_text())["probe"] == "x"
    for fake in fakes:
        with pytest.raises(probes.ProbeReportError):
            probes.write_probe_report(tmp_path / "bad.json", {"oops": f"has {fake}"}, fakes)
    assert not (tmp_path / "bad.json").exists()


BILLING = "x-anthropic-billing-header: cc_version=2.1.303.3a9; cc_entrypoint=sdk-cli;"


def rec(index: int, route: str, body: object, headers: list | None = None, **extra) -> dict:
    return {"index": index, "route": route, "method": "POST", "path": "/v1/messages?beta=true", "body": json.dumps(body),
            "headers": headers or [], "content_encoding": None, "chunked": False, "content_length_header": 10,
            "transfer_encoding": None, "response_status": 200, **extra}


ANTHROPIC = {
    "model": "m", "max_tokens": 1, "stream": True, "metadata": {"user_id": "{}"}, "thinking": {"type": "adaptive"},
    "system": [{"type": "text", "text": BILLING}, {"type": "text", "text": "You are an agent", "cache_control": {"type": "ephemeral"}}],
    "tools": [{"name": "Read", "description": "d", "input_schema": {}}, {"type": "bash_20250124", "name": "bash"}],
    "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "mocksig00000001"},
                                           {"type": "redacted_thinking", "data": "opaque-x"},
                                           {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "a"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}]},
            {"type": "text", "text": "reminder"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "plain string"}]}],
}


def test_fr_proof_04_g2_analysis_reads_the_attribution_block() -> None:
    other = json.loads(json.dumps(ANTHROPIC))
    other["system"][0]["text"] = BILLING.replace("3a9", "7bc")
    got = g2.attribution([rec(0, "anthropic.messages", ANTHROPIC), rec(1, "anthropic.messages", other)])
    assert got["found"] and got["position"] == 0 and got["own_block"] and not got["has_cache_control"]
    assert got["leading_text"] == "x-anthropic-billing-header: cc_version="
    assert got["template"] == "x-anthropic-billing-header: cc_version=<version>.<hex3>; cc_entrypoint=<entrypoint>;"
    assert got["example"] == BILLING and got["stable_across_requests"] is False and got["requests_checked"] == 2
    assert g2.attribution([rec(0, "anthropic.messages", {"system": "plain string", "messages": []})])["found"] is False
    assert g2.attribution([])["found"] is False


def test_fr_proof_04_g2_analysis_lists_the_block_types_an_agent_sent() -> None:
    got = g2.block_types([rec(0, "anthropic.messages", ANTHROPIC)])["anthropic"]
    assert got["content_block_types"] == ["redacted_thinking", "text", "thinking", "tool_result", "tool_use"]
    assert got["tool_result_content_forms"] == ["blocks", "string"] and got["tool_result_block_types"] == ["image", "text"]
    assert got["system_form"] == "blocks" and got["system_blocks"] == 2 and got["message_string_content_roles"] == ["user"]
    assert got["tool_names"] == ["Read", "bash"]
    assert "tools[].type=bash_20250124" in got["tool_kinds"] and "tools[].custom" in got["tool_kinds"]
    assert {"model", "max_tokens", "messages", "system", "tools", "metadata", "thinking", "stream"} <= set(got["top_level_keys"])
    assert got["thinking_param"] == {"type": "str"}


def test_fr_proof_04_g2_analysis_lists_the_item_types_a_responses_client_sent() -> None:
    body = {"model": "m", "stream": True, "store": False, "include": ["reasoning.encrypted_content"], "prompt_cache_key": "k",
            "tools": [{"type": "function", "name": "exec_command", "parameters": {}}, {"type": "web_search"},
                      {"type": "namespace", "name": "ns", "tools": []}],
            "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "x"}]},
                      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "y"}]},
                      {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "mockcipher00000001"},
                      {"type": "function_call", "call_id": "c", "name": "exec_command", "arguments": "{}"},
                      {"type": "function_call_output", "call_id": "c", "output": "ok"}]}
    got = g2.block_types([rec(0, "openai.responses", body)])["responses"]
    assert got["input_item_types"] == ["function_call", "function_call_output", "message", "reasoning"]
    assert got["message_roles"] == ["developer", "user"] and got["content_part_types"] == ["input_text"]
    assert got["tool_types"] == ["function", "namespace", "web_search"] and got["tool_names"] == ["exec_command", "ns"]
    assert got["has_previous_response_id"] is False and got["store"] == [False] and got["include"] == [["reasoning.encrypted_content"]]
    assert got["input_form"] == "items"
    with_prev = dict(body, previous_response_id="resp_1")
    assert g2.block_types([rec(0, "openai.responses", with_prev)])["responses"]["has_previous_response_id"] is True


def test_fr_proof_04_g2_analysis_summarizes_transport() -> None:
    plain = rec(0, "anthropic.messages", ANTHROPIC, [["Content-Type", "application/json"], ["x-api-key", "<present>"]])
    chunked = rec(1, "openai.responses", {}, [["Transfer-Encoding", "chunked"]], chunked=True, content_length_header=None,
                  transfer_encoding="chunked", path="/v1/responses")
    zst = rec(2, "openai.responses", {}, [["Content-Encoding", "zstd"]], content_encoding="zstd", path="/v1/responses")
    ws = {"index": 3, "route": "websocket-upgrade", "method": "GET", "path": "/v1/responses", "body": "", "response_status": 426,
          "headers": [["Upgrade", "websocket"], ["Sec-WebSocket-Extensions", "permessage-deflate; client_max_window_bits"]],
          "content_encoding": None, "chunked": False, "content_length_header": None, "transfer_encoding": None}
    got = g2.transport([plain, chunked, zst, ws])
    assert got["requests"] == 4 and got["content_encodings"] == {"none": 3, "zstd": 1}
    assert got["framing"] == {"content_length": 2, "chunked": 1, "none": 1}
    assert got["upgrade_requests"] == 1 and got["websocket_extensions_offered"] == ["permessage-deflate; client_max_window_bits"]
    assert got["endpoints"] == ["GET /v1/responses", "POST /v1/messages", "POST /v1/responses"]
    assert "x-api-key" in got["header_names"] and "content-type" in got["header_names"]
    assert got["response_statuses"] == {"200": 3, "426": 1}


def test_fr_proof_04_g2_analysis_sees_whether_thinking_is_sent_back() -> None:
    first = dict(ANTHROPIC, messages=[{"role": "user", "content": "go"}])
    got = g2.thinking_resent([rec(0, "anthropic.messages", first), rec(1, "anthropic.messages", ANTHROPIC)],
                             {"thinking": ["mocksig00000001"], "redacted": ["opaque-x"], "reasoning": []})
    assert got == {"thinking_resent": True, "redacted_thinking_resent": True, "first_request_with_it": 1,
                   "reasoning_items_resent": None}
    stripped = json.loads(json.dumps(ANTHROPIC))
    stripped["messages"][1]["content"] = [stripped["messages"][1]["content"][2]]
    none = g2.thinking_resent([rec(0, "anthropic.messages", stripped)],
                              {"thinking": ["mocksig00000001"], "redacted": ["opaque-x"], "reasoning": []})
    assert none["thinking_resent"] is False and none["redacted_thinking_resent"] is False
    responses = {"input": [{"type": "reasoning", "id": "rs", "summary": [], "encrypted_content": "mockcipher00000001"}]}
    got = g2.thinking_resent([rec(0, "openai.responses", responses)], {"thinking": [], "redacted": [], "reasoning": ["mockcipher00000001"]})
    assert got["reasoning_items_resent"] is True and got["thinking_resent"] is None


def test_fr_proof_04_g2_analysis_shows_a_json_header_by_shape_and_never_by_value() -> None:
    meta = json.dumps({"installation_id": "abc", "session_id": "def", "workspaces": {"/some/path": {"has_changes": True}},
                       "sandbox_mode": "read-only", "turn_started_at_unix_ms": 123, "list": [1, 2]})
    records = [rec(0, "openai.responses", {}, [["x-codex-turn-metadata", meta], ["user-agent", "u"]])]
    shape = g2.header_shape(records, "x-codex-turn-metadata")
    assert shape == {"installation_id": "str", "session_id": "str", "workspaces": {"<key>": {"has_changes": "bool"}},
                     "sandbox_mode": "str", "turn_started_at_unix_ms": "int", "list": ["int"]}
    assert "abc" not in json.dumps(shape) and "/some/path" not in json.dumps(shape)
    assert g2.header_shape(records, "user-agent") == "str" and g2.header_shape(records, "absent") is None


def test_fr_proof_04_g2_analysis_says_where_a_planted_string_turned_up_without_printing_it() -> None:
    needle = "https://user:probe-pw-0123456789@example.test/r.git"
    records = [rec(0, "openai.responses", {"client_metadata": {"x-codex-turn-metadata": json.dumps({"remote": needle})},
                                           "input": [{"type": "message", "role": "user", "content": "hi"}]},
                   [["x-codex-turn-metadata", json.dumps({"remote": needle})], ["user-agent", "u"]]),
               rec(1, "openai.responses", {"input": "clean"}, [["user-agent", "u"]])]
    where = g2.where_found(records, needle)
    assert where == [{"request": 0, "where": "header", "name": "x-codex-turn-metadata"},
                     {"request": 0, "where": "body", "name": "client_metadata.x-codex-turn-metadata{json}.remote"}]
    assert needle not in json.dumps(where) and g2.where_found(records, "not-there-1234567") == []


def test_fr_proof_04_g2_analysis_checks_a_header_comes_back() -> None:
    records = [rec(0, "openai.responses", {}, [["x-codex-turn-state", "opaque-state-1"]]),
               rec(1, "openai.responses", {}, [["user-agent", "u"]]),
               rec(2, "openai.responses", {}, [["X-Codex-Turn-State", "opaque-state-1"]])]
    assert g2.header_values(records, "x-codex-turn-state") == {0: "opaque-state-1", 2: "opaque-state-1"}
    assert g2.header_echoed(records, "x-codex-turn-state", "opaque-state-1") == [0, 2]
    assert g2.header_echoed(records, "x-codex-turn-state", "other") == []


def test_fr_proof_04_g2_analysis_reads_the_turn_metadata_facts_the_gate_needs() -> None:
    meta = {"installation_id": "i", "request_kind": "turn", "workspaces": {"/abs/path": {
        "associated_remote_urls": {"origin": "https://example.test/r.git"}, "latest_git_commit_hash": "a" * 40, "has_changes": True}}}
    raw = json.dumps(meta)
    records = [rec(0, "openai.responses", {"client_metadata": {"x-codex-turn-metadata": raw, "thread_id": "t"}},
                   [["x-codex-turn-metadata", raw]]),
               rec(1, "openai.responses", {}, [["x-codex-turn-metadata", json.dumps(dict(meta, request_kind="compaction"))]])]
    facts = g2.turn_metadata_facts(records)
    assert facts == {"remote_url_has_userinfo": False, "remote_url_scheme_host_path_only": True, "commit_hash_hex_length": 40,
                     "workspace_keys_are_absolute_paths": True, "same_json_in_body_client_metadata": True,
                     "request_kinds": {"compaction": 1, "turn": 1}}
    assert "example.test" not in json.dumps({k: v for k, v in facts.items()})
    leaky = dict(meta)
    leaky["workspaces"] = {"/abs/path": {"associated_remote_urls": {"origin": "https://user:pw@example.test/r.git"}}}
    got = g2.turn_metadata_facts([rec(0, "openai.responses", {}, [["x-codex-turn-metadata", json.dumps(leaky)]])])
    assert got["remote_url_has_userinfo"] is True and got["same_json_in_body_client_metadata"] is False
    assert g2.turn_metadata_facts([])["request_kinds"] == {}


def test_fr_proof_04_g2_analysis_notes_system_role_messages_by_content_form() -> None:
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}]},
                         {"role": "system", "content": "# Environment ..."},
                         {"role": "system", "content": [{"type": "text", "text": "tail", "cache_control": {"type": "ephemeral"}}]}]}
    got = g2.block_types([rec(0, "anthropic.messages", body)])["anthropic"]
    assert got["system_role_message_forms"] == ["blocks", "string"]
    assert g2.block_types([rec(0, "anthropic.messages", ANTHROPIC)])["anthropic"]["system_role_message_forms"] == []


def test_fr_proof_04_g2_analysis_reads_the_request_kind_of_one_request() -> None:
    assert g2.request_kind(rec(0, "openai.responses", {}, [["X-Codex-Turn-Metadata", json.dumps({"request_kind": "compaction"})]])) == "compaction"
    assert g2.request_kind(rec(0, "openai.responses", {}, [["x-codex-turn-metadata", "not json"]])) is None
    assert g2.request_kind(rec(0, "openai.responses", {}, [])) is None
    assert g2.request_kind(rec(0, "openai.responses", {}, [["x-codex-turn-metadata", json.dumps({"request_kind": 3})]])) is None
