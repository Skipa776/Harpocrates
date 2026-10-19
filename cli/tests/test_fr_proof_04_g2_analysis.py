"""FR-PROOF-04 (G2): reading a mock recording for the parser-spec and transport-note verify items.

Each function answers one verify item from names and shapes only. The recordings below are made up.
"""
from __future__ import annotations

import json

from bench import g2_analysis as g2

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
