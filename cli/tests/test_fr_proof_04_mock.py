"""FR-PROOF-04: the mock provider (bench/mock_provider.py) as a loopback server, its JSON and SSE replies, and what it records."""
from __future__ import annotations

import glob
import gzip
import hashlib
import http.client
import json
import signal
import socket
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Iterable, Optional

import pytest

from bench import mock_provider as mock

FAKE = "FAKE-CRED-0123456789abcdef"


@pytest.mark.parametrize("name", [
    "Authorization", "authorization", "Proxy-Authorization", "x-api-key", "X-Api-Key", "api-key", "openai-api-key",
    "Cookie", "Set-Cookie", "x-auth-token", "X-Amz-Security-Token", "x-claude-code-session-id", "x-session-affinity",
    "x-client-secret", "ChatGPT-Account-ID", "chatgpt-account-id", "x-goog-api-key", "anthropic-auth-bearer",
])
def test_fr_proof_04_mock_redacts_every_credential_header_name(name: str) -> None:
    assert mock.is_credential_header(name)
    assert mock.redact_headers([(name, FAKE)]) == [[name, "<present>"]]


@pytest.mark.parametrize("name", [
    "anthropic-version", "anthropic-beta", "content-type", "content-length", "content-encoding", "user-agent",
    "accept", "host", "x-stainless-lang", "x-codex-turn-state", "x-codex-turn-metadata", "openai-beta", "version",
    "x-client-request-id", "transfer-encoding", "connection", "accept-encoding",
])
def test_fr_proof_04_mock_keeps_other_headers_verbatim(name: str) -> None:
    assert not mock.is_credential_header(name)
    assert mock.redact_headers([(name, "some-value")]) == [[name, "some-value"]]


def test_fr_proof_04_mock_redaction_keeps_order_and_duplicates() -> None:
    got = mock.redact_headers([("A", "1"), ("Cookie", FAKE), ("A", "2"), ("x-api-key", FAKE)])
    assert got == [["A", "1"], ["Cookie", "<present>"], ["A", "2"], ["x-api-key", "<present>"]]
    assert FAKE not in repr(got)


@pytest.mark.parametrize("target, expected", [
    ("/v1/messages?beta=true", "/v1/messages?beta=true"),
    ("/v1/models?limit=1000", "/v1/models?limit=1000"),
    ("/v1/messages?key=" + FAKE, "/v1/messages?key=%3Cpresent%3E"),
    ("/x?a=1&api_key=" + FAKE + "&token=" + FAKE + "&b=2", "/x?a=1&api_key=%3Cpresent%3E&token=%3Cpresent%3E&b=2"),
    ("/x?session_id=" + FAKE, "/x?session_id=%3Cpresent%3E"),
])
def test_fr_proof_04_mock_redacts_credential_query_values(target: str, expected: str) -> None:
    got = mock.redact_target(target)
    assert got == expected and FAKE not in got


# --- decoding the request body for the recording ---------------------------------------------------------------


def test_fr_proof_04_mock_decodes_identity_and_absent_encoding() -> None:
    for encoding in (None, "", "identity", "IDENTITY"):
        assert mock.decode_body(b'{"a":1}', encoding, 1024) == mock.Decoded(b'{"a":1}', "ok")


def test_fr_proof_04_mock_decodes_gzip() -> None:
    raw = gzip.compress(b'{"hello": "world"}')
    assert mock.decode_body(raw, "gzip", 1024) == mock.Decoded(b'{"hello": "world"}', "ok")
    assert mock.decode_body(raw, "GZIP", 1024).status == "ok"


def test_fr_proof_04_mock_gzip_bomb_is_too_large_not_expanded() -> None:
    bomb = gzip.compress(b"\0" * (8 * 1024 * 1024))
    got = mock.decode_body(bomb, "gzip", 1024 * 1024)
    assert got.status == "too-large" and got.body is None


def test_fr_proof_04_mock_identity_over_the_cap_is_too_large() -> None:
    assert mock.decode_body(b"x" * 11, None, 10) == mock.Decoded(None, "too-large")
    assert mock.decode_body(b"x" * 10, None, 10).status == "ok"


@pytest.mark.parametrize("raw", [b"this is not gzip", gzip.compress(b"abcdef" * 50)[:-6], b"", b"\x1f\x8b\x08\x00junk"])
def test_fr_proof_04_mock_wrong_or_truncated_gzip_is_bad_encoding(raw: bytes) -> None:
    got = mock.decode_body(raw, "gzip", 1024)
    assert got == mock.Decoded(None, "bad-encoding")


@pytest.mark.parametrize("encoding", ["br", "deflate", "gzip, zstd", "compress", "unknown"])
def test_fr_proof_04_mock_other_encodings_are_unsupported(encoding: str) -> None:
    assert mock.decode_body(zlib.compress(b"x"), encoding, 1024) == mock.Decoded(None, "unsupported-encoding")


def test_fr_proof_04_mock_zstd_is_flagged_when_no_module_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock, "_load_zstd", lambda: None)
    assert mock.decode_body(b"\x28\xb5\x2f\xfd\x00", "zstd", 1024) == mock.Decoded(None, "zstd-unavailable")


def test_fr_proof_04_mock_zstd_is_decoded_when_a_module_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock, "_load_zstd", lambda: lambda data, cap: b'{"z": 1}')
    assert mock.decode_body(b"fake-zstd-frame", "zstd", 1024) == mock.Decoded(b'{"z": 1}', "ok")

    def fail(data: bytes, cap: int) -> bytes:
        raise ValueError("corrupt frame")

    monkeypatch.setattr(mock, "_load_zstd", lambda: fail)
    assert mock.decode_body(b"x", "zstd", 1024) == mock.Decoded(None, "bad-encoding")
    monkeypatch.setattr(mock, "_load_zstd", lambda: lambda data, cap: b"x" * (cap + 1))
    assert mock.decode_body(b"x", "zstd", 8) == mock.Decoded(None, "too-large")


# --- the scripted replies ------------------------------------------------------------------------------------------


def write_script(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "script.json"
    path.write_text(json.dumps(payload))
    return path


def test_fr_proof_04_mock_script_loads_a_list_of_turns(tmp_path: Path) -> None:
    turns = [{"tool_calls": [{"name": "Read", "input": {"file_path": ".env"}}]},
             {"text": "done"}, {"error": {"status": 400, "message": "blocked"}}]
    assert mock.load_script(write_script(tmp_path, turns)) == turns


@pytest.mark.parametrize("payload", [
    {"turns": []}, "text", [1], [{"surprise": 1}], [{"text": 5}], [{"tool_calls": "Read"}], [{"tool_calls": [{}]}],
    [{"tool_calls": [{"name": "Read", "input": []}]}], [{"error": {"status": 200, "message": "x"}}],
    [{"error": {"status": "400", "message": "x"}}], [{"error": {"status": 400}}],
])
def test_fr_proof_04_mock_script_rejects_a_bad_shape(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ValueError):
        mock.load_script(write_script(tmp_path, payload))


def test_fr_proof_04_mock_script_rejects_unreadable_or_non_json_files(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError):
        mock.load_script(tmp_path / "bad.json")
    with pytest.raises(ValueError):
        mock.load_script(tmp_path / "missing.json")


def test_fr_proof_04_mock_counts_tool_results_in_an_anthropic_body() -> None:
    body = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "text", "text": "x"}, {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a"}, {"type": "text", "text": "y"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "b"}]}]},
    ]}
    assert mock.count_tool_results(body, "anthropic") == 2
    assert mock.count_tool_results({"messages": [{"role": "user", "content": "hi"}]}, "anthropic") == 0
    assert mock.count_tool_results({}, "anthropic") == 0 and mock.count_tool_results([1], "anthropic") == 0


def test_fr_proof_04_mock_counts_tool_outputs_in_a_responses_body() -> None:
    body = {"input": [
        {"type": "message", "role": "user", "content": "go"},
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "a"},
        {"type": "custom_tool_call_output", "call_id": "c2", "output": "b"},
        {"type": "local_shell_call_output", "id": "c3", "output": "c"},
    ]}
    assert mock.count_tool_results(body, "responses") == 3
    assert mock.count_tool_results({"input": "just text"}, "responses") == 0
    assert mock.count_tool_results({}, "responses") == 0


def test_fr_proof_04_mock_picks_the_turn_by_tool_results_already_in_the_request() -> None:
    turns = [{"tool_calls": [{"name": "Read"}]}, {"tool_calls": [{"name": "Bash"}, {"name": "Grep"}]}, {"text": "done"}]
    assert mock.select_turn(turns, 0) is turns[0]
    assert mock.select_turn(turns, 1) is turns[1]  # the first call has an answer
    assert mock.select_turn(turns, 3) is turns[2]  # both calls of turn 2 have answers
    assert mock.select_turn(turns, 2) is turns[2]  # between turns: the next one that starts at or after 2
    assert mock.select_turn(turns, 99) is turns[2]  # past the end: the last turn, again and again
    assert mock.select_turn([], 5) == {}


def test_fr_proof_04_mock_a_request_without_tools_gets_a_plain_text_turn() -> None:
    turns = [{"tool_calls": [{"name": "Read"}]}]
    assert mock.turn_for({"messages": [], "tools": [{"name": "Read"}]}, "anthropic", turns) is turns[0]
    assert mock.turn_for({"messages": []}, "anthropic", turns) == {}
    assert mock.turn_for({"messages": [], "tools": []}, "anthropic", turns) == {}
    assert mock.turn_for({"input": "x", "tools": [{"type": "function"}]}, "responses", turns) is turns[0]
    assert mock.turn_for({"input": "x"}, "responses", turns) == {}
    assert mock.turn_for("not an object", "responses", turns) == {}


# --- no header outside the redaction set can hold a credential -------------------------------------------------------


# Header names that the agents were seen to send (G2 probe run 20261019T175710Z-g2) and that are not redacted. Each one
# carries a version, an id, a content-negotiation value or a fixed flag; none is a credential. A header that is neither
# credential-named nor in this set makes the test below fail, so a new agent version cannot slip a new header, whose value
# might be a credential, into the recording unreviewed.
BENIGN_HEADERS = frozenset({
    "accept", "accept-encoding", "anthropic-beta", "anthropic-dangerous-direct-browser-access", "anthropic-version",
    "cache-control", "connection", "content-encoding", "content-length", "content-type", "host", "openai-beta", "originator", "thread-id",
    "transfer-encoding", "upgrade", "user-agent", "version", "x-app", "x-client-request-id", "x-codex-beta-features",
    "x-codex-routing-hint", "x-codex-turn-metadata", "x-codex-turn-state", "x-codex-window-id",
})
BENIGN_PREFIXES = ("x-stainless-", "sec-websocket-")


def test_fr_proof_04_mock_no_header_outside_the_redaction_set_could_hold_a_credential() -> None:
    reports = sorted(glob.glob(str(Path(mock.__file__).resolve().parent / "results" / "proof" / "g2-probes-*.json")))
    assert reports, "the committed G2 probe report is missing"
    seen: set[str] = set()
    for path in reports:
        for probe in json.loads(Path(path).read_text())["probes"]:
            transport = probe.get("observations", {}).get("transport")
            if transport:
                seen |= set(transport["header_names"])
    assert len(seen) > 20
    kept = {name for name in seen if not mock.is_credential_header(name)}
    unreviewed = {name for name in kept if name not in BENIGN_HEADERS and not name.startswith(BENIGN_PREFIXES)}
    assert not unreviewed, f"review these headers, then add them to BENIGN_HEADERS or to the redaction words: {sorted(unreviewed)}"
    # And the ones that were redacted really are credential-named: the session ids, keys and tokens among them.
    redacted = seen - kept
    assert {"x-api-key", "authorization", "session-id", "x-claude-code-session-id", "chatgpt-account-id"} <= redacted


TEXT_TURN = {"text": "Hello from the mock."}
TOOL_TURN = {"text": "Reading it.", "tool_calls": [{"name": "Read", "input": {"file_path": ".env", "limit": 5}}]}
TWO_TOOLS = {"tool_calls": [{"name": "Bash", "input": {"command": "cat a"}}, {"name": "Grep", "input": {"pattern": "TOKEN"}}]}
TURNS = [TEXT_TURN, TOOL_TURN, TWO_TOOLS]


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    """Split an SSE body into (event name, data) pairs, requiring the wire format the clients parse."""
    assert raw.endswith(b"\n\n")
    out = []
    for frame in raw.decode("utf-8").split("\n\n")[:-1]:
        lines = frame.split("\n")
        assert len(lines) == 2 and lines[0].startswith("event: ") and lines[1].startswith("data: "), frame
        data = json.loads(lines[1][len("data: "):])
        assert data["type"] == lines[0][len("event: "):]
        out.append((data["type"], data))
    return out


# --- Anthropic Messages ----------------------------------------------------------------------------------------

def check_anthropic_message(msg: dict, turn: dict) -> None:
    assert msg["type"] == "message" and msg["role"] == "assistant"
    assert msg["id"].startswith("msg_") and isinstance(msg["model"], str)
    assert msg["stop_sequence"] is None
    assert isinstance(msg["usage"]["input_tokens"], int) and isinstance(msg["usage"]["output_tokens"], int)
    calls = turn.get("tool_calls", [])
    assert msg["stop_reason"] == ("tool_use" if calls else "end_turn")
    blocks = msg["content"]
    texts = [b for b in blocks if b["type"] == "text"]
    uses = [b for b in blocks if b["type"] == "tool_use"]
    assert len(texts) == (1 if turn.get("text") or not calls else 0) and len(uses) == len(calls)
    for block, call in zip(uses, calls):
        assert block["id"].startswith("toolu_") and block["name"] == call["name"] and block["input"] == call["input"]
    assert len({b["id"] for b in uses}) == len(uses)
    if calls and texts:
        assert blocks.index(texts[0]) < blocks.index(uses[0])  # text first, then the tool calls


@pytest.mark.parametrize("turn", TURNS)
def test_fr_proof_04_mock_anthropic_json_reply_has_the_message_shape(turn: dict) -> None:
    msg = mock.anthropic_message(turn, "claude-mock-1", 7, 123)
    check_anthropic_message(json.loads(json.dumps(msg)), turn)
    assert msg["model"] == "claude-mock-1" and msg["usage"]["input_tokens"] == 123


@pytest.mark.parametrize("turn", TURNS)
def test_fr_proof_04_mock_anthropic_stream_follows_the_documented_event_order(turn: dict) -> None:
    msg = mock.anthropic_message(turn, "claude-mock-1", 7, 123)
    events = parse_sse(mock.sse_bytes(mock.anthropic_events(msg)))
    names = [name for name, _ in events]
    assert names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"]
    assert events[0][1]["message"]["content"] == [] and events[0][1]["message"]["stop_reason"] is None
    assert set(names) <= {"message_start", "content_block_start", "ping", "content_block_delta", "content_block_stop",
                          "message_delta", "message_stop"}
    # Rebuild the message from the events, as a client does, and compare it with the JSON reply.
    rebuilt: list[dict] = []
    partial: dict[int, str] = {}
    open_index = None
    for name, data in events[1:-2]:
        if name == "content_block_start":
            assert data["index"] == len(rebuilt) and open_index is None
            rebuilt.append(dict(data["content_block"]))
            partial[data["index"]] = ""
            open_index = data["index"]
        elif name == "content_block_delta":
            assert data["index"] == open_index
            delta = data["delta"]
            if delta["type"] == "text_delta":
                rebuilt[open_index]["text"] += delta["text"]
            else:
                assert delta["type"] == "input_json_delta"
                partial[open_index] += delta["partial_json"]
        elif name == "content_block_stop":
            assert data["index"] == open_index
            if rebuilt[open_index]["type"] == "tool_use":
                rebuilt[open_index]["input"] = json.loads(partial[open_index])
            open_index = None
    assert open_index is None and rebuilt == msg["content"]
    last = events[-2][1]
    assert last["delta"]["stop_reason"] == msg["stop_reason"] and isinstance(last["usage"]["output_tokens"], int)


def test_fr_proof_04_mock_anthropic_tool_use_block_starts_with_empty_input_like_the_api() -> None:
    msg = mock.anthropic_message(TOOL_TURN, "m", 1, 1)
    events = parse_sse(mock.sse_bytes(mock.anthropic_events(msg)))
    starts = [d["content_block"] for n, d in events if n == "content_block_start"]
    tool = next(b for b in starts if b["type"] == "tool_use")
    assert tool["input"] == {} and tool["id"].startswith("toolu_")


def test_fr_proof_04_mock_anthropic_default_turn_is_a_short_text_reply() -> None:
    msg = mock.anthropic_message({}, "m", 1, 1)
    assert msg["stop_reason"] == "end_turn" and msg["content"] and msg["content"][0]["type"] == "text"


# --- OpenAI Responses ------------------------------------------------------------------------------------------

def check_response(resp: dict, turn: dict, status: str = "completed") -> None:
    assert resp["object"] == "response" and resp["id"].startswith("resp_") and resp["status"] == status
    assert isinstance(resp["model"], str) and isinstance(resp["created_at"], int)
    assert resp["error"] is None and resp["incomplete_details"] is None and isinstance(resp["output"], list)
    calls = turn.get("tool_calls", [])
    for item in resp["output"]:
        if status == "completed":
            assert item["status"] == "completed"
    messages = [i for i in resp["output"] if i["type"] == "message"]
    functions = [i for i in resp["output"] if i["type"] == "function_call"]
    assert len(messages) == (1 if turn.get("text") or not calls else 0) and len(functions) == len(calls)
    for item in messages:
        assert item["role"] == "assistant" and item["id"].startswith("msg_")
        assert item["content"][0]["type"] == "output_text" and item["content"][0]["annotations"] == []
    for item, call in zip(functions, calls):
        assert item["id"].startswith("fc_") and item["call_id"].startswith("call_") and item["name"] == call["name"]
        assert isinstance(item["arguments"], str) and json.loads(item["arguments"]) == call["input"]
    usage = resp["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert "cached_tokens" in usage["input_tokens_details"] and "reasoning_tokens" in usage["output_tokens_details"]


@pytest.mark.parametrize("turn", TURNS)
def test_fr_proof_04_mock_responses_json_reply_has_the_response_shape(turn: dict) -> None:
    resp = mock.responses_object(turn, "gpt-mock", 9, 55)
    check_response(json.loads(json.dumps(resp)), turn)
    assert resp["model"] == "gpt-mock" and resp["usage"]["input_tokens"] == 55


@pytest.mark.parametrize("turn", TURNS)
def test_fr_proof_04_mock_responses_stream_follows_the_documented_event_order(turn: dict) -> None:
    resp = mock.responses_object(turn, "gpt-mock", 9, 55)
    events = parse_sse(mock.sse_bytes(mock.responses_events(resp)))
    names = [name for name, _ in events]
    assert names[:2] == ["response.created", "response.in_progress"] and names[-1] == "response.completed"
    assert [d["sequence_number"] for _, d in events] == list(range(len(events)))
    assert events[0][1]["response"]["status"] == "in_progress" and events[0][1]["response"]["output"] == []
    assert events[-1][1]["response"] == resp
    # Per output item: added, its content events, done, in output order.
    done = [d for n, d in events if n == "response.output_item.done"]
    added = [d for n, d in events if n == "response.output_item.added"]
    assert [d["item"] for d in done] == resp["output"]
    assert [d["output_index"] for d in added] == list(range(len(resp["output"])))
    assert [d["output_index"] for d in done] == list(range(len(resp["output"])))
    assert all(d["item"]["status"] == "in_progress" for d in added)
    for index, item in enumerate(resp["output"]):
        mine = [n for n, d in events if d.get("output_index") == index]
        if item["type"] == "message":
            assert mine == ["response.output_item.added", "response.content_part.added", "response.output_text.delta",
                            "response.output_text.done", "response.content_part.done", "response.output_item.done"]
            text = "".join(d["delta"] for n, d in events if n == "response.output_text.delta" and d["output_index"] == index)
            assert text == item["content"][0]["text"]
        else:
            assert mine == ["response.output_item.added", "response.function_call_arguments.delta",
                            "response.function_call_arguments.done", "response.output_item.done"]
            args = "".join(d["delta"] for n, d in events
                           if n == "response.function_call_arguments.delta" and d["output_index"] == index)
            assert args == item["arguments"]
    for n, d in events:
        if "item_id" in d:
            assert d["item_id"] in {i["id"] for i in resp["output"]}


def test_fr_proof_04_mock_responses_default_turn_is_a_short_text_message() -> None:
    resp = mock.responses_object({}, "m", 1, 1)
    assert [i["type"] for i in resp["output"]] == ["message"]


def test_fr_proof_04_mock_ids_are_stable_for_a_serial_and_differ_between_serials() -> None:
    a1, a2 = mock.anthropic_message(TOOL_TURN, "m", 3, 1), mock.anthropic_message(TOOL_TURN, "m", 3, 1)
    assert a1 == a2
    assert mock.anthropic_message(TOOL_TURN, "m", 4, 1)["id"] != a1["id"]
    r1, r2 = mock.responses_object(TOOL_TURN, "m", 3, 1), mock.responses_object(TOOL_TURN, "m", 4, 1)
    assert r1["id"] != r2["id"] and r1["output"][1]["call_id"] != r2["output"][1]["call_id"]


# --- thinking blocks and reasoning items (G2 verify items V4 and V6) -----------------------------------------------

THINKING_TURN = {"thinking": "I should look at the file.", "text": "Reading.",
                 "tool_calls": [{"name": "Bash", "input": {"command": "ls"}}]}
REDACTED_TURN = {"redacted_thinking": "opaque-redacted-payload", "tool_calls": [{"name": "Bash", "input": {"command": "ls"}}]}
BOTH_TURN = {"thinking": "first", "redacted_thinking": "second", "text": "hi"}


def test_fr_proof_04_mock_anthropic_thinking_blocks_come_first_and_carry_a_signature() -> None:
    msg = mock.anthropic_message(THINKING_TURN, "m", 5, 10)
    kinds = [b["type"] for b in msg["content"]]
    assert kinds == ["thinking", "text", "tool_use"]
    thinking = msg["content"][0]
    assert thinking["thinking"] == "I should look at the file." and thinking["signature"].startswith("mocksig")
    msg = mock.anthropic_message(BOTH_TURN, "m", 5, 10)
    assert [b["type"] for b in msg["content"]] == ["thinking", "redacted_thinking", "text"]
    assert msg["content"][1] == {"type": "redacted_thinking", "data": "second"}
    assert msg["stop_reason"] == "end_turn"


@pytest.mark.parametrize("turn", [THINKING_TURN, REDACTED_TURN, BOTH_TURN])
def test_fr_proof_04_mock_anthropic_thinking_stream_rebuilds_to_the_json_message(turn: dict) -> None:
    msg = mock.anthropic_message(turn, "m", 5, 10)
    events = parse_sse(mock.sse_bytes(mock.anthropic_events(msg)))
    rebuilt: list[dict] = []
    partial = ""
    for name, data in events:
        if name == "content_block_start":
            rebuilt.append(dict(data["content_block"]))
            partial = ""
        elif name == "content_block_delta":
            delta = data["delta"]
            if delta["type"] == "thinking_delta":
                rebuilt[-1]["thinking"] += delta["thinking"]
            elif delta["type"] == "signature_delta":
                rebuilt[-1]["signature"] += delta["signature"]
            elif delta["type"] == "text_delta":
                rebuilt[-1]["text"] += delta["text"]
            else:
                partial += delta["partial_json"]
        elif name == "content_block_stop" and rebuilt[-1]["type"] == "tool_use":
            rebuilt[-1]["input"] = json.loads(partial)
    assert rebuilt == msg["content"]
    starts = [d["content_block"] for n, d in events if n == "content_block_start"]
    thinking = [b for b in starts if b["type"] == "thinking"]
    assert all(b["thinking"] == "" and b["signature"] == "" for b in thinking)  # filled by the deltas, as the API does


def test_fr_proof_04_mock_responses_reasoning_items_come_first_with_encrypted_content() -> None:
    resp = mock.responses_object(THINKING_TURN, "m", 5, 10)
    assert [i["type"] for i in resp["output"]] == ["reasoning", "message", "function_call"]
    reasoning = resp["output"][0]
    assert reasoning["summary"] == [{"type": "summary_text", "text": "I should look at the file."}]
    assert reasoning["encrypted_content"].startswith("mockcipher") and reasoning["id"].startswith("rs_")
    only = mock.responses_object(REDACTED_TURN, "m", 5, 10)["output"][0]
    assert only["type"] == "reasoning" and only["summary"] == [] and only["encrypted_content"] == "opaque-redacted-payload"


@pytest.mark.parametrize("turn", [THINKING_TURN, REDACTED_TURN, BOTH_TURN])
def test_fr_proof_04_mock_responses_reasoning_stream_has_the_documented_events(turn: dict) -> None:
    resp = mock.responses_object(turn, "m", 5, 10)
    events = parse_sse(mock.sse_bytes(mock.responses_events(resp)))
    assert [d["sequence_number"] for _, d in events] == list(range(len(events)))
    done = [d["item"] for n, d in events if n == "response.output_item.done"]
    assert done == resp["output"] and events[-1][1]["response"] == resp
    for index, item in enumerate(resp["output"]):
        if item["type"] != "reasoning":
            continue
        mine = [n for n, d in events if d.get("output_index") == index]
        if item["summary"]:
            assert mine == ["response.output_item.added", "response.reasoning_summary_part.added",
                            "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
                            "response.reasoning_summary_part.done", "response.output_item.done"]
        else:
            assert mine == ["response.output_item.added", "response.output_item.done"]


def test_fr_proof_04_mock_script_accepts_thinking_keys(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps([THINKING_TURN, REDACTED_TURN]))
    assert mock.load_script(path) == [THINKING_TURN, REDACTED_TURN]
    path.write_text(json.dumps([{"thinking": 5}]))
    with pytest.raises(ValueError):
        mock.load_script(path)
    path.write_text(json.dumps([{"redacted_thinking": ["x"]}]))
    with pytest.raises(ValueError):
        mock.load_script(path)


def test_fr_proof_04_mock_responses_call_carries_a_namespace_only_when_scripted(tmp_path) -> None:
    """An MCP tool sits in a namespace in Codex's tool list; the scripted call names it, and a plain call has none."""
    turn = {"tool_calls": [{"name": "show_file", "namespace": "mcp__media", "input": {"path": "a"}},
                           {"name": "exec_command", "input": {"cmd": "ls"}}]}
    calls = [item for item in mock.responses_object(turn, "m", 1, 1)["output"] if item["type"] == "function_call"]
    assert [(c["name"], c.get("namespace")) for c in calls] == [("show_file", "mcp__media"), ("exec_command", None)]
    path = tmp_path / "s.json"
    path.write_text(json.dumps([turn]))
    assert mock.load_script(path) == [turn]
    path.write_text(json.dumps([{"tool_calls": [{"name": "x", "namespace": 3}]}]))
    with pytest.raises(ValueError):
        mock.load_script(path)


@pytest.fixture
def server(tmp_path: Path):
    srv = mock.start_server(record_dir=tmp_path / "rec")
    yield srv
    srv.close()


def send(srv, method: str, path: str, body: Optional[bytes] = None, headers: Optional[dict] = None,
         chunks: Optional[Iterable[bytes]] = None) -> tuple[int, dict, bytes]:
    """One request over a fresh connection; chunks sends a chunked body. Returns (status, headers, body)."""
    conn = http.client.HTTPConnection(srv.host, srv.port, timeout=10)
    try:
        if chunks is not None:
            conn.request(method, path, body=iter(chunks), headers={**(headers or {}), "Transfer-Encoding": "chunked"},
                         encode_chunked=True)
        else:
            conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


def recorded(srv) -> list[dict]:
    """The recording as the file holds it: one JSON object per line."""
    lines = (Path(srv.record_dir) / "requests.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


# --- binding ---------------------------------------------------------------------------------------------------

def test_fr_proof_04_mock_binds_loopback_only(server) -> None:
    assert server.host == "127.0.0.1" and server.server.server_address[0] == "127.0.0.1"
    assert server.url == f"http://127.0.0.1:{server.port}" and server.port > 0
    # Not reachable on this machine's other addresses.
    try:
        other = socket.gethostbyname(socket.gethostname())
    except OSError:
        other = "127.0.0.1"
    if not other.startswith("127."):
        with pytest.raises(OSError):
            socket.create_connection((other, server.port), timeout=2).close()


@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", "192.168.1.10", "10.0.0.1", "8.8.8.8", "example.com", "127.0.0.1.evil.test"])
def test_fr_proof_04_mock_refuses_to_start_on_a_non_loopback_address(host: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        mock.start_server(host=host, record_dir=tmp_path / "rec")
    assert not (tmp_path / "rec").exists()  # refused before it created anything


@pytest.mark.parametrize("host, shown", [("localhost", "127.0.0.1"), ("127.0.0.1", "127.0.0.1")])
def test_fr_proof_04_mock_accepts_loopback_addresses(host: str, shown: str, tmp_path: Path) -> None:
    with mock.start_server(host=host, record_dir=tmp_path / "rec") as srv:
        assert srv.host == shown


@pytest.mark.parametrize("host, ok", [("127.0.0.1", True), ("127.0.0.2", True), ("::1", True), ("localhost", True),
                                      ("0.0.0.0", False), ("::", False), ("", False), ("192.168.0.1", False),
                                      ("::ffff:8.8.8.8", False), ("localhost.evil.test", False), ("127.0.0.1 ", False)])
def test_fr_proof_04_mock_loopback_check(host: str, ok: bool) -> None:
    if ok:
        assert mock.ensure_loopback(host) in (host, "127.0.0.1")
    else:
        with pytest.raises(ValueError, match="loopback"):
            mock.ensure_loopback(host)


def test_fr_proof_04_mock_makes_no_outbound_connection() -> None:
    source = Path(mock.__file__).read_text()
    for needle in ("urlopen", "http.client", "create_connection", "import requests", "import httpx", ".connect("):
        assert needle not in source, needle


# --- unknown routes and the recording ---------------------------------------------------------------------------

def test_fr_proof_04_mock_records_unknown_routes(server) -> None:
    status, _, body = send(server, "POST", "/v1/surprise?x=1", b'{"hello": "world"}', {"Content-Type": "application/json"})
    assert status == 404 and json.loads(body)["error"]["type"] == "not_found_error"
    assert send(server, "GET", "/nowhere")[0] == 404
    assert send(server, "PUT", "/v1/files/1", b"data")[0] == 404
    assert send(server, "DELETE", "/v1/files/1")[0] == 404
    status, _, head_body = send(server, "HEAD", "/api/hello")
    assert status == 404 and head_body == b""
    rows = recorded(server)
    assert [(r["method"], r["path"], r["route"], r["response_status"]) for r in rows] == [
        ("POST", "/v1/surprise?x=1", "unknown", 404), ("GET", "/nowhere", "unknown", 404),
        ("PUT", "/v1/files/1", "unknown", 404), ("DELETE", "/v1/files/1", "unknown", 404),
        ("HEAD", "/api/hello", "unknown", 404)]
    first = rows[0]
    assert first["body"] == '{"hello": "world"}' and first["body_size"] == 18 and first["body_status"] == "ok"
    assert first["body_sha256"] == hashlib.sha256(b'{"hello": "world"}').hexdigest()
    assert [r["index"] for r in rows] == [0, 1, 2, 3, 4] and all(r["time"].endswith("+00:00") for r in rows)
    assert server.records() == rows  # the in-memory copy matches the file


def test_fr_proof_04_mock_never_records_credentials(server) -> None:
    creds = {"x-api-key": FAKE, "Authorization": f"Bearer {FAKE}", "cookie": f"sid={FAKE}", "ChatGPT-Account-ID": FAKE,
             "X-Auth-Token": FAKE, "x-claude-code-session-id": FAKE, "openai-api-key": FAKE, "Set-Cookie": FAKE}
    plain = {"anthropic-version": "2023-06-01", "user-agent": "unit-test", "x-codex-turn-state": "opaque-state"}
    send(server, "POST", f"/v1/unknown?key={FAKE}&beta=true", b"{}", {**creds, **plain})
    blob = (Path(server.record_dir) / "requests.jsonl").read_text()
    assert FAKE not in blob and FAKE not in repr(server.records())
    row = recorded(server)[0]
    headers = {name.lower(): value for name, value in row["headers"]}
    for name in creds:
        assert headers[name.lower()] == "<present>", name
    for name, value in plain.items():
        assert headers[name] == value
    assert row["path"] == "/v1/unknown?key=%3Cpresent%3E&beta=true"


# --- reading the body --------------------------------------------------------------------------------------------

def test_fr_proof_04_mock_records_a_chunked_body_as_chunked(server) -> None:
    send(server, "POST", "/v1/unknown", chunks=[b'{"a":', b' "b"', b"}"])
    row = recorded(server)[0]
    assert row["chunked"] is True and row["content_length_header"] is None and row["transfer_encoding"] == "chunked"
    assert row["body"] == '{"a": "b"}' and row["body_size"] == 10 and row["body_status"] == "ok"


def test_fr_proof_04_mock_records_content_length_and_gzip(server) -> None:
    raw = gzip.compress(b'{"zipped": true}')
    send(server, "POST", "/v1/unknown", raw, {"Content-Encoding": "gzip"})
    row = recorded(server)[0]
    assert row["chunked"] is False and row["content_length_header"] == len(raw) and row["transfer_encoding"] is None
    assert row["content_encoding"] == "gzip" and row["body"] == '{"zipped": true}'
    assert row["body_size"] == len(raw) and row["body_sha256"] == hashlib.sha256(raw).hexdigest()


def test_fr_proof_04_mock_survives_bad_bodies(tmp_path: Path) -> None:
    with mock.start_server(record_dir=tmp_path / "rec", max_body=1000) as srv:
        send(srv, "POST", "/v1/unknown", b"x" * 1001)  # over the cap, Content-Length
        send(srv, "POST", "/v1/unknown", chunks=[b"y" * 600, b"y" * 600])  # over the cap, chunked
        send(srv, "POST", "/v1/unknown", gzip.compress(b"\0" * 100_000), {"Content-Encoding": "gzip"})  # a bomb
        send(srv, "POST", "/v1/unknown", b"this is not gzip", {"Content-Encoding": "gzip"})  # wrong encoding
        send(srv, "POST", "/v1/unknown", b"\xff\xfe\x00 not utf-8 \xc3\x28")  # not UTF-8
        send(srv, "POST", "/v1/unknown", b"opaque", {"Content-Encoding": "br"})  # an encoding it cannot decode
        send(srv, "POST", "/v1/unknown", chunks=[])  # an empty chunked body
        rows = recorded(srv)
        assert [r["body_status"] for r in rows] == ["too-large", "too-large", "too-large", "bad-encoding", "not-utf8",
                                                    "unsupported-encoding", "empty"]
        assert [r["response_status"] for r in rows] == [400, 400, 400, 400, 404, 404, 404]
        assert rows[0]["body"] is None and rows[0]["body_size"] == 1001
        assert rows[0]["body_sha256"] == hashlib.sha256(b"x" * 1001).hexdigest()  # hashed while discarded
        assert rows[1]["body_size"] == 1200 and rows[1]["body_sha256"] == hashlib.sha256(b"y" * 1200).hexdigest()
        assert rows[3]["body"] is None
        assert rows[4]["body"] is not None and rows[4]["body_size"] == 16  # decoded with replacement characters
        assert rows[5]["body"] is None and rows[5]["body_raw_b64"] == "b3BhcXVl"
        # A bad Content-Length, and a client that promises more than it sends, do not stop the server.
        conn = http.client.HTTPConnection(srv.host, srv.port, timeout=10)
        conn.putrequest("POST", "/v1/unknown")
        conn.putheader("Content-Length", "not-a-number")
        conn.endheaders()
        assert conn.getresponse().status == 400
        conn.close()
        sock = socket.create_connection((srv.host, srv.port), timeout=5)
        sock.sendall(b"POST /v1/unknown HTTP/1.1\r\nHost: x\r\nContent-Length: 500\r\n\r\nonly a little")
        sock.shutdown(socket.SHUT_WR)
        sock.recv(4096)
        sock.close()
        assert send(srv, "GET", "/still-alive")[0] == 404
        tail = [r["body_status"] for r in recorded(srv)[-3:]]
        assert tail == ["bad-length", "truncated", "empty"]


def test_fr_proof_04_mock_answers_a_websocket_upgrade_with_426_and_records_it(server) -> None:
    status, _, _ = send(server, "GET", "/v1/responses", headers={
        "Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": FAKE, "Sec-WebSocket-Extensions": "permessage-deflate"})
    assert status == 426
    row = recorded(server)[0]
    assert row["route"] == "websocket-upgrade" and row["response_status"] == 426
    names = {n.lower(): v for n, v in row["headers"]}
    assert names["upgrade"] == "websocket" and names["sec-websocket-extensions"] == "permessage-deflate"
    assert names["sec-websocket-key"] == "<present>" and FAKE not in json.dumps(row)


# --- the provider routes -----------------------------------------------------------------------------------------


JSON = {"Content-Type": "application/json"}
ANTHROPIC_REQUEST = {"model": "claude-x", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
RESPONSES_REQUEST = {"model": "gpt-x", "input": "hi", "store": False}
READ_TOOL = [{"name": "Read", "description": "d", "input_schema": {"type": "object"}}]
SHELL_TOOL = [{"type": "function", "name": "shell", "parameters": {"type": "object"}}]


def post_json(srv, path: str, payload: object, headers: Optional[dict] = None) -> tuple[int, dict, bytes]:
    return send(srv, "POST", path, json.dumps(payload).encode(), {**JSON, **(headers or {})})


def test_fr_proof_04_mock_answers_both_formats(server) -> None:
    status, hdrs, body = post_json(server, "/v1/messages?beta=true", ANTHROPIC_REQUEST)
    assert status == 200 and hdrs["content-type"].startswith("application/json")
    message = json.loads(body)
    check_anthropic_message(message, {})
    assert message["model"] == "claude-x"

    status, hdrs, body = post_json(server, "/v1/messages", {**ANTHROPIC_REQUEST, "stream": True})
    assert status == 200 and hdrs["content-type"].startswith("text/event-stream") and "content-length" not in hdrs
    events = parse_sse(body)
    names = [name for name, _ in events]
    assert names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"]
    assert events[0][1]["message"]["model"] == "claude-x"

    status, hdrs, body = post_json(server, "/v1/responses", RESPONSES_REQUEST)
    assert status == 200 and hdrs["content-type"].startswith("application/json")
    response = json.loads(body)
    check_response(response, {})
    assert response["model"] == "gpt-x"

    status, hdrs, body = post_json(server, "/v1/responses", {**RESPONSES_REQUEST, "stream": True})
    assert status == 200 and hdrs["content-type"].startswith("text/event-stream")
    events = parse_sse(body)
    assert events[0][0] == "response.created" and events[-1][0] == "response.completed"
    check_response(events[-1][1]["response"], {})
    rows = recorded(server)
    assert [(r["route"], r["stream"], r["response_status"]) for r in rows] == [
        ("anthropic.messages", False, 200), ("anthropic.messages", True, 200),
        ("openai.responses", False, 200), ("openai.responses", True, 200)]
    assert json.loads(rows[0]["body"]) == ANTHROPIC_REQUEST  # the body is stored decoded


def test_fr_proof_04_mock_follows_the_script_one_turn_per_model_call(tmp_path: Path) -> None:
    turns = [{"tool_calls": [{"name": "Read", "input": {"file_path": ".env"}}]},
             {"tool_calls": [{"name": "Bash", "input": {"command": "cat config/settings.py"}}]}, {"text": "done"}]
    with mock.start_server(record_dir=tmp_path / "rec", turns=turns) as srv:
        convo = [{"role": "user", "content": "go"}]
        seen = []
        for _ in range(3):
            status, _, body = post_json(srv, "/v1/messages", {**ANTHROPIC_REQUEST, "messages": convo, "tools": READ_TOOL})
            assert status == 200
            msg = json.loads(body)
            seen.append(msg["stop_reason"])
            uses = [b for b in msg["content"] if b["type"] == "tool_use"]
            convo = convo + [{"role": "assistant", "content": msg["content"]}]
            if uses:
                convo.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"], "content": "ok"}]})
        assert seen == ["tool_use", "tool_use", "end_turn"] and msg["content"][0]["text"] == "done"
        assert [r["reply_turn"] for r in recorded(srv)] == [0, 1, 2]
        # The same for Responses, with function_call items and their outputs.
        items = [{"type": "message", "role": "user", "content": "go"}]
        seen = []
        for _ in range(3):
            status, _, body = post_json(srv, "/v1/responses", {**RESPONSES_REQUEST, "input": items, "tools": SHELL_TOOL})
            out = json.loads(body)["output"]
            calls = [i for i in out if i["type"] == "function_call"]
            seen.append(calls[0]["name"] if calls else "text")
            items = items + out
            if calls:
                items.append({"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "ok"})
        assert seen == ["Read", "Bash", "text"]  # the same script, now spoken as function_call items
        # A streamed tool call carries the same input.
        _, _, body = post_json(srv, "/v1/messages", {**ANTHROPIC_REQUEST, "tools": READ_TOOL, "stream": True})
        events = parse_sse(body)
        assert "".join(d["delta"]["partial_json"] for n, d in events if n == "content_block_delta") == '{"file_path":".env"}'


def test_fr_proof_04_mock_a_scripted_error_turn_is_a_provider_shaped_http_error(tmp_path: Path) -> None:
    turns = [{"error": {"status": 400, "message": "blocked by policy"}}]
    with mock.start_server(record_dir=tmp_path / "rec", turns=turns) as srv:
        status, _, body = post_json(srv, "/v1/messages", {**ANTHROPIC_REQUEST, "tools": READ_TOOL})
        assert status == 400 and json.loads(body)["type"] == "error"
        assert json.loads(body)["error"] == {"type": "invalid_request_error", "message": "blocked by policy"}
        status, _, body = post_json(srv, "/v1/responses", {**RESPONSES_REQUEST, "tools": SHELL_TOOL})
        assert status == 400 and json.loads(body)["error"]["message"] == "blocked by policy"
        assert [r["response_status"] for r in recorded(srv)] == [400, 400]


def test_fr_proof_04_mock_counts_tokens_and_lists_models(tmp_path: Path) -> None:
    with mock.start_server(record_dir=tmp_path / "rec") as srv:
        status, _, body = post_json(srv, "/v1/messages/count_tokens", ANTHROPIC_REQUEST)
        assert status == 200 and json.loads(body) == {"input_tokens": 42}
        status, _, body = send(srv, "GET", "/v1/models?limit=1000", headers={"anthropic-version": "2023-06-01"})
        listing = json.loads(body)
        assert status == 200 and listing["data"][0]["type"] == "model" and listing["has_more"] is False
        status, _, body = send(srv, "GET", "/v1/models")
        listing = json.loads(body)
        assert listing["object"] == "list" and listing["data"][0]["object"] == "model"
    with mock.start_server(record_dir=tmp_path / "rec2", count_tokens_status=400) as srv:
        status, _, body = post_json(srv, "/v1/messages/count_tokens", ANTHROPIC_REQUEST)
        assert status == 400 and json.loads(body)["type"] == "error"
        assert post_json(srv, "/v1/messages", ANTHROPIC_REQUEST)[0] == 200  # only count_tokens is affected


@pytest.mark.parametrize("path, family", [("/v1/messages", "anthropic"), ("/v1/responses", "openai")])
def test_fr_proof_04_mock_malformed_json_is_recorded_without_its_text_and_answered_400(server, path: str, family: str) -> None:
    for raw in (b'{"model": "x", ', b"[1, 2]", b'"just a string"', b"\xff\xfe not json", b"null"):
        status, _, body = send(server, "POST", path, raw, JSON)
        assert status == 400
        payload = json.loads(body)
        assert ("type" in payload and payload["type"] == "error") if family == "anthropic" else "message" in payload["error"]
    rows = recorded(server)
    assert [r["body_status"] for r in rows] == ["malformed-json"] * 5 and all(r["body"] is None for r in rows)
    assert rows[0]["body_size"] == 15 and rows[0]["body_sha256"] == hashlib.sha256(b'{"model": "x", ').hexdigest()
    assert send(server, "POST", path, json.dumps(RESPONSES_REQUEST if family == "openai" else ANTHROPIC_REQUEST).encode(), JSON)[0] == 200


def test_fr_proof_04_mock_still_answers_when_it_cannot_decode_the_body(server, monkeypatch) -> None:
    monkeypatch.setattr(mock, "_load_zstd", lambda: None)  # as on a Python without a zstd module
    status, _, body = send(server, "POST", "/v1/responses", b"opaque-frame", {**JSON, "Content-Encoding": "zstd"})
    assert status == 200 and json.loads(body)["object"] == "response"  # the session goes on
    row = recorded(server)[-1]
    assert row["body"] is None and row["body_status"] == "zstd-unavailable" and row["content_encoding"] == "zstd"
    assert row["body_raw_b64"] == "b3BhcXVlLWZyYW1l"
    status, _, body = send(server, "POST", "/v1/messages", gzip.compress(json.dumps(ANTHROPIC_REQUEST).encode()),
                           {**JSON, "Content-Encoding": "gzip"})
    assert status == 200 and json.loads(recorded(server)[-1]["body"]) == ANTHROPIC_REQUEST


def test_fr_proof_04_mock_decodes_a_zstd_body_when_a_module_is_importable(server) -> None:
    zstandard = pytest.importorskip("zstandard")
    raw = zstandard.ZstdCompressor().compress(json.dumps(RESPONSES_REQUEST).encode())
    status, _, _ = send(server, "POST", "/v1/responses", raw, {**JSON, "Content-Encoding": "zstd"})
    row = recorded(server)[-1]
    assert status == 200 and row["body_status"] == "ok" and json.loads(row["body"]) == RESPONSES_REQUEST


# --- the command line ------------------------------------------------------------------------------------------


MOCK_PY = str(Path(mock.__file__).resolve())


def test_fr_proof_04_mock_cli_refuses_a_non_loopback_address(tmp_path: Path, capsys) -> None:
    assert mock.main(["--host", "0.0.0.0", "--record", str(tmp_path / "rec")]) == 2
    err = capsys.readouterr().err
    assert "loopback" in err and not (tmp_path / "rec").exists()


def test_fr_proof_04_mock_cli_rejects_a_bad_script(tmp_path: Path, capsys) -> None:
    (tmp_path / "s.json").write_text('[{"surprise": 1}]')
    assert mock.main(["--script", str(tmp_path / "s.json"), "--record", str(tmp_path / "rec")]) == 2
    assert "turn 0" in capsys.readouterr().err


def test_fr_proof_04_mock_shuts_down_cleanly_on_sigint(tmp_path: Path) -> None:
    script = tmp_path / "script.json"
    script.write_text(json.dumps([{"text": "scripted hello"}]))
    proc = subprocess.Popen([sys.executable, MOCK_PY, "--port", "0", "--record", str(tmp_path / "rec"),
                             "--script", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        info = json.loads(proc.stdout.readline())
        host, port = info["listening"].removeprefix("http://").split(":")
        assert host == "127.0.0.1"
        conn = http.client.HTTPConnection(host, int(port), timeout=10)
        conn.request("POST", "/v1/messages", json.dumps({**ANTHROPIC_REQUEST, "tools": READ_TOOL}), JSON)
        reply = json.loads(conn.getresponse().read())
        conn.close()
        assert reply["content"][0]["text"] == "scripted hello"
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0 and "Traceback" not in err and err == ""
    rows = [json.loads(line) for line in (tmp_path / "rec" / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["route"] == "anthropic.messages"


# --- reply headers (G2 verify item T9: does the agent send back a header the server set?) ------------------------------

def test_fr_proof_04_mock_sends_the_configured_reply_headers_on_json_and_sse(tmp_path: Path) -> None:
    with mock.start_server(record_dir=tmp_path / "rec", reply_headers={"x-codex-turn-state": "opaque-state-1"}) as srv:
        _, hdrs, _ = post_json(srv, "/v1/responses", RESPONSES_REQUEST)
        assert hdrs["x-codex-turn-state"] == "opaque-state-1"
        _, hdrs, _ = post_json(srv, "/v1/responses", {**RESPONSES_REQUEST, "stream": True})
        assert hdrs["x-codex-turn-state"] == "opaque-state-1" and hdrs["content-type"].startswith("text/event-stream")
        _, hdrs, _ = send(srv, "GET", "/nowhere")
        assert hdrs["x-codex-turn-state"] == "opaque-state-1"


@pytest.mark.parametrize("name, value", [("bad name", "v"), ("x-a", "line\r\nx-evil: 1"), ("", "v"), ("x-a", "v\n")])
def test_fr_proof_04_mock_refuses_a_reply_header_that_could_split_the_response(name: str, value: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        mock.start_server(record_dir=tmp_path / "rec", reply_headers={name: value})


def test_fr_proof_04_mock_cli_takes_reply_headers(tmp_path: Path, capsys) -> None:
    assert mock.main(["--reply-header", "no-equals-sign", "--record", str(tmp_path / "rec")]) == 2
    assert "NAME=VALUE" in capsys.readouterr().err
