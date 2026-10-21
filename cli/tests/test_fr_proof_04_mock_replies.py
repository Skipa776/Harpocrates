"""FR-PROOF-04: the mock provider's replies have the shape each provider documents, as JSON and as SSE.

The checks below follow the Anthropic Messages API (message object, streaming events) and the OpenAI Responses API
(response object, response.* streaming events). No SDK is needed: the shapes are spelled out here.
"""
from __future__ import annotations

import json

import pytest

from bench import mock_provider as mock

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
