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
    assert len(texts) == (1 if turn.get("text") else 0) and len(uses) == len(calls)
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
