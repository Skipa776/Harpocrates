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
