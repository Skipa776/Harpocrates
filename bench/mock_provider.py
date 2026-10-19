#!/usr/bin/env python3
"""A mock LLM provider that records every request an agent sends it (FR-PROOF-04, G2).

    python bench/mock_provider.py [--port N] [--record DIR] [--script FILE]

Stdlib only. It binds a loopback address and makes no outbound connection. It answers the routes Claude Code
(Anthropic Messages) and Codex (OpenAI Responses) call, as JSON or as SSE, so an agent can finish a turn without
a real model, and it can follow a script that asks the agent to run tools. Each request becomes one line of
RECORD_DIR/requests.jsonl: time, method, path, headers, sizes, the SHA-256 of the raw body and the decompressed
body.

Credentials are never written. A header whose name holds auth, key, token, cookie, secret or session, and
chatgpt-account-id, is stored as "<present>". Bodies are stored, because the harness counts canaries in them, and
a live session also sends the agent's own system prompt and memory files. So recordings go to
bench/results/mock/<run>/, which is gitignored and never committed.

Bad input never crashes it. A malformed, oversized (over 64 MiB) or wrongly encoded body is recorded and answered
400. It shuts down cleanly on SIGINT.
"""
from __future__ import annotations

import importlib
import io
import json
import urllib.parse
import zlib
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional

PRESENT = "<present>"
MAX_BODY_BYTES = 64 * 1024 * 1024
CREDENTIAL_WORDS = ("auth", "key", "token", "cookie", "secret", "session")
CREDENTIAL_NAMES = ("chatgpt-account-id",)
DEFAULT_TEXT = "Mock reply."
CREATED_AT = 1760000000  # a fixed timestamp keeps replies reproducible


def is_credential_header(name: str) -> bool:
    """
    Description: Decide whether a header (or query parameter) name may carry a credential, so its value is never stored.
    Parameters:
        name: the header or query parameter name, in any case.
    Output: True when the lower-cased name contains auth, key, token, cookie, secret or session, or is
        chatgpt-account-id; False otherwise. No side effects.
    """
    lowered = name.lower()
    return lowered in CREDENTIAL_NAMES or any(word in lowered for word in CREDENTIAL_WORDS)


def redact_headers(headers: Iterable[tuple[str, str]]) -> list[list[str]]:
    """
    Description: Turn the request headers into the rows the recording stores, with every credential value replaced.
    Parameters:
        headers: (name, value) pairs in the order they were received, duplicates included.
    Output: A list of [name, value] rows in the same order; a credential header's value becomes "<present>". The
        name keeps the case it was sent in. No side effects.
    """
    return [[name, PRESENT if is_credential_header(name) else value] for name, value in headers]


def redact_target(target: str) -> str:
    """
    Description: Replace the value of every credential-named query parameter in a request target.
    Parameters:
        target: the request line's path with its query string, such as "/v1/messages?beta=true".
    Output: The target with each non-empty value of a credential-named parameter set to "<present>" (URL-encoded);
        other parameters keep their order and values. A target without a query string is returned unchanged.
    """
    path, sep, query = target.partition("?")
    if not sep:
        return target
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    kept = [(k, PRESENT if v and is_credential_header(k) else v) for k, v in pairs]
    return path + "?" + urllib.parse.urlencode(kept, quote_via=urllib.parse.quote)


class Decoded(NamedTuple):
    """
    Description: The outcome of decoding one request body for the recording.
    Parameters:
        body: the decompressed bytes, or None when they could not or should not be produced.
        status: ok, too-large, bad-encoding, unsupported-encoding or zstd-unavailable.
    Output: An immutable pair; body is None for every status except ok.
    """

    body: Optional[bytes]
    status: str


def _load_zstd() -> Optional[Callable[[bytes, int], bytes]]:
    """
    Description: Find a zstd decompressor in the standard library (Python 3.14) or the optional zstandard package.
    Parameters: None
    Output: A function (data, cap) -> bytes that returns at most cap + 1 bytes of output and raises on a corrupt frame,
        or None when no zstd module can be imported. Imports a module; no other side effects.
    """
    try:
        stdlib = importlib.import_module("compression.zstd")
        return lambda data, cap: stdlib.ZstdDecompressor().decompress(data, max_length=cap + 1)
    except ImportError:
        pass
    try:
        third = importlib.import_module("zstandard")
        return lambda data, cap: third.ZstdDecompressor().stream_reader(io.BytesIO(data)).read(cap + 1)
    except ImportError:
        return None


def decode_body(raw: bytes, content_encoding: Optional[str], cap: int = MAX_BODY_BYTES) -> Decoded:
    """
    Description: Decompress a request body so the recording holds the text the agent sent, never expanding past the cap.
    Parameters:
        raw: the body bytes exactly as received.
        content_encoding: the Content-Encoding header value, or None.
        cap: the largest decompressed size accepted, in bytes.
    Output: Decoded(body, "ok") for an identity or gzip body (and zstd when a module is importable) within the cap;
        Decoded(None, "too-large") over the cap (a compression bomb is never fully expanded);
        Decoded(None, "bad-encoding") for a corrupt or truncated stream; Decoded(None, "unsupported-encoding") for
        any other value, a list included; Decoded(None, "zstd-unavailable") for zstd with no module. No side effects.
    """
    encoding = (content_encoding or "").strip().lower()
    if encoding in ("", "identity"):
        return Decoded(raw, "ok") if len(raw) <= cap else Decoded(None, "too-large")
    if encoding in ("gzip", "x-gzip"):
        try:
            stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
            out = stream.decompress(raw, cap + 1)
            complete = stream.eof
        except zlib.error:
            return Decoded(None, "bad-encoding")
        if len(out) > cap:
            return Decoded(None, "too-large")
        return Decoded(out, "ok") if complete else Decoded(None, "bad-encoding")
    if encoding == "zstd":
        decompress = _load_zstd()
        if decompress is None:
            return Decoded(None, "zstd-unavailable")
        try:
            out = decompress(raw, cap)
        except Exception:  # noqa: BLE001 - any decoder failure is a bad stream, never a crash
            return Decoded(None, "bad-encoding")
        return Decoded(out, "ok") if len(out) <= cap else Decoded(None, "too-large")
    return Decoded(None, "unsupported-encoding")


def _output_tokens(turn: dict) -> int:
    """
    Description: Estimate the output token count of a scripted turn, so usage fields hold plausible numbers.
    Parameters:
        turn: the scripted turn (text and tool_calls).
    Output: A positive integer: a quarter of the text length plus a fixed amount per tool call.
    """
    return max(1, len(turn.get("text") or DEFAULT_TEXT) // 4 + 12 * len(turn.get("tool_calls", [])))


def anthropic_message(turn: dict, model: str, serial: int, input_tokens: int) -> dict:
    """
    Description: Build an Anthropic Messages API reply (the non-streaming message object) for one scripted turn.
    Parameters:
        turn: {"text": str, "tool_calls": [{"name": str, "input": dict}]}, every key optional; an empty turn is a
            short text reply.
        model: the model name to echo, taken from the request.
        serial: a counter that makes the ids unique per reply and identical when replayed.
        input_tokens: the figure to report as the request's input tokens.
    Output: A message dict: text block first, then a tool_use block per call, stop_reason tool_use when there are
        calls and end_turn otherwise. No side effects.
    """
    calls = turn.get("tool_calls", [])
    content: list[dict] = []
    if turn.get("text") or not calls:
        content.append({"type": "text", "text": turn.get("text") or DEFAULT_TEXT})
    for index, call in enumerate(calls):
        content.append({"type": "tool_use", "id": f"toolu_mock{serial:08d}{index:02d}", "name": call["name"],
                        "input": call.get("input", {})})
    return {"id": f"msg_mock{serial:08d}", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                      "output_tokens": _output_tokens(turn)}}


def anthropic_events(message: dict) -> list[tuple[str, dict]]:
    """
    Description: Expand an Anthropic message into its streaming events, in the documented order.
    Parameters:
        message: a message dict from anthropic_message.
    Output: (event name, data) pairs: message_start, then for each content block content_block_start, content_block_delta
        (text_delta, or input_json_delta in two pieces for a tool call) and content_block_stop, with one ping after the
        first block starts, then message_delta and message_stop. A tool_use block starts with an empty input.
    """
    usage = message["usage"]
    start = dict(message, content=[], stop_reason=None, usage=dict(usage, output_tokens=1))
    events: list[tuple[str, dict]] = [("message_start", {"type": "message_start", "message": start})]
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                   "content_block": {"type": "text", "text": ""}}))
            deltas = [{"type": "text_delta", "text": block["text"]}]
        else:
            events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                   "content_block": dict(block, input={})}))
            raw = json.dumps(block["input"], separators=(",", ":"))
            cut = max(1, len(raw) // 2)
            deltas = [{"type": "input_json_delta", "partial_json": raw[:cut]},
                      {"type": "input_json_delta", "partial_json": raw[cut:]}]
        if index == 0:
            events.append(("ping", {"type": "ping"}))
        events += [("content_block_delta", {"type": "content_block_delta", "index": index, "delta": d}) for d in deltas]
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                                     "usage": {"input_tokens": usage["input_tokens"],
                                               "output_tokens": usage["output_tokens"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return events


def responses_object(turn: dict, model: str, serial: int, input_tokens: int) -> dict:
    """
    Description: Build an OpenAI Responses API reply (the response object) for one scripted turn.
    Parameters:
        turn: {"text": str, "tool_calls": [{"name": str, "input": dict}]}, every key optional; an empty turn is a
            short text reply.
        model: the model name to echo, taken from the request.
        serial: a counter that makes the ids unique per reply and identical when replayed.
        input_tokens: the figure to report as the request's input tokens.
    Output: A response dict with status completed: a message item (output_text) first when there is text or no call,
        then a function_call item per call, whose arguments are a JSON string. No side effects.
    """
    calls = turn.get("tool_calls", [])
    output: list[dict] = []
    if turn.get("text") or not calls:
        output.append({"type": "message", "id": f"msg_mock{serial:08d}", "status": "completed", "role": "assistant",
                       "content": [{"type": "output_text", "text": turn.get("text") or DEFAULT_TEXT,
                                    "annotations": []}]})
    for index, call in enumerate(calls):
        output.append({"type": "function_call", "id": f"fc_mock{serial:08d}{index:02d}",
                       "call_id": f"call_mock{serial:08d}{index:02d}", "name": call["name"],
                       "arguments": json.dumps(call.get("input", {}), separators=(",", ":")), "status": "completed"})
    out_tokens = _output_tokens(turn)
    return {"id": f"resp_mock{serial:08d}", "object": "response", "created_at": CREATED_AT, "status": "completed",
            "error": None, "incomplete_details": None, "instructions": None, "max_output_tokens": None,
            "model": model, "output": output, "parallel_tool_calls": True, "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None}, "store": False, "temperature": 1.0,
            "text": {"format": {"type": "text"}}, "tool_choice": "auto", "tools": [], "top_p": 1.0,
            "truncation": "disabled", "metadata": {}, "user": None,
            "usage": {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": out_tokens, "output_tokens_details": {"reasoning_tokens": 0},
                      "total_tokens": input_tokens + out_tokens}}


def responses_events(response: dict) -> list[tuple[str, dict]]:
    """
    Description: Expand an OpenAI response into its streaming events, in the documented order.
    Parameters:
        response: a response dict from responses_object.
    Output: (event name, data) pairs with a running sequence_number: response.created and response.in_progress (status
        in_progress, no output), then per output item output_item.added, its content events (content_part.added,
        output_text.delta, output_text.done, content_part.done for a message; function_call_arguments.delta and .done for
        a call) and output_item.done, then response.completed carrying the full response.
    """
    events: list[tuple[str, dict]] = []

    def emit(kind: str, /, **fields: object) -> None:
        """
        Description: Append one event with the next sequence number.
        Parameters:
            kind: the event type, which is also its SSE event name.
            **fields: the remaining members of the event's data object.
        Output: None; extends the enclosing events list.
        """
        events.append((kind, {"type": kind, "sequence_number": len(events), **fields}))

    skeleton = dict(response, status="in_progress", output=[], usage=None)
    emit("response.created", response=skeleton)
    emit("response.in_progress", response=skeleton)
    for index, item in enumerate(response["output"]):
        if item["type"] == "message":
            part = item["content"][0]
            emit("response.output_item.added", output_index=index, item=dict(item, status="in_progress", content=[]))
            emit("response.content_part.added", item_id=item["id"], output_index=index, content_index=0,
                 part=dict(part, text=""))
            emit("response.output_text.delta", item_id=item["id"], output_index=index, content_index=0,
                 delta=part["text"], logprobs=[])
            emit("response.output_text.done", item_id=item["id"], output_index=index, content_index=0,
                 text=part["text"], logprobs=[])
            emit("response.content_part.done", item_id=item["id"], output_index=index, content_index=0, part=part)
        else:
            emit("response.output_item.added", output_index=index, item=dict(item, status="in_progress", arguments=""))
            emit("response.function_call_arguments.delta", item_id=item["id"], output_index=index,
                 delta=item["arguments"])
            emit("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                 name=item["name"], arguments=item["arguments"])
        emit("response.output_item.done", output_index=index, item=item)
    emit("response.completed", response=response)
    return events


def sse_bytes(events: Iterable[tuple[str, dict]]) -> bytes:
    """
    Description: Encode streaming events in the SSE wire format both providers use.
    Parameters:
        events: (event name, data) pairs.
    Output: Bytes holding "event: NAME" and "data: JSON" lines, each event ended by a blank line. No side effects.
    """
    return b"".join(f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode() for name, data in events)


_TURN_KEYS = {"text", "tool_calls", "error"}


def _check_turn(index: int, turn: object) -> None:
    """
    Description: Validate one scripted turn.
    Parameters:
        index: the turn's position in the script, for the error message.
        turn: the parsed JSON value.
    Output: None. Raises ValueError naming the turn when it is not an object with only text (str), tool_calls (a list of
        {"name": str, "input": object}) and error ({"status": 400 to 599, "message": str}).
    """
    where = f"turn {index}"
    if not isinstance(turn, dict) or not set(turn) <= _TURN_KEYS:
        raise ValueError(f"{where}: must be an object with only {sorted(_TURN_KEYS)}")
    if "text" in turn and not isinstance(turn["text"], str):
        raise ValueError(f"{where}: text must be a string")
    calls = turn.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError(f"{where}: tool_calls must be a list")
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("input", {}), dict):
            raise ValueError(f"{where}: each tool call needs a name and an optional input object")
    if "error" in turn:
        err = turn["error"]
        ok = (isinstance(err, dict) and isinstance(err.get("status"), int) and not isinstance(err.get("status"), bool)
              and 400 <= err["status"] <= 599 and isinstance(err.get("message"), str))
        if not ok:
            raise ValueError(f"{where}: error needs a status from 400 to 599 and a message")


def load_script(path: Path) -> list[dict]:
    """
    Description: Read a scripted-replies file: a JSON list of turns, answered one per model call.
    Parameters:
        path: the file to read.
    Output: The list of turns. Raises ValueError when the file cannot be read, is not JSON, is not a list, or a turn is
        malformed (see _check_turn).
    """
    try:
        turns = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the script {path}: {exc}") from exc
    if not isinstance(turns, list):
        raise ValueError("the script must be a JSON list of turns")
    for index, turn in enumerate(turns):
        _check_turn(index, turn)
    return turns


def count_tool_results(body: object, fmt: str) -> int:
    """
    Description: Count the tool results already in a request, which says how far through the script the agent is (the
        history is re-sent on every call, so the count survives retries and side requests).
    Parameters:
        body: the parsed request body.
        fmt: "anthropic" (tool_result blocks in messages) or "responses" (input items whose type ends in _call_output).
    Output: The number of tool results; 0 for anything that is not the expected shape.
    """
    if not isinstance(body, dict):
        return 0
    if fmt == "anthropic":
        messages = body.get("messages")
        blocks = [b for m in messages if isinstance(m, dict) and isinstance(m.get("content"), list)
                  for b in m["content"]] if isinstance(messages, list) else []
        return sum(1 for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result")
    items = body.get("input")
    return sum(1 for i in items if isinstance(i, dict) and str(i.get("type", "")).endswith("_call_output")) \
        if isinstance(items, list) else 0


def select_turn(turns: list[dict], results: int) -> dict:
    """
    Description: Pick the scripted turn to answer with, given how many tool results the request already holds.
    Parameters:
        turns: the script.
        results: count_tool_results of the request.
    Output: The first turn whose tool calls start at or after `results` calls (a turn with two calls takes two results);
        the last turn when the request is past the end; {} (a short text reply) for an empty script. No side effects.
    """
    start = 0
    for turn in turns:
        if start >= results:
            return turn
        start += len(turn.get("tool_calls", []))
    return turns[-1] if turns else {}


def turn_for(body: object, fmt: str, turns: list[dict]) -> dict:
    """
    Description: Choose the reply for one model request: the scripted turn, or a plain text turn when the request offers no
        tools (a title or summary side request that could not use a tool call).
    Parameters:
        body: the parsed request body.
        fmt: "anthropic" or "responses".
        turns: the script.
    Output: A turn dict; {} when body is not an object or its tools list is missing or empty.
    """
    if not isinstance(body, dict) or not body.get("tools"):
        return {}
    return select_turn(turns, count_tool_results(body, fmt))
