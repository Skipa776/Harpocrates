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

import argparse
import base64
import hashlib
import importlib
import io
import ipaddress
import json
import os
import re
import signal
import socket
import sys
import threading
import urllib.parse
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        turn: {"text": str, "thinking": str, "redacted_thinking": str, "tool_calls": [{"name": str, "input": dict}]},
            every key optional; an empty turn is a short text reply.
        model: the model name to echo, taken from the request.
        serial: a counter that makes the ids unique per reply and identical when replayed.
        input_tokens: the figure to report as the request's input tokens.
    Output: A message dict: a thinking block (with a signature) and a redacted_thinking block when the turn has them, then
        the text block, then a tool_use block per call; stop_reason tool_use when there are calls and end_turn otherwise.
        No side effects.
    """
    calls = turn.get("tool_calls", [])
    content: list[dict] = []
    if turn.get("thinking"):
        content.append({"type": "thinking", "thinking": turn["thinking"], "signature": f"mocksig{serial:08d}"})
    if turn.get("redacted_thinking"):
        content.append({"type": "redacted_thinking", "data": turn["redacted_thinking"]})
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
        (text_delta; thinking_delta then signature_delta for thinking; input_json_delta in two pieces for a tool call;
        none for redacted_thinking, which starts complete) and content_block_stop, with one ping after the first block
        starts, then message_delta and message_stop. A tool_use block starts with an empty input.
    """
    usage = message["usage"]
    start = dict(message, content=[], stop_reason=None, usage=dict(usage, output_tokens=1))
    events: list[tuple[str, dict]] = [("message_start", {"type": "message_start", "message": start})]
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                   "content_block": {"type": "text", "text": ""}}))
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif block["type"] == "thinking":
            events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                   "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
            deltas = [{"type": "thinking_delta", "thinking": block["thinking"]},
                      {"type": "signature_delta", "signature": block["signature"]}]
        elif block["type"] == "redacted_thinking":
            events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": block}))
            deltas = []
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
        turn: {"text": str, "tool_calls": [{"name": str, "input": dict, "namespace": str}]}, every key optional (namespace
            names the tool group an MCP tool sits in, as Codex offers them); an empty turn is a short text reply.
        model: the model name to echo, taken from the request.
        serial: a counter that makes the ids unique per reply and identical when replayed.
        input_tokens: the figure to report as the request's input tokens.
    Output: A response dict with status completed: a reasoning item for thinking (a summary and encrypted_content) and one
        for redacted_thinking (no summary), then a message item (output_text) when there is text or no call, then a
        function_call item per call, whose arguments are a JSON string. No side effects.
    """
    calls = turn.get("tool_calls", [])
    output: list[dict] = []
    if turn.get("thinking"):
        output.append({"type": "reasoning", "id": f"rs_mock{serial:08d}", "encrypted_content": f"mockcipher{serial:08d}",
                       "summary": [{"type": "summary_text", "text": turn["thinking"]}]})
    if turn.get("redacted_thinking"):
        output.append({"type": "reasoning", "id": f"rs_mock{serial:08d}x", "summary": [],
                       "encrypted_content": turn["redacted_thinking"]})
    if turn.get("text") or not calls:
        output.append({"type": "message", "id": f"msg_mock{serial:08d}", "status": "completed", "role": "assistant",
                       "content": [{"type": "output_text", "text": turn.get("text") or DEFAULT_TEXT,
                                    "annotations": []}]})
    for index, call in enumerate(calls):
        output.append({"type": "function_call", "id": f"fc_mock{serial:08d}{index:02d}",
                       "call_id": f"call_mock{serial:08d}{index:02d}", "name": call["name"],
                       **({"namespace": call["namespace"]} if "namespace" in call else {}),
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
        output_text.delta, output_text.done, content_part.done for a message; reasoning_summary_part.added,
        reasoning_summary_text.delta and .done, reasoning_summary_part.done for a reasoning item with a summary;
        function_call_arguments.delta and .done for a call) and output_item.done, then response.completed carrying the
        full response.
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
        if item["type"] == "reasoning":
            emit("response.output_item.added", output_index=index, item=dict(item, summary=[]))
            for part_index, part in enumerate(item["summary"]):
                emit("response.reasoning_summary_part.added", item_id=item["id"], output_index=index,
                     summary_index=part_index, part=dict(part, text=""))
                emit("response.reasoning_summary_text.delta", item_id=item["id"], output_index=index,
                     summary_index=part_index, delta=part["text"])
                emit("response.reasoning_summary_text.done", item_id=item["id"], output_index=index,
                     summary_index=part_index, text=part["text"])
                emit("response.reasoning_summary_part.done", item_id=item["id"], output_index=index,
                     summary_index=part_index, part=part)
        elif item["type"] == "message":
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


_TURN_KEYS = {"text", "thinking", "redacted_thinking", "tool_calls", "error"}


def _check_turn(index: int, turn: object) -> None:
    """
    Description: Validate one scripted turn.
    Parameters:
        index: the turn's position in the script, for the error message.
        turn: the parsed JSON value.
    Output: None. Raises ValueError naming the turn when it is not an object with only text, thinking and redacted_thinking (str), tool_calls (a list of
        {"name": str, "input": object, "namespace": str}) and error ({"status": 400 to 599, "message": str}).
    """
    where = f"turn {index}"
    if not isinstance(turn, dict) or not set(turn) <= _TURN_KEYS:
        raise ValueError(f"{where}: must be an object with only {sorted(_TURN_KEYS)}")
    for key in ("text", "thinking", "redacted_thinking"):
        if key in turn and not isinstance(turn[key], str):
            raise ValueError(f"{where}: {key} must be a string")
    calls = turn.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError(f"{where}: tool_calls must be a list")
    for call in calls:
        if (not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("input", {}), dict)
                or not isinstance(call.get("namespace", ""), str)):
            raise ValueError(f"{where}: each tool call needs a name, an optional input object and an optional namespace")
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


# --- the server ------------------------------------------------------------------------------------------------

READ_CHUNK = 1 << 20
FAILED_BODY = ("too-large", "bad-encoding", "bad-length", "bad-chunking", "truncated")  # answered 400


def ensure_loopback(host: str) -> str:
    """
    Description: Refuse any bind address that is not on the loopback interface, so the mock is never reachable from
        another machine.
    Parameters:
        host: the address to bind: "localhost" or an IP literal.
    Output: The address to bind ("localhost" becomes 127.0.0.1). Raises ValueError, naming loopback, for an empty host,
        a wildcard (0.0.0.0, ::), a non-loopback address or a name that is not an address. No name lookup is made.
    """
    if host == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is None or not address.is_loopback:
        raise ValueError(f"the mock provider binds a loopback address only (127.0.0.1 or ::1), not {host!r}")
    return host


_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")


def check_reply_headers(headers: dict[str, str]) -> tuple:
    """
    Description: Validate extra response headers, so a value can never split the response.
    Parameters:
        headers: {name: value} to add to every response.
    Output: A tuple of (name, value) pairs. Raises ValueError for an empty or non-token name, or a value with CR, LF or NUL.
    """
    for name, value in headers.items():
        if not _HEADER_NAME.match(name) or any(ch in value for ch in "\r\n\0"):
            raise ValueError(f"reply header {name!r} is not a valid name and single-line value")
    return tuple(headers.items())


@dataclass(frozen=True)
class Config:
    """
    Description: The settings a running mock was started with.
    Parameters:
        turns: the scripted replies, one per model call (see load_script); empty means a short text reply every time.
        max_body: the largest request body accepted, raw or decompressed, in bytes.
        count_tokens_status: when set, /v1/messages/count_tokens answers this HTTP error status instead of a count.
        reply_headers: (name, value) pairs added to every response, such as a sticky-routing header the agent should echo.
    Output: An immutable record read by every request handler.
    """

    turns: tuple = ()
    max_body: int = MAX_BODY_BYTES
    count_tokens_status: Optional[int] = None
    reply_headers: tuple = ()


class RawBody(NamedTuple):
    """
    Description: A request body as read off the wire, before any decompression.
    Parameters:
        data: the bytes, or None when the body was over the cap, empty-and-failed or unreadable.
        size: how many bytes arrived.
        sha256: the hex SHA-256 of every byte that arrived (also of bytes not kept).
        status: ok, empty, too-large, bad-length, bad-chunking or truncated.
        chunked: whether the body used chunked transfer encoding.
        content_length: the Content-Length header as an int, or None when absent or invalid.
    Output: An immutable record used to build the recording line.
    """

    data: Optional[bytes]
    size: int
    sha256: str
    status: str
    chunked: bool
    content_length: Optional[int]


class _Reader:
    """
    Description: Accumulate the size, hash and (while under the cap) the bytes of a body as it is read in pieces.
    Parameters:
        cap: the most bytes kept; beyond it the pieces are hashed and counted but dropped.
    Output: An object whose add() takes a piece and whose result() gives (data or None, size, sha256).
    """

    def __init__(self, cap: int) -> None:
        """
        Description: Start an empty accumulation.
        Parameters:
            cap: the most bytes to keep.
        Output: None.
        """
        self.cap, self.size, self.hash, self.kept = cap, 0, hashlib.sha256(), bytearray()

    def add(self, piece: bytes) -> None:
        """
        Description: Count and hash one piece, keeping it only while the total is within the cap.
        Parameters:
            piece: the bytes just read.
        Output: None. Once the total passes the cap, the bytes kept so far are dropped.
        """
        self.hash.update(piece)
        self.size += len(piece)
        if self.size <= self.cap:
            self.kept += piece
        else:
            self.kept = bytearray()

    def result(self) -> tuple[Optional[bytes], int, str]:
        """
        Description: Report what was read.
        Parameters: None
        Output: (bytes or None when over the cap, total size, hex SHA-256).
        """
        return (bytes(self.kept) if self.size <= self.cap else None), self.size, self.hash.hexdigest()


def _read_fixed(rfile: io.BufferedIOBase, length: int, cap: int) -> RawBody:
    """
    Description: Read a body of a known length in pieces, hashing all of it and keeping it only within the cap.
    Parameters:
        rfile: the connection's input stream.
        length: the Content-Length value.
        cap: the most bytes to keep.
    Output: A RawBody with status ok, empty (length 0), too-large or truncated (the client sent fewer bytes).
    """
    reader = _Reader(cap)
    remaining = length
    while remaining > 0:
        piece = rfile.read(min(remaining, READ_CHUNK))
        if not piece:
            data, size, digest = reader.result()
            return RawBody(None, size, digest, "truncated", False, length)
        reader.add(piece)
        remaining -= len(piece)
    data, size, digest = reader.result()
    status = "empty" if length == 0 else "ok" if data is not None else "too-large"
    return RawBody(data, size, digest, status, False, length)


def _read_chunked(rfile: io.BufferedIOBase, cap: int) -> RawBody:
    """
    Description: Read a chunked body to its end, hashing all of it and keeping it only within the cap.
    Parameters:
        rfile: the connection's input stream.
        cap: the most bytes to keep.
    Output: A RawBody with status ok, empty (no data chunks), too-large, bad-chunking (a malformed size line or missing
        CRLF) or truncated (the stream ended early); chunked is True.
    """
    reader = _Reader(cap)

    def finish(status: str) -> RawBody:
        """
        Description: Build the result for the chunks read so far.
        Parameters:
            status: the status to report, replaced by too-large or empty when that applies to an ok read.
        Output: A RawBody for the enclosing read.
        """
        data, size, digest = reader.result()
        if status == "ok":
            status = "empty" if size == 0 else "ok" if data is not None else "too-large"
        return RawBody(data if status == "ok" else None, size, digest, status, True, None)

    while True:
        line = rfile.readline(65537)
        if not line.endswith(b"\n"):
            return finish("truncated" if not line else "bad-chunking")
        try:
            remaining = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            return finish("bad-chunking")
        if remaining < 0:
            return finish("bad-chunking")
        if remaining == 0:
            while True:  # trailers, up to the blank line
                trailer = rfile.readline(65537)
                if trailer in (b"\r\n", b"\n"):
                    return finish("ok")
                if not trailer.endswith(b"\n"):
                    return finish("truncated" if not trailer else "bad-chunking")
        while remaining > 0:
            piece = rfile.read(min(remaining, READ_CHUNK))
            if not piece:
                return finish("truncated")
            reader.add(piece)
            remaining -= len(piece)
        if rfile.read(2) != b"\r\n":
            return finish("bad-chunking")


def error_body(family: str, kind: str, message: str) -> dict:
    """
    Description: Build an error body in the shape a provider uses.
    Parameters:
        family: "anthropic" ({"type": "error", "error": {...}}) or anything else for the OpenAI shape ({"error": {...}}).
        kind: the error type, such as invalid_request_error or not_found_error.
        message: the human-readable message.
    Output: The JSON-serializable error body. No side effects.
    """
    if family == "anthropic":
        return {"type": "error", "error": {"type": kind, "message": message}, "request_id": "req_mock"}
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}


ROUTES: dict[tuple[str, str], str] = {
    ("POST", "/v1/messages"): "anthropic.messages",
    ("POST", "/v1/messages/count_tokens"): "anthropic.count_tokens",
    ("POST", "/v1/responses"): "openai.responses",
    ("GET", "/v1/models"): "models",
}  # (method, path) -> route name
JSON_ROUTES = ("anthropic.messages", "anthropic.count_tokens", "openai.responses")  # routes whose body must be a JSON object
ERROR_KINDS = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
               413: "request_too_large", 429: "rate_limit_error", 529: "overloaded_error"}


def classify(method: str, target: str, headers: dict[str, str]) -> str:
    """
    Description: Name the route a request is for.
    Parameters:
        method: the HTTP method.
        target: the request target with its query string.
        headers: the request headers, keys lower-cased.
    Output: websocket-upgrade for a request that asks to upgrade to a WebSocket, otherwise the route name from ROUTES,
        otherwise "unknown".
    """
    if "websocket" in headers.get("upgrade", "").lower():
        return "websocket-upgrade"
    return ROUTES.get((method, target.partition("?")[0]), "unknown")


class Plan(NamedTuple):
    """
    Description: How the mock will answer one request.
    Parameters:
        status: the HTTP status.
        payload: the JSON body, or None when events carry the answer.
        events: the SSE events for a streamed reply, or None.
        stream: whether the request asked for a stream (None on routes that do not stream).
        reply_turn: the index of the scripted turn used, or None.
    Output: An immutable record built by plan_reply and carried out by the handler.
    """

    status: int
    payload: Optional[dict]
    events: Optional[list]
    stream: Optional[bool]
    reply_turn: Optional[int]


def _models_payload(headers: dict[str, str]) -> dict:
    """
    Description: Build the fixed model list in the shape of whichever provider asked.
    Parameters:
        headers: the request headers, keys lower-cased; an anthropic-version header selects the Anthropic shape.
    Output: The JSON-serializable model list.
    """
    if "anthropic-version" in headers:
        return {"data": [{"type": "model", "id": "claude-mock", "display_name": "Claude Mock",
                          "created_at": "2025-01-01T00:00:00Z"}],
                "has_more": False, "first_id": "claude-mock", "last_id": "claude-mock"}
    return {"object": "list", "data": [{"id": "gpt-mock", "object": "model", "created": CREATED_AT, "owned_by": "mock"}]}


def plan_reply(route: str, body_status: str, parsed: Optional[dict], headers: dict[str, str], config: Config,
               serial: int, input_tokens: int) -> Plan:
    """
    Description: Decide the answer to a request from its route and what was read of its body.
    Parameters:
        route: the name from classify().
        body_status: how reading the body went (a failed status is answered 400).
        parsed: the request body as a JSON object, or None when it is not one.
        headers: the request headers, keys lower-cased.
        config: the settings the mock runs with.
        serial: the counter that makes reply ids unique.
        input_tokens: the figure to report as the request's input tokens.
    Output: A Plan: 400 for a failed or malformed body; 426 for a WebSocket upgrade; the model list for /v1/models;
        a fixed count (or the configured error) for count_tokens; for the two model routes, the scripted turn (or a text
        reply) as JSON, or as SSE events when the request asked for a stream, or the scripted HTTP error; 404 otherwise.
    """
    family = "anthropic" if route.startswith("anthropic") else "openai"
    if body_status in FAILED_BODY or body_status == "malformed-json":
        return Plan(400, error_body(family, "invalid_request_error", f"request body rejected: {body_status}"), None, None, None)
    if route == "websocket-upgrade":
        return Plan(426, error_body("openai", "upgrade_required", "WebSocket is not supported; use HTTP"), None, None, None)
    if route == "models":
        return Plan(200, _models_payload(headers), None, None, None)
    if route == "anthropic.count_tokens":
        if config.count_tokens_status is not None:
            status = config.count_tokens_status
            return Plan(status, error_body("anthropic", ERROR_KINDS.get(status, "api_error"), "count_tokens refused"),
                        None, None, None)
        return Plan(200, {"input_tokens": 42}, None, None, None)
    if route not in ("anthropic.messages", "openai.responses"):
        return Plan(404, error_body("anthropic", "not_found_error", "no such route in the mock provider"), None, None, None)
    fmt = "anthropic" if route == "anthropic.messages" else "responses"
    turns = list(config.turns)
    turn = turn_for(parsed, fmt, turns)
    index = next((i for i, candidate in enumerate(turns) if candidate is turn), None)
    stream = bool(parsed and parsed.get("stream") is True)
    if "error" in turn:
        err = turn["error"]
        kind = err.get("type") or ERROR_KINDS.get(err["status"], "api_error")
        return Plan(err["status"], error_body(family, kind, err["message"]), None, stream, index)
    model = parsed.get("model") if parsed and isinstance(parsed.get("model"), str) else "mock-model"
    if fmt == "anthropic":
        message = anthropic_message(turn, model, serial, input_tokens)
        return Plan(200, None if stream else message, anthropic_events(message) if stream else None, stream, index)
    response = responses_object(turn, model, serial, input_tokens)
    return Plan(200, None if stream else response, responses_events(response) if stream else None, stream, index)


class _Httpd(ThreadingHTTPServer):
    """
    Description: The HTTP server: a thread per connection, with the recording state the handlers share.
    Parameters:
        address: (host, port) to bind, already checked to be loopback.
        config: the Config the mock runs with.
        record_dir: where requests.jsonl goes, or None to keep the recording in memory only.
    Output: A server whose entries list and optional file hold one dict per request.
    """

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], config: Config, record_dir: Optional[Path]) -> None:
        """
        Description: Bind the socket and open the recording file.
        Parameters:
            address: (host, port) to bind.
            config: the Config the mock runs with.
            record_dir: the folder for requests.jsonl, created here with mode 0700, or None.
        Output: None. Raises OSError when the address cannot be bound.
        """
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, _Handler)
        self.config, self.record_dir = config, record_dir
        self.lock = threading.Lock()
        self.entries: list[dict] = []
        self.serial = 0
        self.record_file = None
        if record_dir is not None:
            Path(record_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(Path(record_dir) / "requests.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self.record_file = os.fdopen(fd, "a", encoding="utf-8")

    def record(self, entry: dict) -> dict:
        """
        Description: Number a request's entry and append it to the recording.
        Parameters:
            entry: the dict describing the request and its answer.
        Output: The entry with its "index" and "time" set. Writes one flushed JSON line when a file is open.
        """
        with self.lock:
            full = {"index": len(self.entries), "time": datetime.now(timezone.utc).isoformat(), **entry}
            self.entries.append(full)
            if self.record_file is not None:
                self.record_file.write(json.dumps(full, separators=(",", ":")) + "\n")
                self.record_file.flush()
        return full

    def next_serial(self) -> int:
        """
        Description: Hand out the next reply counter, which keeps reply ids unique.
        Parameters: None
        Output: 1 for the first call, then 2, 3, and so on. Thread-safe.
        """
        with self.lock:
            self.serial += 1
            return self.serial


class _Handler(BaseHTTPRequestHandler):
    """
    Description: Handle one connection's requests: read the body, record the request, answer it.
    Parameters:
        request: the connected socket (set by the server).
    Output: One recording line and one response per request. Never raises on bad input.
    """

    protocol_version = "HTTP/1.1"
    server_version = "mock-provider"
    disable_nagle_algorithm = True  # headers and body are two writes; Nagle plus delayed ACK would add 40 ms to each reply

    def log_message(self, format: str, *args: object) -> None:
        """
        Description: Silence the default per-request log line, which would add noise and the request target.
        Parameters:
            format: the log format string (ignored).
            *args: its arguments (ignored).
        Output: None.
        """

    def _body(self) -> RawBody:
        """
        Description: Read this request's body, whichever way it was sent.
        Parameters: None
        Output: A RawBody; chunked transfer encoding is read chunk by chunk, otherwise Content-Length bytes, and a
            request with neither has an empty body. A non-numeric or negative Content-Length is bad-length.
        """
        cap = self.server.config.max_body
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            return _read_chunked(self.rfile, cap)
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return RawBody(b"", 0, hashlib.sha256(b"").hexdigest(), "empty", False, None)
        try:
            length = int(raw_length)
            if length < 0:
                raise ValueError
        except ValueError:
            return RawBody(None, 0, hashlib.sha256(b"").hexdigest(), "bad-length", False, None)
        return _read_fixed(self.rfile, length, cap)

    def _send(self, status: int, payload: Optional[dict] = None, extra: Optional[dict] = None) -> None:
        """
        Description: Send a JSON response (no body for HEAD) and ignore a client that has gone.
        Parameters:
            status: the HTTP status.
            payload: the JSON body, or None for an empty one.
            extra: additional response headers.
        Output: None. Writes to the connection.
        """
        data = b"" if payload is None else json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in {**dict(self.server.config.reply_headers), **(extra or {})}.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except OSError:
            self.close_connection = True

    def _send_sse(self, events: list) -> None:
        """
        Description: Send a 200 response as an SSE stream in chunked transfer encoding, one chunk per event.
        Parameters:
            events: the (event name, data) pairs to send.
        Output: None. Writes to the connection; ignores a client that has gone.
        """
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            for name, value in self.server.config.reply_headers:
                self.send_header(name, value)
            self.end_headers()
            for event in events:
                data = sse_bytes([event])
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            self.close_connection = True

    def _decode(self, raw: RawBody, headers: dict[str, str], route: str) -> tuple[Optional[str], str, Optional[str], Optional[dict]]:
        """
        Description: Turn the bytes read into the text to record and, on a JSON route, the parsed request object.
        Parameters:
            raw: the body as read.
            headers: the request headers, keys lower-cased.
            route: the route name.
        Output: (text to record or None, final body status, base64 of the raw body when it could not be decoded,
            the parsed JSON object or None). A JSON route whose body is not a JSON object has status malformed-json and no
            text recorded: size and hash only.
        """
        if raw.status == "empty":
            return "", "empty", None, None
        if raw.status != "ok":
            return None, raw.status, None, None
        decoded = decode_body(raw.data, headers.get("content-encoding"), self.server.config.max_body)
        if decoded.status in ("unsupported-encoding", "zstd-unavailable"):
            return None, decoded.status, base64.b64encode(raw.data).decode("ascii"), None
        if decoded.status != "ok":
            return None, decoded.status, None, None
        if route in JSON_ROUTES:
            try:
                parsed = json.loads(decoded.body)
            except (ValueError, RecursionError):
                parsed = None
            if not isinstance(parsed, dict):
                return None, "malformed-json", None, None
            return decoded.body.decode("utf-8"), "ok", None, parsed
        try:
            return decoded.body.decode("utf-8"), "ok", None, None
        except UnicodeDecodeError:
            return decoded.body.decode("utf-8", errors="replace"), "not-utf8", None, None

    def _serve(self) -> None:
        """
        Description: Answer any request: read it, record it, reply.
        Parameters: None
        Output: None. Records one line (see the module docstring), then sends the Plan from plan_reply. The line is written
            before the reply goes out, so a client that hangs up still leaves its record.
        """
        headers = {k.lower(): v for k, v in self.headers.items()}
        route = classify(self.command, self.path, headers)
        raw = self._body()
        text, body_status, raw_b64, parsed = self._decode(raw, headers, route)
        plan = plan_reply(route, body_status, parsed, headers, self.server.config, self.server.next_serial(),
                          max(1, raw.size // 4))
        if raw.status in ("bad-length", "bad-chunking", "truncated"):
            self.close_connection = True
        entry = {"method": self.command, "path": redact_target(self.path), "http_version": self.request_version,
                 "route": route, "headers": redact_headers(self.headers.items()),
                 "content_length_header": raw.content_length, "transfer_encoding": "chunked" if raw.chunked else None,
                 "chunked": raw.chunked, "content_encoding": headers.get("content-encoding"),
                 "body_size": raw.size, "body_sha256": raw.sha256, "body_status": body_status, "body": text,
                 "stream": plan.stream, "reply_turn": plan.reply_turn, "response_status": plan.status}
        if raw_b64 is not None:
            entry["body_raw_b64"] = raw_b64
        self.server.record(entry)
        if plan.events is not None:
            self._send_sse(plan.events)
        else:
            self._send(plan.status, plan.payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _serve  # noqa: N815 - http.server names


class MockServer:
    """
    Description: A running mock provider and its recording.
    Parameters:
        httpd: the bound server.
        thread: the thread serving it.
    Output: An object with host, port, url, record_dir, records() and close(); usable as a context manager.
    """

    def __init__(self, httpd: _Httpd, thread: threading.Thread) -> None:
        """
        Description: Wrap a started server.
        Parameters:
            httpd: the bound server.
            thread: the thread running serve_forever.
        Output: None.
        """
        self.server, self.thread = httpd, thread
        self.host, self.port = httpd.server_address[0], httpd.server_address[1]
        self.record_dir = httpd.record_dir

    @property
    def url(self) -> str:
        """
        Description: The base URL an agent should be pointed at.
        Parameters: None
        Output: "http://HOST:PORT" (IPv6 hosts in brackets).
        """
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    def records(self) -> list[dict]:
        """
        Description: Copy the requests recorded so far.
        Parameters: None
        Output: A list of entry dicts in arrival order.
        """
        with self.server.lock:
            return list(self.server.entries)

    def close(self) -> None:
        """
        Description: Stop serving, release the port and close the recording file.
        Parameters: None
        Output: None; safe to call twice.
        """
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=10)
        self.server.server_close()
        if self.server.record_file is not None and not self.server.record_file.closed:
            self.server.record_file.close()

    def __enter__(self) -> "MockServer":
        """
        Description: Enter the context manager.
        Parameters: None
        Output: This server.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """
        Description: Leave the context manager, closing the server.
        Parameters:
            *exc: the exception information (ignored).
        Output: None.
        """
        self.close()


def start_server(host: str = "127.0.0.1", port: int = 0, record_dir: Optional[Path] = None,
                 turns: Iterable[dict] = (), max_body: int = MAX_BODY_BYTES,
                 count_tokens_status: Optional[int] = None, reply_headers: Optional[dict] = None) -> MockServer:
    """
    Description: Start the mock provider on a loopback address in a background thread.
    Parameters:
        host: the address to bind; must be loopback (see ensure_loopback).
        port: the port, 0 to let the system choose one.
        record_dir: the folder for requests.jsonl, or None to record in memory only.
        turns: the scripted replies (see load_script).
        max_body: the largest request body accepted, in bytes.
        count_tokens_status: an HTTP error status for count_tokens, or None for a fixed count.
        reply_headers: {name: value} added to every response (see check_reply_headers).
    Output: The running MockServer. Raises ValueError for a non-loopback host or a bad reply header before anything is
        created, OSError when the port is taken.
    """
    bind = ensure_loopback(host)
    config = Config(tuple(turns), max_body, count_tokens_status, check_reply_headers(reply_headers or {}))
    httpd = _Httpd((bind, port), config, record_dir)
    if not ipaddress.ip_address(httpd.server_address[0]).is_loopback:  # belt and braces after the bind
        httpd.server_close()
        raise ValueError("the mock provider bound a non-loopback address; refusing to serve")
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    return MockServer(httpd, thread)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Run the mock provider until it is interrupted, printing one JSON line with its address.
    Parameters:
        argv: command-line arguments, or None to use sys.argv.
    Output: Exit status: 0 after a clean shutdown on SIGINT or SIGTERM; 2 for a non-loopback host or a bad script (a
        message on stderr, nothing created); 1 when the port cannot be bound. Prints {"listening": URL, "record": PATH}
        to stdout once it serves. Never prints a request.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="loopback address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="port; 0 lets the system choose")
    parser.add_argument("--record", type=Path, default=None, help="folder for requests.jsonl (default: memory only)")
    parser.add_argument("--script", type=Path, default=None, help="JSON list of scripted turns")
    parser.add_argument("--max-body", type=int, default=MAX_BODY_BYTES, help="largest accepted body in bytes")
    parser.add_argument("--count-tokens-status", type=int, default=None, help="answer count_tokens with this HTTP error")
    parser.add_argument("--reply-header", action="append", default=[], metavar="NAME=VALUE",
                        help="add this header to every response (repeatable)")
    args = parser.parse_args(argv)
    try:
        ensure_loopback(args.host)
        extra = dict(pair.split("=", 1) if "=" in pair else (pair, None) for pair in args.reply_header)
        if None in extra.values():
            raise ValueError("--reply-header needs NAME=VALUE")
        turns = load_script(args.script) if args.script else []
        server = start_server(args.host, args.port, args.record, turns, args.max_body, args.count_tokens_status, extra)
    except ValueError as exc:
        print(f"mock_provider: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"mock_provider: cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    record = str(Path(args.record) / "requests.jsonl") if args.record else None
    print(json.dumps({"listening": server.url, "record": record}), flush=True)
    stop.wait()
    server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
