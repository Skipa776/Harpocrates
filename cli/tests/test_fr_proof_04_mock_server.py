"""FR-PROOF-04: the mock provider as a running server: loopback only, one recording line per request, no credential kept.

A real ThreadingHTTPServer runs on 127.0.0.1 for each test. The credential values are made-up strings.
"""
from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import socket
from pathlib import Path
from typing import Iterable, Optional

import pytest

from bench import mock_provider as mock

FAKE = "FAKE-CRED-0123456789abcdef"


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

from test_fr_proof_04_mock_replies import (  # noqa: E402
    check_anthropic_message,
    check_response,
    parse_sse,
)

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
