"""FR-GATE-01, SEC-03, FR-INSTALL-04, FR-INSTALL-05, FR-GATE-06: what the gate forwards and how it streams the reply back,
end to end in front of bench/mock_provider.py (and a scripted raw-socket upstream for what the mock cannot do:
a stream cut mid-chunk, a silent server, a garbage reply). Every credential and canary here is a made-up string."""
from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import re
import socket
import threading
import time

import pytest
import zstandard
from gate_rig import (
    ANTHROPIC,
    JSON,
    MESSAGES,
    RECOVERY_PHRASES,
    RESPONSES,
    STREAM_MESSAGES,
    STREAM_RESPONSES,
    passthrough_scan,
    running_rig,
)
from gate_support import ScriptedUpstream, chunk, free_closed_port

from bench import mock_provider as mock
from Harpocrates.gate.scan import Forward
from Harpocrates.gate.session_token import TOKEN_HEADER

FORWARD_CASES = [
    pytest.param("/v1/messages?beta=true", MESSAGES, ANTHROPIC, "application/json", id="anthropic-json"),
    pytest.param("/v1/messages", STREAM_MESSAGES, ANTHROPIC, "text/event-stream", id="anthropic-sse"),
    pytest.param("/v1/responses", RESPONSES, {}, "application/json", id="responses-json"),
    pytest.param("/v1/responses", STREAM_RESPONSES, {}, "text/event-stream", id="responses-sse"),
]


def normalise(raw: bytes) -> bytes:
    """Make two mock replies comparable: the mock numbers every reply id, so a reply fetched later has other ids."""
    return re.sub(rb"mock(sig|cipher)?\d{8}", rb"mockN", raw)


def direct(upstream, target: str, body: bytes, headers: dict) -> tuple[int, list, bytes]:
    conn = http.client.HTTPConnection(upstream.host, upstream.port, timeout=10)
    try:
        conn.request("POST", target, body=body, headers={**JSON, **headers})
        response = conn.getresponse()
        return response.status, response.getheaders(), response.read()
    finally:
        conn.close()


@pytest.mark.parametrize("target,body,headers,content_type", FORWARD_CASES)
def test_fr_gate_01_forwards_to_upstream_and_streams_back(target, body, headers, content_type) -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        reply = rig.call("POST", target, body, {**JSON, **headers})
        (record,) = rig.mock.records()
        assert reply.status == 200 and reply.header("content-type").startswith(content_type)
        # The provider got exactly what the agent sent: same method, path and query, same body bytes.
        assert record["method"] == "POST" and record["path"] == target
        assert record["body_sha256"] == hashlib.sha256(body).hexdigest() and record["body_status"] == "ok"
        # The agent got exactly what the provider says to a direct caller.
        status, direct_headers, direct_body = direct(rig.mock, target, body, headers)
        assert status == 200 and normalise(reply.body) == normalise(direct_body)
        if content_type == "text/event-stream":
            assert reply.header("transfer-encoding") == "chunked" and reply.header("cache-control") == "no-cache"
            assert reply.body.startswith(b"event: ") and reply.body.count(b"\n\n") >= 4
        else:
            assert json.loads(reply.body)  # whole JSON document
            assert reply.header("content-length") == str(len(reply.body))


def test_fr_gate_01_models_is_passed_through_for_both_providers() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        openai = rig.call("GET", "/v1/models")
        anthropic = rig.call("GET", "/v1/models?limit=1000", None, ANTHROPIC)
        assert openai.status == anthropic.status == 200
        assert json.loads(openai.body)["data"][0]["id"] == "gpt-mock"
        assert json.loads(anthropic.body)["data"][0]["id"] == "claude-mock"
        paths = [r["path"] for r in rig.mock.records()]
        assert paths == ["/v1/models", "/v1/models?limit=1000"] and all(r["body_size"] == 0 for r in rig.mock.records())


def test_fr_gate_01_a_chunked_request_body_reaches_the_provider_with_a_length() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        parts = [MESSAGES[:10], MESSAGES[10:40], MESSAGES[40:]]
        reply = rig.call("POST", "/v1/messages", None, {**JSON, **ANTHROPIC}, chunks=parts)
        (record,) = rig.mock.records()
        assert reply.status == 200 and record["body_sha256"] == hashlib.sha256(MESSAGES).hexdigest()
        assert record["chunked"] is False and record["content_length_header"] == len(MESSAGES)


def test_fr_gate_01_a_body_at_the_cap_is_forwarded_and_one_byte_over_is_not() -> None:
    pad = lambda size: json.dumps({"model": "m", "input": "x" * (size - 27)}).encode()  # noqa: E731
    at_cap, over = pad(4096), pad(4097)
    assert len(at_cap) == 4096 and len(over) == 4097
    with running_rig(scan_step=passthrough_scan, max_body=4096) as rig:
        assert rig.call("POST", "/v1/responses", at_cap, JSON).status == 200
        assert rig.call("POST", "/v1/responses", over, JSON).status == 400
        assert len(rig.mock.records()) == 1


def test_fr_gate_01_a_scan_step_may_rewrite_the_body_and_the_length_follows() -> None:
    rewritten = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "REDACTED"}]})
    with running_rig(scan_step=lambda body, route: Forward(rewritten.encode())) as rig:
        reply = rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        (record,) = rig.mock.records()
        assert reply.status == 200 and record["body"] == rewritten
        assert record["content_length_header"] == len(rewritten)


def test_fr_gate_06_provider_errors_pass_through_with_status_headers_and_body() -> None:
    turns = [{"error": {"status": 529, "message": "Overloaded, upstream wording"}}]
    with running_rig(scan_step=passthrough_scan, mock_kwargs={"turns": turns}) as rig:
        with_tools = json.dumps({"model": "m", "max_tokens": 1, "tools": [{"name": "t"}], "messages": []}).encode()
        reply = rig.call("POST", "/v1/messages", with_tools, {**JSON, **ANTHROPIC})  # the mock scripts replies for tool requests
        assert reply.status == 529
        assert json.loads(reply.body)["error"]["message"] == "Overloaded, upstream wording"
        assert json.loads(reply.body)["type"] == "error"  # the provider's own error shape, untouched


def test_sec_03_token_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock, "is_credential_header", lambda name: False)  # record header values as they arrive
    with running_rig(scan_step=passthrough_scan) as rig:
        rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        rig.call("GET", "/v1/models", None, {"X-Harpo-Gate-Token": rig.token}, token=False)
        rig.call("POST", "/v1/responses", RESPONSES, {**JSON, "X-HARPO-GATE-TOKEN": rig.token}, token=False)
        records = rig.mock.records()
        assert len(records) == 3
        for record in records:
            assert not [h for h in record["headers"] if h[0].lower() == TOKEN_HEADER]
            assert rig.token not in json.dumps(record)  # not in a header, the path or the body


def test_fr_install_05_auth_headers_forwarded_byte_identical_and_never_logged(monkeypatch: pytest.MonkeyPatch,
                                                                              capfd) -> None:
    monkeypatch.setattr(mock, "is_credential_header", lambda name: False)  # record header values as they arrive
    canaries = {
        "x-api-key": "FAKE-ANTHROPIC-KEY-9f3a1c77d0b2", "Authorization": "Bearer FAKE.token+with/odd=chars",
        "anthropic-version": "2023-06-01", "anthropic-beta": "oauth-2025-04-20,claude-code-20250219,FAKE-BETA-77",
        "ChatGPT-Account-ID": "acct-FAKE-0042", "OpenAI-Organization": "org-FAKE-1", "OpenAI-Project": "proj-FAKE-2",
        "OpenAI-Beta": "responses=FAKE", "x-codex-turn-state": "FAKE-STATE-5521", "session-id": "FAKE-SESSION-88",
        "x-codex-turn-metadata": '{"workspaces":{"/FAKE/path":{"latest_git_commit_hash":"FAKEHASH"}}}',
        "X-Mixed-Case": "café ÿ  inner  spaces",
    }
    sent = {**JSON, **canaries}
    body_canary, query_canary = "FAKE-BODY-CANARY-31337", "FAKE-QUERY-CANARY-4242"
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": body_canary}]}).encode()
    with running_rig(scan_step=passthrough_scan) as rig:
        reply = rig.call("POST", f"/v1/messages?k={query_canary}", body, sent)
        assert reply.status == 200
        # Blocked requests carry the same headers; they must not be logged either.
        rig.call("POST", "/v1/chat/completions", body, sent)
        rig.call("POST", "/v1/messages", body, sent, token="wrong-token")
        (record,) = rig.mock.records()
        received = [tuple(h) for h in record["headers"]]
        for name, value in canaries.items():
            assert (name, value) in received, name  # same name spelling, same value bytes
        order = [h for h in received if h[0] in canaries]
        assert [h[0] for h in order] == [n for n in sent if n in canaries]  # the agent's header order is kept
        lines = rig.wait_log(3)
        assert len(lines) == 3  # one line for each of the three requests, so the check below is not vacuous
        out, err = capfd.readouterr()
        logged = "".join(lines) + out + err
        for secret in (*canaries.values(), body_canary, query_canary, rig.token):
            assert secret not in logged


def test_fr_install_05_headers_are_forwarded_in_order_with_duplicates_and_expect_is_not_forwarded() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        head = (f"POST /v1/messages HTTP/1.1\r\nHost: gate.test\r\n{TOKEN_HEADER}: {rig.token}\r\nX-Dup: one\r\n"
                f"anthropic-version: 2023-06-01\r\nx-dup: two\r\nExpect: 100-continue\r\nConnection: keep-alive\r\n"
                f"Keep-Alive: timeout=5\r\nContent-Length: {len(MESSAGES)}\r\n\r\n").encode()
        raw = rig.raw(head + MESSAGES)
        assert raw.startswith(b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK")
        (record,) = rig.mock.records()
        names = [h[0] for h in record["headers"]]
        assert [tuple(h) for h in record["headers"] if h[0].lower() == "x-dup"] == [("X-Dup", "one"), ("x-dup", "two")]
        assert names.index("X-Dup") < names.index("anthropic-version") < names.index("x-dup")
        assert not [n for n in names if n.lower() in ("expect", "connection", "keep-alive", TOKEN_HEADER)]
        assert ("Host", f"{rig.mock.host}:{rig.mock.port}") in [tuple(h) for h in record["headers"]]


@pytest.mark.parametrize("body,encoding", [(MESSAGES, "identity"), (MESSAGES, "gzip"), (MESSAGES, "zstd")])
def test_fr_install_04_forwarded_body_is_re_encoded_as_received(body: bytes, encoding: str) -> None:
    wire = {"identity": body, "gzip": gzip.compress(body), "zstd": zstandard.ZstdCompressor().compress(body)}[encoding]
    headers = {**JSON, **ANTHROPIC, **({} if encoding == "identity" else {"Content-Encoding": encoding})}
    with running_rig(scan_step=passthrough_scan) as rig:
        assert rig.call("POST", "/v1/messages", wire, headers).status == 200
        (record,) = rig.mock.records()
        assert record["body_status"] == "ok" and record["body"] == body.decode()  # the provider can read it
        assert record["content_encoding"] == (None if encoding == "identity" else encoding)
        assert record["content_length_header"] == record["body_size"]  # the length is for the forwarded bytes
    with running_rig(scan_step=lambda b, r: Forward(b'{"rewritten":true}')) as rig:
        assert rig.call("POST", "/v1/messages", wire, headers).status == 200
        (record,) = rig.mock.records()
        assert record["body"] == '{"rewritten":true}' and record["content_encoding"] == (
            None if encoding == "identity" else encoding)


@pytest.mark.parametrize("path,body,headers,shape", [
    ("/v1/messages", MESSAGES, ANTHROPIC, "anthropic"), ("/v1/responses", RESPONSES, {}, "openai")])
def test_fr_gate_06_upstream_failure_502(path, body, headers, shape) -> None:
    dead = f"http://127.0.0.1:{free_closed_port()}"
    silent_release = threading.Event()
    with ScriptedUpstream(lambda conn, head, b: silent_release.wait(5)) as silent, \
            ScriptedUpstream(lambda conn, head, b: conn.sendall(b"garbage\r\n\r\n")) as garbage:
        for url, extra in ((dead, {}), (silent.url, {"read_timeout": 0.4}), (garbage.url, {})):
            with running_rig(scan_step=passthrough_scan, anthropic_upstream=url, openai_upstream=url,
                             **extra) as rig:
                started = time.monotonic()
                reply = rig.call("POST", path, body, {**JSON, **headers})
                assert reply.status == 502 and time.monotonic() - started < 6
                error = json.loads(reply.body)
                assert "HARPO-UPSTREAM-FAILED" in reply.body.decode()
                assert (error["type"] == "error" and error["error"]["type"] == "api_error") if shape == "anthropic" \
                    else error["error"]["type"] == "server_error"
                assert not [p for p in RECOVERY_PHRASES if p in reply.body.decode().lower()]
                assert "127.0.0.1" not in reply.body.decode() and str(rig.token) not in reply.body.decode()
        silent_release.set()


def head_and_events(sock: socket.socket) -> bytes:
    """Read from a socket until the first SSE event (or EOF)."""
    data = b""
    while b"first-event" not in data:
        piece = sock.recv(65536)
        if not piece:
            break
        data += piece
    return data


def test_fr_gate_06_the_first_event_reaches_the_agent_before_the_provider_sends_the_next() -> None:
    release = threading.Event()
    upstream_saw_client_read: list[bool] = []

    def script(conn, head, body) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                     + chunk(b"event: first-event\ndata: {}\n\n"))
        upstream_saw_client_read.append(release.wait(8))
        conn.sendall(chunk(b"event: second-event\ndata: {}\n\n") + b"0\r\n\r\n")

    with ScriptedUpstream(script) as up, running_rig(scan_step=passthrough_scan, anthropic_upstream=up.url) as rig:
        request = (f"POST /v1/messages HTTP/1.1\r\nHost: g\r\n{TOKEN_HEADER}: {rig.token}\r\nanthropic-version: x\r\n"
                   f"Content-Length: {len(MESSAGES)}\r\n\r\n").encode() + MESSAGES
        with socket.create_connection((rig.gate.host, rig.gate.port), timeout=10) as sock:
            sock.sendall(request)
            first = head_and_events(sock)
            assert b"HTTP/1.1 200 OK" in first and b"first-event" in first and b"second-event" not in first
            release.set()  # only now may the provider go on: the agent already holds the first event
            rest = b""
            while not rest.endswith(b"0\r\n\r\n"):
                piece = sock.recv(65536)
                if not piece:
                    break
                rest += piece
            assert b"second-event" in rest and rest.endswith(b"0\r\n\r\n")
        assert upstream_saw_client_read == [True]


def test_fr_gate_06_mid_stream_drop_closes() -> None:
    def script(conn, head, body) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                     + chunk(b"event: first-event\ndata: {}\n\n") + b"40\r\nevent: cut-off-ev")  # then the provider dies

    with ScriptedUpstream(script) as up, running_rig(scan_step=passthrough_scan, anthropic_upstream=up.url) as rig:
        request = (f"POST /v1/messages HTTP/1.1\r\nHost: g\r\n{TOKEN_HEADER}: {rig.token}\r\nanthropic-version: x\r\n"
                   f"Content-Length: {len(MESSAGES)}\r\n\r\n").encode() + MESSAGES
        started = time.monotonic()
        raw = rig.raw(request, half_close=False, timeout=10)  # returns when the gate closes the connection
        assert time.monotonic() - started < 5  # it closed; it did not leave the agent waiting on a dead stream
        assert raw.startswith(b"HTTP/1.1 200 OK") and b"first-event" in raw
        assert not raw.endswith(b"0\r\n\r\n")  # no invented ending
        # An HTTP client sees an incomplete read, never a finished response.
        conn = http.client.HTTPConnection(rig.gate.host, rig.gate.port, timeout=10)
        conn.request("POST", "/v1/messages", body=MESSAGES, headers={TOKEN_HEADER: rig.token, "anthropic-version": "x"})
        response = conn.getresponse()
        with pytest.raises(http.client.IncompleteRead):
            response.read()
        conn.close()


def test_fr_gate_06_an_agent_that_hangs_up_cancels_the_provider_request() -> None:
    upstream_failed = threading.Event()

    def script(conn, head, body) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        for _ in range(400):  # a long stream; a live agent would read it all
            try:
                conn.sendall(chunk(b"event: first-event\ndata: " + b"x" * 4000 + b"\n\n"))
            except OSError:
                upstream_failed.set()
                return
            time.sleep(0.01)

    with ScriptedUpstream(script) as up, running_rig(scan_step=passthrough_scan, anthropic_upstream=up.url) as rig:
        request = (f"POST /v1/messages HTTP/1.1\r\nHost: g\r\n{TOKEN_HEADER}: {rig.token}\r\nanthropic-version: x\r\n"
                   f"Content-Length: {len(MESSAGES)}\r\n\r\n").encode() + MESSAGES
        sock = socket.create_connection((rig.gate.host, rig.gate.port), timeout=10)
        sock.sendall(request)
        assert b"first-event" in head_and_events(sock)
        sock.close()  # the agent goes away mid-stream
        assert upstream_failed.wait(8)  # the gate dropped the provider connection, so the provider's send failed
