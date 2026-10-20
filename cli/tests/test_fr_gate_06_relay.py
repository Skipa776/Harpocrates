"""FR-GATE-06: the response goes back to the agent as it arrives, with the provider's status, headers and body
unchanged, and a stream that breaks is cut off, never finished with invented bytes. Unit tests of relay_response."""
from __future__ import annotations

import io
import threading

import pytest
from gate_support import ScriptedUpstream, chunk

from Harpocrates.gate.config import GateConfig, Upstream
from Harpocrates.gate.relay import Relayed, relay_response
from Harpocrates.gate.upstream import open_upstream

FAST = GateConfig(connect_timeout=2, read_timeout=2)


def fetch(reply, http10: bool = False, out: io.BufferedIOBase | None = None):
    """Run reply(conn) as the upstream, relay its response into `out`, and return (Relayed, raw bytes written)."""
    sink = out if out is not None else io.BytesIO()
    with ScriptedUpstream(lambda conn, head, body: reply(conn)) as up:
        conn, response = open_upstream(Upstream("http", up.host, up.port), FAST, "POST", "/v1/messages", [("Content-Length", "2")], b"{}")
        try:
            relayed = relay_response(response, sink, http10)
        finally:
            conn.close()
    return relayed, sink.getvalue() if isinstance(sink, io.BytesIO) else b""


def split(raw: bytes) -> tuple[str, list[tuple[str, str]], bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    return lines[0], [tuple(line.split(": ", 1)) for line in lines[1:]], body


def dechunk(body: bytes) -> tuple[list[bytes], bool]:
    """Chunks of a chunked body, and whether the terminating chunk was there."""
    chunks, rest = [], body
    while rest:
        size_line, _, rest = rest.partition(b"\r\n")
        size = int(size_line, 16)
        if size == 0:
            return chunks, rest == b"\r\n"
        chunks.append(rest[:size])
        rest = rest[size + 2:]
    return chunks, False


def test_fr_gate_06_a_length_delimited_reply_passes_status_headers_and_body_unchanged() -> None:
    reply = (b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nretry-after: 12\r\n"
             b"x-should-retry: true\r\nanthropic-ratelimit-unified-status: rejected\r\nContent-Length: 11\r\n"
             b"Connection: keep-alive\r\nKeep-Alive: timeout=5\r\n\r\n{\"e\":\"x\"}\n\n")
    relayed, raw = fetch(lambda conn: conn.sendall(reply))
    status, headers, body = split(raw)
    assert status == "HTTP/1.1 429 Too Many Requests"
    assert headers == [("Content-Type", "application/json"), ("retry-after", "12"), ("x-should-retry", "true"),
                       ("anthropic-ratelimit-unified-status", "rejected"), ("Content-Length", "11")]
    assert body == b'{"e":"x"}\n\n'
    assert relayed == Relayed(429, 11, "complete")


@pytest.mark.parametrize("status,reason", [(200, "OK"), (400, "Bad Request"), (401, "Unauthorized"), (500, "Internal Server Error"),
                                           (529, "Overloaded")])
def test_fr_gate_06_error_statuses_and_bodies_pass_unchanged(status: int, reason: str) -> None:
    body = b'{"type":"error","error":{"type":"x","message":"upstream wording stays"}}'
    reply = f"HTTP/1.1 {status} {reason}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
    relayed, raw = fetch(lambda conn: conn.sendall(reply))
    assert split(raw)[0] == f"HTTP/1.1 {status} {reason}" and split(raw)[2] == body
    assert relayed.status == status and relayed.ended == "complete"


def test_fr_gate_06_a_chunked_stream_is_forwarded_chunk_by_chunk() -> None:
    events = [b"event: message_start\ndata: {}\n\n", b"event: ping\ndata: {}\n\n", b"event: message_stop\ndata: {}\n\n"]

    def reply(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                     b"Transfer-Encoding: chunked\r\n\r\n")
        for event in events:
            conn.sendall(chunk(event))
        conn.sendall(b"0\r\n\r\n")

    relayed, raw = fetch(reply)
    status, headers, body = split(raw)
    assert status == "HTTP/1.1 200 OK"
    assert ("Transfer-Encoding", "chunked") in headers and ("Content-Type", "text/event-stream") in headers
    chunks, terminated = dechunk(body)
    assert b"".join(chunks) == b"".join(events) and terminated
    assert relayed == Relayed(200, sum(len(e) for e in events), "complete")


def test_fr_gate_06_the_first_event_reaches_the_agent_before_the_provider_sends_the_second() -> None:
    first_written = threading.Event()
    seen_before_second: list[bool] = []

    class Out(io.BytesIO):
        def write(self, data):
            if b"first-event" in data:
                first_written.set()
            return super().write(data)

    def reply(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"first-event"))
        seen_before_second.append(first_written.wait(5))  # a buffering relay would never get here with True
        conn.sendall(chunk(b"second-event") + b"0\r\n\r\n")

    out = Out()
    relayed, _ = fetch(reply, out=out)
    assert seen_before_second == [True] and relayed.ended == "complete"
    assert b"second-event" in out.getvalue()


def test_fr_gate_06_a_stream_the_provider_cuts_off_is_cut_off_without_a_terminating_chunk() -> None:
    def reply(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"event one\n\n"))
        conn.sendall(b"20\r\nhalf a chunk")  # promises 32 bytes, sends 12, then the connection closes

    relayed, raw = fetch(reply)
    _, _, body = split(raw)
    assert relayed.ended == "upstream-drop" and relayed.status == 200
    assert relayed.bytes_back == len(b"event one\n\n") + len(b"half a chunk")  # the bytes that did arrive, as they arrived
    assert not body.endswith(b"0\r\n\r\n")
    assert b"event one" in body


def test_fr_gate_06_a_stream_that_ends_without_its_last_chunk_is_a_drop() -> None:
    def reply(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"event one\n\n"))

    relayed, raw = fetch(reply)
    assert relayed.ended == "upstream-drop" and not raw.endswith(b"0\r\n\r\n")


def test_fr_gate_06_a_length_delimited_body_cut_short_is_a_drop() -> None:
    relayed, raw = fetch(lambda conn: conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nonly ten b"))
    assert relayed.ended == "upstream-drop" and relayed.bytes_back == 10
    assert split(raw)[2] == b"only ten b"


def test_fr_gate_06_a_reply_without_framing_is_sent_chunked_and_ends_at_the_close() -> None:
    relayed, raw = fetch(lambda conn: conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nuntil close"))
    status, headers, body = split(raw)
    assert ("Transfer-Encoding", "chunked") in headers and not [h for h in headers if h[0].lower() == "content-length"]
    assert dechunk(body) == ([b"until close"], True) and relayed.ended == "complete"


def test_fr_gate_06_an_http_1_0_agent_gets_the_body_raw_without_chunk_framing() -> None:
    def reply(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"abc") + chunk(b"def") + b"0\r\n\r\n")

    relayed, raw = fetch(reply, http10=True)
    status, headers, body = split(raw)
    assert body == b"abcdef" and not [h for h in headers if h[0].lower() in ("transfer-encoding", "content-length")]
    assert relayed.ended == "complete"


@pytest.mark.parametrize("status,reason", [(204, "No Content"), (304, "Not Modified")])
def test_fr_gate_06_replies_without_a_body_get_no_framing(status: int, reason: str) -> None:
    relayed, raw = fetch(lambda conn: conn.sendall(f"HTTP/1.1 {status} {reason}\r\nETag: abc\r\n\r\n".encode()))
    head, headers, body = split(raw)
    assert head == f"HTTP/1.1 {status} {reason}" and body == b"" and headers == [("ETag", "abc")]
    assert relayed == Relayed(status, 0, "complete")


def test_fr_gate_06_a_second_or_invalid_content_length_from_the_provider_is_not_passed_on() -> None:
    relayed, raw = fetch(lambda conn: conn.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nContent-Length: 5\r\n\r\nhello"))
    _, headers, body = split(raw)
    assert [h for h in headers if h[0].lower() == "content-length"] == [("Content-Length", "5")] and body == b"hello"
    relayed, raw = fetch(lambda conn: conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: abc\r\n\r\nhello"))
    _, headers, body = split(raw)
    assert not [h for h in headers if h[0].lower() == "content-length"] and dechunk(body) == ([b"hello"], True)


def test_fr_gate_06_an_agent_that_hangs_up_stops_the_relay_quietly() -> None:
    class Gone(io.BytesIO):
        writes = 0

        def write(self, data):
            Gone.writes += 1
            if Gone.writes > 1:
                raise BrokenPipeError("client went away")
            return super().write(data)

    def reply(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"one") + chunk(b"two") + b"0\r\n\r\n")

    relayed, _ = fetch(reply, out=Gone())
    assert relayed.ended == "client-gone"
