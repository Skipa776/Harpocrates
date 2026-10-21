"""FR-GATE-06 and NFR-01 (docs/design.md): the gate keeps idle upstream connections and
reuses them, but only after a reply was read to its framed end on HTTP/1.1 without Connection: close or an error; a
reused connection that fails before any reply byte is retried once on a new connection; a new connection that fails is
not retried.

The upstream is a keep-alive server on 127.0.0.1 written here, which counts accepted connections and the requests on each.
"""
from __future__ import annotations

import http.client
import json
import shutil
import socket
import ssl
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import pytest
from gate_rig import passthrough_scan
from gate_support import read_http_request

import Harpocrates.gate.upstream as upstream_module
from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.request_log import format_entry
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token

BODY = json.dumps({"model": "m", "input": "hi", "store": False}).encode()
JSON = {"Content-Type": "application/json", "Authorization": "Bearer sk-fake-api-key-0000"}


def ok(payload: bytes = b"{}", close: bool = False) -> bytes:
    extra = b"Connection: close\r\n" if close else b""
    return (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(payload)).encode()
            + b"\r\n" + extra + b"\r\n" + payload)


# A script decides each reply: script(connection_number, request_number_on_that_connection, head) -> bytes to send, or
# None to close the connection without answering. It may also return a callable(conn) that does the sending itself and
# returns whether to keep the connection.
Script = Callable[[int, int, bytes], object]


class KeepAliveUpstream:
    """A loopback HTTP/1.1 server that serves many requests per connection and records (connection, request) pairs."""

    def __init__(self, script: Script, tls: Optional[ssl.SSLContext] = None) -> None:
        self.script = script
        self.tls = tls
        self.accepted = 0
        self.requests: list[tuple[int, int]] = []
        self.closed_by_peer: list[int] = []
        self.lock = threading.Lock()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(32)
        self.url = f"{'https' if tls else 'http'}://127.0.0.1:{self.listener.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            with self.lock:
                self.accepted += 1
                number = self.accepted
            threading.Thread(target=self._serve, args=(conn, number), daemon=True).start()

    def _serve(self, conn: socket.socket, number: int) -> None:
        conn.settimeout(20)
        if self.tls is not None:
            try:
                conn = self.tls.wrap_socket(conn, server_side=True)
            except OSError:
                conn.close()
                return
        served = 0
        try:
            while True:
                head, _ = read_http_request(conn)
                if not head:
                    with self.lock:
                        self.closed_by_peer.append(number)
                    return
                served += 1
                with self.lock:
                    self.requests.append((number, served))
                action = self.script(number, served, head)
                if action is None:
                    return
                if callable(action):
                    if not action(conn):
                        return
                    continue
                conn.sendall(action)
                if b"Connection: close" in action:
                    return
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self.listener.close()


class Rig:
    """A gate (pass-through scan step) whose OpenAI upstream is a KeepAliveUpstream."""

    def __init__(self, script: Script, tls: Optional[tuple] = None, **config: object) -> None:
        self.upstream = KeepAliveUpstream(script, tls[0] if tls else None)
        self.token = new_token()
        self.log_lines: list[str] = []
        self.gate = start_gate(GateConfig(openai_upstream=self.upstream.url, anthropic_upstream=self.upstream.url, **config),
                               self.token, scan_step=passthrough_scan, log_sink=self.log_lines.append)
        if tls:  # trust the throwaway certificate on the gate's side
            self.gate.server.pool.close()
            self.gate.server.pool = upstream_module.UpstreamPool(tls[1])

    def __enter__(self) -> "Rig":
        return self

    def __exit__(self, *exc: object) -> None:
        self.gate.close()
        self.upstream.close()

    def call(self, read_all: bool = True) -> tuple[int, bytes]:
        """One request; returns once the gate has logged it, which happens after it handed its connection back."""
        logged = len(self.log_lines)
        conn = http.client.HTTPConnection(self.gate.host, self.gate.port, timeout=20)
        try:
            conn.request("POST", "/v1/responses", body=BODY, headers={TOKEN_HEADER: self.token, **JSON})
            response = conn.getresponse()
            result = response.status, (response.read() if read_all else response.read(8))
        finally:
            conn.close()
        deadline = time.monotonic() + 5
        while len(self.log_lines) <= logged and time.monotonic() < deadline:
            time.sleep(0.005)
        return result

    def log(self, count: int) -> list[dict]:
        deadline = time.monotonic() + 3
        while len(self.log_lines) < count and time.monotonic() < deadline:
            time.sleep(0.01)
        return [json.loads(line) for line in self.log_lines]


def always_ok(number: int, served: int, head: bytes) -> bytes:
    return ok(json.dumps({"connection": number, "request": served}).encode())


def test_fr_gate_06_second_request_reuses_upstream_connection() -> None:
    with Rig(always_ok) as rig:
        replies = [rig.call() for _ in range(3)]
        log = rig.log(3)
    assert [status for status, _ in replies] == [200, 200, 200]
    assert [json.loads(body) for _, body in replies] == [{"connection": 1, "request": n} for n in (1, 2, 3)]
    assert rig.upstream.accepted == 1
    assert [entry["upstream_reused"] for entry in log] == [False, True, True]
    assert log[0]["ms_connect"] > 0 and log[1]["ms_connect"] == 0 and log[2]["ms_connect"] == 0


def test_fr_gate_06_connection_close_not_reused() -> None:
    with Rig(lambda number, served, head: ok(close=True)) as rig:
        assert [rig.call()[0] for _ in range(2)] == [200, 200]
        log = rig.log(2)
    assert rig.upstream.accepted == 2 and rig.upstream.requests == [(1, 1), (2, 1)]
    assert [entry["upstream_reused"] for entry in log] == [False, False]


def test_fr_gate_06_http_1_0_reply_not_reused() -> None:
    reply = b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
    with Rig(lambda number, served, head: reply) as rig:
        assert [rig.call()[0] for _ in range(2)] == [200, 200]
    assert rig.upstream.accepted == 2


def slow_stream(conn: socket.socket) -> bool:
    """A chunked reply that keeps sending for about 3 s, as a model streaming tokens."""
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
    for _ in range(30):
        event = b"data: {}\n\n"
        conn.sendall(f"{len(event):x}\r\n".encode() + event + b"\r\n")
        time.sleep(0.1)
    conn.sendall(b"0\r\n\r\n")
    return True


def test_fr_gate_06_cut_stream_connection_not_reused() -> None:
    """The agent hangs up mid-stream: the upstream connection, still sending, must be closed, not pooled."""
    def script(number: int, served: int, head: bytes) -> object:
        return slow_stream if number == 1 else always_ok(number, served, head)

    with Rig(script) as rig:
        status, _ = rig.call(read_all=False)  # read a few bytes, then hang up
        assert status == 200
        time.sleep(0.5)
        status, body = rig.call()
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}
    assert rig.upstream.accepted == 2


def test_fr_gate_06_error_not_reused() -> None:
    """A reply that does not come in time is an upstream error (502); that connection is discarded, so a late reply on it
    can never be read as the answer to the next request."""
    def script(number: int, served: int, head: bytes) -> object:
        if number == 1:
            def late(conn: socket.socket) -> bool:
                time.sleep(1.5)
                conn.sendall(ok(b'{"late": true}'))
                return True
            return late
        return always_ok(number, served, head)

    with Rig(script, read_timeout=0.5) as rig:
        assert rig.call()[0] == 502
        status, body = rig.call()
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}


def test_fr_gate_06_stale_pooled_connection_retried_once() -> None:
    """The provider closes an idle connection: the next request fails on it before any reply byte, goes once more on a
    new connection, succeeds, and the provider sees it only once."""
    def script(number: int, served: int, head: bytes) -> object:
        def answer_then_hang_up(conn: socket.socket) -> bool:
            conn.sendall(always_ok(number, served, head))
            time.sleep(0.1)
            return False  # close the connection the gate may have pooled
        return answer_then_hang_up if number == 1 else always_ok(number, served, head)

    with Rig(script) as rig:
        assert rig.call()[0] == 200
        time.sleep(0.4)
        status, body = rig.call()
        log = rig.log(2)
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}
    assert rig.upstream.requests == [(1, 1), (2, 1)]
    assert log[1]["upstream_reused"] is False


def test_fr_gate_06_reused_connection_dropped_before_reply_is_retried_once() -> None:
    """The trade-off in the spec: a provider that read the request and dropped the reused connection without a byte
    sees it twice, and the agent gets the one reply."""
    def script(number: int, served: int, head: bytes) -> object:
        if number == 1 and served == 2:
            return None  # read it, then close without answering
        return always_ok(number, served, head)

    with Rig(script) as rig:
        assert rig.call()[0] == 200
        status, body = rig.call()
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}
    assert rig.upstream.requests == [(1, 1), (1, 2), (2, 1)]


def test_fr_gate_06_retry_that_also_fails_is_a_502_with_no_third_attempt() -> None:
    def script(number: int, served: int, head: bytes) -> object:
        return always_ok(number, served, head) if (number, served) == (1, 1) else None

    with Rig(script) as rig:
        assert rig.call()[0] == 200
        assert rig.call()[0] == 502
        time.sleep(0.3)
    assert rig.upstream.requests == [(1, 1), (1, 2), (2, 1)]


def test_fr_gate_06_fresh_connection_failure_not_retried() -> None:
    with Rig(lambda number, served, head: None) as rig:
        assert rig.call()[0] == 502
        time.sleep(0.3)
    assert rig.upstream.accepted == 1 and rig.upstream.requests == [(1, 1)]


def test_fr_gate_06_idle_connection_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upstream_module, "POOL_IDLE_SECONDS", 0.2)
    with Rig(always_ok) as rig:
        assert rig.call()[0] == 200
        time.sleep(0.5)
        status, body = rig.call()
        log = rig.log(2)
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}
    assert rig.upstream.requests == [(1, 1), (2, 1)]  # the expired one was never tried
    assert log[1]["upstream_reused"] is False
    assert 1 in rig.upstream.closed_by_peer  # and it was closed, not leaked


def test_fr_gate_06_at_most_four_idle_connections_per_upstream() -> None:
    def slow_ok(number: int, served: int, head: bytes) -> object:
        def answer(conn: socket.socket) -> bool:
            time.sleep(0.5)
            conn.sendall(always_ok(number, served, head))
            return True
        return answer

    def burst(rig: Rig) -> None:
        threads = [threading.Thread(target=rig.call) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)

    assert upstream_module.POOL_MAX_IDLE == 4
    with Rig(slow_ok) as rig:
        burst(rig)
        assert rig.upstream.accepted == 6
        burst(rig)
    assert rig.upstream.accepted == 8  # four reused, two new


def test_fr_gate_06_gate_close_closes_pooled_connections() -> None:
    rig = Rig(always_ok)
    try:
        assert rig.call()[0] == 200
    finally:
        rig.gate.close()
    deadline = time.monotonic() + 3
    while not rig.upstream.closed_by_peer and time.monotonic() < deadline:
        time.sleep(0.02)
    rig.upstream.close()
    assert rig.upstream.closed_by_peer == [1]


def test_fr_gate_06_pool_constants_match_the_spec() -> None:
    assert upstream_module.POOL_MAX_IDLE == 4 and upstream_module.POOL_IDLE_SECONDS == 30.0


@pytest.mark.parametrize("value,kept", [(True, True), (False, True), (1, False), ("yes", False), (None, False)])
def test_fr_gate_06_log_keeps_upstream_reused_only_as_a_bool(value: Optional[object], kept: bool) -> None:
    line = json.loads(format_entry({"upstream_reused": value, "ms_connect": 12.5}))
    assert ("upstream_reused" in line) is kept and line["ms_connect"] == 12.5
    if kept:
        assert line["upstream_reused"] is value


def _then_on_idle(extra: bytes):
    """A script: answer the first request on connection 1 normally, then send `extra` on the idle connection."""
    def script(number: int, served: int, head: bytes) -> object:
        if (number, served) == (1, 1):
            def answer(conn: socket.socket) -> bool:
                conn.sendall(always_ok(number, served, head))
                time.sleep(0.2)
                conn.sendall(extra)
                return b"Connection: close" not in extra
            return answer
        return always_ok(number, served, head)
    return script


def test_fr_gate_06_stray_reply_on_an_idle_connection_is_never_delivered() -> None:
    """Bytes a provider sends on an idle connection are not the reply to the next request, which may be another agent
    session's: the readable connection is closed, and the request goes on a new one."""
    stray = ok(b'{"stray": "meant for no one"}')
    with Rig(_then_on_idle(stray)) as rig:
        assert rig.call()[0] == 200
        time.sleep(0.5)
        status, body = rig.call()
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}
    assert (1, 2) not in rig.upstream.requests


def test_fr_gate_06_idle_timeout_notice_is_not_delivered() -> None:
    notice = b"HTTP/1.1 408 Request Timeout\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    with Rig(_then_on_idle(notice)) as rig:
        assert rig.call()[0] == 200
        time.sleep(0.5)
        status, body = rig.call()
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}


def test_fr_gate_06_reset_after_reply_bytes_is_not_retried() -> None:
    """A reused connection that breaks after some reply bytes may follow a processed request: 502, no second send."""
    def script(number: int, served: int, head: bytes) -> object:
        if (number, served) == (1, 2):
            def half_head(conn: socket.socket) -> bool:
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-")
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # reset, not FIN
                return False
            return half_head
        return always_ok(number, served, head)

    with Rig(script) as rig:
        assert rig.call()[0] == 200
        assert rig.call()[0] == 502
        time.sleep(0.3)
    assert rig.upstream.requests == [(1, 1), (1, 2)]


def test_fr_gate_06_quiet_check_on_an_idle_socket() -> None:
    left, right = socket.socketpair()
    try:
        class Conn:
            sock = left
        assert upstream_module._quiet(Conn())
        right.sendall(b"x")
        time.sleep(0.05)
        assert not upstream_module._quiet(Conn())
        Conn.sock = None
        assert not upstream_module._quiet(Conn())
    finally:
        left.close()
        right.close()


def test_fr_gate_06_cut_stream_with_a_silent_upstream_not_reused() -> None:
    """The agent hangs up mid-stream and the upstream then goes quiet with its stream unfinished: the connection is
    closed, not pooled, even though nothing is waiting to be read on it."""
    def script(number: int, served: int, head: bytes) -> object:
        if number == 1:
            def stall(conn: socket.socket) -> bool:
                conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nfirst\r\n")
                time.sleep(0.6)
                conn.sendall(b"6\r\nsecond\r\n")  # the gate's write to the gone agent fails here
                time.sleep(5)  # then silent, stream unfinished, socket open
                return False
            return stall
        return always_ok(number, served, head)

    with Rig(script, read_timeout=2) as rig:
        assert rig.call(read_all=False)[0] == 200
        status, body = rig.call()
    assert status == 200 and json.loads(body) == {"connection": 2, "request": 1}
    assert (1, 2) not in rig.upstream.requests


def test_fr_gate_06_a_reply_still_in_flight_when_the_gate_closes_is_not_pooled() -> None:
    def script(number: int, served: int, head: bytes) -> object:
        def slow(conn: socket.socket) -> bool:
            time.sleep(0.6)
            conn.sendall(always_ok(number, served, head))
            return True
        return slow

    rig = Rig(script)
    try:
        caller = threading.Thread(target=rig.call)
        caller.start()
        time.sleep(0.2)
        rig.gate.close()
        caller.join(10)
        deadline = time.monotonic() + 3
        while not rig.upstream.closed_by_peer and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        rig.upstream.close()
    assert rig.upstream.closed_by_peer == [1]


class FakeConnection:
    """A stand-in HTTPConnection whose send or read step raises a chosen error."""

    def __init__(self, send_error: Optional[BaseException] = None, read_error: Optional[BaseException] = None) -> None:
        self.send_error, self.read_error, self.closed = send_error, read_error, False

    def putrequest(self, *args: object, **kwargs: object) -> None:
        pass

    def putheader(self, *args: object) -> None:
        pass

    def endheaders(self, message_body: object = None) -> None:
        if self.send_error:
            raise self.send_error

    def getresponse(self) -> object:
        if self.read_error:
            raise self.read_error
        return "response"

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("error", [BrokenPipeError(), ConnectionResetError(), ssl.SSLEOFError()],
                         ids=["broken_pipe", "reset", "tls_eof"])
def test_fr_gate_06_stale_failure_while_sending_is_retried(error) -> None:
    conn = FakeConnection(send_error=error)
    assert upstream_module.UpstreamPool._on_pooled(conn, "POST", "/v1/responses", [], b"{}") is None and conn.closed


@pytest.mark.parametrize("error", [TimeoutError(), http.client.BadStatusLine("x")], ids=["timeout", "bad_status"])
def test_fr_gate_06_other_failure_while_sending_is_not_retried(error) -> None:
    conn = FakeConnection(send_error=error)
    with pytest.raises(upstream_module.UpstreamError):
        upstream_module.UpstreamPool._on_pooled(conn, "POST", "/v1/responses", [], b"{}")
    assert conn.closed


@pytest.mark.parametrize("error,retried", [(http.client.RemoteDisconnected("x"), True), (ConnectionResetError(), False),
                                           (TimeoutError(), False), (ssl.SSLEOFError(), False)],
                         ids=["no_reply_byte", "reset_mid_reply", "timeout", "tls_eof_mid_reply"])
def test_fr_gate_06_while_reading_only_a_close_with_no_reply_byte_is_retried(error, retried) -> None:
    conn = FakeConnection(read_error=error)
    if retried:
        assert upstream_module.UpstreamPool._on_pooled(conn, "POST", "/v1/responses", [], b"{}") is None
    else:
        with pytest.raises(upstream_module.UpstreamError):
            upstream_module.UpstreamPool._on_pooled(conn, "POST", "/v1/responses", [], b"{}")
    assert conn.closed


def test_fr_gate_06_take_is_last_in_first_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upstream_module, "_quiet", lambda connection: True)
    pool = upstream_module.UpstreamPool()
    up = upstream_module.Upstream("http", "127.0.0.1", 1)
    older, newer = FakeConnection(), FakeConnection()
    now = time.monotonic()
    pool._idle[up] = [(older, now), (newer, now)]
    assert pool._take(up) is newer and pool._take(up) is older and pool._take(up) is None


def _throwaway_tls(tmp_path: Path) -> tuple:
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
                    "-days", "1", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
                   check=True, capture_output=True)
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(str(cert), str(key))
    return server, ssl.create_default_context(cafile=str(cert))


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command to make a throwaway certificate")
def test_fr_gate_06_tls_connection_is_reused_and_a_stray_record_is_not_delivered(tmp_path: Path) -> None:
    def script(number: int, served: int, head: bytes) -> object:
        if (number, served) == (1, 2):
            def answer_then_stray(conn: socket.socket) -> bool:
                conn.sendall(always_ok(number, served, head))
                time.sleep(0.2)
                conn.sendall(ok(b'{"stray": "meant for no one"}'))
                return True
            return answer_then_stray
        return always_ok(number, served, head)

    with Rig(script, tls=_throwaway_tls(tmp_path)) as rig:
        replies = [rig.call() for _ in range(2)]
        time.sleep(0.5)
        replies.append(rig.call())
        log = rig.log(3)
    assert [json.loads(body) for _, body in replies] == [{"connection": 1, "request": 1}, {"connection": 1, "request": 2},
                                                         {"connection": 2, "request": 1}]
    assert [entry["upstream_reused"] for entry in log] == [False, True, False]
