"""FR-GATE-06 (transport): the request to the provider. Headers go out in order with duplicates, the body goes out
unchanged, the certificate is always verified for https, and every failure before a response is one UpstreamError."""
from __future__ import annotations

import hashlib
import shutil
import socket
import ssl
import subprocess
import threading
from pathlib import Path

import pytest
from gate_support import ScriptedUpstream, free_closed_port

from bench import mock_provider as mock
from Harpocrates.gate.config import GateConfig, Upstream
from Harpocrates.gate.upstream import UpstreamError, make_ssl_context, open_upstream

BODY = b'{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
FAST = GateConfig(connect_timeout=2, read_timeout=0.5)


@pytest.fixture
def mock_server():
    server = mock.start_server()
    yield server
    server.close()


def target_of(server) -> Upstream:
    return Upstream("http", server.host, server.port)


def test_fr_gate_06_request_reaches_the_upstream_with_path_headers_and_body(mock_server) -> None:
    headers = [("X-Test-One", "a"), ("x-test-one", "b"), ("Content-Type", "application/json"),
               ("Content-Length", str(len(BODY)))]
    conn, response = open_upstream(target_of(mock_server), FAST, "POST", "/v1/messages?beta=true", headers, BODY)
    try:
        assert response.status == 200
        assert b"Mock reply." in response.read()
    finally:
        conn.close()
    (record,) = mock_server.records()
    assert record["method"] == "POST" and record["path"] == "/v1/messages?beta=true"
    assert record["body_sha256"] == hashlib.sha256(BODY).hexdigest()
    sent = [tuple(h) for h in record["headers"]]
    assert [h for h in sent if h[0].lower() == "x-test-one"] == [("X-Test-One", "a"), ("x-test-one", "b")]
    assert ("Host", f"{mock_server.host}:{mock_server.port}") in sent


def test_fr_gate_06_no_accept_encoding_is_added_and_a_get_has_no_body(mock_server) -> None:
    conn, response = open_upstream(target_of(mock_server), FAST, "GET", "/v1/models", [("anthropic-version", "x")], None)
    try:
        assert response.status == 200
        response.read()
    finally:
        conn.close()
    (record,) = mock_server.records()
    names = [h[0].lower() for h in record["headers"]]
    assert "accept-encoding" not in names and "content-length" not in names
    assert record["body_size"] == 0


def test_fr_gate_06_a_redirect_is_returned_and_never_followed() -> None:
    def redirect(conn, head, body):
        conn.sendall(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/elsewhere\r\nContent-Length: 0\r\n\r\n")

    with ScriptedUpstream(redirect) as up:
        conn, response = open_upstream(Upstream("http", up.host, up.port), FAST, "GET", "/v1/models", [], None)
        try:
            assert response.status == 302 and response.getheader("Location") == "http://127.0.0.1:1/elsewhere"
        finally:
            conn.close()
        assert len(up.heads) == 1


def test_fr_gate_06_a_refused_connection_is_an_upstream_error() -> None:
    with pytest.raises(UpstreamError):
        open_upstream(Upstream("http", "127.0.0.1", free_closed_port()), FAST, "POST", "/v1/messages", [], BODY)


def test_fr_gate_06_a_server_that_never_answers_times_out_as_an_upstream_error() -> None:
    gate = threading.Event()
    with ScriptedUpstream(lambda conn, head, body: gate.wait(5)) as up:
        with pytest.raises(UpstreamError):
            open_upstream(Upstream("http", up.host, up.port), GateConfig(connect_timeout=1, read_timeout=0.3),
                          "POST", "/v1/messages", [("Content-Length", "2")], b"{}")
        gate.set()


@pytest.mark.parametrize("reply", [b"", b"garbage\r\n\r\n", b"HTTP/1.1 abc OK\r\n\r\n", b"\x00\x01\x02"])
def test_fr_gate_06_no_reply_or_a_garbage_reply_is_an_upstream_error(reply: bytes) -> None:
    with ScriptedUpstream(lambda conn, head, body: conn.sendall(reply) if reply else None) as up:
        with pytest.raises(UpstreamError):
            open_upstream(Upstream("http", up.host, up.port), FAST, "POST", "/v1/messages", [], BODY)


def test_fr_gate_06_an_upstream_error_names_only_a_kind_never_the_host_or_a_header() -> None:
    secret_headers = [("x-api-key", "FAKE-KEY-SHOULD-NOT-APPEAR")]
    with pytest.raises(UpstreamError) as caught:
        open_upstream(Upstream("http", "127.0.0.1", free_closed_port()), FAST, "POST", "/v1/messages?k=FAKE-Q",
                      secret_headers, BODY)
    text = str(caught.value)
    assert "FAKE" not in text and "127.0.0.1" not in text and "/v1/" not in text


@pytest.mark.parametrize("name", ["Bad Name", "", "a:b", "x\r\ny", " lead", "caf\u00e9"])
def test_fr_gate_06_a_bad_header_name_is_an_upstream_error_and_nothing_is_sent(mock_server, name: str) -> None:
    with pytest.raises(UpstreamError):
        open_upstream(target_of(mock_server), FAST, "POST", "/v1/messages", [(name, "v")], BODY)
    assert mock_server.records() == []


def test_fr_gate_06_a_header_value_with_a_line_break_is_an_upstream_error(mock_server) -> None:
    with pytest.raises(UpstreamError):
        open_upstream(target_of(mock_server), FAST, "POST", "/v1/messages", [("X-A", "v\r\nX-Injected: 1")], BODY)
    assert mock_server.records() == []


def test_fr_gate_06_plain_http_to_a_non_loopback_host_is_refused_before_connecting() -> None:
    with pytest.raises(ValueError, match="loopback"):
        open_upstream(Upstream("http", "example.com", 80), FAST, "GET", "/v1/models", [], None)


def test_fr_gate_06_the_tls_context_always_verifies_the_certificate_and_name() -> None:
    context = make_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command to make a throwaway certificate")
def test_fr_gate_06_a_certificate_the_system_does_not_trust_fails_and_nothing_is_sent(tmp_path: Path) -> None:
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
                    "-days", "1", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
                   check=True, capture_output=True)
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(cert), str(key))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    received: list[bytes] = []

    def serve() -> None:
        while True:
            try:
                raw, _ = listener.accept()
            except OSError:
                return
            try:
                with server_ctx.wrap_socket(raw, server_side=True) as tls:
                    received.append(tls.recv(65536))
                    tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            except (ssl.SSLError, OSError):
                pass

    threading.Thread(target=serve, daemon=True).start()
    try:
        target = Upstream("https", "127.0.0.1", port)
        with pytest.raises(UpstreamError):
            open_upstream(target, FAST, "POST", "/v1/messages", [("x-api-key", "FAKE")], BODY)
        assert received == []  # the handshake failed before any request byte was sent
        trusting = ssl.create_default_context(cafile=str(cert))  # control: the same server works when trusted
        conn, response = open_upstream(target, FAST, "POST", "/v1/messages", [], BODY, ssl_context=trusting)
        try:
            assert response.status == 200 and response.read() == b"ok"
        finally:
            conn.close()
        assert len(received) == 1
    finally:
        listener.close()
