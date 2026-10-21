"""Helpers for the gate tests: a scripted raw-socket upstream, raw HTTP exchanges and a pass-through scan step.

Nothing here contacts a real provider. Every upstream is on 127.0.0.1: either bench/mock_provider.py or a
ScriptedUpstream below, which exists for the cases the mock cannot produce (a stream cut mid-chunk, a silent server,
a garbage reply, a TLS server with a self-signed certificate).
"""
from __future__ import annotations

import socket
import threading
from typing import Callable


def read_http_request(conn: socket.socket) -> tuple[bytes, bytes]:
    """Read one request from a socket: (head without the blank line, body by Content-Length or chunked)."""
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        piece = conn.recv(65536)
        if not piece:
            return buffer, b""
        buffer += piece
    head, _, rest = buffer.partition(b"\r\n\r\n")
    lowered = head.lower()
    if b"transfer-encoding: chunked" in lowered:
        while not rest.endswith(b"0\r\n\r\n"):
            piece = conn.recv(65536)
            if not piece:
                break
            rest += piece
        return head, rest
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    while len(rest) < length:
        piece = conn.recv(65536)
        if not piece:
            break
        rest += piece
    return head, rest[:length]


class ScriptedUpstream:
    """A loopback TCP server that runs script(conn, head, body) for each connection, then closes it."""

    def __init__(self, script: Callable[[socket.socket, bytes, bytes], None]) -> None:
        self.script = script
        self.heads: list[bytes] = []
        self.bodies: list[bytes] = []
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.host, self.port = self.listener.getsockname()
        self.url = f"http://127.0.0.1:{self.port}"
        self.threads: list[threading.Thread] = []
        self.accept_thread = threading.Thread(target=self._accept, daemon=True)
        self.accept_thread.start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            thread = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _serve(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(15)
            head, body = read_http_request(conn)
            self.heads.append(head)
            self.bodies.append(body)
            self.script(conn, head, body)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self.listener.close()

    def __enter__(self) -> "ScriptedUpstream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def free_closed_port() -> int:
    """A loopback port nothing listens on: bind, note the number, close."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def chunk(data: bytes) -> bytes:
    """One chunked-transfer chunk."""
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"
