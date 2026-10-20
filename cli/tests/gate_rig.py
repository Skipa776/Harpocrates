"""The whole gate in front of the mock provider, for the end-to-end gate tests: both on 127.0.0.1, with a client
that talks to the gate. Nothing here contacts a real provider."""
from __future__ import annotations

import http.client
import socket
from contextlib import contextmanager
from typing import Iterable, Iterator, NamedTuple, Optional

from bench import mock_provider as mock
from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.scan import Forward, default_scan_step
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token


def passthrough_scan(body: bytes, route: object) -> Forward:
    """A scan step that lets everything through. Only tests use it: the gate itself ships no such step."""
    return Forward(body)


class Reply(NamedTuple):
    """What a client got back from the gate."""

    status: int
    headers: list
    body: bytes

    def header(self, name: str) -> Optional[str]:
        for key, value in self.headers:
            if key.lower() == name.lower():
                return value
        return None


class Rig:
    """The gate in front of the mock provider, both on 127.0.0.1, plus a client that talks to the gate."""

    def __init__(self, upstream, gate, token: str, log_lines: list) -> None:
        self.mock, self.gate, self.token, self.log_lines = upstream, gate, token, log_lines

    def call(self, method: str, target: str, body: Optional[bytes] = None, headers: Optional[dict] = None,
             token: object = True, chunks: Optional[Iterable[bytes]] = None, timeout: float = 15) -> Reply:
        """One request over a fresh connection. token=True sends the right token, False none, a str that value."""
        sent = dict(headers or {})
        if token is True:
            sent[TOKEN_HEADER] = self.token
        elif isinstance(token, str):
            sent[TOKEN_HEADER] = token
        conn = http.client.HTTPConnection(self.gate.host, self.gate.port, timeout=timeout)
        try:
            if chunks is not None:
                conn.request(method, target, body=iter(chunks), headers={**sent, "Transfer-Encoding": "chunked"},
                             encode_chunked=True)
            else:
                conn.request(method, target, body=body, headers=sent)
            response = conn.getresponse()
            return Reply(response.status, response.getheaders(), response.read())
        finally:
            conn.close()

    def raw(self, data: bytes, half_close: bool = True, timeout: float = 15) -> bytes:
        """Send raw bytes to the gate and return everything it sends back until it closes."""
        with socket.create_connection((self.gate.host, self.gate.port), timeout=timeout) as sock:
            sock.sendall(data)
            if half_close:
                sock.shutdown(socket.SHUT_WR)
            received = b""
            try:
                while True:
                    piece = sock.recv(65536)
                    if not piece:
                        return received
                    received += piece
            except (TimeoutError, ConnectionResetError):
                return received


@contextmanager
def running_rig(scan_step=default_scan_step, mock_kwargs: Optional[dict] = None, **config: object) -> Iterator[Rig]:
    """Start the mock provider and a gate pointed at it. The default scan step is the gate's own (the stub that
    raises); a test that wants requests to go through passes passthrough_scan."""
    upstream = mock.start_server(**(mock_kwargs or {}))
    token, log_lines = new_token(), []
    settings = GateConfig(anthropic_upstream=upstream.url, openai_upstream=upstream.url, **config)
    gate = start_gate(settings, token, scan_step=scan_step, log_sink=log_lines.append)
    try:
        yield Rig(upstream, gate, token, log_lines)
    finally:
        gate.close()
        upstream.close()
