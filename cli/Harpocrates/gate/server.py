"""The gate server (FR-GATE-01): a threaded HTTP server on 127.0.0.1 and the handler that runs each request through
the checks in order. Every error path ends in a block; there is no override."""
from __future__ import annotations

import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from Harpocrates.gate.body_encoding import Zstd, load_zstd
from Harpocrates.gate.config import BIND_HOST, GateConfig
from Harpocrates.gate.errors import error_response
from Harpocrates.gate.request_log import RequestLog
from Harpocrates.gate.routes import Route, classify
from Harpocrates.gate.scan import ScanStep, default_scan_step
from Harpocrates.gate.session_token import TOKEN_HEADER, check_token

_DRAIN_LIMIT = 1 << 20
_DRAIN_SECONDS = 1.0


class GateServer(ThreadingHTTPServer):
    """
    Description: The gate's HTTP server: one thread per connection, bound to 127.0.0.1.
    Parameters:
        config: the checked settings.
        token: the session token an agent must send.
        scan_step: the scan step (the default raises, so nothing is forwarded).
        log: where one line per request goes.
        zstd: the zstd functions, or None when no zstd module is available.
    Output: A bound server. Raises OSError when the port is taken.
    """

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, config: GateConfig, token: str, scan_step: ScanStep, log: RequestLog, zstd: Optional[Zstd]) -> None:
        """
        Description: Bind the socket and keep what the handlers need.
        Parameters:
            config: the checked settings.
            token: the session token.
            scan_step: the scan step.
            log: the request log.
            zstd: the zstd functions or None.
        Output: None.
        """
        super().__init__((config.host, config.port), GateHandler)
        self.config, self.token, self.scan_step, self.log, self.zstd = config, token, scan_step, log, zstd

    def handle_error(self, request: object, client_address: object) -> None:
        """
        Description: Report an error that escaped a handler without printing its message or traceback, which can hold
            request text.
        Parameters:
            request: the socket (ignored).
            client_address: the peer (ignored).
        Output: None. Writes one line with the exception class name to stderr.
        """
        kind = sys.exc_info()[0]
        sys.stderr.write(f"harpo-gate: internal error: {kind.__name__ if kind else 'unknown'}\n")


class GateHandler(BaseHTTPRequestHandler):
    """
    Description: Handles the requests of one agent connection.
    Parameters:
        request: the connected socket (set by the server).
    Output: One response per request. Never forwards a request that has not passed every check.
    """

    protocol_version = "HTTP/1.1"
    server_version = "harpo-gate"
    sys_version = ""
    disable_nagle_algorithm = True  # an SSE event must not wait for a delayed ACK

    def version_string(self) -> str:
        """
        Description: Name the server in responses the gate writes itself.
        Parameters: None
        Output: "harpo-gate", without the Python version.
        """
        return self.server_version

    def log_message(self, format: str, *args: object) -> None:
        """
        Description: Silence the default access log, which would print the request line and so the query string.
        Parameters:
            format: the log format (ignored).
            *args: its arguments (ignored).
        Output: None.
        """

    def _discard_pending(self) -> None:
        """
        Description: After an answer was sent without reading the whole request, let the agent finish sending, so
            closing the connection does not reset it before the agent has read the answer.
        Parameters: None
        Output: None. Reads and drops at most 1 MiB for at most about a second per read; a socket error is ignored.
        """
        try:
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_WR)
            self.connection.settimeout(_DRAIN_SECONDS)
            left = _DRAIN_LIMIT
            while left > 0:
                piece = self.connection.recv(min(65536, left))
                if not piece:
                    break
                left -= len(piece)
        except OSError:
            pass

    def _refuse(self, route: Route, code: str, body_unread: bool) -> None:
        """
        Description: Answer with the gate's own error and close the connection.
        Parameters:
            route: the route of the request, which picks the error shape.
            code: the reason code.
            body_unread: True when part of the request body may still be on the connection.
        Output: None. Writes the answer; the connection is closed afterwards.
        """
        status, payload = error_response(route.provider, code)
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        if body_unread:
            self._discard_pending()

    def _serve(self) -> None:
        """
        Description: Answer one request.
        Parameters: None
        Output: None. Checks the token first (SEC-03): a missing, repeated or wrong token gets 401 and nothing else is
            read. At this stage nothing after the token check exists yet, so every other request is blocked.
        """
        lowered = {name.lower(): value for name, value in self.headers.items()}
        route = classify(self.command, self.path, lowered)
        if not check_token(self.headers.get_all(TOKEN_HEADER, []), self.server.token):
            return self._refuse(route, "HARPO-BLOCK-TOKEN", body_unread=True)
        self._refuse(route, "HARPO-BLOCK-INTERNAL", body_unread=True)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _serve  # noqa: N815 - http.server names


class RunningGate:
    """
    Description: A gate that is serving.
    Parameters:
        server: the bound server.
        thread: the thread running serve_forever.
    Output: An object with host, port, url and close(); usable as a context manager.
    """

    def __init__(self, server: GateServer, thread: threading.Thread) -> None:
        """
        Description: Wrap a started server.
        Parameters:
            server: the bound server.
            thread: the thread serving it.
        Output: None.
        """
        self.server, self.thread = server, thread
        self.host, self.port = server.server_address[0], server.server_address[1]

    @property
    def url(self) -> str:
        """
        Description: The base URL an agent should be pointed at.
        Parameters: None
        Output: "http://127.0.0.1:PORT".
        """
        return f"http://{self.host}:{self.port}"

    def close(self) -> None:
        """
        Description: Stop serving and release the port.
        Parameters: None
        Output: None; safe to call twice.
        """
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=10)
        self.server.server_close()

    def __enter__(self) -> "RunningGate":
        """
        Description: Enter the context manager.
        Parameters: None
        Output: This gate.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """
        Description: Leave the context manager, closing the gate.
        Parameters:
            *exc: the exception information (ignored).
        Output: None.
        """
        self.close()


def start_gate(config: GateConfig, token: str, scan_step: ScanStep = default_scan_step,
               log_sink: Optional[Callable[[str], None]] = None) -> RunningGate:
    """
    Description: Bind 127.0.0.1 and start serving in a background thread.
    Parameters:
        config: the checked settings.
        token: the session token (see session_token.new_token).
        scan_step: the scan step. The default raises, so every request is blocked; only a test passes another.
        log_sink: a function that takes one request-log line, or None to log nothing.
    Output: The running gate. Raises ValueError, before anything listens, when the settings do not name 127.0.0.1 or the
        socket ends up bound elsewhere; OSError when the port is taken.
    """
    if config.host != BIND_HOST:
        raise ValueError(f"host: the gate binds {BIND_HOST} only")
    server = GateServer(config, token, scan_step, RequestLog(log_sink), load_zstd())
    if server.server_address[0] != BIND_HOST:  # belt and braces after the bind
        server.server_close()
        raise ValueError(f"host: the gate bound a socket other than {BIND_HOST}; refusing to serve")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    return RunningGate(server, thread)
