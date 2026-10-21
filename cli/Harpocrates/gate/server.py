"""The gate server (FR-GATE-01): a threaded HTTP server on 127.0.0.1 and the handler that runs each request through
the checks in order. Every error path ends in a block; there is no override."""
from __future__ import annotations

import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from Harpocrates.gate.body_encoding import Zstd, decode_body, encode_body, load_zstd
from Harpocrates.gate.body_read import read_body
from Harpocrates.gate.config import BIND_HOST, CHATGPT_PATH_PREFIX, GateConfig
from Harpocrates.gate.errors import UPSTREAM_FAILED, error_response
from Harpocrates.gate.headers import upstream_request_headers
from Harpocrates.gate.relay import relay_response
from Harpocrates.gate.request_log import RequestLog
from Harpocrates.gate.request_scan import RequestScanner
from Harpocrates.gate.routes import Route, classify
from Harpocrates.gate.scan import Block, RequestHead, ScanStep, run_scan
from Harpocrates.gate.session_token import TOKEN_HEADER, check_token
from Harpocrates.gate.upstream import UpstreamError, UpstreamPool

_BODY_CODES = {"too-large": "HARPO-BLOCK-BODY-TOO-LARGE", "bad-framing": "HARPO-BLOCK-BODY-FRAMING",
               "truncated": "HARPO-BLOCK-BODY-TRUNCATED"}
_ENCODING_CODES = {"unsupported": "HARPO-BLOCK-ENCODING", "zstd-unavailable": "HARPO-BLOCK-ENCODING",
                   "corrupt": "HARPO-BLOCK-ENCODING-CORRUPT", "too-large": "HARPO-BLOCK-BODY-TOO-LARGE"}
_DRAIN_LIMIT = 1 << 20
_DRAIN_SECONDS = 1.0


class GateServer(ThreadingHTTPServer):
    """
    Description: The gate's HTTP server: one thread per connection, bound to 127.0.0.1.
    Parameters:
        config: the checked settings.
        token: the session token an agent must send.
        scan_step: the scan step.
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
        Output: None. Also makes the pool of idle provider connections (closed with the server).
        """
        super().__init__((config.host, config.port), GateHandler)
        self.config, self.token, self.scan_step, self.log, self.zstd = config, token, scan_step, log, zstd
        self.pool = UpstreamPool()

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

    def setup(self) -> None:
        """
        Description: Apply the agent-side timeout before the connection's streams are made.
        Parameters: None
        Output: None. A client that sends nothing for client_timeout seconds raises a timeout in the next read.
        """
        self.timeout = self.server.config.client_timeout
        super().setup()

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
        Output: None. Writes the answer and notes the status, decision and reason for the log; the connection is closed
            afterwards (after the log line is written, when part of the body must still be drained).
        """
        status, payload = error_response(route.provider, code)
        decision = {"HARPO-BLOCK-UPGRADE": "upgrade-refused", UPSTREAM_FAILED: "upstream-error"}.get(code, "blocked")
        self._facts.update(status=status, decision=decision, reason=code)
        self._responded = True
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self._drain = body_unread
        self.wfile.write(payload)

    def _forward(self, route: Route, body: bytes, encoding: str) -> None:
        """
        Description: Send a scanned request to the provider and stream the reply back to the agent.
        Parameters:
            route: the route of the request, which picks the upstream.
            body: the scanned body, already in the encoding the request arrived in.
            encoding: "identity", "gzip" or "zstd".
        Output: None. The agent gets the provider's reply unchanged; if the provider cannot be reached or does not answer
            in time it gets 502. A reply that breaks off mid-stream closes the agent's connection. Notes the sizes,
            timings, status and decision for the log.
        """
        config = self.server.config
        has_body = route.action != "pass"
        headers = upstream_request_headers(list(self.headers.items()), encoding, len(body) if has_body else None)
        self._facts["bytes_out"] = len(body) if has_body else 0
        provider, target = self._destination(route)
        upstream = config.upstream_for(provider)
        sent_at = time.monotonic()
        try:
            exchange = self.server.pool.exchange(upstream, config, self.command, target, headers,
                                                 body if has_body else None)
        except UpstreamError:
            return self._refuse(route, UPSTREAM_FAILED, body_unread=False)
        self._facts.update(ms_upstream=(time.monotonic() - sent_at) * 1000, upstream_reused=exchange.reused,
                           ms_connect=exchange.ms_connect)
        self._responded = True
        relayed = None
        try:
            relayed = relay_response(exchange.response, self.wfile, self.request_version == "HTTP/1.0")
        finally:
            self.server.pool.release(upstream, exchange, relayed is not None and relayed.ended == "complete")
        self._facts.update(status=relayed.status, decision="forwarded", bytes_back=relayed.bytes_back)
        if relayed.ended != "complete":
            self.close_connection = True

    def _destination(self, route: Route) -> tuple[str, str]:
        """
        Description: Choose where a request goes (G5 spec, part 2): an OpenAI route that carries chatgpt-account-id is a
            Codex ChatGPT login, which goes to the ChatGPT upstream with /v1 replaced by /backend-api/codex.
        Parameters:
            route: the request's route.
        Output: (provider for GateConfig.upstream_for, request target). Every other request keeps its provider and its
            target as received. Only the header's presence is read, never its value.
        """
        # Only /v1/responses and /v1/models reach here (routes.classify), so the prefix is always "/v1/".
        chatgpt = route.provider == "openai" and any(name.lower() == "chatgpt-account-id" for name in self.headers.keys())
        if chatgpt and self.path.startswith("/v1/"):
            return "chatgpt", CHATGPT_PATH_PREFIX + self.path[len("/v1"):]
        return route.provider, self.path

    def _head(self) -> RequestHead:
        """
        Description: Collect what the scan step is told about the request besides its body (decision 12).
        Parameters: None
        Output: A RequestHead with the headers as received (without the gate token header), the path and the query.
        """
        path, _, query = self.path.partition("?")
        kept = tuple((name, value) for name, value in self.headers.items() if name.lower() != TOKEN_HEADER)
        return RequestHead(kept, path, query)

    def _pipeline(self, route: Route) -> None:
        """
        Description: Run one request through the checks in order and forward it if every one passes.
        Parameters:
            route: the route of the request.
        Output: None. Checks the token first (SEC-03): a missing, repeated or wrong token gets 401 and nothing else is
            read. Then the route: a WebSocket upgrade gets 426 and any route the gate does not know gets 400, both
            without reading the body. Then the body is read (not at all on a route that takes none): bad framing,
            a body over the cap and a body cut short get 400. Then the body is decoded (identity, gzip, zstd): any other
            Content-Encoding, a corrupt stream and a body that passes the cap once decoded get 400. Then the scan
            step runs on the decoded body (on every route, an empty body included): an error, a timeout or a Block
            from it gets 400 (FR-GATE-03). What the step lets through goes to the provider and the reply streams back.
        """
        if not check_token(self.headers.get_all(TOKEN_HEADER, []), self.server.token):
            return self._refuse(route, "HARPO-BLOCK-TOKEN", body_unread=True)
        if route.action == "upgrade":
            return self._refuse(route, "HARPO-BLOCK-UPGRADE", body_unread=True)
        if route.action == "block":
            return self._refuse(route, "HARPO-BLOCK-ROUTE", body_unread=True)
        cap = 0 if route.action == "pass" else self.server.config.max_body
        raw = read_body(self.rfile, self.headers.get_all("Content-Length", []),
                        self.headers.get_all("Transfer-Encoding", []), cap)
        if raw.status != "ok":
            return self._refuse(route, _BODY_CODES[raw.status], body_unread=True)
        decoded = decode_body(raw.data, self.headers.get_all("Content-Encoding", []), cap, self.server.zstd)
        if decoded.status != "ok":
            return self._refuse(route, _ENCODING_CODES[decoded.status], body_unread=False)
        self._facts["bytes_in"] = len(decoded.body)
        scan_started = time.monotonic()
        outcome = run_scan(self.server.scan_step, decoded.body, route, self._head(), self.server.config.scan_timeout)
        self._facts["ms_scan"] = (time.monotonic() - scan_started) * 1000
        if isinstance(outcome, Block):
            return self._refuse(route, outcome.reason, body_unread=False)
        if outcome.media:
            self._facts["media"] = [{"kind": note.kind, "bytes": note.size} for note in outcome.media]
        self._forward(route, encode_body(outcome.body, decoded.encoding, self.server.zstd), decoded.encoding)

    def parse_request(self) -> bool:
        """
        Description: Parse the request line and headers, forgetting the previous request's headers first so a refusal
            never takes its error shape from them.
        Parameters: None
        Output: True when the request parsed; False when it was refused (send_error has already answered).
        """
        self.headers = None
        return super().parse_request()

    def send_error(self, code: int, message: Optional[str] = None, explain: Optional[str] = None) -> None:
        """
        Description: Answer a request the HTTP parser itself refused (a bad request line, an oversize line or header
            block, an unsupported version, a method the gate has no handler for) with the gate's own error instead of
            the standard library's HTML page.
        Parameters:
            code: the status the parser chose (ignored except that 501 means an unknown method).
            message: the parser's short message (ignored: it can echo request text).
            explain: the parser's long message (ignored).
        Output: None. Writes 400 (with a status line even when the request line did not parse) in the caller's error
            shape with HARPO-BLOCK-ROUTE (unknown method) or HARPO-BLOCK-REQUEST (anything else), logs one line, and
            closes the connection.
        """
        started = time.monotonic()
        self._responded = False
        self._drain = False
        self._facts = {"method": self.command or "", "route": "unknown"}
        headers = getattr(self, "headers", None)  # missing when the request line was too long to parse
        caller = "anthropic" if headers is not None and "anthropic-version" in headers else "openai"
        self.request_version = "HTTP/1.1"  # the parser leaves HTTP/0.9 here, which would send no status line
        try:
            self._refuse(Route("unknown", caller, "block"),
                         "HARPO-BLOCK-ROUTE" if code == 501 else "HARPO-BLOCK-REQUEST", body_unread=True)
        finally:
            self._facts["ms_total"] = (time.monotonic() - started) * 1000
            self.server.log.write(self._facts)
            if self._drain:
                self._discard_pending()

    def _serve(self) -> None:
        """
        Description: Answer one request, and log it.
        Parameters: None
        Output: None. Runs the request through _pipeline. An unexpected error is a block: 400 HARPO-BLOCK-INTERNAL when
            nothing was answered yet, otherwise the connection is closed with nothing added; the error text is never
            shown or logged. A socket error means the agent went away. Always writes one line to the request log.
        """
        started = time.monotonic()
        self._responded = False
        self._drain = False
        self._facts: dict[str, object] = {"method": self.command, "route": "unknown"}
        route = Route("unknown", "openai", "block")
        try:
            route = classify(self.command, self.path, {name.lower(): value for name, value in self.headers.items()})
            self._facts["route"] = route.name
            self._pipeline(route)
        except OSError:
            self.close_connection = True
        except Exception:  # fail closed: whatever went wrong, the request is not forwarded unscanned
            self.close_connection = True
            if self._responded:
                self._facts.update(decision="blocked", reason="HARPO-BLOCK-INTERNAL")
            else:
                self._refuse(route, "HARPO-BLOCK-INTERNAL", body_unread=True)
        finally:
            self._facts["ms_total"] = (time.monotonic() - started) * 1000
            self.server.log.write(self._facts)
            if self._drain:
                self._discard_pending()

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
        self.server.pool.close()

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


def start_gate(config: GateConfig, token: str, scan_step: Optional[ScanStep] = None,
               log_sink: Optional[Callable[[str], None]] = None) -> RunningGate:
    """
    Description: Bind 127.0.0.1 and start serving in a background thread.
    Parameters:
        config: the checked settings.
        token: the session token (see session_token.new_token).
        scan_step: the scan step, or None for a new RequestScanner: the real scanner, with a session key made now (SEC-02)
            that dies with this gate. Only a test passes another. There is no way to skip scanning (FR-GATE-03).
        log_sink: a function that takes one request-log line, or None to log nothing.
    Output: The running gate. Raises ValueError, before anything listens, when the settings do not name 127.0.0.1 or the
        socket ends up bound elsewhere; OSError when the port is taken.
    """
    if config.host != BIND_HOST:
        raise ValueError(f"host: the gate binds {BIND_HOST} only")
    server = GateServer(config, token, scan_step if scan_step is not None else RequestScanner(), RequestLog(log_sink),
                        load_zstd())
    if server.server_address[0] != BIND_HOST:  # belt and braces after the bind
        server.server_close()
        raise ValueError(f"host: the gate bound a socket other than {BIND_HOST}; refusing to serve")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    return RunningGate(server, thread)
