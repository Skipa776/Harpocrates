"""The request to the provider (FR-GATE-06, transport): connect, send, and read the response head; and the pool of idle
connections reused across requests (G5 spec, part 3, NFR-01)."""
from __future__ import annotations

import http.client
import re
import select
import ssl
import threading
import time
from typing import NamedTuple, Optional, Sequence

from Harpocrates.gate.config import GateConfig, Upstream, is_loopback_ip

_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")  # RFC 9110 token
POOL_MAX_IDLE = 4  # idle connections kept per upstream
POOL_IDLE_SECONDS = 30.0  # an idle connection older than this is closed, not reused
# Failures sending on a reused connection that mean it went stale while idle: retried once on a new connection. While
# reading the reply only RemoteDisconnected (closed with no reply byte) is retried; anything later may follow a processed
# request.
_STALE_SEND = (ConnectionError, ssl.SSLEOFError, ssl.SSLZeroReturnError)


class UpstreamError(Exception):
    """
    Description: The provider could not be reached or did not answer with a readable response head.
    Parameters:
        kind: the exception class name that caused it, for the log. Never a host, header or message from the cause.
    Output: An exception whose text is the kind only.
    """


class UpstreamReply(NamedTuple):
    """
    Description: An open exchange with the provider.
    Parameters:
        connection: the connection; the caller closes it.
        response: the response, with its head read and its body not yet read.
    Output: An immutable pair built by open_upstream.
    """

    connection: http.client.HTTPConnection
    response: http.client.HTTPResponse


def make_ssl_context() -> ssl.SSLContext:
    """
    Description: Make the TLS settings for https upstreams: the system trust store, certificate and host name checked.
    Parameters: None
    Output: A default client SSLContext (verify_mode CERT_REQUIRED, check_hostname True). There is no way to turn
        verification off.
    """
    return ssl.create_default_context()


def _connect(upstream: Upstream, config: GateConfig,
             ssl_context: Optional[ssl.SSLContext]) -> http.client.HTTPConnection:
    """
    Description: Open a new connection to the provider.
    Parameters:
        upstream: where to connect.
        config: supplies connect_timeout and read_timeout.
        ssl_context: TLS settings, or None for the system trust store.
    Output: A connected HTTPConnection with the read timeout set. Raises ValueError for plain http to a host that is not
        loopback, before connecting; OSError and http.client.HTTPException from the connect (the connection is closed).
    """
    if upstream.scheme == "http":
        if not is_loopback_ip(upstream.host):
            raise ValueError("plain http is allowed only to a loopback address")
        connection: http.client.HTTPConnection = http.client.HTTPConnection(
            upstream.host, upstream.port, timeout=config.connect_timeout)
    else:
        connection = http.client.HTTPSConnection(
            upstream.host, upstream.port, timeout=config.connect_timeout, context=ssl_context or make_ssl_context())
    try:
        connection.connect()
        connection.sock.settimeout(config.read_timeout)
    except BaseException:
        connection.close()
        raise
    return connection


def _put(connection: http.client.HTTPConnection, method: str, target: str, headers: Sequence[tuple[str, str]],
         body: Optional[bytes]) -> None:
    """
    Description: Send one request on an open connection.
    Parameters:
        connection: the connection.
        method: the HTTP method.
        target: the request target with its query string.
        headers: the request headers, in order, duplicates kept.
        body: the request body, or None.
    Output: None. Raises OSError, http.client.HTTPException or ValueError; the caller closes.
    """
    connection.putrequest(method, target, skip_accept_encoding=True)
    for name, value in headers:
        connection.putheader(name, value)
    connection.endheaders(message_body=body)


def _send(connection: http.client.HTTPConnection, method: str, target: str, headers: Sequence[tuple[str, str]],
          body: Optional[bytes]) -> http.client.HTTPResponse:
    """
    Description: Send one request on an open connection and read the response head.
    Parameters:
        connection: the connection.
        method: the HTTP method.
        target: the request target with its query string.
        headers: the request headers, in order, duplicates kept.
        body: the request body, or None.
    Output: The response, head read. Raises OSError, http.client.HTTPException or ValueError; the caller closes.
    """
    _put(connection, method, target, headers, body)
    return connection.getresponse()


def _fresh(upstream: Upstream, config: GateConfig, ssl_context: Optional[ssl.SSLContext]) -> http.client.HTTPConnection:
    """
    Description: Open a new connection, turning every failure into the gate's own error.
    Parameters:
        upstream: where to connect.
        config: supplies the timeouts.
        ssl_context: TLS settings, or None for the system trust store.
    Output: The connection. Raises ValueError for plain http to a host that is not loopback (before connecting), and
        UpstreamError naming only the exception kind for anything else.
    """
    try:
        return _connect(upstream, config, ssl_context)
    except ValueError:
        if upstream.scheme == "http" and not is_loopback_ip(upstream.host):
            raise
        raise UpstreamError("ValueError") from None
    except (OSError, http.client.HTTPException) as error:
        raise UpstreamError(type(error).__name__) from None


def _quiet(connection: http.client.HTTPConnection) -> bool:
    """
    Description: Tell whether an idle connection is still open and has nothing to read. A peer that closed it, sent a TLS
        alert, an idle-timeout notice (such as a 408) or any stray reply makes it readable; its bytes must never be read
        as the reply to the next request.
    Parameters:
        connection: an idle connection.
    Output: True when it may be reused.
    """
    sock = connection.sock
    if sock is None:
        return False
    if isinstance(sock, ssl.SSLSocket) and sock.pending():
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return False
    return not readable


def open_upstream(upstream: Upstream, config: GateConfig, method: str, target: str,
                  headers: Sequence[tuple[str, str]], body: Optional[bytes],
                  ssl_context: Optional[ssl.SSLContext] = None) -> UpstreamReply:
    """
    Description: Send one request to the provider on a new connection and read the response head.
    Parameters:
        upstream: where to connect.
        config: supplies connect_timeout and read_timeout.
        method: the HTTP method.
        target: the request target with its query string, forwarded as received.
        headers: the request headers, in order, duplicates kept. Host is added from the upstream; Accept-Encoding is
            not added; Content-Length comes from the headers given.
        body: the request body, or None when there is none.
        ssl_context: TLS settings for tests that trust a throwaway certificate; the default verifies against the
            system trust store.
    Output: An UpstreamReply. Raises UpstreamError for a refused or timed-out connection, a failed TLS handshake
        (including an untrusted certificate: there is no fallback), a header name that is not an HTTP token or a value that
        cannot be sent (nothing is connected for a bad name), a silent server and an unreadable response head. Raises ValueError, before any connection, for plain http to a host that is not loopback.
    """
    if any(not _HEADER_NAME.fullmatch(name) for name, _ in headers):
        raise UpstreamError("InvalidHeaderName")
    connection = _fresh(upstream, config, ssl_context)
    try:
        return UpstreamReply(connection, _send(connection, method, target, headers, body))
    except (OSError, http.client.HTTPException, ValueError) as error:
        connection.close()
        raise UpstreamError(type(error).__name__) from None


class Exchange(NamedTuple):
    """
    Description: An open exchange from the pool.
    Parameters:
        connection: the connection; hand it back with UpstreamPool.release when the reply was read through.
        response: the response, head read, body unread.
        reused: whether the connection that answered came from the pool.
        ms_connect: milliseconds spent opening a new connection (TCP and TLS); 0 when the pooled one answered.
    Output: An immutable record built by UpstreamPool.exchange.
    """

    connection: http.client.HTTPConnection
    response: http.client.HTTPResponse
    reused: bool
    ms_connect: float


class UpstreamPool:
    """
    Description: Idle provider connections kept for reuse, per upstream (scheme, host, port): last in, first out, at most
        POOL_MAX_IDLE each, none older than POOL_IDLE_SECONDS. Connections carry no credentials (auth is a header on each
        request), so one pool serves every agent session of a gate. Safe from many threads.
    Parameters:
        ssl_context: TLS settings for new https connections, or None for the system trust store.
    Output: An object with exchange(), release() and close().
    """

    def __init__(self, ssl_context: Optional[ssl.SSLContext] = None) -> None:
        """
        Description: Start with no idle connections.
        Parameters:
            ssl_context: TLS settings, or None.
        Output: None.
        """
        self._ssl_context = ssl_context
        self._idle: dict[Upstream, list[tuple[http.client.HTTPConnection, float]]] = {}
        self._lock = threading.Lock()
        self._closed = False

    def _take(self, upstream: Upstream) -> Optional[http.client.HTTPConnection]:
        """
        Description: Take the most recently used idle connection to an upstream, closing any that are too old or that
            have something to read (see _quiet).
        Parameters:
            upstream: the upstream.
        Output: A connection, or None when none is usable.
        """
        with self._lock:
            now = time.monotonic()
            idle = self._idle.get(upstream, [])
            fresh = [(connection, since) for connection, since in idle if now - since <= POOL_IDLE_SECONDS]
            expired = [connection for connection, since in idle if now - since > POOL_IDLE_SECONDS]
            self._idle[upstream] = fresh
        for connection in expired:
            connection.close()
        while True:
            with self._lock:
                fresh = self._idle.get(upstream, [])
                found = fresh.pop()[0] if fresh else None
            if found is None or _quiet(found):
                return found
            found.close()  # readable while idle: closed, sent a notice or a stray reply

    def exchange(self, upstream: Upstream, config: GateConfig, method: str, target: str,
                 headers: Sequence[tuple[str, str]], body: Optional[bytes]) -> Exchange:
        """
        Description: Send one request, on an idle connection when there is one.
        Parameters:
            upstream: where to send it.
            config: supplies the timeouts.
            method: the HTTP method.
            target: the request target with its query string.
            headers: the request headers, in order, duplicates kept.
            body: the request body (already scanned), or None.
        Output: An Exchange. A reused connection that proves stale (see _on_pooled) is closed and the same bytes are
            sent once more on a new connection; a new connection that fails raises UpstreamError, as open_upstream does.
            Any other failure on a reused connection (a timeout, a reset mid-reply) raises UpstreamError without a retry.
            ValueError for plain http to a non-loopback host, before connecting.
        """
        if any(not _HEADER_NAME.fullmatch(name) for name, _ in headers):
            raise UpstreamError("InvalidHeaderName")
        pooled = self._take(upstream)
        if pooled is not None:
            reply = self._on_pooled(pooled, method, target, headers, body)
            if reply is not None:
                return Exchange(pooled, reply, True, 0.0)
        started = time.monotonic()
        connection = _fresh(upstream, config, self._ssl_context)
        ms_connect = (time.monotonic() - started) * 1000
        try:
            return Exchange(connection, _send(connection, method, target, headers, body), False, ms_connect)
        except (OSError, http.client.HTTPException, ValueError) as error:
            connection.close()
            raise UpstreamError(type(error).__name__) from None

    @staticmethod
    def _on_pooled(connection: http.client.HTTPConnection, method: str, target: str,
                   headers: Sequence[tuple[str, str]], body: Optional[bytes]) -> Optional[http.client.HTTPResponse]:
        """
        Description: Try one request on a pooled connection.
        Parameters:
            connection: the pooled connection.
            method: the HTTP method.
            target: the request target.
            headers: the request headers.
            body: the request body, or None.
        Output: The response, head read; or None when the connection proved stale, so the caller tries once more on a new
            one: a stale failure while sending (_STALE_SEND), or a close with no reply byte (RemoteDisconnected) while
            reading. Raises UpstreamError for any other failure, with no retry, since the provider may have processed the
            request. The connection is closed on every failure.
        """
        try:
            _put(connection, method, target, headers, body)
        except _STALE_SEND:
            connection.close()
            return None
        except (OSError, http.client.HTTPException, ValueError) as error:
            connection.close()
            raise UpstreamError(type(error).__name__) from None
        try:
            return connection.getresponse()
        except http.client.RemoteDisconnected:
            connection.close()
            return None
        except (OSError, http.client.HTTPException, ValueError) as error:
            connection.close()
            raise UpstreamError(type(error).__name__) from None

    def release(self, upstream: Upstream, exchange: Exchange, complete: bool) -> None:
        """
        Description: Hand a connection back after its reply: keep it for reuse only when it is safe, otherwise close it.
        Parameters:
            upstream: where it is connected.
            exchange: the exchange it answered.
            complete: True when the reply was relayed to its framed end (relay ended "complete").
        Output: None. Kept only when complete, the response is fully read and closed, the reply was HTTP/1.1, the
            connection was not told to close (Connection: close), the pool is open and holds fewer than POOL_MAX_IDLE
            for this upstream. Every other case closes the connection.
        """
        connection, response = exchange.connection, exchange.response
        if complete and response.length == 0 and not response.isclosed():
            # A reply read to its end: Python 3.10's read1 leaves one read to its Content-Length open (3.11 closes it),
            # and a reply with no body is never read. _quiet still checks the connection before it is reused.
            response.close()
        reusable = (complete and response.isclosed() and response.version == 11 and not response.will_close
                    and connection.sock is not None)
        if reusable:
            with self._lock:
                idle = self._idle.get(upstream, [])
                if not self._closed and len(idle) < POOL_MAX_IDLE:
                    self._idle[upstream] = idle + [(connection, time.monotonic())]
                    return
        connection.close()

    def close(self) -> None:
        """
        Description: Close every idle connection and keep no more.
        Parameters: None
        Output: None; safe to call twice.
        """
        with self._lock:
            self._closed = True
            idle = [connection for entries in self._idle.values() for connection, _ in entries]
            self._idle.clear()
        for connection in idle:
            connection.close()
