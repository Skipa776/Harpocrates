"""The request to the provider (FR-GATE-06, transport): connect, send, and read the response head."""
from __future__ import annotations

import http.client
import re
import ssl
from typing import NamedTuple, Optional, Sequence

from Harpocrates.gate.config import GateConfig, Upstream, is_loopback_ip

_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")  # RFC 9110 token


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


def open_upstream(upstream: Upstream, config: GateConfig, method: str, target: str,
                  headers: Sequence[tuple[str, str]], body: Optional[bytes],
                  ssl_context: Optional[ssl.SSLContext] = None) -> UpstreamReply:
    """
    Description: Send one request to the provider and read the response head.
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
        connection.putrequest(method, target, skip_accept_encoding=True)
        for name, value in headers:
            connection.putheader(name, value)
        connection.endheaders(message_body=body)
        return UpstreamReply(connection, connection.getresponse())
    except (OSError, http.client.HTTPException, ValueError) as error:
        connection.close()
        raise UpstreamError(type(error).__name__) from None
