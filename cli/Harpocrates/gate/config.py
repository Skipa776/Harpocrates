"""The gate's settings (FR-GATE-01): where it binds, where it forwards, and its size and time limits."""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import NamedTuple
from urllib.parse import urlsplit

BIND_HOST = "127.0.0.1"
MAX_BODY_BYTES = 64 * 1024 * 1024  # decision 9: applied after decoding
_HOST_CHARS = re.compile(r"[A-Za-z0-9.\-]+|[0-9A-Fa-f:.]+")
_DEFAULT_PORTS = {"http": 80, "https": 443}


class Upstream(NamedTuple):
    """
    Description: Where one provider's requests are forwarded.
    Parameters:
        scheme: "https", or "http" for a loopback host only.
        host: the host name, or the IP literal without brackets.
        port: the TCP port.
    Output: An immutable record built by parse_upstream.
    """

    scheme: str
    host: str
    port: int


def parse_upstream(url: str) -> Upstream:
    """
    Description: Read an upstream base URL and refuse any form the gate does not support.
    Parameters:
        url: scheme://host[:port], such as "https://api.anthropic.com" or "http://127.0.0.1:8081".
    Output: The Upstream. Raises ValueError for another scheme, a missing host, credentials, a path, a query, a
        fragment, a bad port, or plain http to a host that is not a loopback IP literal. The message never repeats the URL.
    """
    parts = urlsplit(url)
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
        raise ValueError("an upstream must be http:// or https:// followed by a host")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError("an upstream URL must not carry credentials")
    if parts.path not in ("", "/") or "?" in url or "#" in url:
        raise ValueError("an upstream URL must be only scheme://host[:port], with no path, query or fragment")
    try:
        declared = parts.port
    except ValueError:
        raise ValueError("an upstream URL has a port that is not a number") from None
    port = _DEFAULT_PORTS[parts.scheme] if declared is None else declared
    if not 0 < port < 65536:
        raise ValueError("an upstream URL has a port outside 1 to 65535")
    host = parts.hostname
    if not _HOST_CHARS.fullmatch(host):
        raise ValueError("an upstream URL has a host with characters a host name cannot hold")
    if parts.scheme == "http" and not _is_loopback_ip(host):
        raise ValueError("plain http is allowed only for a loopback IP address (127.0.0.1 or ::1); use https")
    return Upstream(parts.scheme, host, port)


def _is_loopback_ip(host: str) -> bool:
    """
    Description: Tell whether a host is a loopback IP literal, without any name lookup.
    Parameters:
        host: a host name or IP literal.
    Output: True for 127.0.0.0/8 and ::1; False for a name or any other address.
    """
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class GateConfig:
    """
    Description: Every setting the gate runs with. Checked when built, so a bad value never reaches a socket.
    Parameters:
        host: the bind address; only 127.0.0.1 is accepted.
        port: the TCP port, 0 to let the system choose.
        anthropic_upstream: base URL for Anthropic routes (default https://api.anthropic.com).
        openai_upstream: base URL for OpenAI routes (default https://api.openai.com).
        max_body: the largest request body, raw and after decoding, in bytes (default 64 MiB).
        scan_timeout: seconds the scan step may take before the request is blocked (default 5).
        connect_timeout: seconds to reach the provider before answering 502.
        read_timeout: seconds the provider may stay silent, before the first byte or mid-stream. Long, because a model
            can think a while; the agents have their own idle timeouts.
        client_timeout: seconds the agent may stay silent while sending a request.
    Output: An immutable record. Raises ValueError from the constructor when a value is not usable.
    """

    host: str = BIND_HOST
    port: int = 0
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    max_body: int = MAX_BODY_BYTES
    scan_timeout: float = 5.0
    connect_timeout: float = 10.0
    read_timeout: float = 600.0
    client_timeout: float = 120.0

    def __post_init__(self) -> None:
        """
        Description: Refuse a non-loopback bind, a bad port, a non-positive limit and an unusable upstream.
        Parameters: None
        Output: None. Raises ValueError naming the field.
        """
        if self.host != BIND_HOST:
            raise ValueError(f"host: the gate binds {BIND_HOST} only")
        if not 0 <= self.port <= 65535:
            raise ValueError("port: must be 0 to 65535")
        for field in ("max_body", "scan_timeout", "connect_timeout", "read_timeout", "client_timeout"):
            if not getattr(self, field) > 0:
                raise ValueError(f"{field}: must be greater than zero")
        parse_upstream(self.anthropic_upstream)
        parse_upstream(self.openai_upstream)

    def upstream_for(self, provider: str) -> Upstream:
        """
        Description: Pick the upstream for a provider.
        Parameters:
            provider: "anthropic" or "openai".
        Output: The Upstream parsed from that provider's URL. Raises ValueError for any other provider name.
        """
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"provider: {provider!r} is not anthropic or openai")
        return parse_upstream(self.anthropic_upstream if provider == "anthropic" else self.openai_upstream)
