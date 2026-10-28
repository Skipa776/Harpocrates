"""Header rules (FR-INSTALL-05): what changes on the way to the provider and back, and nothing else."""
from __future__ import annotations

from typing import Optional, Sequence

from Harpocrates.gate.session_token import TOKEN_HEADER

# RFC 9110 section 7.6.1 hop-by-hop headers. A header only named in "Connection" is NOT removed: the fixed list is
# used on purpose, so a Connection value can never strip a credential header.
HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
                        "te", "trailer", "transfer-encoding", "upgrade"})
# Dropped from the request because the gate replaces them: Host (the upstream's), the body's length and encoding (fixed
# after the scan step), Expect (the gate already answered it) and the gate token (SEC-03).
_REPLACED = frozenset({"host", "content-length", "content-encoding", "expect", TOKEN_HEADER})


def upstream_request_headers(pairs: Sequence[tuple[str, str]], encoding: str,
                             body_length: Optional[int]) -> list[tuple[str, str]]:
    """
    Description: Build the header list for the request sent to the provider.
    Parameters:
        pairs: the request headers as received: (name, value) in order, duplicates and case kept.
        encoding: "identity", "gzip" or "zstd": the encoding the forwarded body is in.
        body_length: the forwarded body's size in bytes, or None when the request has no body.
    Output: A new list. Every header is kept in order with its name and value untouched, except the hop-by-hop headers,
        Host, Content-Length, Content-Encoding, Expect and the gate token header, which are removed. Then
        Content-Encoding is added for gzip and zstd, and Content-Length when body_length is not None. Auth headers
        are never touched.
    """
    out = [(name, value) for name, value in pairs
           if name.lower() not in HOP_BY_HOP and name.lower() not in _REPLACED]
    if encoding != "identity":
        out.append(("Content-Encoding", encoding))
    if body_length is not None:
        out.append(("Content-Length", str(body_length)))
    return out


def client_response_headers(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Description: Pick the provider's response headers to pass back to the agent.
    Parameters:
        pairs: the response headers as received: (name, value) in order.
    Output: A new list of the same pairs without the hop-by-hop headers (Connection, Keep-Alive, Transfer-Encoding and
        the rest of HOP_BY_HOP), which belong to one connection and are set again by the gate. Content-Length,
        Content-Encoding, retry-after, rate-limit headers and every other header pass unchanged.
    """
    return [(name, value) for name, value in pairs if name.lower() not in HOP_BY_HOP]
