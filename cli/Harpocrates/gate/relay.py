"""The response path (FR-GATE-06): the provider's reply goes back to the agent as it arrives, unchanged."""
from __future__ import annotations

import http.client
from typing import BinaryIO, NamedTuple

from Harpocrates.gate.headers import client_response_headers

_PIECE = 65536


class Relayed(NamedTuple):
    """
    Description: How relaying one response went.
    Parameters:
        status: the provider's HTTP status.
        bytes_back: how many body bytes were written to the agent (framing not counted).
        ended: "complete"; "upstream-drop" (the provider's stream broke, so the agent's was left unfinished); or
            "client-gone" (the agent hung up).
    Output: An immutable record built by relay_response. When ended is not "complete" the caller must close the
        agent's connection.
    """

    status: int
    bytes_back: int
    ended: str


def _head(response: http.client.HTTPResponse, framing: str, declared: int) -> bytes:
    """
    Description: Write the status line and headers of the reply for the agent.
    Parameters:
        response: the provider's response, head read.
        framing: "none" (no body), "length", "chunked" or "close".
        declared: the provider's Content-Length for "length" framing, otherwise ignored.
    Output: The head as bytes ending in a blank line. Headers keep their order and values; the hop-by-hop headers are
        removed; Content-Length is written once for "length" and not at all for the other body framings; "chunked" adds
        Transfer-Encoding: chunked.
    """
    pairs = client_response_headers(response.getheaders())
    lines = [f"HTTP/1.1 {response.status} {response.reason}"]
    length_written = False
    for name, value in pairs:
        if name.lower() == "content-length" and framing in ("length", "chunked", "close"):
            if framing == "length" and not length_written:
                lines.append(f"{name}: {declared}")
                length_written = True
            continue
        lines.append(f"{name}: {value}")
    if framing == "chunked":
        lines.append("Transfer-Encoding: chunked")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1", "replace")


def relay_response(response: http.client.HTTPResponse, out: BinaryIO, http10: bool) -> Relayed:
    """
    Description: Send the provider's response to the agent, writing and flushing each piece as it is read, so a stream
        is never held back.
    Parameters:
        response: the provider's response, head read and body unread.
        out: the agent's output stream.
        http10: True when the agent speaks HTTP/1.0, which cannot take chunked bodies.
    Output: A Relayed. Status, headers and body pass unchanged; a body of unknown length is sent chunked (raw for
        HTTP/1.0). If the provider's stream breaks (a reset, a timeout, a cut chunk, or fewer bytes than the
        Content-Length) the relay stops without writing a terminating chunk, so the agent sees a truncated stream and
        never an invented ending. If the agent hangs up the relay stops quietly. Never raises for a connection error.
    """
    status = response.status
    bodiless = status in (204, 304) or 100 <= status < 200
    declared = response.length if response.length is not None else -1
    if bodiless:
        framing = "none"
    elif response.chunked or response.length is None:
        framing = "close" if http10 else "chunked"
    else:
        framing = "length"
    try:
        out.write(_head(response, framing, declared))
        out.flush()
    except OSError:
        return Relayed(status, 0, "client-gone")
    if framing == "none":
        return Relayed(status, 0, "complete")
    sent = 0
    while True:
        try:
            piece = response.read1(_PIECE)
        except (http.client.HTTPException, OSError):
            return Relayed(status, sent, "upstream-drop")
        if not piece:
            break
        try:
            out.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n" if framing == "chunked" else piece)
            out.flush()
        except OSError:
            return Relayed(status, sent, "client-gone")
        sent += len(piece)
    if framing == "length" and sent != declared:
        return Relayed(status, sent, "upstream-drop")
    if framing == "chunked":
        try:
            out.write(b"0\r\n\r\n")
            out.flush()
        except OSError:
            return Relayed(status, sent, "client-gone")
    return Relayed(status, sent, "complete")
