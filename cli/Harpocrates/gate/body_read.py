"""Reading a request body off the connection (FR-GATE-01): by Content-Length or chunked, bounded by the cap."""
from __future__ import annotations

import re
from typing import BinaryIO, NamedTuple, Optional, Sequence

_PIECE = 1 << 20
_LINE_LIMIT = 4096
_TRAILER_LIMIT = 64
_HEX = re.compile(rb"[0-9A-Fa-f]{1,16}")
_MAX_LENGTH_DIGITS = 18
_CRLF = b"\r\n"


class BodyRead(NamedTuple):
    """
    Description: The outcome of reading one request body.
    Parameters:
        status: "ok"; "too-large" (over the cap); "bad-framing" (a Content-Length or chunk that is not valid, or both
            framings at once); "truncated" (the sender stopped, timed out or hung up before the end).
        data: the body bytes when status is "ok", otherwise b"".
    Output: An immutable record built by read_body.
    """

    status: str
    data: bytes


def _fail(status: str) -> BodyRead:
    """
    Description: Build the result for any outcome other than ok.
    Parameters:
        status: too-large, bad-framing or truncated.
    Output: A BodyRead with that status and no data.
    """
    return BodyRead(status, b"")


def _read_exact(rfile: BinaryIO, count: int) -> Optional[bytes]:
    """
    Description: Read exactly count bytes, in pieces so a large body never needs one huge read call.
    Parameters:
        rfile: the connection's input stream.
        count: how many bytes to read.
    Output: The bytes, or None when the stream ends first. A timeout or reset propagates as OSError.
    """
    pieces: list[bytes] = []
    remaining = count
    while remaining > 0:
        piece = rfile.read(min(remaining, _PIECE))
        if not piece:
            return None
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


def _read_line(rfile: BinaryIO) -> tuple[Optional[bytes], str]:
    """
    Description: Read one CRLF-terminated line of a chunked body, at most 4096 bytes.
    Parameters:
        rfile: the connection's input stream.
    Output: (line without its CRLF, "ok"); (None, "truncated") when the stream ended mid-line; (None, "bad-framing")
        for a line that is too long or ends with a bare LF.
    """
    line = rfile.readline(_LINE_LIMIT + 1)
    if line.endswith(_CRLF):
        return line[:-2], "ok"
    if len(line) > _LINE_LIMIT or b"\n" in line:
        return None, "bad-framing"
    return None, "truncated"


def _read_chunked(rfile: BinaryIO, cap: int) -> BodyRead:
    """
    Description: Read a chunked body to its last chunk and its trailers, which are dropped.
    Parameters:
        rfile: the connection's input stream.
        cap: the most data bytes accepted.
    Output: A BodyRead; see read_body. Reads nothing past the blank line that ends the trailers.
    """
    pieces: list[bytes] = []
    total = 0
    while True:
        line, state = _read_line(rfile)
        if line is None:
            return _fail(state)
        size_text = line.split(b";", 1)[0]
        if not _HEX.fullmatch(size_text):
            return _fail("bad-framing")
        size = int(size_text, 16)
        if size == 0:
            break
        total += size
        if total > cap:
            return _fail("too-large")
        data = _read_exact(rfile, size)
        tail = _read_exact(rfile, 2) if data is not None else None
        if tail is None:
            return _fail("truncated")
        if tail != _CRLF:
            return _fail("bad-framing")
        pieces.append(data)
    for _ in range(_TRAILER_LIMIT + 1):
        line, state = _read_line(rfile)
        if line is None:
            return _fail(state)
        if line == b"":
            return BodyRead("ok", b"".join(pieces))
    return _fail("bad-framing")


def read_body(rfile: BinaryIO, content_lengths: Sequence[str], transfer_encodings: Sequence[str], cap: int) -> BodyRead:
    """
    Description: Read the body of a request, whichever way it was framed, without reading past its end or past the cap.
    Parameters:
        rfile: the connection's input stream, positioned after the request headers.
        content_lengths: every Content-Length header value received.
        transfer_encodings: every Transfer-Encoding header value received.
        cap: the most bytes accepted.
    Output: A BodyRead. With neither header the body is empty and nothing is read. Content-Length together with
        Transfer-Encoding, a repeated Content-Length, a Transfer-Encoding other than a single "chunked", a Content-Length
        that is not plain ASCII digits and a malformed chunk are "bad-framing". A body over the cap is "too-large": a
        Content-Length over the cap is refused before any byte is read. A body that ends early, a timeout and a reset are
        "truncated".
    """
    try:
        if content_lengths and transfer_encodings:
            return _fail("bad-framing")
        if transfer_encodings:
            if len(transfer_encodings) != 1 or transfer_encodings[0].strip().lower() != "chunked":
                return _fail("bad-framing")
            return _read_chunked(rfile, cap)
        if not content_lengths:
            return BodyRead("ok", b"")
        if len(content_lengths) != 1:
            return _fail("bad-framing")
        text = content_lengths[0].strip(" \t")
        if not (text.isascii() and text.isdigit()):
            return _fail("bad-framing")
        if len(text) > _MAX_LENGTH_DIGITS or int(text) > cap:
            return _fail("too-large")
        data = _read_exact(rfile, int(text))
        return _fail("truncated") if data is None else BodyRead("ok", data)
    except OSError:
        return _fail("truncated")
