"""Reading a request body within the limits (FR-GATE-01), and decoding its content encoding (FR-INSTALL-04)."""
from __future__ import annotations

import gzip
import importlib
import re
import zlib
from typing import Any, BinaryIO, Callable, NamedTuple, Optional, Sequence

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


_ENCODINGS = ("identity", "gzip", "zstd")
_ZSTD_SLICE = 1024  # input fed per step; one step cannot expand past about 32 MiB, so the cap is checked often


class Zstd(NamedTuple):
    """
    Description: The zstd functions found on this interpreter.
    Parameters:
        new_decompressor: makes an object with decompress(bytes) -> bytes, eof and unused_data (zstandard's
            decompressobj, or compression.zstd.ZstdDecompressor on Python 3.14).
        compress: (bytes) -> one zstd frame.
    Output: An immutable record built by load_zstd.
    """

    new_decompressor: Callable[[], Any]
    compress: Callable[[bytes], bytes]


class Decoded(NamedTuple):
    """
    Description: The outcome of decoding one request body.
    Parameters:
        status: "ok"; "unsupported" (an encoding the gate does not decode, or more than one); "zstd-unavailable" (zstd
            with no zstd module); "corrupt" (bad, truncated or padded data); "too-large" (the decoded size passes the cap).
        body: the decoded bytes when status is "ok", otherwise b"".
        encoding: "identity", "gzip" or "zstd" when the header named one of them, otherwise "".
    Output: An immutable record built by decode_body.
    """

    status: str
    body: bytes
    encoding: str


def load_zstd() -> Optional[Zstd]:
    """
    Description: Find a zstd module: Python 3.14's compression.zstd first, then the zstandard package.
    Parameters: None
    Output: A Zstd, or None when neither can be imported. Imports a module; no other side effects.
    """
    try:
        std = importlib.import_module("compression.zstd")
        return Zstd(std.ZstdDecompressor, std.compress)
    except ImportError:
        pass
    try:
        third = importlib.import_module("zstandard")
        return Zstd(lambda: third.ZstdDecompressor().decompressobj(), lambda data: third.ZstdCompressor().compress(data))
    except ImportError:
        return None


def _decode_gzip(raw: bytes, cap: int) -> Decoded:
    """
    Description: Decode one gzip member, producing at most cap + 1 bytes so a bomb is never fully expanded.
    Parameters:
        raw: the compressed body.
        cap: the largest decoded size accepted.
    Output: A Decoded: too-large past the cap; corrupt for a bad, truncated or empty stream, or any bytes after the member.
    """
    stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = stream.decompress(raw, cap + 1)
    except zlib.error:
        return Decoded("corrupt", b"", "gzip")
    if len(out) > cap:
        return Decoded("too-large", b"", "gzip")
    if not stream.eof or stream.unused_data:
        return Decoded("corrupt", b"", "gzip")
    return Decoded("ok", out, "gzip")


def _decode_zstd(raw: bytes, cap: int, codec: Zstd) -> Decoded:
    """
    Description: Decode exactly one zstd frame in small steps, so a bomb is stopped at the cap.
    Parameters:
        raw: the compressed body.
        cap: the largest decoded size accepted.
        codec: the zstd functions to use.
    Output: A Decoded: too-large past the cap; corrupt for a bad or truncated frame, an empty body, or any bytes after
        the frame (a second frame included).
    """
    decompressor = codec.new_decompressor()
    parts: list[bytes] = []
    total = 0
    for start in range(0, len(raw), _ZSTD_SLICE):
        try:
            piece = decompressor.decompress(raw[start:start + _ZSTD_SLICE])
        except Exception:  # each zstd module has its own error class; any failure is a corrupt body
            return Decoded("corrupt", b"", "zstd")
        total += len(piece)
        if total > cap:
            return Decoded("too-large", b"", "zstd")
        parts.append(piece)
        if decompressor.eof:
            if decompressor.unused_data or start + _ZSTD_SLICE < len(raw):
                return Decoded("corrupt", b"", "zstd")
            break
    if not decompressor.eof:
        return Decoded("corrupt", b"", "zstd")
    return Decoded("ok", b"".join(parts), "zstd")


def decode_body(raw: bytes, content_encodings: Sequence[str], cap: int, zstd: Optional[Zstd]) -> Decoded:
    """
    Description: Turn a request body into the bytes the model would see, never expanding past the cap.
    Parameters:
        raw: the body exactly as received.
        content_encodings: every Content-Encoding header value received.
        cap: the largest decoded size accepted.
        zstd: the zstd functions, or None when no zstd module is available.
    Output: A Decoded. No header, or "identity", passes the bytes through. "gzip" and "zstd" (any case, padding
        trimmed) are decoded. Anything else is "unsupported": "br", "deflate", "x-gzip", a list, an empty value or
        more than one header. zstd with no module is "zstd-unavailable". No side effects.
    """
    if not content_encodings:
        name = "identity"
    elif len(content_encodings) == 1 and content_encodings[0].strip().lower() in _ENCODINGS:
        name = content_encodings[0].strip().lower()
    else:
        return Decoded("unsupported", b"", "")
    if name == "identity":
        return Decoded("ok", raw, name) if len(raw) <= cap else Decoded("too-large", b"", name)
    if name == "gzip":
        return _decode_gzip(raw, cap)
    if zstd is None:
        return Decoded("zstd-unavailable", b"", name)
    return _decode_zstd(raw, cap, zstd)


def encode_body(body: bytes, encoding: str, zstd: Optional[Zstd]) -> bytes:
    """
    Description: Put a body back into the encoding it arrived in, so the provider receives it as the agent sent it.
    Parameters:
        body: the bytes to forward.
        encoding: "identity", "gzip" or "zstd" (the Decoded.encoding of the request).
        zstd: the zstd functions, needed only for "zstd".
    Output: The encoded bytes. gzip output is the same for the same input (no timestamp). Raises ValueError for zstd with
        no module or for any other encoding.
    """
    if encoding == "identity":
        return body
    if encoding == "gzip":
        return gzip.compress(body, mtime=0)
    if encoding == "zstd" and zstd is not None:
        return zstd.compress(body)
    raise ValueError("cannot encode this body")
