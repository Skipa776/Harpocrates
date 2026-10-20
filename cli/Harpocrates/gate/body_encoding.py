"""Request body encodings (FR-INSTALL-04): decode identity, gzip and zstd under the size cap, and encode back."""
from __future__ import annotations

import gzip
import importlib
import zlib
from typing import Any, Callable, NamedTuple, Optional, Sequence

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
