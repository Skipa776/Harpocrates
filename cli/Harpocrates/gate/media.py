"""Media in a request (FR-GATE-07, G5 spec part 1): strict base64 only; a payload passes unscanned only when its first bytes
are a known binary format, and is noted by that format and its size; every other payload is read as text and scanned."""
from __future__ import annotations

import base64
import binascii
import re
from typing import NamedTuple, Optional

from Harpocrates.gate.blocks import MEDIA_MALFORMED, RefusalError

KINDS = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf", "audio"})
_STRICT = re.compile(r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
# Signatures at the start of the decoded bytes. A RIFF container is named by bytes 8 to 12.
_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF87a", "image/gif"),
          (b"GIF89a", "image/gif"), (b"%PDF-", "application/pdf"), (b"ID3", "audio"), (b"OggS", "audio"),
          (b"fLaC", "audio"), (b"\xff\xfb", "audio"), (b"\xff\xf3", "audio"), (b"\xff\xf2", "audio"))
_RIFF = {b"WEBP": "image/webp", b"WAVE": "audio"}


class MediaNote(NamedTuple):
    """
    Description: The warning for one binary payload that passed unscanned.
    Parameters:
        kind: one of KINDS.
        size: the decoded size in bytes.
    Output: An immutable record.
    """

    kind: str
    size: int


def decode_strict(payload: str) -> bytes:
    """
    Description: Decode a media payload that must be strict base64 to its end (no line breaks, no readable tail).
    Parameters:
        payload: the base64 text, without a data URL prefix.
    Output: The bytes. Raises RefusalError(MEDIA_MALFORMED) for anything else.
    """
    if not _STRICT.fullmatch(payload):
        raise RefusalError(MEDIA_MALFORMED)
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise RefusalError(MEDIA_MALFORMED) from None


def binary_kind(data: bytes) -> Optional[str]:
    """
    Description: Name a payload by its first bytes, whatever its label says, so no request text can reach the log.
    Parameters:
        data: the decoded bytes.
    Output: One of KINDS for a known image, PDF or audio signature; None for anything else, which must be scanned.
    """
    if data[:4] == b"RIFF" and data[8:12] in _RIFF:
        return _RIFF[data[8:12]]
    return next((kind for magic, kind in _MAGIC if data.startswith(magic)), None)


def readings(data: bytes) -> list[str]:
    """
    Description: The ways a payload without a binary signature could be text, for the scan.
    Parameters:
        data: the decoded bytes.
    Output: UTF-8 with bad bytes replaced; when the bytes hold a NUL, also UTF-16 little- and big-endian, so a UTF-16
        file is read as the text it is. A secret in any reading blocks.
    """
    texts = [data.decode("utf-8", errors="replace")]
    if b"\x00" in data:
        texts += [data.decode("utf-16-le", errors="replace"), data.decode("utf-16-be", errors="replace")]
    return texts
