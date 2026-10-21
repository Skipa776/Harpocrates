"""Base64 inside text fields (FR-GATE-11, docs/design.md): find long base64 runs in a string, decode them
at every alignment, and keep the text in each decoding, for the scan step to scan."""
from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Iterator

MIN_RUN = 24
MAX_RUN_CHARS = 2 * 1024 * 1024  # per request; below what one 5 s scan can decode, so more blocks fast as TOO-LARGE
MIN_STRETCH = 32
_CH = r"[A-Za-z0-9+/_-]"
# A run: alphabet characters, optionally in long pieces (16 or more) joined by a line break (real, or the two-character
# escapes \n and \r\n left in text), spaces or tabs, as MIME wrapping and pretty-printing split them; a short last piece
# is allowed. Separators are removed before decoding.
_SEP = r"(?:\r|\n|\\r|\\n|[ \t])+"
_RUN = re.compile(rf"{_CH}{{16,}}(?:{_SEP}{_CH}{{16,}})*(?:{_SEP}{_CH}{{1,15}}(?!{_CH}))?")
_STRIP = re.compile(r"\r|\n|\\r|\\n|[ \t]")
_TO_STANDARD = str.maketrans("-_", "+/")
_PRINTABLE_SHARE = 0.9
_NOT_PRINTABLE = r"\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f�"
_STRETCH = re.compile(rf"[^{_NOT_PRINTABLE}]{{{MIN_STRETCH},}}")


def _is_text(text: str) -> bool:
    """
    Description: Tell whether a whole decoding reads as text: at least 90% printable, tabs and line breaks counted as
        printable.
    Parameters:
        text: the decoding, with undecodable bytes already replaced by U+FFFD (which counts as not printable).
    Output: True for text, False for binary, garbage or an empty string.
    """
    if not text:
        return False
    printable = sum((ch.isprintable() and ch != "�") or ch in "\t\n\r" for ch in text)
    return printable >= _PRINTABLE_SHARE * len(text)


def runs(text: str) -> Iterator[str]:
    """
    Description: Find the base64 runs in a string.
    Parameters:
        text: the string, as the model will see it (JSON escapes already decoded).
    Output: Each run of at least MIN_RUN alphabet characters once its separators are removed. Padding is not part of
        a run.
    """
    for match in _RUN.finditer(text):
        run = _STRIP.sub("", match.group())
        if len(run) >= MIN_RUN:
            yield run


def _readings(data: bytes) -> Iterator[str]:
    """
    Description: The ways decoded bytes can be text: UTF-8, and UTF-16 when the NUL bytes fall the way UTF-16 text puts
        them (every other byte), as PowerShell and other Windows tools write files.
    Parameters:
        data: the decoded bytes.
    Output: The UTF-8 reading, then a UTF-16 little- or big-endian reading where the pattern fits; bad bytes replaced.
    """
    yield data.decode("utf-8", errors="replace")
    half = len(data) // 2
    if half and data.count(0) >= half * 0.4:
        if data[1::2].count(0) >= half * 0.4:
            yield data.decode("utf-16-le", errors="replace")
        if data[0::2].count(0) >= half * 0.4:
            yield data.decode("utf-16-be", errors="replace")


def _decode(run: str) -> Iterator[bytes]:
    """
    Description: Decode one run at offsets 0 to 3, so a run glued to stray characters or cut mid-group still decodes.
    Parameters:
        run: the run (standard or URL-safe alphabet, no padding).
    Output: The bytes of each alignment. A partial last group is padded; a single leftover character, which holds no
        whole byte, is dropped.
    """
    standard = run.translate(_TO_STANDARD)
    for offset in range(4):
        piece = standard[offset:]
        if len(piece) % 4 == 1:
            piece = piece[:-1]
        if not piece:
            continue
        try:
            yield base64.b64decode(piece + "=" * (-len(piece) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue


def _text_of(reading: str) -> str:
    """
    Description: The part of a reading to scan.
    Parameters:
        reading: one text reading of a decoding.
    Output: The whole reading when it is text (_is_text); otherwise its printable stretches of MIN_STRETCH or more
        characters, one per line, so a secret next to stray or invalid bytes is still seen; "" when there are none.
    """
    if _is_text(reading):
        return reading
    return "\n".join(_STRETCH.findall(reading))


def texts_of_run(run: str) -> list[str]:
    """
    Description: The text in the decodings of one run.
    Parameters:
        run: a run from runs().
    Output: Distinct texts, in order, from every alignment and reading. One level only: a decoding is not searched
        for further base64.
    """
    found: dict[str, None] = {}
    for data in _decode(run):
        for reading in _readings(data):
            kept = _text_of(reading)
            if kept:
                found.setdefault(kept)
    return list(found)


def decoded_texts(text: str) -> list[str]:
    """
    Description: The text in the decodings of every base64 run in a string.
    Parameters:
        text: the string.
    Output: Distinct texts, in order (texts_of_run over runs()).
    """
    found: dict[str, None] = {}
    for run in runs(text):
        for kept in texts_of_run(run):
            found.setdefault(kept)
    return list(found)


def strings_of(text: str, as_json: bool) -> list[str]:
    """
    Description: The strings to search in one leaf: a JSON-in-string leaf (tool-call arguments) is parsed, so escapes
        such as \\n cannot break or shift a run.
    Parameters:
        text: the leaf text.
        as_json: whether it holds JSON.
    Output: The leaf itself, plus, when it parses as JSON, every key and string value in it. JSON that does not parse,
        or nests too deep to parse, gives the leaf alone (still searched as text).
    """
    if not as_json:
        return [text]
    try:
        tree = json.loads(text)
    except (ValueError, RecursionError):
        return [text]
    found = [text]
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            found.extend(key for key in node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found
