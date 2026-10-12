"""Filesystem-path entropy tokens: recognise a path so it is not sent on as a secret candidate.

The entropy tokenizer's alphabet keeps '/' (base64 needs it), so a long path such as
``/private/tmp/claude-501/-Users-me-proj/<uuid>/scratchpad`` becomes one 100-char token whose
length and mixed character classes clear the entropy floor, and the recall-first gate model
then passes it. A path is recognised structurally, never by tuning on data:

1. Positive evidence it is a path: at least two ``/``-separated segments that are made of
   words (lowercase runs, Capitalised words, digits, joined by ``- _ +``), e.g. ``private``,
   ``node_modules``, ``-Users-me-proj``, ``DerivedData``. Base64 blobs that happen to contain
   '/' almost never have two such segments, and '=' (padding) rules a token out outright.
2. No segment would be a secret candidate on its own: a segment that clears the entropy floor
   and has letters and digits, or mixed case, or ``-``/``_`` with 16+ letters and digits
   (base32/base62/urlsafe machine tokens), keeps the whole token a candidate. UUIDs and pure
   hex are not counted: they are content-addressed or generated directory names, and
   lowercase hex 32-64 still reaches the model through the hex rule.
3. URLs are left to the URL rules: the rule does not apply on a line containing "://", or when
   the token is followed by a query ``?`` / fragment ``#`` or preceded by userinfo ``@``.
4. A secret-named key before the path on the line (``secret: /path/...``, ``TOKEN=/...``)
   switches the rule off for that line: the name is evidence the value is the secret.

Trade-off: recall first. Any random-looking segment keeps the path a candidate, so generated
directory names that mix letters and digits (macOS /var/folders, pnpm and Nix store hashes,
tempfile names) still go to the model and can still be redacted; the rule removes paths whose
segments are words, UUIDs, hex or short names (the agent temp-dir paths that were refused).
"""
from __future__ import annotations

import re

from Harpocrates.detectors.entropy_detector import looks_like_secret

# NAME= prefix the tokenizer glues to an unquoted value (TMP=/var/...); '=' is in its alphabet.
_NAME_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# A word segment: words (lowercase run, optionally Capitalised, or digits) joined by - _ +,
# with optional leading/trailing separators (-Users-me-proj, node_modules, claude-501).
# The lookaheads make every run maximal, so there is one parse and matching stays linear
# (tokens on minified lines can be very long).
_WORD = r"(?:[A-Z]?[a-z]+(?![a-z])|[0-9]+(?![0-9]))"
_SEP = r"[-_+]*(?![-_+])"
_WORD_SEGMENT = re.compile(rf"{_SEP}{_WORD}(?:{_SEP}{_WORD})*{_SEP}")
_UUID = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
_HEX = re.compile(r"[0-9a-fA-F]+")
# Two word segments of 3+ chars: one could be chance in a base64 blob, two is a path's shape.
_MIN_WORD_SEGMENTS = 2
_MIN_WORD_LEN = 3


# A credential-named key followed by ':' or '=' (optionally quoted), anywhere before the path.
_SECRET_KEY = re.compile(r"(?i)(?:secret|token|key|passw(?:or)?d|credential|auth)[\w.\-]*['\"]?\s*[:=]")
# Entropy floor for one segment. Sample entropy of a short string is biased low (about
# log2(k) - (k-1)/(2n ln 2) for n draws from k symbols), so the 4.0 whole-token floor misses
# ~11% of random 32-char base32 and ~6% of 24-char base62 segments; at 3.5 none in 2,000
# generated samples of either. Cost: more short random-ish directory names stay candidates.
_SEGMENT_ENTROPY = 3.5
# urlsafe tokens: a '-'/'_' segment counts once it holds this many letters and digits.
_MIN_URLSAFE_ALNUM = 16


def _segment_is_candidate(segment: str) -> bool:
    """Would this segment alone look like a machine-generated secret?"""
    if _UUID.fullmatch(segment) or _HEX.fullmatch(segment):
        return False
    if not any(c.isdigit() for c in segment) and _WORD_SEGMENT.fullmatch(segment):
        return False  # a digit-free name built from words: -Users-me-proj, DerivedData
    if not looks_like_secret(segment, threshold=_SEGMENT_ENTROPY):
        return False
    letters = any(c.isalpha() for c in segment)
    digits = any(c.isdigit() for c in segment)
    mixed = any(c.isupper() for c in segment) and any(c.islower() for c in segment)
    urlsafe = ("-" in segment or "_" in segment) and sum(c.isalnum() for c in segment) >= _MIN_URLSAFE_ALNUM
    return (letters and digits) or mixed or urlsafe


def _in_url_position(token: str, line: str) -> bool:
    """URL/query/userinfo position, or a secret-named key earlier on the line: leave it to the other rules."""
    if "://" in line:
        return True
    at = line.find(token)
    if at < 0:  # token not found verbatim: decide nothing, keep the candidate
        return True
    after = line[at + len(token): at + len(token) + 1]
    before = line[at - 1: at] if at else ""
    return after in ("?", "#") or before == "@" or bool(_SECRET_KEY.search(line, 0, at))


def benign_path_token(token: str, line: str) -> bool:
    """True when an entropy token is a filesystem path with no secret-looking segment."""
    body = _NAME_PREFIX.sub("", token, count=1)
    if "/" not in body or "=" in body:
        return False
    segments = [s for s in body.split("/") if s]
    if len(segments) < 2 or _in_url_position(token, line):
        return False
    words = sum(1 for s in segments if len(s) >= _MIN_WORD_LEN and _WORD_SEGMENT.fullmatch(s))
    if words < _MIN_WORD_SEGMENTS:
        return False
    return not any(_segment_is_candidate(s) for s in segments)


__all__ = ["benign_path_token"]
