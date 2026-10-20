"""Strict JSON parsing and byte-exact splicing for the scan step (FR-GATE-02 decision 15, FR-GATE-05).

The body is decoded to text once, parsed into a plain tree with the standard library's C scanner, and changed only by
replacing character spans of that text. Nothing is ever re-serialized, so key order, whitespace, number spelling and
every escape outside a replaced span stay as the client wrote them.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

MAX_DEPTH = 128  # decision 15: a body nested deeper than this is refused
_JSON_WHITESPACE = " \t\n\r"
# Everything but brackets, for the depth count: whole string literals and every run of other characters. Run on text the
# parser has already accepted, so each quote that is left is a string delimiter.
_NOT_BRACKETS = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[^\[\]{}"]+', re.S)
_EMPTY_PAIR = re.compile(r"\[\]|\{\}")
_NEEDS_ESCAPE = re.compile('["\\\\\x00-\x1f\ud800-\udfff]')
_SHORT_ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


class BodyParseError(ValueError):
    """
    Description: A body or a nested JSON string that decision 15 refuses.
    Parameters:
        None: the message is a fixed phrase naming the kind of failure, never any of the text.
    Output: Raised by parse_strict and parse_text_strict.
    """


def _no_duplicate_keys(pairs: list) -> dict:
    """
    Description: Build an object from its pairs, refusing a key that occurs twice (two parsers could disagree on which
        value wins).
    Parameters:
        pairs: the (key, value) pairs of one JSON object, in order.
    Output: The dict. Raises BodyParseError for a repeated key (compared after unescaping).
    """
    result = dict(pairs)
    if len(result) != len(pairs):
        raise BodyParseError("a JSON object repeats a key")
    return result


def _no_constant(_name: str) -> object:
    """
    Description: Refuse NaN, Infinity and -Infinity, which the JSON grammar does not have.
    Parameters:
        _name: the constant as the parser read it.
    Output: Never returns. Raises BodyParseError.
    """
    raise BodyParseError("a JSON constant that is not JSON")


def nesting_depth(text: str) -> int:
    """
    Description: Measure how deep the containers of valid JSON text are nested.
    Parameters:
        text: JSON text that already parsed (brackets inside strings are skipped).
    Output: 0 for a scalar, 1 for a flat array or object, and so on. Stops counting at MAX_DEPTH + 1.
    """
    brackets = _NOT_BRACKETS.sub("", text)
    depth = 0
    while brackets:
        depth += 1
        if depth > MAX_DEPTH:
            break
        shorter = _EMPTY_PAIR.sub("", brackets)  # one round removes the innermost level everywhere
        if len(shorter) == len(brackets):
            raise BodyParseError("unbalanced JSON brackets")
        brackets = shorter
    return depth


def parse_text_strict(text: str) -> object:
    """
    Description: Parse JSON text under decision 15's rules.
    Parameters:
        text: the JSON text.
    Output: The parsed value. Raises BodyParseError for invalid JSON, a repeated object key, NaN or Infinity, a number too
        long to read, or nesting deeper than MAX_DEPTH.
    """
    try:
        tree = json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except BodyParseError:
        raise
    except (ValueError, RecursionError):  # invalid JSON, a BOM, an over-long number, a parser that gave up on depth
        raise BodyParseError("the text is not valid JSON") from None
    if nesting_depth(text) > MAX_DEPTH:
        raise BodyParseError("the JSON is nested too deep")
    return tree


def parse_strict(data: bytes) -> tuple[str, object]:
    """
    Description: Decode and parse a request body under decision 15's rules.
    Parameters:
        data: the decoded request body.
    Output: (the text, the tree). Raises BodyParseError for bytes that are not UTF-8 and for everything parse_text_strict
        refuses. The text is what spans are counted in; encode it with UTF-8 to get the bytes back.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise BodyParseError("the body is not UTF-8") from None
    return text, parse_text_strict(text)


def encode_literal(text: str) -> str:
    """
    Description: Write text as a JSON string literal, escaping only what JSON requires.
    Parameters:
        text: the decoded string.
    Output: The literal with its quotes. Quote, backslash and control characters are escaped, and so is a lone surrogate
        (it cannot be written as UTF-8); every other character, accented letters and emoji included, stays as it is.
    """
    return '"' + _NEEDS_ESCAPE.sub(lambda m: _SHORT_ESCAPES.get(m.group()) or f"\\u{ord(m.group()):04x}", text) + '"'


def apply_edits(text: str, edits: Sequence[tuple[int, int, str]]) -> str:
    """
    Description: Replace character spans of a text, leaving every other character exactly as it was.
    Parameters:
        text: the original text.
        edits: (start, end, replacement) triples in any order; the spans must not overlap (an insertion with
            start == end may touch a neighbour).
    Output: The new text; the same object when edits is empty. Raises ValueError for a span outside the text, one that
        ends before it starts, or two that overlap.
    """
    if not edits:
        return text
    pieces, cursor = [], 0
    for start, end, replacement in sorted(edits, key=lambda e: (e[0], e[1])):
        if not 0 <= start <= end <= len(text) or start < cursor:
            raise ValueError("edit spans overlap or lie outside the text")
        pieces += [text[cursor:start], replacement]
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
