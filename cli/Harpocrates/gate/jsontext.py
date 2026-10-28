"""JSON text for the scan step: strict parsing, finding a value's span, and JSON inside a string (FR-GATE-02, FR-GATE-05).

The body is decoded to text once, parsed into a plain tree with the standard library's C scanner, and changed only by
replacing character spans of that text. Nothing is ever re-serialized, so key order, whitespace, number spelling and
every escape outside a replaced span stay as the client wrote them.

locate() turns the scan step's paths into character spans of the original text. It walks only the containers on the way
to a wanted path and skips every other value with the C scanner, so locating a few values in a megabyte costs a fraction
of a parse.

JSON inside a string (parser spec rule 3, decision 2) covers tool-call arguments such as function_call.arguments: the
string is parsed, every string leaf goes through the finder, the new leaves are spliced into the string, and the caller
encodes the whole string once as its own literal. A raw-string span could start or end inside an escape (a backslash and
its letter) and break the arguments; a decoded leaf cannot.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence

from Harpocrates.gate.blocks import JSON_ARGS, KEY_SECRET, RefusalError
from Harpocrates.gate.finder import Finder

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



_SCAN_ONCE = json.JSONDecoder().scan_once
_WHITESPACE = re.compile(r"[ \t\n\r]*")


class _Node:
    """
    Description: One step of the trie of wanted paths.
    Parameters:
        None: built empty.
    Output: An object with children (key -> _Node), want (this path is a value to find) and drop (this path is an array
        element to cut out).
    """

    __slots__ = ("children", "want", "drop")

    def __init__(self) -> None:
        """
        Description: Make an empty node.
        Parameters: None
        Output: None.
        """
        self.children: dict = {}
        self.want = False
        self.drop = False


def _trie(paths: Sequence[tuple], drops: Sequence[tuple]) -> _Node:
    """
    Description: Merge the wanted paths into one trie.
    Parameters:
        paths: paths of values to find.
        drops: paths of array elements to cut out.
    Output: The root node.
    """
    root = _Node()
    for group, is_drop in ((paths, False), (drops, True)):
        for path in group:
            node = root
            for step in path:
                node = node.children.setdefault(step, _Node())
            if is_drop:
                node.drop = True
            else:
                node.want = True
    return root


def _skip(text: str, index: int) -> int:
    """
    Description: Step over JSON whitespace.
    Parameters:
        text: the JSON text.
        index: where to start.
    Output: The index of the first character that is not whitespace.
    """
    return _WHITESPACE.match(text, index).end()


def _children(text: str, start: int) -> list[tuple[object, int, int]]:
    """
    Description: List the direct children of the array or object that starts at an index.
    Parameters:
        text: the JSON text (already known to be valid).
        start: the index of its opening bracket.
    Output: (key or position, start, end) for each child value, in order; the key is the member name for an object and
        the position for an array.
    """
    opening, out = text[start], []
    closing = "]" if opening == "[" else "}"
    index = _skip(text, start + 1)
    while text[index] != closing:
        key: object = len(out)
        if opening == "{":
            key, index = _SCAN_ONCE(text, index)
            index = _skip(text, index) + 1  # the colon
            index = _skip(text, index)
        _, end = _SCAN_ONCE(text, index)
        out.append((key, index, end))
        index = _skip(text, end)
        if text[index] == ",":
            index = _skip(text, index + 1)
    return out


def _cut_spans(children: list[tuple[object, int, int]], dropped: set) -> list[tuple[int, int]]:
    """
    Description: Work out which spans to delete so that the dropped elements, and the commas that went with them, vanish.
    Parameters:
        children: the array's children as _children lists them.
        dropped: the positions to cut.
    Output: Disjoint (start, end) spans. A run of dropped elements takes the separator after it, or, at the end of the
        array, the one before it; dropping everything empties the array's interior.
    """
    spans, index = [], 0
    while index < len(children):
        if index not in dropped:
            index += 1
            continue
        run_end = index
        while run_end + 1 < len(children) and run_end + 1 in dropped:
            run_end += 1
        if run_end + 1 < len(children):
            spans.append((children[index][1], children[run_end + 1][1]))
        elif index > 0:
            spans.append((children[index - 1][2], children[run_end][2]))
        else:
            spans.append((children[0][1], children[run_end][2]))
        index = run_end + 1
    return spans


def _descend(text: str, start: int, node: _Node, path: tuple, found: dict, cuts: list) -> None:
    """
    Description: Walk into the container at an index and record what the trie asks for.
    Parameters:
        text: the JSON text.
        start: the index where the container starts.
        node: the trie node for this container.
        path: the path to this container.
        found: filled with path -> (start, end) for each wanted value.
        cuts: filled with (start, end) spans to delete.
    Output: None. Raises LookupError for a wanted step that is not in the document.
    """
    if text[start] not in "[{":
        raise LookupError("a path goes through a value that is not an array or object")
    children = _children(text, start)
    by_key = {key: (s, e) for key, s, e in children}
    dropped = {key for key in node.children if node.children[key].drop}
    for key, child in node.children.items():
        if key not in by_key:
            raise LookupError("a path is not in the document")
        span = by_key[key]
        if child.want:
            found[path + (key,)] = span
        if child.children:
            _descend(text, span[0], child, path + (key,), found, cuts)
    if dropped:
        if text[start] != "[":
            raise LookupError("only an array element can be cut out")
        cuts.extend(_cut_spans(children, dropped))


def locate(text: str, paths: Sequence[tuple], drops: Sequence[tuple]) -> tuple[dict, list]:
    """
    Description: Find the spans of the wanted values and of the array elements to cut.
    Parameters:
        text: valid JSON text.
        paths: paths (tuples of member names and array positions from the root) of values whose span is wanted.
        drops: paths of array elements to cut out, together with the comma that separates them from their neighbours.
    Output: ({path: (start, end)}, [(start, end), ...]): the character span of each wanted value (a string literal includes
        its quotes) and the disjoint spans to delete. Raises LookupError for a path the document does not have.
    """
    root, found, cuts = _trie(paths, drops), {}, []
    start = _skip(text, 0)
    if root.want:
        found[()] = (start, _SCAN_ONCE(text, start)[1])
    if root.children:
        _descend(text, start, root, (), found, cuts)
    return found, cuts



MAX_LEVELS = 3  # a leaf that is itself JSON is followed this many levels down; deeper text is scanned as plain text


def _container(text: str) -> Optional[object]:
    """
    Description: Parse a string as a JSON object or array, if it is one.
    Parameters:
        text: the string.
    Output: The dict or list, or None when the string is not strict JSON or is a JSON scalar.
    """
    if text.lstrip()[:1] not in ("{", "["):
        return None
    try:
        tree = parse_text_strict(text)
    except BodyParseError:
        return None
    return tree if isinstance(tree, (dict, list)) else None


def _strings(node: object, path: tuple, values: list, keys: list) -> None:
    """
    Description: Collect every string value, with its path, and every object key under a JSON value.
    Parameters:
        node: the JSON value.
        path: the path to node.
        values: appended with (path, string) for each string value.
        keys: appended with each object key.
    Output: None.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            keys.append(key)
            _strings(value, path + (key,), values, keys)
    elif isinstance(node, list):
        for position, value in enumerate(node):
            _strings(value, path + (position,), values, keys)
    elif isinstance(node, str):
        values.append((path, node))


def redact_json_string(text: str, finder: Finder, level: int = 1) -> Optional[str]:
    """
    Description: Redact the secrets in a string that holds JSON.
    Parameters:
        text: the decoded string (the arguments).
        finder: the session's finder.
        level: how deep this string is in JSON-in-JSON (1 for the arguments themselves).
    Output: The new string, or None when nothing in it needed a change. A string that is not strict JSON, or is only a JSON
        scalar, is scanned as plain text and its spans replaced. Otherwise only string leaves change, each re-encoded
        as a whole; a leaf that is itself JSON is followed (to MAX_LEVELS). Raises RefusalError(KEY_SECRET) for a secret in an
        object key and RefusalError(JSON_ARGS) if the result no longer parses.
    """
    tree = _container(text)
    if tree is None:
        return finder.redact(text, "R")
    values: list[tuple[tuple, str]] = []
    keys: list[str] = []
    _strings(tree, (), values, keys)
    for key in keys:
        if finder.has_secret(key, "K"):
            raise RefusalError(KEY_SECRET)
    changed: dict[tuple, str] = {}
    for path, leaf in values:
        follow = level < MAX_LEVELS and leaf.lstrip()[:1] in ("{", "[")
        new = redact_json_string(leaf, finder, level + 1) if follow else finder.redact(leaf, "R")
        if new is not None:
            changed[path] = new
    if not changed:
        return None
    found, _ = locate(text, list(changed), [])
    result = apply_edits(text, [(*found[path], encode_literal(new)) for path, new in changed.items()])
    try:
        parse_text_strict(result)
    except BodyParseError:
        raise RefusalError(JSON_ARGS) from None
    return result
