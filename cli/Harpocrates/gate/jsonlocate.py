"""Find where a value sits in JSON text, by its path (FR-GATE-05).

The scan step decides what to change by walking the parsed tree; this module turns those paths into character spans of the
original text. It walks only the containers on the way to a wanted path and skips every other value with the standard
library's C scanner, so locating a few values in a megabyte costs a fraction of a parse.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

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
