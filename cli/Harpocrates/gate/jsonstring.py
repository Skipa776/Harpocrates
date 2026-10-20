"""JSON inside a string (parser spec rule 3, decision 2): tool-call arguments such as function_call.arguments.

The string is parsed, every string leaf goes through the finder, the new leaves are spliced into the string, and the caller
encodes the whole string once as its own literal. A raw-string span could start or end inside an escape (a backslash and
its letter) and break the arguments; a decoded leaf cannot.
"""
from __future__ import annotations

from typing import Optional

from Harpocrates.gate.blocks import JSON_ARGS, KEY_SECRET, RefusalError
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.jsonlocate import locate
from Harpocrates.gate.jsonspan import BodyParseError, apply_edits, encode_literal, parse_text_strict

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
