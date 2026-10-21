"""What the format walkers hand to the scan step, and the helpers they share (parser spec, "Field classes").

A walker looks at a parsed request and yields operations; it never looks for secrets. Leaf is a string to scan (class R:
rewrite it; class S: scan only, a finding blocks), KeyText an object key to scan (a finding blocks), Drop an array element to
cut out, Media a base64 payload in a field the tables mark M (FR-GATE-07). Anything a walker does not yield is not
scanned: identifiers (I), ciphertext and signatures (O) and connector credentials (K). A walker raises RefusalError for a shape or type the parser spec does not list.
"""
from __future__ import annotations

from typing import Callable, Iterator, NamedTuple, Optional, Union

from Harpocrates.gate.blocks import PARSE, UNKNOWN_TYPE, RefusalError

# Signatures and ciphertext (class O) and cache markers (class I). A block that carries them says so (deep=OPAQUE); a field
# the user fills (metadata, tool arguments, schemas) may use the same words as keys and is scanned in full.
OPAQUE = frozenset({"signature", "encrypted_content", "encrypted_index", "encrypted_stdout", "cache_control"})
CACHE = frozenset({"cache_control"})
NO_SKIP: frozenset = frozenset()


class Leaf(NamedTuple):
    """
    Description: A string the scan step must look at.
    Parameters:
        path: where it is in the request (member names and array positions from the root).
        text: the decoded string.
        cls: "R" (a finding is replaced) or "S" (a finding blocks the request).
        as_json: True when the string holds JSON (tool-call arguments) to be parsed and redacted leaf by leaf.
    Output: An immutable record.
    """

    path: tuple
    text: str
    cls: str
    as_json: bool = False


class KeyText(NamedTuple):
    """
    Description: An object key the scan step must look at; a finding blocks the request.
    Parameters:
        path: the path of the member the key names.
        text: the key.
    Output: An immutable record.
    """

    path: tuple
    text: str


class Drop(NamedTuple):
    """
    Description: An array element that is removed from the request (a thinking block, or a message left empty).
    Parameters:
        path: the path of the element, ending in its position in the array.
    Output: An immutable record.
    """

    path: tuple


def refuse_type() -> RefusalError:
    """
    Description: Make the error for a block, item, tool or source type the parser spec does not list.
    Parameters: None
    Output: RefusalError(UNKNOWN_TYPE).
    """
    return RefusalError(UNKNOWN_TYPE)


def refuse_shape() -> RefusalError:
    """
    Description: Make the error for a request that is not shaped as its format says.
    Parameters: None
    Output: RefusalError(PARSE).
    """
    return RefusalError(PARSE)


class Media(NamedTuple):
    """
    Description: A base64 payload in a field the parser tables mark M (FR-GATE-07): checked, and scanned when it is text.
    Parameters:
        path: where it is in the request.
        payload: the base64 text, without any data URL prefix.
    Output: An immutable record.
    """

    path: tuple
    payload: str


Op = Union[Leaf, KeyText, Drop, Media]
Special = dict[str, Callable[[object, tuple], Iterator[Op]]]


def media(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a media field. A data URL gives its payload after ";base64,"; a plain string is the payload.
    Parameters:
        value: the field's value.
        path: the field's path.
    Output: One Media for a string; a value of another type is scanned like any other (strings()), so nothing in it
        escapes the scan.
    """
    if not isinstance(value, str):
        yield from strings(value, path)
        return
    _, marker, payload = value.partition(";base64,")
    yield Media(path, payload if value.startswith("data:") and marker else value)


def media_url(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk an image URL field: a base64 data URL is media, any other URL is text (it may hold a token).
    Parameters:
        value: the field's value.
        path: its path.
    Output: One Media for "data:...;base64,...", otherwise what strings() yields.
    """
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        yield from media(value, path)
    else:
        yield from strings(value, path)


def leaf(path: tuple, text: str, cls: str = "R", as_json: bool = False) -> Iterator[Op]:
    """
    Description: Yield one string as a Leaf unless it is empty. A string that looks like a data URL is text here too: media
        is decided by the field (media()), not by the content (FR-GATE-07).
    Parameters:
        path: its path.
        text: the string.
        cls: "R" or "S".
        as_json: whether it holds JSON to parse.
    Output: Zero or one Leaf.
    """
    if text:
        yield Leaf(path, text, cls, as_json)


def strings(node: object, path: tuple, cls: str = "R", skip: frozenset = NO_SKIP, keys: bool = False,
            deep: frozenset = CACHE) -> Iterator[Op]:
    """
    Description: Yield every string under a JSON value (the open field list: a string field the tables do not name is
        scanned like its neighbours).
    Parameters:
        node: the value.
        path: its path.
        cls: "R" or "S" for the leaves.
        skip: member names not to enter in the first object reached (an array passes this on to its elements; objects
            below start with none), for the identifier fields of one block.
        keys: also yield each object key as a KeyText (free-form names, such as tool-call arguments).
        deep: member names not to enter at any depth (default: cache markers; pass OPAQUE inside a provider block that
            carries signatures or ciphertext).
    Output: Leaf and KeyText operations in document order.
    """
    if isinstance(node, str):
        yield from leaf(path, node, cls)
    elif isinstance(node, dict):
        for name, value in node.items():
            if name in skip or name in deep:
                continue
            if keys:
                yield KeyText(path + (name,), name)
            yield from strings(value, path + (name,), cls, NO_SKIP, keys, deep)
    elif isinstance(node, list):
        for position, value in enumerate(node):
            yield from strings(value, path + (position,), cls, skip, keys, deep)


def fields(block: dict, path: tuple, skip: frozenset = NO_SKIP, special: Optional[Special] = None,
           cls: str = "R", json_keys: frozenset = NO_SKIP) -> Iterator[Op]:
    """
    Description: Walk the members of one block: skip the identifiers, give named members their own rule, scan the rest.
    Parameters:
        block: the block (an object).
        path: its path.
        skip: member names that are identifiers or ciphertext and are not scanned. Signatures and cache markers are
            always skipped.
        special: member name -> function(value, path) for members with their own rule.
        cls: "R" or "S" for the members scanned generically.
        json_keys: member names whose string value holds JSON (tool-call arguments), yielded with as_json=True.
    Output: Operations in document order.
    """
    for name, value in block.items():
        if name in skip or name in OPAQUE:
            continue
        if special and name in special:
            yield from special[name](value, path + (name,))
        elif name in json_keys and isinstance(value, str):
            yield from leaf(path + (name,), value, cls, as_json=True)
        else:
            yield from strings(value, path + (name,), cls)
