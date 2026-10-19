"""Find planted canaries in recorded request bodies and say where, without ever writing a value (FR-PROOF-04).

A canary is "received" when its value, or an encoded form of it, appears in a string of a request body the mock
provider recorded. The functions here match in memory and return places (a field path and a block type), never the
value. Keys that hold a canary, or that are not plain identifiers, are written as "<key>" in a path.

Known-format canaries have a distinctive prefix or structure that a signature regex matches on its own (an AWS key
id, a GitHub token, a JWT, a PEM block, a URL with credentials). ML-only canaries have no such marker: a 40-character
AWS secret, a Twilio auth token, a database password, a human-chosen password. Only context or the model finds those.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.parse
from typing import NamedTuple

KNOWN_FORMAT_KINDS = frozenset({
    "aws_access_key_id", "github_token", "openai_key", "stripe_key", "slack_token", "sendgrid_key", "gcp_api_key",
    "npm_token", "jwt", "private_key", "db_url",
})
ML_ONLY_KINDS = frozenset({"aws_secret_access_key", "twilio_auth_token", "db_password", "postgres_password"})
MIN_LEN = 8  # a shorter value is too likely to appear by chance
MIN_LINE = 16  # a line of a multi-line value counts as a leak when it is this long and is not armor
MAX_JSON_DEPTH = 3  # JSON inside a JSON string, inside a string, is followed this deep
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.\-]{1,48}$")


class Hit(NamedTuple):
    """
    Description: One place a canary was found.
    Parameters:
        path: the field path, such as messages[3].content[0].content[0].text; {json} marks JSON inside a string.
        block_type: the type of the nearest enclosing block or item, message(role) for a message, or the top-level field.
        form: plain, line (one line of a multi-line value), base64, hex or urlencoded; (unparsed) bodies carry plain.
    Output: An immutable record holding no value.
    """

    path: str
    block_type: str
    form: str


def kind_class(kind: str) -> str:
    """
    Description: Say whether a canary kind is found by its format or only by context and the model.
    Parameters:
        kind: a key of bench/canary_repo.py's canaries.
    Output: "known-format" or "ml-only". Raises ValueError for a kind in neither set.
    """
    if kind in KNOWN_FORMAT_KINDS:
        return "known-format"
    if kind in ML_ONLY_KINDS:
        return "ml-only"
    raise ValueError(f"unknown canary kind {kind!r}")


def needles(value: str) -> list[tuple[str, str]]:
    """
    Description: List the texts whose presence in a body means the value was received.
    Parameters:
        value: the canary value.
    Output: (form, text) pairs: plain; for a multi-line value each non-armor line of at least 16 characters (line);
        base64, hex and URL-encoded spellings when they differ from plain. Empty for a value under 8 characters.
    """
    if len(value) < MIN_LEN:
        return []
    found = [("plain", value)]
    lines = value.splitlines()
    if len(lines) > 1:
        found += [("line", ln) for ln in lines if len(ln) >= MIN_LINE and not ln.startswith("-----")]
    found.append(("base64", base64.b64encode(value.encode()).decode()))
    found.append(("hex", value.encode().hex()))
    quoted = urllib.parse.quote(value, safe="")
    if quoted != value:
        found.append(("urlencoded", quoted))
    return found


def _forms_in(text: str, value: str) -> list[str]:
    """
    Description: Name the forms of a value that occur in a piece of text.
    Parameters:
        text: a decoded string leaf, a key or a whole body.
        value: the canary value.
    Output: Each form name once, in the order of needles(); "line" is left out when the plain value is there, since it adds nothing.
    """
    hits = list(dict.fromkeys(form for form, needle in needles(value) if needle in text))
    return [f for f in hits if not (f == "line" and "plain" in hits)]


def _segment(key: str, value: str) -> str:
    """
    Description: Turn a dictionary key into a path segment that cannot leak a canary.
    Parameters:
        key: the key as it appears in the body.
        value: the canary being searched for.
    Output: The key when it is a plain identifier that holds no form of the value, otherwise "<key>".
    """
    return key if _SAFE_KEY.match(key) and not _forms_in(key, value) else "<key>"


def _walk(node: object, path: str, block: str, value: str, depth: int, hits: list[Hit]) -> None:
    """
    Description: Visit every string and key under a JSON value, collecting where the canary occurs.
    Parameters:
        node: the JSON value.
        path: the path to node.
        block: the nearest enclosing block type so far.
        value: the canary value.
        depth: how many JSON-in-string levels deep this is.
        hits: the list to append to.
    Output: None; appends a Hit per form found in a string or a key. A string that holds a JSON object or array is parsed
        and visited too, to MAX_JSON_DEPTH levels, and when that finds the value only the inner places are reported.
    """
    if isinstance(node, dict):
        kind = node.get("type")
        role = node.get("role")
        here = kind if isinstance(kind, str) else f"message({role})" if isinstance(role, str) and not block else block
        for key, child in node.items():
            seg = _segment(key, value)
            child_path = f"{path}.{seg}" if path else seg
            hits += [Hit(child_path, here or seg, form) for form in _forms_in(key, value)]
            _walk(child, child_path, here, value, depth, hits)
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _walk(child, f"{path}[{index}]", block, value, depth, hits)
    elif isinstance(node, str):
        if depth < MAX_JSON_DEPTH and node.lstrip()[:1] in ("{", "["):
            try:
                inner = json.loads(node)
            except ValueError:
                inner = None
            if isinstance(inner, (dict, list)):
                before = len(hits)
                _walk(inner, path + "{json}", block, value, depth + 1, hits)
                if len(hits) > before:  # the inner place is more exact than the string that holds it
                    return
        label = block or path.split(".")[0].split("[")[0]
        hits += [Hit(path, label, form) for form in _forms_in(node, value)]


def locate(body_text: str, value: str) -> list[Hit]:
    """
    Description: Find where a canary appears in one recorded request body.
    Parameters:
        body_text: the decompressed body as the recording holds it.
        value: the canary value; never returned or logged.
    Output: A Hit per place and form. A body that is not a JSON object or array is searched as text and reports one
        (unparsed body) hit per form. A value found in the raw text but in no string (an unusual escape) reports a
        (body text) hit, so nothing is undercounted. Empty when the value is under 8 characters or absent.
    """
    if len(value) < MIN_LEN:
        return []
    try:
        parsed = json.loads(body_text)
    except ValueError:
        parsed = None
    if not isinstance(parsed, (dict, list)):
        return [Hit("(unparsed body)", "(unparsed)", form) for form in _forms_in(body_text, value)]
    hits: list[Hit] = []
    _walk(parsed, "", "", value, 0, hits)
    if not hits:
        hits = [Hit("(body text)", "(unlocated)", form) for form in _forms_in(body_text, value)]
    return hits
