"""Scan the parts of a request that are not the body (decision 12): header values, the path and the query.

A header cannot be rewritten safely, so a finding in one blocks the request. The user's own credentials travel in headers
on purpose; those headers are skipped, not scanned.
"""
from __future__ import annotations

from typing import Iterator
from urllib.parse import unquote, unquote_plus

from Harpocrates.gate.blocks import HEADER_SECRET, RefusalError
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.scan import RequestHead

CREDENTIAL_WORDS = ("auth", "key", "token", "cookie", "secret")  # x-api-key, Authorization, x-codex-turn-state, set-cookie
CREDENTIAL_NAMES = frozenset({"chatgpt-account-id"})
CREDENTIAL_PREFIXES = ("anthropic-", "openai-")  # anthropic-version and -beta, OpenAI-Organization, -Project, -Beta


def is_credential_header(name: str) -> bool:
    """
    Description: Decide whether a header carries a credential or protocol setting the user sends on purpose.
    Parameters:
        name: the header name, any case.
    Output: True when the name contains auth, key, token, cookie or secret, is ChatGPT-Account-ID, or starts with
        anthropic- or openai-. Such a header is forwarded byte for byte and not scanned.
    """
    lowered = name.lower()
    return (lowered in CREDENTIAL_NAMES or lowered.startswith(CREDENTIAL_PREFIXES)
            or any(word in lowered for word in CREDENTIAL_WORDS))


def head_texts(head: RequestHead) -> Iterator[str]:
    """
    Description: List the strings of a request head that can hold a secret.
    Parameters:
        head: the headers, path and query.
    Output: Each non-credential header value, then the path and the query; a path or query that changes when its percent
        escapes are decoded is listed in both forms, so an encoded secret is found too.
    """
    for name, value in head.headers:
        if value and not is_credential_header(name):
            yield value
    for part, decode in ((head.path, unquote), (head.query, unquote_plus)):
        if part:
            yield part
            decoded = decode(part)
            if decoded != part:
                yield decoded


def scan_head(head: RequestHead, finder: Finder) -> None:
    """
    Description: Block a request whose head holds a secret.
    Parameters:
        head: the headers, path and query.
        finder: the session's finder.
    Output: None. Raises RefusalError(HEADER_SECRET) at the first finding; never names the header.
    """
    for text in head_texts(head):
        if finder.has_secret(text, "H"):
            raise RefusalError(HEADER_SECRET)
