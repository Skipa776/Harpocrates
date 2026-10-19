#!/usr/bin/env python3
"""A mock LLM provider that records every request an agent sends it (FR-PROOF-04, G2).

    python bench/mock_provider.py [--port N] [--record DIR] [--script FILE]

Stdlib only. It binds a loopback address and makes no outbound connection. It answers the routes Claude Code
(Anthropic Messages) and Codex (OpenAI Responses) call, as JSON or as SSE, so an agent can finish a turn without
a real model, and it can follow a script that asks the agent to run tools. Each request becomes one line of
RECORD_DIR/requests.jsonl: time, method, path, headers, sizes, the SHA-256 of the raw body and the decompressed
body.

Credentials are never written. A header whose name holds auth, key, token, cookie, secret or session, and
chatgpt-account-id, is stored as "<present>". Bodies are stored, because the harness counts canaries in them, and
a live session also sends the agent's own system prompt and memory files. So recordings go to
bench/results/mock/<run>/, which is gitignored and never committed.

Bad input never crashes it. A malformed, oversized (over 64 MiB) or wrongly encoded body is recorded and answered
400. It shuts down cleanly on SIGINT.
"""
from __future__ import annotations

import urllib.parse
from typing import Iterable

PRESENT = "<present>"
CREDENTIAL_WORDS = ("auth", "key", "token", "cookie", "secret", "session")
CREDENTIAL_NAMES = ("chatgpt-account-id",)


def is_credential_header(name: str) -> bool:
    """
    Description: Decide whether a header (or query parameter) name may carry a credential, so its value is never stored.
    Parameters:
        name: the header or query parameter name, in any case.
    Output: True when the lower-cased name contains auth, key, token, cookie, secret or session, or is
        chatgpt-account-id; False otherwise. No side effects.
    """
    lowered = name.lower()
    return lowered in CREDENTIAL_NAMES or any(word in lowered for word in CREDENTIAL_WORDS)


def redact_headers(headers: Iterable[tuple[str, str]]) -> list[list[str]]:
    """
    Description: Turn the request headers into the rows the recording stores, with every credential value replaced.
    Parameters:
        headers: (name, value) pairs in the order they were received, duplicates included.
    Output: A list of [name, value] rows in the same order; a credential header's value becomes "<present>". The
        name keeps the case it was sent in. No side effects.
    """
    return [[name, PRESENT if is_credential_header(name) else value] for name, value in headers]


def redact_target(target: str) -> str:
    """
    Description: Replace the value of every credential-named query parameter in a request target.
    Parameters:
        target: the request line's path with its query string, such as "/v1/messages?beta=true".
    Output: The target with each non-empty value of a credential-named parameter set to "<present>" (URL-encoded);
        other parameters keep their order and values. A target without a query string is returned unchanged.
    """
    path, sep, query = target.partition("?")
    if not sep:
        return target
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    kept = [(k, PRESENT if v and is_credential_header(k) else v) for k, v in pairs]
    return path + "?" + urllib.parse.urlencode(kept, quote_via=urllib.parse.quote)
