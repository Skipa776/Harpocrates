"""FR-INSTALL-05: auth headers cross the gate byte-identical. Only hop-by-hop headers, Host, the length and encoding
of the body, and the gate token header are changed. Unit tests of the header rules."""
from __future__ import annotations

import pytest

from Harpocrates.gate.headers import client_response_headers, upstream_request_headers

# Names the transport notes list as passing untouched, in the case each agent sends them.
CREDENTIAL_PAIRS = [
    ("x-api-key", "FAKE-ANTHROPIC-KEY-0123456789abcdef"),
    ("Authorization", "Bearer FAKE-TOKEN.with+odd/chars=="),
    ("anthropic-version", "2023-06-01"),
    ("anthropic-beta", "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14"),
    ("ChatGPT-Account-ID", "acct-fake-0001"),
    ("chatgpt-account-id", "acct-fake-0002"),
    ("OpenAI-Organization", "org-fake"),
    ("OpenAI-Project", "proj-fake"),
    ("OpenAI-Beta", "responses=experimental"),
    ("version", "0.163.0"),
    ("x-codex-installation-id", "00000000-0000-4000-8000-000000000000"),
    ("x-codex-turn-state", "opaque-state-value"),
    ("x-codex-turn-metadata", '{"workspaces":{"/tmp/x":{"has_changes":false}}}'),
    ("x-codex-window-id", "w:1"),
    ("x-codex-parent-thread-id", "t-0"),
    ("x-openai-subagent", "review"),
    ("x-client-request-id", "req-0001"),
    ("x-claude-code-session-id", "s-0001"),
    ("session-id", "s-0002"),
    ("thread-id", "t-0002"),
    ("originator", "codex_cli_rs"),
    ("x-stainless-retry-count", "0"),
    ("User-Agent", "claude-cli/2.1.303 (external, sdk-cli)"),
]


def test_fr_install_05_credential_and_agent_headers_are_forwarded_byte_identical_in_order() -> None:
    pairs = [("Accept", "application/json"), *CREDENTIAL_PAIRS, ("Content-Type", "application/json")]
    assert upstream_request_headers(pairs, "identity", None) == pairs


def test_fr_install_05_case_duplicates_spacing_and_high_bytes_are_kept() -> None:
    pairs = [("X-Api-Key", "k1 "), ("x-api-key", "k2"), ("X-Odd", "caf\xe9 \x80\xff  x"), ("X-Empty", ""),
             ("Accept", "a"), ("Accept", "b")]
    assert upstream_request_headers(pairs, "identity", None) == pairs


@pytest.mark.parametrize("name", ["Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
                                  "Proxy-Connection", "TE", "Trailer", "Transfer-Encoding", "Upgrade"])
def test_fr_install_05_hop_by_hop_headers_are_dropped(name: str) -> None:
    for spelling in (name, name.lower(), name.upper()):
        assert upstream_request_headers([(spelling, "x"), ("x-api-key", "k")], "identity", None) == [("x-api-key", "k")]


def test_fr_install_05_connection_naming_a_header_does_not_remove_it() -> None:
    pairs = [("Connection", "keep-alive, x-api-key"), ("x-api-key", "k")]
    assert upstream_request_headers(pairs, "identity", None) == [("x-api-key", "k")]


@pytest.mark.parametrize("name", ["Host", "host", "Content-Length", "content-length", "Content-Encoding",
                                  "CONTENT-ENCODING", "Expect", "X-Harpo-Gate-Token", "x-harpo-gate-token"])
def test_fr_install_05_host_body_headers_expect_and_the_gate_token_are_dropped(name: str) -> None:
    assert upstream_request_headers([(name, "v"), ("x-api-key", "k")], "identity", None) == [("x-api-key", "k")]


def test_fr_install_05_content_length_is_set_from_the_forwarded_bytes() -> None:
    out = upstream_request_headers([("Content-Length", "999"), ("x-api-key", "k")], "identity", 12)
    assert out == [("x-api-key", "k"), ("Content-Length", "12")]
    assert upstream_request_headers([("x-api-key", "k")], "identity", 0)[-1] == ("Content-Length", "0")


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_fr_install_05_content_encoding_is_restored_when_the_body_arrived_encoded(encoding: str) -> None:
    out = upstream_request_headers([("Content-Encoding", "GZIP"), ("x-api-key", "k")], encoding, 5)
    assert out == [("x-api-key", "k"), ("Content-Encoding", encoding), ("Content-Length", "5")]


def test_fr_install_05_response_headers_pass_unchanged_except_hop_by_hop() -> None:
    pairs = [
        ("Content-Type", "text/event-stream; charset=utf-8"), ("Cache-Control", "no-cache"), ("retry-after", "12"),
        ("x-should-retry", "true"), ("anthropic-ratelimit-unified-status", "allowed"), ("x-codex-turn-state", "s"),
        ("Set-Cookie", "a=1"), ("Set-Cookie", "b=2"), ("Content-Length", "10"), ("Content-Encoding", "gzip"),
        ("Date", "Mon, 19 Oct 2026 00:00:00 GMT"), ("Server", "upstream"),
    ]
    hops = [("Connection", "keep-alive"), ("Keep-Alive", "timeout=5"), ("Transfer-Encoding", "chunked"),
            ("Upgrade", "h2c"), ("Trailer", "X"), ("TE", "trailers"), ("Proxy-Authenticate", "x"),
            ("Proxy-Connection", "x")]
    assert client_response_headers(pairs[:6] + hops + pairs[6:]) == pairs
