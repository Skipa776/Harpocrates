"""FR-PROOF-04: the mock provider (bench/mock_provider.py) records what an agent sends, never a credential.

Everything here runs against 127.0.0.1 only. The credential values are made-up strings.
"""
from __future__ import annotations

import pytest

from bench import mock_provider as mock

FAKE = "FAKE-CRED-0123456789abcdef"


@pytest.mark.parametrize("name", [
    "Authorization", "authorization", "Proxy-Authorization", "x-api-key", "X-Api-Key", "api-key", "openai-api-key",
    "Cookie", "Set-Cookie", "x-auth-token", "X-Amz-Security-Token", "x-claude-code-session-id", "x-session-affinity",
    "x-client-secret", "ChatGPT-Account-ID", "chatgpt-account-id", "x-goog-api-key", "anthropic-auth-bearer",
])
def test_fr_proof_04_mock_redacts_every_credential_header_name(name: str) -> None:
    assert mock.is_credential_header(name)
    assert mock.redact_headers([(name, FAKE)]) == [[name, "<present>"]]


@pytest.mark.parametrize("name", [
    "anthropic-version", "anthropic-beta", "content-type", "content-length", "content-encoding", "user-agent",
    "accept", "host", "x-stainless-lang", "x-codex-turn-state", "x-codex-turn-metadata", "openai-beta", "version",
    "x-client-request-id", "transfer-encoding", "connection", "accept-encoding",
])
def test_fr_proof_04_mock_keeps_other_headers_verbatim(name: str) -> None:
    assert not mock.is_credential_header(name)
    assert mock.redact_headers([(name, "some-value")]) == [[name, "some-value"]]


def test_fr_proof_04_mock_redaction_keeps_order_and_duplicates() -> None:
    got = mock.redact_headers([("A", "1"), ("Cookie", FAKE), ("A", "2"), ("x-api-key", FAKE)])
    assert got == [["A", "1"], ["Cookie", "<present>"], ["A", "2"], ["x-api-key", "<present>"]]
    assert FAKE not in repr(got)


@pytest.mark.parametrize("target, expected", [
    ("/v1/messages?beta=true", "/v1/messages?beta=true"),
    ("/v1/models?limit=1000", "/v1/models?limit=1000"),
    ("/v1/messages?key=" + FAKE, "/v1/messages?key=%3Cpresent%3E"),
    ("/x?a=1&api_key=" + FAKE + "&token=" + FAKE + "&b=2", "/x?a=1&api_key=%3Cpresent%3E&token=%3Cpresent%3E&b=2"),
    ("/x?session_id=" + FAKE, "/x?session_id=%3Cpresent%3E"),
])
def test_fr_proof_04_mock_redacts_credential_query_values(target: str, expected: str) -> None:
    got = mock.redact_target(target)
    assert got == expected and FAKE not in got
