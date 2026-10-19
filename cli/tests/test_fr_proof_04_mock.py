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


# --- decoding the request body for the recording ---------------------------------------------------------------

import gzip  # noqa: E402
import zlib  # noqa: E402


def test_fr_proof_04_mock_decodes_identity_and_absent_encoding() -> None:
    for encoding in (None, "", "identity", "IDENTITY"):
        assert mock.decode_body(b'{"a":1}', encoding, 1024) == mock.Decoded(b'{"a":1}', "ok")


def test_fr_proof_04_mock_decodes_gzip() -> None:
    raw = gzip.compress(b'{"hello": "world"}')
    assert mock.decode_body(raw, "gzip", 1024) == mock.Decoded(b'{"hello": "world"}', "ok")
    assert mock.decode_body(raw, "GZIP", 1024).status == "ok"


def test_fr_proof_04_mock_gzip_bomb_is_too_large_not_expanded() -> None:
    bomb = gzip.compress(b"\0" * (8 * 1024 * 1024))
    got = mock.decode_body(bomb, "gzip", 1024 * 1024)
    assert got.status == "too-large" and got.body is None


def test_fr_proof_04_mock_identity_over_the_cap_is_too_large() -> None:
    assert mock.decode_body(b"x" * 11, None, 10) == mock.Decoded(None, "too-large")
    assert mock.decode_body(b"x" * 10, None, 10).status == "ok"


@pytest.mark.parametrize("raw", [b"this is not gzip", gzip.compress(b"abcdef" * 50)[:-6], b"", b"\x1f\x8b\x08\x00junk"])
def test_fr_proof_04_mock_wrong_or_truncated_gzip_is_bad_encoding(raw: bytes) -> None:
    got = mock.decode_body(raw, "gzip", 1024)
    assert got == mock.Decoded(None, "bad-encoding")


@pytest.mark.parametrize("encoding", ["br", "deflate", "gzip, zstd", "compress", "unknown"])
def test_fr_proof_04_mock_other_encodings_are_unsupported(encoding: str) -> None:
    assert mock.decode_body(zlib.compress(b"x"), encoding, 1024) == mock.Decoded(None, "unsupported-encoding")


def test_fr_proof_04_mock_zstd_is_flagged_when_no_module_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock, "_load_zstd", lambda: None)
    assert mock.decode_body(b"\x28\xb5\x2f\xfd\x00", "zstd", 1024) == mock.Decoded(None, "zstd-unavailable")


def test_fr_proof_04_mock_zstd_is_decoded_when_a_module_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock, "_load_zstd", lambda: lambda data, cap: b'{"z": 1}')
    assert mock.decode_body(b"fake-zstd-frame", "zstd", 1024) == mock.Decoded(b'{"z": 1}', "ok")

    def fail(data: bytes, cap: int) -> bytes:
        raise ValueError("corrupt frame")

    monkeypatch.setattr(mock, "_load_zstd", lambda: fail)
    assert mock.decode_body(b"x", "zstd", 1024) == mock.Decoded(None, "bad-encoding")
    monkeypatch.setattr(mock, "_load_zstd", lambda: lambda data, cap: b"x" * (cap + 1))
    assert mock.decode_body(b"x", "zstd", 8) == mock.Decoded(None, "too-large")
