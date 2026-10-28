"""Tests for Harpocrates.utils.redaction (PRD-01 Task 4)."""
from __future__ import annotations

import pytest

from Harpocrates.core.result import EvidenceType, Finding, Severity
from Harpocrates.utils.redaction import redact_token

# ---------------------------------------------------------------------------
# redact_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n"])
def test_redact_token_returns_none_for_empty_or_whitespace_input(value) -> None:
    assert redact_token(value) is None


@pytest.mark.parametrize(
    "token",
    ["a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefg", "\u200b"],
)
def test_redact_token_replaces_too_short_tokens_with_placeholder(token) -> None:
    # Tokens under 8 chars would over-disclose if sliced (e.g. "ab" -> "a...b").
    # They must be replaced wholesale so no original byte leaks. Includes
    # zero-width and other non-strip()-whitespace single chars.
    assert redact_token(token) == "[REDACTED]"


def test_redact_token_handles_multibyte_unicode() -> None:
    # Multibyte chars are still code points; an 8-char unicode token slices safely.
    token = "\u00e9" * 8  # eight é characters
    assert redact_token(token) == "\u00e9...\u00e9"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        # 4-char multibyte token must hit the short-token floor.
        pytest.param(
            "\u00e9\u00e9\u00e9\u00e9", "[REDACTED]", id="short_unicode_token_is_redacted"
        ),
        # len == 8 -> short: first char + "..." + last char
        pytest.param("abcdefgh", "a...h", id="short_token_uses_first_and_last_char"),
        pytest.param("12345678", "1...8", id="boundary_eight_char_token_is_short"),
        # 8 < len <= 10
        pytest.param(
            "abcdefghij", "ab...ij", id="medium_token_uses_first_and_last_two_chars"
        ),
        pytest.param("1234567890", "12...90", id="boundary_ten_char_token_is_medium"),
        pytest.param(
            "AKIAIOSFODNN7EXAMPLE",
            "AKIA...MPLE",
            id="long_token_uses_first_and_last_four_chars",
        ),
    ],
)
def test_redact_token(token: str, expected: str) -> None:
    assert redact_token(token) == expected


def test_redact_token_preserves_pattern_for_very_long_tokens() -> None:
    token = "ghp_" + "x" * 100 + "1234"
    redacted = redact_token(token)
    assert redacted == "ghp_...1234"


# ---------------------------------------------------------------------------
# Finding helper
# ---------------------------------------------------------------------------


def _make_finding(token: str | None) -> Finding:
    return Finding(
        type="TEST",
        snippet="TEST=redacted",
        evidence=EvidenceType.REGEX,
        severity=Severity.HIGH,
        token=token,
    )


# ---------------------------------------------------------------------------
# Finding.redacted_token delegates to the helper
# ---------------------------------------------------------------------------


def test_finding_redacted_token_matches_helper() -> None:
    finding = _make_finding("AKIAIOSFODNN7EXAMPLE")
    assert finding.redacted_token == redact_token(finding.token)


def test_finding_redacted_token_none_when_no_token() -> None:
    finding = _make_finding(None)
    assert finding.redacted_token is None
