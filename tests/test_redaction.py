"""Tests for Harpocrates.utils.redaction (PRD-01 Task 4)."""
from __future__ import annotations

import pytest

from Harpocrates.core.result import EvidenceType, Finding, Severity
from Harpocrates.utils.redaction import (
    redact_finding,
    redact_findings,
    redact_token,
)


# ---------------------------------------------------------------------------
# redact_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, ""])
def test_redact_token_returns_none_for_empty_input(value) -> None:
    assert redact_token(value) is None


def test_redact_token_short_token_uses_first_and_last_char() -> None:
    # len == 8 -> short: first char + "..." + last char
    assert redact_token("abcdefgh") == "a...h"


def test_redact_token_boundary_eight_char_token_is_short() -> None:
    assert redact_token("12345678") == "1...8"


def test_redact_token_medium_token_uses_first_and_last_two_chars() -> None:
    # 8 < len <= 10
    assert redact_token("abcdefghij") == "ab...ij"


def test_redact_token_boundary_ten_char_token_is_medium() -> None:
    assert redact_token("1234567890") == "12...90"


def test_redact_token_long_token_uses_first_and_last_four_chars() -> None:
    assert redact_token("AKIAIOSFODNN7EXAMPLE") == "AKIA...MPLE"


def test_redact_token_preserves_pattern_for_very_long_tokens() -> None:
    token = "ghp_" + "x" * 100 + "1234"
    redacted = redact_token(token)
    assert redacted == "ghp_...1234"


# ---------------------------------------------------------------------------
# redact_finding
# ---------------------------------------------------------------------------


def _make_finding(token: str | None) -> Finding:
    return Finding(
        type="TEST",
        snippet="TEST=redacted",
        evidence=EvidenceType.REGEX,
        severity=Severity.HIGH,
        token=token,
    )


def test_redact_finding_returns_new_instance_with_redacted_token() -> None:
    original = _make_finding("AKIAIOSFODNN7EXAMPLE")
    redacted = redact_finding(original)

    assert redacted is not original
    assert redacted.token == "AKIA...MPLE"
    # Original must not be mutated.
    assert original.token == "AKIAIOSFODNN7EXAMPLE"


def test_redact_finding_preserves_all_non_token_fields() -> None:
    original = _make_finding("AKIAIOSFODNN7EXAMPLE")
    redacted = redact_finding(original)

    assert redacted.type == original.type
    assert redacted.snippet == original.snippet
    assert redacted.evidence == original.evidence
    assert redacted.severity == original.severity


def test_redact_finding_handles_missing_token() -> None:
    original = _make_finding(None)
    redacted = redact_finding(original)

    assert redacted.token is None
    assert redacted is not original


# ---------------------------------------------------------------------------
# redact_findings
# ---------------------------------------------------------------------------


def test_redact_findings_returns_list_in_order() -> None:
    findings = [
        _make_finding("abcdefgh"),
        _make_finding(None),
        _make_finding("AKIAIOSFODNN7EXAMPLE"),
    ]

    result = redact_findings(findings)

    assert [f.token for f in result] == ["a...h", None, "AKIA...MPLE"]


def test_redact_findings_accepts_any_iterable() -> None:
    findings = (_make_finding("abcdefgh"), _make_finding("1234567890"))
    result = redact_findings(iter(findings))

    assert isinstance(result, list)
    assert [f.token for f in result] == ["a...h", "12...90"]


# ---------------------------------------------------------------------------
# Finding.redacted_token delegates to the helper
# ---------------------------------------------------------------------------


def test_finding_redacted_token_matches_helper() -> None:
    finding = _make_finding("AKIAIOSFODNN7EXAMPLE")
    assert finding.redacted_token == redact_token(finding.token)


def test_finding_redacted_token_none_when_no_token() -> None:
    finding = _make_finding(None)
    assert finding.redacted_token is None
