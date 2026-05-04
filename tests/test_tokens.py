"""Tests for TokenMatch dataclass, CodeContext.token_match, and Finding offsets."""
from __future__ import annotations

import pytest

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.tokens import TokenMatch


def test_tokenmatch_is_frozen() -> None:
    """TokenMatch must be immutable (frozen=True)."""
    tm = TokenMatch(token="abc", start=5, end=8)
    with pytest.raises((AttributeError, TypeError)):
        tm.start = 0  # type: ignore[misc]


def test_tokenmatch_kind_defaults_to_none() -> None:
    """kind field is optional and defaults to None."""
    tm = TokenMatch(token="tok", start=0, end=3)
    assert tm.kind is None


def test_codecontext_token_match_default_none() -> None:
    """CodeContext.token_match must default to None for backward compat."""
    ctx = CodeContext(line_content="some line")
    assert ctx.token_match is None


def test_codecontext_token_match_assignable() -> None:
    """CodeContext.token_match can be set after construction (not frozen)."""
    ctx = CodeContext(line_content="api_key = 'secretvalue'")
    tm = TokenMatch(token="secretvalue", start=11, end=22, kind="sensitive_assignment")
    ctx.token_match = tm
    assert ctx.token_match is tm
    assert ctx.token_match.start == 11


def test_finding_token_offsets_default_none() -> None:
    """New token_start / token_end fields default to None."""
    f = Finding(type="T", snippet="s", evidence=EvidenceType.REGEX)
    assert f.token_start is None
    assert f.token_end is None


def test_finding_token_offsets_excluded_from_json_dict() -> None:
    """token_start and token_end are internal scaffolding, not API contract."""
    f = Finding(
        type="AWS_ACCESS_KEY_ID",
        snippet="AKIAIOSFODNN7EXAMPLE",
        evidence=EvidenceType.REGEX,
        token="AKIAIOSFODNN7EXAMPLE",
        token_start=0,
        token_end=20,
    )
    d = f.to_json_dict(include_token=True)
    assert "token_start" not in d
    assert "token_end" not in d
    assert "token" in d  # token itself present when include_token=True


def test_finding_token_offsets_excluded_without_include_token() -> None:
    """Offsets absent from JSON dict in default (redacted) mode too."""
    f = Finding(
        type="T",
        snippet="s",
        evidence=EvidenceType.REGEX,
        token_start=1,
        token_end=5,
    )
    d = f.to_json_dict()
    assert "token_start" not in d
    assert "token_end" not in d
    assert "token" not in d


def test_detector_populates_token_offsets_for_regex_findings() -> None:
    """CRITICAL and HIGH regex findings must have token_start/token_end populated."""
    from Harpocrates.core.detector import _collect_text_findings

    github_token = "ghp_" + "a" * 36
    line = f"token = {github_token}"
    findings = _collect_text_findings(line)

    regex_findings = [f for f in findings if f.evidence == EvidenceType.REGEX]
    assert regex_findings, "expected at least one REGEX finding"

    for f in regex_findings:
        assert f.token_start is not None, f"token_start is None for finding {f.type}"
        assert f.token_end is not None, f"token_end is None for finding {f.type}"
        assert f.token_start >= 0
        assert f.token_end > f.token_start
        # Offset round-trip: the token must live at the declared position in stripped line
        stripped = line.strip()
        assert stripped[f.token_start : f.token_end] == f.token


def test_entropy_candidate_has_no_offsets() -> None:
    """ENTROPY_CANDIDATE findings intentionally have token_start=None (ship-dark contract).

    _TOKEN_RE.findall returns strings, not match objects, so entropy candidates
    cannot carry offsets yet. This test pins that invariant so a future accidental
    change doesn't silently populate wrong scan_text-relative offsets.
    """
    from Harpocrates.core.detector import _collect_text_findings

    # High-entropy token that passes looks_like_secret but has no sensitive variable name
    # so it will generate ENTROPY_CANDIDATE, not ML_CANDIDATE
    line = "value = aB3xK9mZqR7wN2pL5vY8cF1hD4gJ6tU0sE"  # mixed-case random string

    findings = _collect_text_findings(line)

    entropy_findings = [f for f in findings if f.evidence == EvidenceType.ENTROPY]
    if not entropy_findings:
        pytest.skip("no ENTROPY_CANDIDATE generated for this input")

    for f in entropy_findings:
        assert f.token_start is None, (
            "ENTROPY_CANDIDATE must have token_start=None until v0.3 findall→finditer migration"
        )
        assert f.token_end is None


def test_detector_populates_token_offsets_for_sensitive_assignment() -> None:
    """ML_CANDIDATE findings from _SENSITIVE_ASSIGNMENT_RE must have offsets."""
    from Harpocrates.core.detector import _collect_text_findings

    line = 'password = "hunter2_long_enough_yes"'
    findings = _collect_text_findings(line)

    ml_findings = [f for f in findings if f.evidence == EvidenceType.ML]
    if not ml_findings:
        pytest.skip("no ML_CANDIDATE generated for this input (entropy threshold not met)")

    for f in ml_findings:
        assert f.token_start is not None
        assert f.token_end is not None
        assert f.token_end > f.token_start
