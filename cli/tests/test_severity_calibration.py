"""Tests for the classification-aware severity helper (Phase 6.1)."""
from __future__ import annotations

from Harpocrates.core.classification import CategoryInference, ViolationCategory
from Harpocrates.core.detector import _severity_from_classification
from Harpocrates.core.result import Severity


def _inf(cat: ViolationCategory, conf: float) -> CategoryInference:
    return CategoryInference(category=cat, reason="layer=test", confidence=conf)


# ---------------------------------------------------------------------------
# INFO band — no/weak signal
# ---------------------------------------------------------------------------

def test_low_confidence_generic_is_info() -> None:
    """cat_conf < 0.5 → INFO regardless of category."""
    assert _severity_from_classification(_inf(ViolationCategory.GENERIC_SECRET, 0.30)) == Severity.INFO


def test_specific_category_low_confidence_is_info() -> None:
    """Even a specific category at cat_conf 0.40 stays INFO."""
    assert _severity_from_classification(_inf(ViolationCategory.PASSWORD, 0.40)) == Severity.INFO


def test_generic_secret_moderate_confidence_is_medium() -> None:
    """GENERIC_SECRET at cat_conf 0.75 → MEDIUM."""
    assert _severity_from_classification(_inf(ViolationCategory.GENERIC_SECRET, 0.75)) == Severity.MEDIUM


# ---------------------------------------------------------------------------
# MEDIUM band — moderate signal
# ---------------------------------------------------------------------------

def test_specific_category_moderate_is_medium() -> None:
    """API_TOKEN at cat_conf 0.75 → MEDIUM."""
    assert _severity_from_classification(_inf(ViolationCategory.API_TOKEN, 0.75)) == Severity.MEDIUM


def test_specific_category_at_boundary_070_is_medium() -> None:
    assert _severity_from_classification(_inf(ViolationCategory.API_TOKEN, 0.70)) == Severity.MEDIUM


def test_specific_category_at_exactly_050_is_info() -> None:
    """cat_conf=0.50 is below the 0.70 MEDIUM floor → INFO."""
    assert _severity_from_classification(_inf(ViolationCategory.API_TOKEN, 0.50)) == Severity.INFO


def test_generic_secret_at_exactly_050_is_info() -> None:
    assert _severity_from_classification(_inf(ViolationCategory.GENERIC_SECRET, 0.50)) == Severity.INFO


# ---------------------------------------------------------------------------
# HIGH band — strong signal
# ---------------------------------------------------------------------------

def test_specific_category_strong_is_high() -> None:
    """API_TOKEN at cat_conf 0.85 → HIGH."""
    assert _severity_from_classification(_inf(ViolationCategory.API_TOKEN, 0.85)) == Severity.HIGH


def test_specific_category_at_boundary_085_is_high() -> None:
    assert _severity_from_classification(_inf(ViolationCategory.PASSWORD, 0.85)) == Severity.HIGH


def test_ml_high_confidence_promotes_moderate_specific_to_high() -> None:
    """ml_confidence 0.90 with API_TOKEN cat_conf 0.75 → HIGH."""
    assert (
        _severity_from_classification(
            _inf(ViolationCategory.API_TOKEN, 0.75), ml_confidence=0.90
        )
        == Severity.HIGH
    )


def test_ml_high_confidence_does_not_promote_generic_secret() -> None:
    """GENERIC_SECRET is never promoted to HIGH by ml_confidence alone."""
    result = _severity_from_classification(
        _inf(ViolationCategory.GENERIC_SECRET, 0.50), ml_confidence=0.95
    )
    assert result != Severity.HIGH


def test_ml_high_confidence_does_not_promote_generic_secret_at_reconstruction_default() -> None:
    """GENERIC_SECRET at 0.30 (the exact _inference_from_finding default) + ml_conf=0.99."""
    result = _severity_from_classification(
        _inf(ViolationCategory.GENERIC_SECRET, 0.30), ml_confidence=0.99
    )
    assert result != Severity.HIGH


# ---------------------------------------------------------------------------
# CRITICAL is unreachable
# ---------------------------------------------------------------------------

def test_critical_unreachable_at_max_confidence() -> None:
    """Even cat_conf=1.0 + ml_confidence=1.0 cannot produce CRITICAL."""
    result = _severity_from_classification(
        _inf(ViolationCategory.API_TOKEN, 1.0), ml_confidence=1.0
    )
    assert result == Severity.HIGH
    assert result != Severity.CRITICAL


# ---------------------------------------------------------------------------
# End-to-end: entropy candidates in detect_text pick up correct severity
# ---------------------------------------------------------------------------

def test_apim_secret_key_real_secret_severity_is_at_least_medium() -> None:
    """APIM_SECRET_KEY with a high-entropy value → MEDIUM or HIGH via lexicon."""
    from Harpocrates.core.detector import detect_text

    line = 'APIM_SECRET_KEY = "40z_9Yw7dtRMVxOr3KpNqBsE2mWa1lFg"\n'
    findings = detect_text(line)
    assert findings, "Expected at least one finding"
    for f in findings:
        assert f.severity in (Severity.MEDIUM, Severity.HIGH), (
            f"Expected MEDIUM or HIGH for APIM_SECRET_KEY, got {f.severity}"
        )


def test_entropy_path_produces_high_for_strong_category_signal() -> None:
    """detect_text entropy findings with strong cat_conf (>=0.85) must reach HIGH.

    _severity_from_entropy now delegates to _severity_from_classification so
    that well-named credentials (DB_PASSWORD, jwt_token) surface as HIGH
    without waiting for ML verification.  CRITICAL remains unreachable from
    the entropy path.
    """
    from Harpocrates.core.detector import detect_text
    from Harpocrates.core.result import EvidenceType

    line = 'db_password = "Xk9mQ2vRpLwYhN3cD7bJsTqFuEiAo8WP"\n'
    findings = detect_text(line)
    entropy_findings = [f for f in findings if f.evidence in (EvidenceType.ENTROPY, EvidenceType.ML)]
    assert any(f.severity == Severity.HIGH for f in entropy_findings), (
        "Strong-signal entropy finding (db_password, cat_conf=0.90) must be HIGH"
    )
    for f in entropy_findings:
        assert f.severity != Severity.CRITICAL, (
            f"CRITICAL is unreachable from entropy path; got {f.severity}"
        )


def test_authenticated_enum_does_not_become_critical() -> None:
    """AUTHENTICATED = 'authenticated' must not surface as CRITICAL."""
    from Harpocrates.core.detector import detect_text
    from Harpocrates.core.result import EvidenceType

    line = 'AUTHENTICATED = "authenticated"\n'
    findings = detect_text(line)
    ml_or_entropy = [f for f in findings if f.evidence in (EvidenceType.ENTROPY, EvidenceType.ML)]
    for f in ml_or_entropy:
        assert f.severity != Severity.CRITICAL, (
            f"AUTHENTICATED enum got CRITICAL severity: {f}"
        )
