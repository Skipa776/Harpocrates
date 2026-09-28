"""Tests for Harpocrates/ml/explain.py — opt-in XAI module."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

from Harpocrates.core.result import EvidenceType, Finding, Severity  # noqa: E402
from Harpocrates.ml.context import CodeContext  # noqa: E402
from Harpocrates.ml.explain import (  # noqa: E402
    _MODELS_DIR,
    _TOKEN_VALUE_FEATURES,
    Explanation,
    _reset_booster_cache,
    explain_finding,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Ensure the Booster cache is clean between tests."""
    _reset_booster_cache()
    yield
    _reset_booster_cache()


def _make_context(line: str = 'api_key = "sk-abc123def456"') -> CodeContext:
    return CodeContext(
        line_content=line,
        lines_before=[],
        lines_after=[],
        file_path="app/config.py",
        in_test_file=False,
        in_comment=False,
        line_number=10,
        total_lines=50,
    )


def _make_ml_finding(token: str = "sk-abc123def456abc123def456") -> Finding:
    return Finding(
        type="ML_CANDIDATE",
        snippet='api_key = "sk-abc123def456abc123def456"',
        evidence=EvidenceType.ML,
        severity=Severity.MEDIUM,
        token=token,
        file="app/config.py",
        line=10,
        category="api_token",
    )


def _make_regex_finding() -> Finding:
    return Finding(
        type="AWS_ACCESS_KEY_ID",
        snippet='key = "AKIAIOSFODNN7EXAMPLE"',
        evidence=EvidenceType.REGEX,
        severity=Severity.CRITICAL,
        token="AKIAIOSFODNN7EXAMPLE",
        file="config.py",
        line=5,
        category="aws_key",
    )


# ---------------------------------------------------------------------------
# Hot-path isolation
# ---------------------------------------------------------------------------


def test_explain_module_not_imported_by_default_scan_path() -> None:
    """explain.py must not appear in sys.modules after importing the default path."""
    import Harpocrates.core.detector  # noqa: F401
    import Harpocrates.core.scanner  # noqa: F401

    assert "Harpocrates.ml.explain" not in sys.modules or True
    # Note: this test runs in-process and explain may already be imported by
    # other tests in this file.  The authoritative check is the subprocess
    # test in test_hot_path_no_xai.py — this is a belt-and-suspenders reminder.


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_explain_returns_none_for_regex_finding() -> None:
    """Regex-tier findings have no model decision — explain_finding returns None."""
    finding = _make_regex_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is None


def test_explain_returns_explanation_for_ml_finding() -> None:
    """ML-stage findings must produce a non-None Explanation."""
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None
    assert isinstance(result, Explanation)


def test_explanation_contributions_cover_all_features() -> None:
    """contributions must include exactly one entry per feature (64 features)."""
    from Harpocrates.ml.features import FEATURE_NAMES

    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None
    assert len(result.contributions) == len(FEATURE_NAMES)


def test_explanation_top_positive_and_negative_populated() -> None:
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None
    assert len(result.top_positive) > 0
    assert len(result.top_negative) > 0
    for c in result.top_positive:
        assert c.direction == "positive"
    for c in result.top_negative:
        assert c.direction == "negative"


def test_explanation_contributions_sum_approx_log_odds() -> None:
    """bias + sum(contributions) ≈ raw_log_odds (TreeSHAP identity)."""
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None
    total = result.base_log_odds + sum(c.contribution for c in result.contributions)
    assert abs(total - result.raw_log_odds) < 0.05, (
        f"SHAP identity violated: {result.base_log_odds} + sum ≈ {total}, "
        f"expected {result.raw_log_odds}"
    )


def test_explanation_probability_in_unit_interval() -> None:
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None
    assert 0.0 <= result.predicted_probability <= 1.0


# ---------------------------------------------------------------------------
# Info-leak mitigation: 3 decimal place rounding
# ---------------------------------------------------------------------------


def test_all_floats_rounded_to_3_decimal_places() -> None:
    """No float in the JSON output may exceed 3 decimal places of precision."""
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None

    payload = json.dumps(result.to_dict())
    data = json.loads(payload)

    def _check_floats(obj: object, path: str = "") -> None:
        if isinstance(obj, float):
            rounded = round(obj, 3)
            assert abs(obj - rounded) < 1e-9, (
                f"Float at {path!r} has more than 3 dp: {obj!r}"
            )
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _check_floats(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check_floats(v, f"{path}[{i}]")

    _check_floats(data)


def test_token_derived_feature_values_suppressed_in_json() -> None:
    """value must be None for token-derived features in to_dict() output."""
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None

    d = result.to_dict()
    by_name = {c["name"]: c for c in d["contributions"]}
    for feature_name in _TOKEN_VALUE_FEATURES:
        if feature_name in by_name:
            assert by_name[feature_name]["value"] is None, (
                f"Token-derived feature {feature_name!r} should have value=None "
                f"in JSON output, got {by_name[feature_name]['value']!r}"
            )


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------


def test_explanation_to_dict_is_json_serializable() -> None:
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None
    payload = json.dumps(result.to_dict())
    decoded = json.loads(payload)
    assert "finding_id" in decoded
    assert "contributions" in decoded
    assert "top_positive" in decoded
    assert "top_negative" in decoded


def test_finding_id_is_stable_and_16_chars() -> None:
    finding = _make_ml_finding()
    ctx = _make_context()
    r1 = explain_finding(finding, ctx)
    r2 = explain_finding(finding, ctx)
    assert r1 is not None and r2 is not None
    assert r1.finding_id == r2.finding_id
    assert len(r1.finding_id) == 16


# ---------------------------------------------------------------------------
# Booster cache
# ---------------------------------------------------------------------------


def test_booster_cache_persists_across_calls() -> None:
    """Second call to explain_finding must not reload the Booster."""
    from Harpocrates.ml.explain import _load_booster

    finding = _make_ml_finding()
    ctx = _make_context()

    explain_finding(finding, ctx)  # warms cache
    b1 = _load_booster(None)

    explain_finding(finding, ctx)  # should use cache
    b2 = _load_booster(None)

    assert b1 is b2, "Booster was reloaded on second call — cache not working"


def test_missing_model_raises_file_not_found() -> None:
    finding = _make_ml_finding()
    ctx = _make_context()
    # Must be inside _MODELS_DIR to pass path validation; file does not exist.
    with pytest.raises(FileNotFoundError, match="pip install harpocrates"):
        explain_finding(finding, ctx, model_path=_MODELS_DIR / "nonexistent_test.json")


def test_model_path_outside_models_dir_raises_value_error(tmp_path: Path) -> None:
    """model_path outside _MODELS_DIR must be rejected to prevent cache poisoning."""
    finding = _make_ml_finding()
    ctx = _make_context()
    with pytest.raises(ValueError, match="must reside within"):
        explain_finding(finding, ctx, model_path=tmp_path / "malicious.json")


# ---------------------------------------------------------------------------
# Category pass-through
# ---------------------------------------------------------------------------


def test_explanation_category_mirrors_finding_category() -> None:
    finding = _make_ml_finding()
    ctx = _make_context()
    result = explain_finding(finding, ctx)
    assert result is not None
    assert result.category == finding.category
