"""
Opt-in explainability for ML-stage findings via native XGBoost TreeSHAP.

This module is imported ONLY when --explain (CLI) or include_contributions=True
(MCP) is passed. The default scan path never imports it — enforced by
tests/test_hot_path_no_xai.py (subprocess + AST layers).

Architectural contracts:
- No shap/lime package imports. Native XGBoost pred_contribs=True only.
- All imports are lazy / inside functions — never triggered by core/detector.
- All float outputs are rounded to 3 decimal places (SHAP info-leak
  mitigation: high-precision values on token-derived features could allow
  mathematical reconstruction of the redacted token via Shannon entropy).
- The XGBoost Booster is cached per resolved path with double-checked locking
  (thread-safe for future async/multi-threaded MCP transports).
- model_path overrides must reside within _MODELS_DIR (cache-poisoning guard).
"""
from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional

if TYPE_CHECKING:
    import xgboost as xgb  # type: ignore[import-untyped]

    from Harpocrates.core.result import Finding
    from Harpocrates.ml.context import CodeContext

_MODELS_DIR = Path(__file__).parent / "models"
_DEFAULT_XGBOOST_PATH = _MODELS_DIR / "xgboost_model.json"
_DEFAULT_HASHES_PATH = _MODELS_DIR / "onnx_model_hashes.json"

# Token-derived continuous features whose values are suppressed in JSON output.
# These encode length, entropy, and character-class distributions that, even
# at 3 decimal places, narrow the token search space enough to assist targeted
# brute-force reconstruction of the redacted token.
_TOKEN_VALUE_FEATURES: FrozenSet[str] = frozenset({
    "token_length",
    "token_entropy",
    "char_class_count",
    "digit_ratio",
    "special_char_ratio",
})

# Per-resolved-path Booster cache — loaded once per (process, path) on first
# --explain call. Keyed by resolved path to prevent stale-cache bugs when
# model_path is overridden between calls.
_BOOSTER_CACHE: Dict[Path, "xgb.Booster"] = {}
_BOOSTER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's SHAP contribution to the model's log-odds output."""

    name: str
    index: int
    value: float        # raw feature value at inference time (3 dp; None in JSON for token-derived features)
    contribution: float  # SHAP value — log-odds delta vs baseline (3 dp)
    direction: str      # "positive" (toward secret) | "negative"


@dataclass(frozen=True)
class Explanation:
    """
    Structured explanation of one ML-stage finding.

    `base_log_odds` + sum(c.contribution for c in contributions) ≈ raw_log_odds.
    `predicted_probability` = sigmoid(raw_log_odds).

    All floats rounded to 3 decimal places — prevents reconstruction of
    the redacted token from high-precision entropy / length values.
    Token-derived feature values (token_length, token_entropy, char_class_count,
    digit_ratio, special_char_ratio) are suppressed to None in to_dict() output.

    Findings produced by the regex tier (evidence != ML) return None from
    explain_finding — they have no model decision to explain.
    """

    finding_id: str
    category: Optional[str]
    base_log_odds: float
    raw_log_odds: float
    predicted_probability: float
    contributions: List[FeatureContribution]  # all 64 features, canonical order
    top_positive: List[FeatureContribution]   # top-k secret-pushing features
    top_negative: List[FeatureContribution]   # top-k not-secret-pushing features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "base_log_odds": self.base_log_odds,
            "raw_log_odds": self.raw_log_odds,
            "predicted_probability": self.predicted_probability,
            "contributions": [_serialize_contribution(c) for c in self.contributions],
            "top_positive": [_serialize_contribution(c) for c in self.top_positive],
            "top_negative": [_serialize_contribution(c) for c in self.top_negative],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_finding(
    finding: "Finding",
    context: "CodeContext",
    *,
    model_path: Optional[Path] = None,
    top_k: int = 5,
) -> Optional[Explanation]:
    """
    Compute TreeSHAP feature contributions for a single ML-stage finding.

    Returns None when the finding was not produced by the ML stage (regex-tier
    findings have no model decision to explain).

    Lazy-imports xgboost — cost paid only on the first call in a process.
    Subsequent calls use the cached Booster (~0.2ms each).

    Args:
        finding:    The Finding to explain.
        context:    The CodeContext used during inference.
        model_path: Override path to xgboost_model.json. Must reside within
                    Harpocrates/ml/models/ (security: prevents cache poisoning).
                    Defaults to the bundled artifact.
        top_k:      Number of features to include in top_positive/top_negative.

    Returns:
        Explanation dataclass or None.
    """
    import numpy as np
    import xgboost as xgb  # lazy — never executed on default scan path

    from Harpocrates.core.result import EvidenceType
    from Harpocrates.ml.features import FEATURE_NAMES, extract_features

    if finding.evidence != EvidenceType.ML:
        return None

    features = extract_features(finding, context).to_array()  # shape (64,)

    booster = _load_booster(model_path)
    dmat = xgb.DMatrix(
        np.array(features, dtype=np.float32).reshape(1, -1),
        feature_names=list(FEATURE_NAMES),
    )

    # pred_contribs=True → shape (1, n_features + 1); last column is bias.
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    bias = float(contribs[-1])
    feat_contribs = contribs[:-1]
    raw_log_odds = float(bias + feat_contribs.sum())

    rows = [
        FeatureContribution(
            name=name,
            index=i,
            value=round(float(features[i]), 3),
            contribution=round(float(feat_contribs[i]), 3),
            direction="positive" if feat_contribs[i] > 0 else "negative",
        )
        for i, name in enumerate(FEATURE_NAMES)
    ]

    top_pos = sorted(
        (r for r in rows if r.contribution > 0), key=lambda r: -r.contribution
    )[:top_k]
    top_neg = sorted(
        (r for r in rows if r.contribution < 0), key=lambda r: r.contribution
    )[:top_k]

    return Explanation(
        finding_id=_finding_id(finding),
        category=finding.category,
        base_log_odds=round(bias, 3),
        raw_log_odds=round(raw_log_odds, 3),
        predicted_probability=round(_sigmoid(raw_log_odds), 3),
        contributions=rows,
        top_positive=top_pos,
        top_negative=top_neg,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialize_contribution(c: FeatureContribution) -> Dict[str, Any]:
    """Serialize a FeatureContribution, suppressing value for token-derived features."""
    d = asdict(c)
    if c.name in _TOKEN_VALUE_FEATURES:
        d["value"] = None
    return d


def _load_booster(model_path: Optional[Path]) -> "xgb.Booster":
    """
    Load and cache an XGBoost Booster. Thread-safe via double-checked locking.

    Cache is keyed by resolved path so different model_path overrides each get
    their own cache slot (prevents stale-cache bugs on test overrides).

    Security: model_path, if provided, must reside within _MODELS_DIR to
    prevent untrusted callers from poisoning the cache with a malicious model.
    """
    import xgboost as xgb

    path = model_path or _DEFAULT_XGBOOST_PATH

    if model_path is not None:
        try:
            path.resolve().relative_to(_MODELS_DIR.resolve())
        except ValueError:
            raise ValueError(
                f"model_path must reside within {_MODELS_DIR}. "
                f"Got: {model_path}"
            )

    resolved = path.resolve()

    if resolved in _BOOSTER_CACHE:
        return _BOOSTER_CACHE[resolved]

    with _BOOSTER_LOCK:
        if resolved in _BOOSTER_CACHE:  # double-checked
            return _BOOSTER_CACHE[resolved]

        if not path.exists():
            raise FileNotFoundError(
                f"TreeSHAP requires {path}. "
                "Reinstall with `pip install harpocrates[ml]` or rebuild via "
                "`python -m Harpocrates.training.train_model`."
            )

        booster = xgb.Booster()
        booster.load_model(str(path))
        _BOOSTER_CACHE[resolved] = booster
        return booster


def _reset_booster_cache() -> None:
    """Reset the Booster cache. Intended for tests only."""
    with _BOOSTER_LOCK:
        _BOOSTER_CACHE.clear()


def _finding_id(finding: "Finding") -> str:
    """Stable 16-char hex identifier for a finding — token participates only in hashing."""
    raw = f"{finding.file}|{finding.line}|{finding.token or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


__all__ = [
    "Explanation",
    "FeatureContribution",
    "explain_finding",
]
