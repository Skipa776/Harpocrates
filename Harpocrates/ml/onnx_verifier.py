"""
ONNX Runtime inference engine for single-stage ML pipeline (v2.1.0).

Replaces native xgboost with onnxruntime:
  - Single ONNX model on all 64 features
  - Dual-threshold decision: SAFE / REVIEW / SECRET
  - Platt-calibrated probabilities from model_config.json
  - SHA-256 hash verification guards against supply-chain attacks

Usage:
    from Harpocrates.ml.onnx_verifier import OnnxVerifier
    verifier = OnnxVerifier.get_instance()
    result = verifier.verify(finding, context)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FEATURE_NAMES, FeatureVector, extract_features
from Harpocrates.ml.verifier import DEFAULT_MODEL_DIR, VerificationResult, Verifier

if TYPE_CHECKING:
    from Harpocrates.core.result import EvidenceType, Finding

logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = DEFAULT_MODEL_DIR / "model.onnx"
ONNX_HASHES_PATH = DEFAULT_MODEL_DIR / "onnx_model_hashes.json"
MODEL_CONFIG_PATH = DEFAULT_MODEL_DIR / "model_config.json"

_DEFAULT_THRESHOLD_LOW = 0.15
_DEFAULT_THRESHOLD_HIGH = 0.85


class OnnxModelSchemaError(ValueError):
    """Raised when model, config, and runtime feature widths disagree."""


def _validate_model_config(
    config: dict[str, Any],
) -> tuple[float, float, float, float, Optional[int]]:
    """Validate routing policy before it can influence model decisions."""

    def finite_number(key: str, default: float) -> float:
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise OnnxModelSchemaError(f"model_config.json '{key}' must be a finite number")
        return float(value)

    threshold_low = finite_number("threshold_low", _DEFAULT_THRESHOLD_LOW)
    threshold_high = finite_number("threshold_high", _DEFAULT_THRESHOLD_HIGH)
    if not 0.0 <= threshold_low < threshold_high <= 1.0:
        raise OnnxModelSchemaError(
            "model_config.json thresholds must satisfy 0 <= threshold_low < threshold_high <= 1"
        )
    platt_a = finite_number("platt_a", 0.0)
    platt_b = finite_number("platt_b", 0.0)
    feature_count = config.get("feature_count")
    if feature_count is not None and (type(feature_count) is not int or feature_count <= 0):
        raise OnnxModelSchemaError("model_config.json 'feature_count' must be a positive integer")
    return threshold_low, threshold_high, platt_a, platt_b, feature_count


def _verify_file_hash(path: Path, expected: Dict[str, str]) -> bytes:
    """Return verified bytes for one manifest-tracked ML artifact."""
    key = path.name
    if key not in expected:
        raise ValueError(
            f"Hash manifest does not contain an entry for '{key}'. "
            "Re-run scripts/convert_to_onnx.py to regenerate the manifest."
        )
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected[key]:
        raise ValueError(
            f"SHA-256 mismatch for '{key}': possible supply-chain tampering. "
            f"Expected {expected[key][:16]}..., got {actual[:16]}..."
        )
    return content


def _validate_model_schema(session, configured_feature_count: Optional[int]) -> int:
    """Validate the loaded session against the live Python feature contract."""
    model_input = session.get_inputs()[0]
    model_width = model_input.shape[-1]
    expected_width = len(FEATURE_NAMES)
    if not isinstance(model_width, int):
        raise OnnxModelSchemaError(
            f"ONNX model input width must be fixed, got {model_width!r} for '{model_input.name}'."
        )
    if configured_feature_count is not None and configured_feature_count != expected_width:
        raise OnnxModelSchemaError(
            "ML feature schema mismatch: model_config.json declares "
            f"{configured_feature_count} features but the extractor emits "
            f"{expected_width}."
        )
    if model_width != expected_width:
        raise OnnxModelSchemaError(
            "ML feature schema mismatch: ONNX model expects "
            f"{model_width} features but the extractor emits {expected_width}. "
            "Re-run scripts/convert_to_onnx.py."
        )
    return model_width


def _apply_platt(raw_prob: float, a: float, b: float) -> float:
    """Apply Platt sigmoid calibration: p = 1 / (1 + exp(a*f + b))."""
    if a == 0.0 and b == 0.0:
        return raw_prob
    return 1.0 / (1.0 + math.exp(a * raw_prob + b))


def _run_session(session: Any, features: List[List[float]]) -> List[float]:
    """Run ONNX InferenceSession and extract positive-class probabilities."""
    import numpy as np

    input_name = session.get_inputs()[0].name
    features_np = np.array(features, dtype=np.float32)
    outputs = session.run(None, {input_name: features_np})

    if len(outputs) < 2:
        raise ValueError(
            f"ONNX model returned {len(outputs)} output(s); expected 2 "
            "(labels at index 0, probabilities at index 1). "
            "Re-convert with onnxmltools ensuring ZipMap output is present."
        )

    probabilities = outputs[1]
    if (
        isinstance(probabilities, list)
        and len(probabilities) > 0
        and isinstance(probabilities[0], dict)
    ):
        positive_probabilities = [float(p[1]) for p in probabilities]
    else:
        positive_probabilities = [
            float(value) for value in np.asarray(probabilities)[:, 1].tolist()
        ]
    if len(positive_probabilities) != len(features):
        raise ValueError(
            "ONNX model returned an unexpected probability count: "
            f"expected {len(features)}, got {len(positive_probabilities)}"
        )
    if any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in positive_probabilities
    ):
        raise ValueError("ONNX model returned a non-finite or out-of-range probability")
    return positive_probabilities


class OnnxVerifier(Verifier):
    """
    ONNX-based single-stage verifier (v2.1.0).

    Decision logic:
      P(secret) < threshold_low  -> SAFE    (exit 0, silent)
      P(secret) > threshold_high -> SECRET  (exit 1, hard block)
      otherwise                  -> REVIEW  (exit 1, override available)
    """

    _instance: Optional["OnnxVerifier"] = None

    def __init__(
        self,
        model_path: Optional[Path] = None,
        hashes_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        lazy_load: bool = True,
    ):
        self._model_path = model_path or ONNX_MODEL_PATH
        self._hashes_path = hashes_path or ONNX_HASHES_PATH
        self._config_path = config_path or MODEL_CONFIG_PATH
        self._session: Any = None
        self._loaded = False
        self._input_feature_count: Optional[int] = None

        self._threshold_low = _DEFAULT_THRESHOLD_LOW
        self._threshold_high = _DEFAULT_THRESHOLD_HIGH
        self._platt_a = 0.0
        self._platt_b = 0.0

        if not lazy_load:
            self._load_session()

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[Path] = None,
    ) -> "OnnxVerifier":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    @classmethod
    def is_available(cls) -> bool:
        return bool(ONNX_MODEL_PATH.exists())

    def _load_session(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. Install with: pip install onnxruntime"
            ) from exc

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self._model_path}. "
                "Run `python scripts/convert_to_onnx.py` to generate it."
            )

        if not self._hashes_path.exists():
            raise FileNotFoundError(
                f"ONNX hash manifest not found: {self._hashes_path}. "
                "Refusing to load an unverified model."
            )
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"ONNX model config not found: {self._config_path}. "
                "Refusing to use implicit routing policy."
            )

        with open(self._hashes_path) as f:
            expected: Dict[str, str] = json.load(f)
        model_bytes = _verify_file_hash(self._model_path, expected)
        config_bytes = _verify_file_hash(self._config_path, expected)
        config = json.loads(config_bytes)
        (
            self._threshold_low,
            self._threshold_high,
            self._platt_a,
            self._platt_b,
            config_width,
        ) = _validate_model_config(config)
        self._session = ort.InferenceSession(model_bytes)

        self._input_feature_count = _validate_model_schema(self._session, config_width)
        self._loaded = True
        logger.info("Loaded ONNX session: %s", self._model_path.name)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_session()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def threshold(self) -> float:
        return self._threshold_high

    @property
    def input_feature_count(self) -> int:
        """Return the validated width expected by the loaded ONNX model."""
        self._ensure_loaded()
        if self._input_feature_count is None:
            raise RuntimeError("ONNX input schema was not initialized")
        return self._input_feature_count

    def _route(self, features: FeatureVector) -> Tuple[bool, float, str]:
        """Single-model routing with Platt calibration and dual thresholds."""
        self._ensure_loaded()

        raw_prob = _run_session(self._session, [features.to_array()])[0]
        prob = _apply_platt(raw_prob, self._platt_a, self._platt_b)

        if prob < self._threshold_low:
            return False, 1.0 - prob, "safe"
        if prob > self._threshold_high:
            return True, prob, "blocked"
        return True, prob, "review"

    def _combine_confidence(
        self,
        original: float,
        ml: float,
        evidence_type: "EvidenceType",
    ) -> float:
        from Harpocrates.core.result import EvidenceType

        if evidence_type == EvidenceType.REGEX:
            return 0.6 * original + 0.4 * ml
        return 0.3 * original + 0.7 * ml

    def verify(
        self,
        finding: "Finding",
        context: CodeContext,
    ) -> VerificationResult:
        self._ensure_loaded()

        features = extract_features(finding, context)
        is_secret, ml_confidence, routing = self._route(features)

        original_confidence = finding.confidence or 0.5
        combined_confidence = self._combine_confidence(
            original_confidence, ml_confidence, finding.evidence
        )

        label = "likely secret" if is_secret else "likely safe"
        explanation = f"{label} ({ml_confidence:.0%} confidence) [ONNX/{routing}]"

        return VerificationResult(
            is_secret=is_secret,
            ml_confidence=ml_confidence,
            original_confidence=original_confidence,
            combined_confidence=combined_confidence,
            features_used=dict(zip(FeatureVector.get_feature_names(), features.to_array())),
            explanation=explanation,
        )

    def verify_batch(
        self,
        findings_with_context: List[Tuple["Finding", CodeContext]],
    ) -> List[VerificationResult]:
        if not findings_with_context:
            return []
        self._ensure_loaded()

        all_features = [extract_features(f, ctx) for f, ctx in findings_with_context]

        raw_probs = _run_session(
            self._session,
            [fv.to_array() for fv in all_features],
        )

        results = []
        for i, (finding, _ctx) in enumerate(findings_with_context):
            prob = _apply_platt(raw_probs[i], self._platt_a, self._platt_b)

            if prob < self._threshold_low:
                is_secret, ml_confidence, routing = False, 1.0 - prob, "safe"
            elif prob > self._threshold_high:
                is_secret, ml_confidence, routing = True, prob, "blocked"
            else:
                is_secret, ml_confidence, routing = True, prob, "review"

            original_confidence = finding.confidence or 0.5
            combined_confidence = self._combine_confidence(
                original_confidence, ml_confidence, finding.evidence
            )
            label = "likely secret" if is_secret else "likely safe"
            results.append(
                VerificationResult(
                    is_secret=is_secret,
                    ml_confidence=ml_confidence,
                    original_confidence=original_confidence,
                    combined_confidence=combined_confidence,
                    features_used=dict(
                        zip(
                            FeatureVector.get_feature_names(),
                            all_features[i].to_array(),
                        )
                    ),
                    explanation=f"{label} ({ml_confidence:.0%} confidence) [ONNX/{routing}]",
                )
            )
        return results


__all__ = [
    "OnnxVerifier",
    "ONNX_MODEL_PATH",
    "ONNX_HASHES_PATH",
    "OnnxModelSchemaError",
    "_validate_model_config",
]
