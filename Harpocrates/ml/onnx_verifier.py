"""
ONNX Runtime inference engine for single-stage ML pipeline (v2.1.0).

Replaces native xgboost with onnxruntime:
  - Single ONNX model on all 65 features
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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FeatureVector, extract_features
from Harpocrates.ml.verifier import DEFAULT_MODEL_DIR, VerificationResult, Verifier

if TYPE_CHECKING:
    from Harpocrates.core.result import EvidenceType, Finding

logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = DEFAULT_MODEL_DIR / "model.onnx"
ONNX_HASHES_PATH = DEFAULT_MODEL_DIR / "onnx_model_hashes.json"
MODEL_CONFIG_PATH = DEFAULT_MODEL_DIR / "model_config.json"

_DEFAULT_THRESHOLD_LOW = 0.15
_DEFAULT_THRESHOLD_HIGH = 0.85


def _apply_platt(raw_prob: float, a: float, b: float) -> float:
    """Apply Platt sigmoid calibration: p = 1 / (1 + exp(a*f + b))."""
    if a == 0.0 and b == 0.0:
        return raw_prob
    return 1.0 / (1.0 + math.exp(a * raw_prob + b))


def _run_session(session, features: List[List[float]]) -> List[float]:
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
        return [float(p[1]) for p in probabilities]
    return np.asarray(probabilities)[:, 1].tolist()


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
        self._session = None
        self._loaded = False

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
        return ONNX_MODEL_PATH.exists()

    def _load_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            ) from exc

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self._model_path}. "
                "Run `python scripts/convert_to_onnx.py` to generate it."
            )

        # Load config for thresholds and Platt parameters
        if self._config_path.exists():
            with open(self._config_path, encoding="utf-8") as f:
                config = json.load(f)
            self._threshold_low = config.get("threshold_low", _DEFAULT_THRESHOLD_LOW)
            self._threshold_high = config.get("threshold_high", _DEFAULT_THRESHOLD_HIGH)
            self._platt_a = config.get("platt_a", 0.0)
            self._platt_b = config.get("platt_b", 0.0)

        # Hash verification with fail-closed policy controlled by env var
        import os
        require_hashes = os.getenv("HARPOCRATES_REQUIRE_HASHES", "").lower() in ("1", "true", "yes")

        if self._hashes_path.exists():
            with open(self._hashes_path, encoding="utf-8") as f:
                expected: Dict[str, str] = json.load(f)

            key = self._model_path.name
            if key not in expected:
                raise ValueError(
                    f"Hash manifest does not contain an entry for '{key}'. "
                    "Re-run scripts/convert_to_onnx.py to regenerate the manifest."
                )
            model_bytes = self._model_path.read_bytes()
            actual = hashlib.sha256(model_bytes).hexdigest()
            if actual != expected[key]:
                raise ValueError(
                    f"SHA-256 mismatch for '{key}': possible supply-chain tampering. "
                    f"Expected {expected[key][:16]}..., got {actual[:16]}..."
                )
            self._session = ort.InferenceSession(model_bytes)
        else:
            if require_hashes:
                raise ValueError(
                    f"Hash manifest not found at {self._hashes_path} and "
                    "HARPOCRATES_REQUIRE_HASHES is set. Cannot load model without "
                    "integrity verification."
                )
            self._session = ort.InferenceSession(str(self._model_path))
            logger.warning(
                "Hash manifest not found at %s — loading model without "
                "integrity verification.",
                self._hashes_path,
            )

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

    def _route(
        self, features: FeatureVector
    ) -> Tuple[bool, float, str]:
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
            features_used=dict(
                zip(FeatureVector.get_feature_names(), features.to_array())
            ),
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
]
