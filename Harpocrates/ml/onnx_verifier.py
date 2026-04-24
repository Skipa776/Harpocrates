"""
ONNX Runtime inference engine for the two-stage ML pipeline (v0.2.0).

Replaces native xgboost/lightgbm with onnxruntime:
  - Model files <500 KB each
  - Single-row inference <1 ms (C-backend tree traversal)
  - Uniform runtime for both Stage A and Stage B
  - SHA-256 hash verification guards against supply-chain attacks

Usage after conversion:
    from Harpocrates.ml.onnx_verifier import OnnxTwoStageVerifier
    verifier = OnnxTwoStageVerifier.get_instance()
    result = verifier.verify(finding, context)
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FeatureVector, extract_features
from Harpocrates.ml.verifier import DEFAULT_MODEL_DIR, VerificationResult, Verifier

if TYPE_CHECKING:
    from Harpocrates.core.result import EvidenceType, Finding

logger = logging.getLogger(__name__)

ONNX_STAGE_A_PATH = DEFAULT_MODEL_DIR / "stageA.onnx"
ONNX_STAGE_B_PATH = DEFAULT_MODEL_DIR / "stageB.onnx"
ONNX_HASHES_PATH = DEFAULT_MODEL_DIR / "onnx_model_hashes.json"

# Thresholds from training run (two_stage_config.json v2.4.0).
# CLAUDE.md §6: "Thresholds stay at training defaults 0.1/0.9. Never tighten them."
_STAGE_A_LOW = 0.1
_STAGE_A_HIGH = 0.9
_STAGE_B_THRESHOLD = 0.30


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_session(session, features: List[List[float]]) -> List[float]:
    """
    Run an ONNX InferenceSession and extract positive-class probabilities.

    Handles both ZipMap dict output (onnxmltools default) and dense ndarray
    output, so the caller does not need to know the conversion format.
    """
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
    # ZipMap output: list of {0: p_neg, 1: p_pos}
    if isinstance(probabilities, list) and len(probabilities) > 0 and isinstance(probabilities[0], dict):
        return [float(p[1]) for p in probabilities]
    # Dense ndarray output shape [n_samples, 2]
    return np.asarray(probabilities)[:, 1].tolist()


class OnnxTwoStageVerifier(Verifier):
    """
    ONNX-based two-stage verifier (v0.2.0 inference engine).

    Cascade:
      Stage A (XGBoost ONNX, 23 token features) — high-recall recall gate
        prob < _STAGE_A_LOW  → fast-reject (not a secret)
        prob > _STAGE_A_HIGH → fast-accept (is a secret)
        otherwise            → forward to Stage B
      Stage B (LightGBM ONNX, 63 full features) — high-precision verifier
        prob >= _STAGE_B_THRESHOLD → secret
    """

    _instance: Optional["OnnxTwoStageVerifier"] = None

    def __init__(
        self,
        stage_a_path: Optional[Path] = None,
        stage_b_path: Optional[Path] = None,
        hashes_path: Optional[Path] = None,
        lazy_load: bool = True,
    ):
        self._stage_a_path = stage_a_path or ONNX_STAGE_A_PATH
        self._stage_b_path = stage_b_path or ONNX_STAGE_B_PATH
        self._hashes_path = hashes_path or ONNX_HASHES_PATH
        self._stage_a_session = None
        self._stage_b_session = None
        self._loaded = False

        if not lazy_load:
            self._load_sessions()

    @classmethod
    def get_instance(
        cls,
        stage_a_path: Optional[Path] = None,
        stage_b_path: Optional[Path] = None,
    ) -> "OnnxTwoStageVerifier":
        """Singleton accessor — reuses loaded sessions across scans."""
        if cls._instance is None:
            cls._instance = cls(stage_a_path=stage_a_path, stage_b_path=stage_b_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton — useful in tests."""
        cls._instance = None

    @classmethod
    def is_available(cls) -> bool:
        """Return True if both ONNX model files exist on disk."""
        return ONNX_STAGE_A_PATH.exists() and ONNX_STAGE_B_PATH.exists()

    def _load_sessions(self) -> None:
        """Load and hash-verify both ONNX sessions."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            ) from exc

        if not self._hashes_path.exists():
            raise FileNotFoundError(
                f"Hash manifest not found: {self._hashes_path}. "
                "Re-run scripts/convert_to_onnx.py to regenerate it."
            )
        with open(self._hashes_path) as f:
            expected: Dict[str, str] = json.load(f)

        model_pairs = [
            (self._stage_a_path, "_stage_a_session"),
            (self._stage_b_path, "_stage_b_session"),
        ]
        for model_path, attr in model_pairs:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {model_path}. "
                    "Run `python scripts/convert_to_onnx.py` to generate it."
                )

            # Supply-chain integrity gate (design doc §3, step 5).
            # Key MUST be present in the manifest — absence is not a silent pass.
            key = model_path.name
            if key not in expected:
                raise ValueError(
                    f"Hash manifest does not contain an entry for '{key}'. "
                    "Re-run scripts/convert_to_onnx.py to regenerate the manifest."
                )
            # HIGH-3 fix: read bytes once → hash → pass bytes to InferenceSession,
            # closing the TOCTOU window between verification and load.
            model_bytes = model_path.read_bytes()
            actual = hashlib.sha256(model_bytes).hexdigest()
            if actual != expected[key]:
                raise ValueError(
                    f"SHA-256 mismatch for '{key}': possible supply-chain tampering. "
                    f"Expected {expected[key][:16]}…, got {actual[:16]}…"
                )

            setattr(self, attr, ort.InferenceSession(model_bytes))
            logger.info("Loaded ONNX session: %s", model_path.name)

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_sessions()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def threshold(self) -> float:
        return _STAGE_B_THRESHOLD

    def _route(
        self, features: FeatureVector
    ) -> Tuple[bool, float, str]:
        """
        Two-stage routing: returns (is_secret, confidence, routing_label).

        Stage A uses only the 23 token-only features; Stage B uses all 63.
        """
        self._ensure_loaded()

        stage_a_prob = _run_session(
            self._stage_a_session, [features.to_token_only_array()]
        )[0]

        if stage_a_prob < _STAGE_A_LOW:
            return False, 1.0 - stage_a_prob, "rejected_by_stage_a"

        if stage_a_prob > _STAGE_A_HIGH:
            return True, stage_a_prob, "accepted_by_stage_a"

        # Ambiguous — consult Stage B (full feature set)
        stage_b_prob = _run_session(
            self._stage_b_session, [features.to_array()]
        )[0]
        is_secret = stage_b_prob >= _STAGE_B_THRESHOLD
        confidence = stage_b_prob if is_secret else (1.0 - stage_b_prob)
        return is_secret, confidence, "decided_by_stage_b"

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

        # Extract features for all findings up-front
        all_features = [extract_features(f, ctx) for f, ctx in findings_with_context]

        # Stage A: single batched ONNX call over all token-only feature rows
        stage_a_probs = _run_session(
            self._stage_a_session,
            [fv.to_token_only_array() for fv in all_features],
        )

        # Identify which rows need Stage B (ambiguous zone)
        ambiguous_idx = [
            i for i, p in enumerate(stage_a_probs)
            if _STAGE_A_LOW <= p <= _STAGE_A_HIGH
        ]

        # Stage B: single batched ONNX call for ambiguous rows only
        stage_b_probs: Dict[int, float] = {}
        if ambiguous_idx:
            b_results = _run_session(
                self._stage_b_session,
                [all_features[i].to_array() for i in ambiguous_idx],
            )
            stage_b_probs = dict(zip(ambiguous_idx, b_results))

        results = []
        for i, (finding, _ctx) in enumerate(findings_with_context):
            a_prob = stage_a_probs[i]
            if a_prob < _STAGE_A_LOW:
                is_secret, ml_confidence, routing = False, 1.0 - a_prob, "rejected_by_stage_a"
            elif a_prob > _STAGE_A_HIGH:
                is_secret, ml_confidence, routing = True, a_prob, "accepted_by_stage_a"
            else:
                b_prob = stage_b_probs[i]
                is_secret = b_prob >= _STAGE_B_THRESHOLD
                ml_confidence = b_prob if is_secret else (1.0 - b_prob)
                routing = "decided_by_stage_b"

            original_confidence = finding.confidence or 0.5
            combined_confidence = self._combine_confidence(
                original_confidence, ml_confidence, finding.evidence
            )
            label = "likely secret" if is_secret else "likely safe"
            results.append(VerificationResult(
                is_secret=is_secret,
                ml_confidence=ml_confidence,
                original_confidence=original_confidence,
                combined_confidence=combined_confidence,
                features_used=dict(zip(
                    FeatureVector.get_feature_names(), all_features[i].to_array()
                )),
                explanation=f"{label} ({ml_confidence:.0%} confidence) [ONNX/{routing}]",
            ))
        return results


__all__ = [
    "OnnxTwoStageVerifier",
    "ONNX_STAGE_A_PATH",
    "ONNX_STAGE_B_PATH",
    "ONNX_HASHES_PATH",
]
