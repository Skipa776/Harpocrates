"""
LightGBM-based verification for secrets detection.

Provides LightGBM-based verification as an alternative/complement to XGBoost,
offering potentially different learned patterns for ensemble combination.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FeatureVector, extract_features
from Harpocrates.ml.verifier import (
    DEFAULT_ML_THRESHOLD,
    DEFAULT_MODEL_DIR,
    VerificationResult,
    Verifier,
)

if TYPE_CHECKING:
    from Harpocrates.core.result import EvidenceType, Finding

# Default LightGBM model paths
DEFAULT_LGBM_MODEL_PATH = DEFAULT_MODEL_DIR / "lightgbm_v1.txt"
DEFAULT_LGBM_CONFIG_PATH = DEFAULT_MODEL_DIR / "lightgbm_config.json"


class LightGBMVerifier(Verifier):
    """
    LightGBM-based finding verifier.

    Uses a trained LightGBM model to classify findings as true secrets
    or false positives based on contextual features.

    LightGBM often learns different patterns than XGBoost due to:
    - Leaf-wise tree growth (vs level-wise in XGBoost)
    - Different regularization strategies
    - Histogram-based splitting

    This makes it valuable for ensemble combination.
    """

    _instance: Optional["LightGBMVerifier"] = None

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        lazy_load: bool = True,
        threshold: float = DEFAULT_ML_THRESHOLD,
    ):
        """
        Initialize the LightGBM verifier.

        Args:
            model_path: Path to LightGBM model file
            config_path: Path to feature config JSON file
            lazy_load: If True, defer model loading until first use
            threshold: Classification threshold (default 0.5)
        """
        self._model = None
        self._feature_config: Optional[Dict] = None
        self._model_path = model_path or DEFAULT_LGBM_MODEL_PATH
        self._config_path = config_path or DEFAULT_LGBM_CONFIG_PATH
        self._threshold = threshold
        self._lazy = lazy_load

        if not lazy_load:
            self._load_model()

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[Path] = None,
        threshold: float = DEFAULT_ML_THRESHOLD,
    ) -> "LightGBMVerifier":
        """
        Get singleton instance of LightGBM verifier.

        Useful for reusing the same model across multiple scans.
        """
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, threshold=threshold)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None

    def _load_model(self) -> None:
        """Load LightGBM model from disk."""
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM is required for this verifier. "
                "Install with: pip install harpocrates[ml]"
            ) from e

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"LightGBM model file not found: {self._model_path}. "
                "Train a model first with: python -m Harpocrates.training.train --model-type lightgbm"
            )

        self._model = lgb.Booster(model_file=str(self._model_path))

        # Load feature config if available
        if self._config_path.exists():
            with open(self._config_path) as f:
                self._feature_config = json.load(f)
                if "threshold" in self._feature_config:
                    self._threshold = self._feature_config["threshold"]

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded (lazy loading)."""
        if self._model is None:
            self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def threshold(self) -> float:
        """Get classification threshold."""
        return self._threshold

    def _predict_proba(self, features_array: List[List[float]]) -> List[float]:
        """
        Get prediction probabilities from model.

        Args:
            features_array: 2D array of feature vectors

        Returns:
            List of probabilities (0.0-1.0) for each sample
        """
        self._ensure_loaded()

        # LightGBM's predict returns probabilities directly for binary classification
        probas = self._model.predict(features_array)

        return probas.tolist()

    def _combine_confidence(
        self,
        original_confidence: float,
        ml_confidence: float,
        evidence_type: "EvidenceType",
    ) -> float:
        """
        Combine original and ML confidence scores.

        Uses different weights based on evidence type:
        - Regex matches: ML has less influence (already high confidence)
        - Entropy matches: ML has more influence (needs context)
        """
        from Harpocrates.core.result import EvidenceType

        if evidence_type == EvidenceType.REGEX:
            # Regex already high confidence, ML validates
            weight_original = 0.6
            weight_ml = 0.4
        else:
            # Entropy needs more ML context
            weight_original = 0.3
            weight_ml = 0.7

        return weight_original * original_confidence + weight_ml * ml_confidence

    def _generate_explanation(
        self,
        features: FeatureVector,
        ml_confidence: float,
        is_secret: bool,
    ) -> str:
        """Generate human-readable explanation for classification."""
        reasons = []

        if is_secret:
            if features.var_contains_secret:
                reasons.append("variable name suggests secret")
            if features.file_is_config:
                reasons.append("found in config file")
            if features.token_entropy > 4.0:
                reasons.append(f"high entropy ({features.token_entropy:.2f})")
            if features.file_extension_risk == 2:
                reasons.append("high-risk file type")
            if features.is_base64_like and features.has_padding:
                reasons.append("base64-encoded content")
            if features.regex_match_type > 0:
                reasons.append("matches known secret pattern")
        else:
            if features.var_contains_safe:
                reasons.append("variable name suggests safe content")
            if features.context_mentions_git:
                reasons.append("git-related context")
            if features.context_mentions_hash:
                reasons.append("hash/checksum context")
            if features.context_mentions_test:
                reasons.append("test/mock context")
            if features.file_is_test:
                reasons.append("test file")
            if features.line_is_comment:
                reasons.append("found in comment")
            if features.token_length == 40 and features.context_mentions_git:
                reasons.append("likely Git SHA based on context")

        if not reasons:
            reasons.append("based on overall context")

        confidence_str = f"{ml_confidence:.0%} confidence"
        classification = "likely secret" if is_secret else "likely not a secret"

        return f"{classification} ({confidence_str}): {', '.join(reasons)}"

    def verify(
        self,
        finding: "Finding",
        context: CodeContext,
    ) -> VerificationResult:
        """
        Verify a single finding with context.

        Args:
            finding: Finding to verify
            context: Code context around the finding

        Returns:
            VerificationResult with classification and confidence
        """
        self._ensure_loaded()

        # Extract features
        features = extract_features(finding, context)
        features_array = [features.to_array()]

        # Get ML prediction
        probas = self._predict_proba(features_array)
        ml_confidence = probas[0]

        # Classify based on threshold
        is_secret = ml_confidence >= self._threshold

        # Combine confidences
        original_confidence = finding.confidence or 0.5
        combined_confidence = self._combine_confidence(
            original_confidence,
            ml_confidence,
            finding.evidence,
        )

        # Generate explanation
        explanation = self._generate_explanation(features, ml_confidence, is_secret)

        # Create feature dict for debugging
        feature_names = FeatureVector.get_feature_names()
        features_dict = dict(zip(feature_names, features.to_array()))

        return VerificationResult(
            is_secret=is_secret,
            ml_confidence=ml_confidence,
            original_confidence=original_confidence,
            combined_confidence=combined_confidence,
            features_used=features_dict,
            explanation=explanation,
        )

    def verify_batch(
        self,
        findings_with_context: List[Tuple["Finding", CodeContext]],
    ) -> List[VerificationResult]:
        """
        Verify multiple findings efficiently.

        Args:
            findings_with_context: List of (finding, context) tuples

        Returns:
            List of VerificationResults in same order
        """
        if not findings_with_context:
            return []

        self._ensure_loaded()

        # Extract all features
        all_features = []
        for finding, context in findings_with_context:
            features = extract_features(finding, context)
            all_features.append(features)

        # Batch prediction
        features_array = [f.to_array() for f in all_features]
        probas = self._predict_proba(features_array)

        # Build results
        results = []
        for i, (finding, context) in enumerate(findings_with_context):
            ml_confidence = probas[i]
            is_secret = ml_confidence >= self._threshold

            original_confidence = finding.confidence or 0.5
            combined_confidence = self._combine_confidence(
                original_confidence,
                ml_confidence,
                finding.evidence,
            )

            explanation = self._generate_explanation(
                all_features[i],
                ml_confidence,
                is_secret,
            )

            feature_names = FeatureVector.get_feature_names()
            features_dict = dict(zip(feature_names, all_features[i].to_array()))

            results.append(
                VerificationResult(
                    is_secret=is_secret,
                    ml_confidence=ml_confidence,
                    original_confidence=original_confidence,
                    combined_confidence=combined_confidence,
                    features_used=features_dict,
                    explanation=explanation,
                )
            )

        return results


__all__ = [
    "LightGBMVerifier",
]
