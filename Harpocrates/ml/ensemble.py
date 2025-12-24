"""
Ensemble verification combining multiple ML models.

Combines XGBoost and LightGBM predictions for improved accuracy
and robustness through model diversity.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FeatureVector, extract_features
from Harpocrates.ml.verifier import (
    DEFAULT_ML_THRESHOLD,
    VerificationResult,
    Verifier,
    XGBoostVerifier,
)

if TYPE_CHECKING:
    from Harpocrates.core.result import EvidenceType, Finding


class EnsembleStrategy(Enum):
    """Strategy for combining model predictions."""

    WEIGHTED_AVERAGE = "weighted_average"  # Weighted average of probabilities
    SOFT_VOTING = "soft_voting"  # Average probabilities, then threshold
    HARD_VOTING = "hard_voting"  # Majority vote of classifications
    MAX_CONFIDENCE = "max_confidence"  # Take most confident prediction


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    xgboost_weight: float = 0.6  # Weight for XGBoost predictions
    lightgbm_weight: float = 0.4  # Weight for LightGBM predictions
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE
    require_both_models: bool = False  # If False, works with single model


class EnsembleVerifier(Verifier):
    """
    Ensemble verifier combining XGBoost and LightGBM.

    Combines predictions from multiple models to improve:
    - Accuracy: Models learn different patterns
    - Robustness: Reduces variance from single model
    - Confidence: Agreement between models increases reliability

    Default strategy: Weighted average with XGBoost 0.6 + LightGBM 0.4
    """

    _instance: Optional["EnsembleVerifier"] = None

    def __init__(
        self,
        xgboost_path: Optional[Path] = None,
        lightgbm_path: Optional[Path] = None,
        config: Optional[EnsembleConfig] = None,
        lazy_load: bool = True,
        threshold: float = DEFAULT_ML_THRESHOLD,
    ):
        """
        Initialize the ensemble verifier.

        Args:
            xgboost_path: Path to XGBoost model file
            lightgbm_path: Path to LightGBM model file
            config: Ensemble configuration
            lazy_load: If True, defer model loading until first use
            threshold: Classification threshold (default 0.5)
        """
        self._config = config or EnsembleConfig()
        self._threshold = threshold
        self._lazy = lazy_load

        # Initialize individual verifiers
        self._xgboost_verifier: Optional[XGBoostVerifier] = None
        self._lightgbm_verifier = None  # Will be LightGBMVerifier when loaded
        self._xgboost_path = xgboost_path
        self._lightgbm_path = lightgbm_path

        if not lazy_load:
            self._load_models()

    @classmethod
    def get_instance(
        cls,
        xgboost_path: Optional[Path] = None,
        lightgbm_path: Optional[Path] = None,
        config: Optional[EnsembleConfig] = None,
        threshold: float = DEFAULT_ML_THRESHOLD,
    ) -> "EnsembleVerifier":
        """Get singleton instance of ensemble verifier."""
        if cls._instance is None:
            cls._instance = cls(
                xgboost_path=xgboost_path,
                lightgbm_path=lightgbm_path,
                config=config,
                threshold=threshold,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        XGBoostVerifier.reset_instance()
        # Reset LightGBM instance if available
        try:
            from Harpocrates.ml.lightgbm_verifier import LightGBMVerifier
            LightGBMVerifier.reset_instance()
        except ImportError:
            pass

    def _load_models(self) -> None:
        """Load all available models."""
        # Load XGBoost (always available)
        try:
            self._xgboost_verifier = XGBoostVerifier(
                model_path=self._xgboost_path,
                lazy_load=False,
                threshold=self._threshold,
            )
        except (ImportError, FileNotFoundError) as e:
            if self._config.require_both_models:
                raise
            logger.warning(
                "XGBoost model failed to load (path=%s): %s",
                self._xgboost_path,
                str(e),
                exc_info=True,
            )
            self._xgboost_verifier = None

        # Load LightGBM (optional)
        try:
            from Harpocrates.ml.lightgbm_verifier import LightGBMVerifier
            self._lightgbm_verifier = LightGBMVerifier(
                model_path=self._lightgbm_path,
                lazy_load=False,
                threshold=self._threshold,
            )
        except (ImportError, FileNotFoundError) as e:
            if self._config.require_both_models:
                raise
            logger.warning(
                "LightGBM model failed to load (path=%s): %s",
                self._lightgbm_path,
                str(e),
                exc_info=True,
            )
            self._lightgbm_verifier = None

        # Ensure at least one model is loaded
        if self._xgboost_verifier is None and self._lightgbm_verifier is None:
            raise RuntimeError(
                "No models available for ensemble. "
                "At least one model (XGBoost or LightGBM) must be trained."
            )

    def _ensure_loaded(self) -> None:
        """Ensure models are loaded (lazy loading)."""
        if self._xgboost_verifier is None and self._lightgbm_verifier is None:
            self._load_models()

    @property
    def is_loaded(self) -> bool:
        """Check if at least one model is loaded."""
        xgb_loaded = self._xgboost_verifier is not None and self._xgboost_verifier.is_loaded
        lgb_loaded = self._lightgbm_verifier is not None and self._lightgbm_verifier.is_loaded
        return xgb_loaded or lgb_loaded

    @property
    def available_models(self) -> List[str]:
        """List of loaded model names."""
        models = []
        if self._xgboost_verifier is not None and self._xgboost_verifier.is_loaded:
            models.append("xgboost")
        if self._lightgbm_verifier is not None and self._lightgbm_verifier.is_loaded:
            models.append("lightgbm")
        return models

    @property
    def threshold(self) -> float:
        """Get classification threshold."""
        return self._threshold

    def _get_model_predictions(
        self,
        features_array: List[List[float]],
    ) -> Dict[str, List[float]]:
        """
        Get predictions from all available models.

        Returns:
            Dict mapping model name to list of probabilities
        """
        import numpy as np

        predictions = {}
        failed_models = []

        # XGBoost predictions
        if self._xgboost_verifier is not None:
            try:
                import xgboost as xgb
                xgb_matrix = xgb.DMatrix(features_array)
                xgb_probas = self._xgboost_verifier._model.predict(xgb_matrix)
                predictions["xgboost"] = xgb_probas.tolist()
            except Exception as e:
                failed_models.append("xgboost")
                logger.error(
                    "XGBoost prediction failed: %s\n%s",
                    str(e),
                    traceback.format_exc(),
                )

        # LightGBM predictions
        if self._lightgbm_verifier is not None:
            try:
                lgb_probas = self._lightgbm_verifier._model.predict(features_array)
                predictions["lightgbm"] = lgb_probas.tolist()
            except Exception as e:
                failed_models.append("lightgbm")
                logger.error(
                    "LightGBM prediction failed: %s\n%s",
                    str(e),
                    traceback.format_exc(),
                )

        # Track failed models for callers to detect
        if failed_models:
            predictions["_failed_models"] = failed_models

        return predictions

    def _combine_predictions(
        self,
        predictions: Dict[str, List[float]],
        sample_idx: int,
    ) -> float:
        """
        Combine predictions from multiple models.

        Args:
            predictions: Dict mapping model name to probabilities
            sample_idx: Index of the sample to combine

        Returns:
            Combined probability (0.0-1.0)
        """
        # Filter out metadata keys (e.g., "_failed_models") to get only model predictions
        model_predictions = {
            k: v for k, v in predictions.items()
            if not k.startswith("_") and isinstance(v, list)
        }

        if not model_predictions:
            return 0.5  # Default to uncertain if no valid predictions

        strategy = self._config.strategy

        if strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            # Weighted average of probabilities
            total_weight = 0.0
            weighted_sum = 0.0

            if "xgboost" in model_predictions:
                weighted_sum += self._config.xgboost_weight * model_predictions["xgboost"][sample_idx]
                total_weight += self._config.xgboost_weight

            if "lightgbm" in model_predictions:
                weighted_sum += self._config.lightgbm_weight * model_predictions["lightgbm"][sample_idx]
                total_weight += self._config.lightgbm_weight

            return weighted_sum / total_weight if total_weight > 0 else 0.5

        elif strategy == EnsembleStrategy.SOFT_VOTING:
            # Simple average of probabilities
            probas = [p[sample_idx] for p in model_predictions.values()]
            return sum(probas) / len(probas) if probas else 0.5

        elif strategy == EnsembleStrategy.HARD_VOTING:
            # Majority vote of classifications
            votes = [p[sample_idx] >= self._threshold for p in model_predictions.values()]
            return 1.0 if votes and sum(votes) > len(votes) / 2 else 0.0

        elif strategy == EnsembleStrategy.MAX_CONFIDENCE:
            # Take most confident prediction (furthest from 0.5)
            probas = [p[sample_idx] for p in model_predictions.values()]
            if not probas:
                return 0.5
            confidences = [abs(p - 0.5) for p in probas]
            max_idx = confidences.index(max(confidences))
            return probas[max_idx]

        return 0.5  # Fallback

    def _combine_confidence(
        self,
        original_confidence: float,
        ml_confidence: float,
        evidence_type: "EvidenceType",
    ) -> float:
        """Combine original and ML confidence scores."""
        from Harpocrates.core.result import EvidenceType

        if evidence_type == EvidenceType.REGEX:
            weight_original = 0.6
            weight_ml = 0.4
        else:
            weight_original = 0.3
            weight_ml = 0.7

        return weight_original * original_confidence + weight_ml * ml_confidence

    def _generate_explanation(
        self,
        features: FeatureVector,
        ml_confidence: float,
        is_secret: bool,
        model_predictions: Dict[str, float],
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

        if not reasons:
            reasons.append("based on overall context")

        # Add model agreement info
        model_info = []
        for model_name, proba in model_predictions.items():
            model_info.append(f"{model_name}={proba:.0%}")

        confidence_str = f"{ml_confidence:.0%} confidence"
        classification = "likely secret" if is_secret else "likely not a secret"
        models_str = f"[{', '.join(model_info)}]"

        return f"{classification} ({confidence_str}) {models_str}: {', '.join(reasons)}"

    def verify(
        self,
        finding: "Finding",
        context: CodeContext,
    ) -> VerificationResult:
        """
        Verify a single finding using ensemble.

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

        # Get predictions from all models
        predictions = self._get_model_predictions(features_array)

        # Combine predictions
        ml_confidence = self._combine_predictions(predictions, 0)

        # Classify based on threshold
        is_secret = ml_confidence >= self._threshold

        # Combine confidences
        original_confidence = finding.confidence or 0.5
        combined_confidence = self._combine_confidence(
            original_confidence,
            ml_confidence,
            finding.evidence,
        )

        # Get individual model predictions for explanation (exclude metadata keys)
        model_probs = {
            name: probs[0] for name, probs in predictions.items()
            if not name.startswith("_") and isinstance(probs, list) and len(probs) > 0
        }

        # Generate explanation
        explanation = self._generate_explanation(
            features, ml_confidence, is_secret, model_probs
        )

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
        Verify multiple findings efficiently using ensemble.

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

        # Batch prediction from all models
        features_array = [f.to_array() for f in all_features]
        predictions = self._get_model_predictions(features_array)

        # Build results
        results = []
        for i, (finding, context) in enumerate(findings_with_context):
            ml_confidence = self._combine_predictions(predictions, i)
            is_secret = ml_confidence >= self._threshold

            original_confidence = finding.confidence or 0.5
            combined_confidence = self._combine_confidence(
                original_confidence,
                ml_confidence,
                finding.evidence,
            )

            # Exclude metadata keys when building model_probs
            model_probs = {
                name: probs[i] for name, probs in predictions.items()
                if not name.startswith("_") and isinstance(probs, list) and len(probs) > i
            }

            explanation = self._generate_explanation(
                all_features[i],
                ml_confidence,
                is_secret,
                model_probs,
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
    "EnsembleStrategy",
    "EnsembleConfig",
    "EnsembleVerifier",
]
