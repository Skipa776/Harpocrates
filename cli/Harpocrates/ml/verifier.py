"""Verifier interface and result type. The shipped implementation is OnnxVerifier (onnx_verifier.py)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from Harpocrates.ml.context import CodeContext

if TYPE_CHECKING:
    from Harpocrates.core.result import Finding

# Default model path relative to package
DEFAULT_MODEL_DIR = Path(__file__).parent / "models"


@dataclass
class VerificationResult:
    """
    Description: Result of ML verification on a finding.
    Parameters:
        is_secret: Final classification.
        ml_confidence: Model confidence from 0.0 to 1.0.
        original_confidence: Detector confidence before ML verification.
        combined_confidence: Weighted detector/model confidence.
        features_used: Optional feature-name/value mapping.
        explanation: Optional human-readable verification reason.
        calibrated: Optional validation-calibrated P(secret), per FR-CORE-03.
    Output: Instances provide classification, confidence scores, optional features/explanation, and
    a confidence delta.
    """

    is_secret: bool  # Final classification
    ml_confidence: float  # ML model confidence (0.0-1.0)
    original_confidence: float  # Pre-ML confidence from detector
    combined_confidence: float  # Weighted combination
    features_used: Optional[Dict[str, float]] = None  # Feature values
    explanation: Optional[str] = None  # Human-readable reason
    calibrated: Optional[float] = None  # P(secret) after the validation-fit map (FR-CORE-03)

    @property
    def confidence_delta(self) -> float:
        """
        Description: Change in confidence from original to combined.
        Parameters: None
        Output: Combined confidence minus original confidence.
        """
        return self.combined_confidence - self.original_confidence


class Verifier(ABC):
    """
    Description: Abstract base class for finding verification.
    Parameters: None
    Output: Concrete subclasses provide single and batch verification plus model readiness; direct
    instantiation raises TypeError because methods are abstract.
    """

    @abstractmethod
    def verify(
        self,
        finding: "Finding",
        context: CodeContext,
    ) -> VerificationResult:
        """
        Description: Define contextual verification of one finding for concrete verifier
        implementations.
        Parameters:
            finding: Finding to verify
            context: Code context around the finding
        Output: Subclasses must return VerificationResult with classification and confidence; this
        abstract stub returns None if called directly.
        Verify a single finding with context.
        VerificationResult with classification and confidence
        """
        pass

    @abstractmethod
    def verify_batch(
        self,
        findings_with_context: List[Tuple["Finding", CodeContext]],
    ) -> List[VerificationResult]:
        """
        Description: Define ordered batch verification so implementations can evaluate multiple
        findings efficiently.
        Parameters:
            findings_with_context: List of (finding, context) tuples
        Output: Subclasses must return VerificationResults in input order; this abstract stub
        returns None if called directly.
        Verify multiple findings for efficiency.
        List of VerificationResults in same order
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Description: Define the model-readiness property for concrete verifier implementations.
        Parameters: None
        Output: Subclasses must return a readiness boolean; this abstract stub returns None if
        called directly.
        Check if model is loaded and ready.
        """
        pass


__all__ = [
    "Verifier",
    "VerificationResult",
]
