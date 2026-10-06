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
    """Result of ML verification on a finding."""

    is_secret: bool  # Final classification
    ml_confidence: float  # ML model confidence (0.0-1.0)
    original_confidence: float  # Pre-ML confidence from detector
    combined_confidence: float  # Weighted combination
    features_used: Optional[Dict[str, float]] = None  # Feature values
    explanation: Optional[str] = None  # Human-readable reason
    calibrated: Optional[float] = None  # P(secret) after the validation-fit map (FR-CORE-03)

    @property
    def confidence_delta(self) -> float:
        """Change in confidence from original to combined."""
        return self.combined_confidence - self.original_confidence


class Verifier(ABC):
    """Abstract base class for finding verification."""

    @abstractmethod
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
        pass

    @abstractmethod
    def verify_batch(
        self,
        findings_with_context: List[Tuple["Finding", CodeContext]],
    ) -> List[VerificationResult]:
        """
        Verify multiple findings for efficiency.

        Args:
            findings_with_context: List of (finding, context) tuples

        Returns:
            List of VerificationResults in same order
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        pass


__all__ = [
    "Verifier",
    "VerificationResult",
]
