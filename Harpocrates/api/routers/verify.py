"""
ML verification endpoint.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from Harpocrates.api.config import settings
from Harpocrates.api.exceptions import MLModelError
from Harpocrates.api.schemas import VerifyRequest, VerifyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verify", tags=["Verification"])


@router.post("", response_model=VerifyResponse)
async def verify_token(request: VerifyRequest) -> VerifyResponse:
    """
    Verify a potential secret using ML.

    Uses the two-stage ML pipeline to classify whether a token
    is likely a real secret or a false positive (like a UUID, hash, etc.).
    """
    if not settings.ml_enabled:
        raise MLModelError("ML verification is disabled")

    try:
        from Harpocrates.ml.context import CodeContext
        from Harpocrates.ml.ensemble import get_verifier

        verifier = get_verifier(settings.ml_mode)
        if verifier is None:
            raise MLModelError("Failed to load ML verifier")

        # Build line content from context
        line_content = f"{request.context_before}{request.token}{request.context_after}"

        # Create context for verification
        # CodeContext expects lines_before/after as lists of strings
        lines_before = request.context_before.split("\n") if request.context_before else []
        lines_after = request.context_after.split("\n") if request.context_after else []

        context = CodeContext(
            line_content=line_content,
            lines_before=lines_before,
            lines_after=lines_after,
            file_path=request.filename,
            line_number=1,
        )

        # Create a minimal finding for the verifier
        from Harpocrates.core.result import EvidenceType, Finding, Severity

        finding = Finding(
            type="UNKNOWN",
            snippet=line_content,
            evidence=EvidenceType.ML,
            token=request.token,
            severity=Severity.MEDIUM,
        )

        # Run verification
        result = verifier.verify(finding, context)

        # VerificationResult has: is_secret, ml_confidence, original_confidence, combined_confidence
        confidence = result.ml_confidence

        # Determine decision path based on confidence
        # Two-stage: high confidence = stage_a decision, medium = stage_b
        decision_path = "single_stage"
        if confidence > 0.85:
            decision_path = "stage_a_high"
        elif confidence < 0.15:
            decision_path = "stage_a_low"
        else:
            decision_path = "stage_b"

        return VerifyResponse(
            is_secret=result.is_secret,
            confidence=confidence,
            stage_a_score=None,  # Not exposed in VerificationResult
            stage_b_score=None,  # Not exposed in VerificationResult
            decision_path=decision_path,
        )

    except ImportError as e:
        logger.error(f"ML import error: {e}")
        raise MLModelError("ML dependencies not installed")
    except Exception as e:
        logger.exception(f"Verification failed: {e}")
        raise MLModelError(f"Verification failed: {str(e)}")
