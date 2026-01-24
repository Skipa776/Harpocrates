"""
LLM verification endpoints.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from Harpocrates.api.config import settings
from Harpocrates.api.exceptions import MLModelError
from Harpocrates.api.schemas import LLMStatusResponse, LLMVerifyRequest, LLMVerifyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])

# Cached LLM provider
_llm_provider = None


def _get_llm_provider():
    """Get or create LLM provider."""
    global _llm_provider

    if _llm_provider is not None:
        return _llm_provider

    if not settings.llm_enabled:
        return None

    try:
        from Harpocrates.llm.providers import get_provider

        _llm_provider = get_provider(
            provider_name=settings.llm_provider,
            model=settings.llm_model,
        )
        return _llm_provider
    except Exception as e:
        logger.warning(f"Failed to initialize LLM provider: {e}")
        return None


@router.get("/status", response_model=LLMStatusResponse)
async def llm_status() -> LLMStatusResponse:
    """
    Get LLM status.

    Returns whether LLM is enabled and available.
    """
    provider = _get_llm_provider()

    if provider is None:
        return LLMStatusResponse(
            enabled=settings.llm_enabled,
            available=False,
            provider=None,
            model=None,
        )

    return LLMStatusResponse(
        enabled=settings.llm_enabled,
        available=provider.is_available(),
        provider=provider.name,
        model=provider.model,
    )


@router.post("/verify", response_model=LLMVerifyResponse)
async def llm_verify(request: LLMVerifyRequest) -> LLMVerifyResponse:
    """
    Verify a potential secret using LLM.

    Uses an LLM to analyze code context and determine if a token
    is a real secret or a false positive. Provides human-readable
    explanations for the decision.

    Note: LLM verification is slower than ML but provides explanations.
    """
    if not settings.llm_enabled:
        raise MLModelError("LLM verification is disabled")

    provider = _get_llm_provider()
    if provider is None:
        raise MLModelError("LLM provider not available")

    try:
        from Harpocrates.core.result import EvidenceType, Finding, Severity
        from Harpocrates.llm.verifier import LLMVerifier
        from Harpocrates.ml.context import CodeContext

        # Build context lines
        lines_before = request.context_before.split("\n") if request.context_before else []
        lines_after = request.context_after.split("\n") if request.context_after else []

        # line_content is the single line containing the token
        last_before = lines_before[-1] if lines_before else ""
        first_after = lines_after[0] if lines_after else ""
        line_content = f"{last_before}{request.token}{first_after}"

        context = CodeContext(
            line_content=line_content,
            lines_before=lines_before[:-1] if lines_before else [],
            lines_after=lines_after[1:] if lines_after else [],
            file_path=request.filename,
            line_number=1,
        )

        # Create finding
        finding = Finding(
            type=request.secret_type or "UNKNOWN",
            snippet=line_content,
            evidence=EvidenceType.ML,
            token=request.token,
            severity=Severity.MEDIUM,
        )

        # Run LLM verification
        verifier = LLMVerifier(provider)
        result = verifier.verify(finding, context)

        return LLMVerifyResponse(
            is_secret=result.is_secret,
            confidence=result.confidence,
            reasoning=result.reasoning,
            model=result.model,
            provider=result.provider,
            latency_ms=result.latency_ms,
        )

    except ImportError as e:
        logger.error(f"LLM import error: {e}")
        raise MLModelError("LLM dependencies not available")
    except Exception as e:
        logger.exception(f"LLM verification failed: {e}")
        raise MLModelError(f"LLM verification failed: {str(e)}")
