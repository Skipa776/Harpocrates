"""
Scanning endpoints.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

from fastapi import APIRouter

from Harpocrates.api.config import settings
from Harpocrates.api.exceptions import BatchTooLargeError, ContentTooLargeError, ScanError
from Harpocrates.api.schemas import (
    BatchScanFileResult,
    BatchScanRequest,
    BatchScanResponse,
    EvidenceType,
    FindingResponse,
    ScanRequest,
    ScanResponse,
    Severity,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scan", tags=["Scanning"])


def _finding_to_response(finding) -> FindingResponse:
    """Convert core Finding to API FindingResponse."""
    # Get redacted token
    token_redacted = finding.redacted_token or "***"

    # Map evidence type
    evidence_type = EvidenceType(finding.evidence.value)

    # Map severity
    severity = Severity(finding.severity.value)

    return FindingResponse(
        secret_type=finding.type,
        token_redacted=token_redacted,
        line_number=finding.line,
        snippet=finding.snippet or "",
        severity=severity,
        confidence=finding.confidence or 0.5,
        evidence_type=evidence_type,
        entropy=finding.entropy,
        ml_verified=getattr(finding, "ml_verified", None),
        ml_confidence=getattr(finding, "ml_confidence", None),
    )


def _scan_content(content: str, filename: str | None, ml_verify: bool) -> ScanResponse:
    """Internal function to scan content."""
    start = time.perf_counter()

    try:
        from Harpocrates.core.detector import detect_text

        # Always start with standard detection
        findings = detect_text(content)

        # Assign filename to findings for ML file-path features
        if filename:
            for f in findings:
                if not f.file:
                    f.file = filename

        # Apply ML verification if enabled
        if ml_verify and settings.ml_enabled and findings:
            from Harpocrates.core.detector import _apply_ml_verification
            from Harpocrates.ml.ensemble import get_verifier

            verifier = get_verifier(settings.ml_mode)
            if verifier:
                findings = _apply_ml_verification(
                    findings=findings,
                    full_content=content,
                    verifier=verifier,
                    ml_threshold=settings.ml_threshold,
                )
    except Exception as e:
        logger.exception(f"Scan failed: {e}")
        raise ScanError(str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000

    finding_responses = [_finding_to_response(f) for f in findings]

    return ScanResponse(
        findings=finding_responses,
        scan_time_ms=elapsed_ms,
        ml_enabled=ml_verify and settings.ml_enabled,
        total_findings=len(finding_responses),
        high_confidence_findings=sum(1 for f in finding_responses if f.confidence > 0.8),
    )


@router.post("", response_model=ScanResponse)
async def scan_content(request: ScanRequest) -> ScanResponse:
    """
    Scan code content for secrets.

    Analyzes the provided content using regex patterns, entropy analysis,
    and optionally ML verification to detect potential secrets.
    """
    # Check content size
    content_size = len(request.content.encode("utf-8"))
    if content_size > settings.max_content_size:
        raise ContentTooLargeError(content_size, settings.max_content_size)

    return _scan_content(request.content, request.filename, request.ml_verify)


@router.post("/batch", response_model=BatchScanResponse)
async def scan_batch(request: BatchScanRequest) -> BatchScanResponse:
    """
    Scan multiple files for secrets.

    Processes each file independently and returns aggregated results.
    """
    # Check batch size
    if len(request.files) > settings.max_batch_size:
        raise BatchTooLargeError(len(request.files), settings.max_batch_size)

    start = time.perf_counter()
    results: Dict[str, BatchScanFileResult] = {}
    total_findings = 0
    files_with_errors = 0

    for file_item in request.files:
        try:
            # Check individual file size
            content_size = len(file_item.content.encode("utf-8"))
            if content_size > settings.max_content_size:
                results[file_item.filename] = BatchScanFileResult(
                    findings=[],
                    scan_time_ms=0,
                    error=f"Content too large: {content_size} bytes",
                )
                files_with_errors += 1
                continue

            scan_result = _scan_content(
                file_item.content,
                file_item.filename,
                request.ml_verify,
            )

            results[file_item.filename] = BatchScanFileResult(
                findings=scan_result.findings,
                scan_time_ms=scan_result.scan_time_ms,
                error=None,
            )
            total_findings += scan_result.total_findings

        except Exception as e:
            logger.exception(f"Failed to scan {file_item.filename}: {e}")
            results[file_item.filename] = BatchScanFileResult(
                findings=[],
                scan_time_ms=0,
                error=str(e),
            )
            files_with_errors += 1

    elapsed_ms = (time.perf_counter() - start) * 1000

    return BatchScanResponse(
        results=results,
        total_files=len(request.files),
        total_findings=total_findings,
        scan_time_ms=elapsed_ms,
        files_with_errors=files_with_errors,
    )
