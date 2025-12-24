"""
Core detection engine for Harpocrates.
Combines regex patterns, entropy analysis, and optional ML verification
to detect secrets with context-aware false positive filtering.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from Harpocrates.core.result import EvidenceType, Finding, Severity
from Harpocrates.detectors.entropy_detector import looks_like_secret, shannon_entropy
from Harpocrates.detectors.regex_patterns import SIGNATURES
from Harpocrates.utils.file_utils import iter_text_lines

if TYPE_CHECKING:
    from Harpocrates.ml.verifier import Verifier

# Token pattern for entropy detection (alphanumeric + common special chars)
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-.]{8,}")


def _assess_severity(
    evidence_type: EvidenceType,
    entropy_val: Optional[float],
    secret_type: str
) -> Severity:
    """
    Determine severity based on detection method and secret type.

    Args:
        evidence_type: How the secret was detected
        entropy_val: Shannon entropy (if available)
        secret_type: Type of secret detected

    Returns:
        Severity level
    """
    # Regex matches are generally high confidence
    if evidence_type == EvidenceType.REGEX:
        # Known high-value secrets
        if any(x in secret_type for x in ["AWS", "GITHUB", "STRIPE", "OPENAI"]):
            return Severity.CRITICAL
        return Severity.HIGH

    # Entropy-only detections are less certain
    if evidence_type == EvidenceType.ENTROPY:
        if entropy_val and entropy_val >= 4.5:
            return Severity.MEDIUM
        return Severity.LOW

    return Severity.MEDIUM


def _calculate_confidence(
    evidence_type: EvidenceType,
    entropy_val: Optional[float],
) -> float:
    """
    Calculate confidence score for a finding.

    Args:
        evidence_type: How the secret was detected
        entropy_val: Shannon entropy (if available)

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if evidence_type == EvidenceType.REGEX:
        # Regex patterns are high confidence (known formats)
        return 0.95

    if evidence_type == EvidenceType.ENTROPY:
        # Entropy-based detection: scale from 0.6 to 0.8 based on entropy
        # Higher entropy = more likely to be a secret
        if entropy_val is None:
            return 0.6
        # Map entropy 4.0-5.5 to confidence 0.6-0.8
        base = 0.6
        if entropy_val >= 5.5:
            return 0.8
        if entropy_val >= 4.0:
            # Linear interpolation: 4.0->0.6, 5.5->0.8
            return base + (entropy_val - 4.0) * (0.2 / 1.5)
        return base

    return 0.5  # Default for unknown evidence types


def _scan_line(line: str, lineno: int, file: Optional[str]) -> List[Finding]:
    """
    Scan a single line for regex signatures and entropy-based candidates.

    Args:
        line: The line of text to scan
        lineno: Line number in the file
        file: File path (None for text scans)

    Returns:
        List of findings detected in this line
    """
    findings: List[Finding] = []
    stripped = line.strip()

    # Skip empty lines and comments
    if not stripped or stripped.startswith('#'):
        return findings

    # Phase 1: Regex pattern matching (high precision)
    for sig_name, pattern in SIGNATURES.items():
        for match in pattern.finditer(stripped):
            token = match.group()
            entropy_val = shannon_entropy(token) if token else 0.0

            finding = Finding(
                type=sig_name,
                file=file,
                line=lineno,
                snippet=stripped[:200],  # Truncate long lines
                entropy=entropy_val,
                evidence=EvidenceType.REGEX,
                severity=_assess_severity(EvidenceType.REGEX, entropy_val, sig_name),
                confidence=_calculate_confidence(EvidenceType.REGEX, entropy_val),
                token=token,
            )
            findings.append(finding)

    # Phase 2: Entropy-based detection (high recall)
    # Only run if no regex matches (avoid duplicates)
    if not findings:
        for token in _TOKEN_RE.findall(stripped):
            if looks_like_secret(token):
                ent = shannon_entropy(token)

                finding = Finding(
                    type="ENTROPY_CANDIDATE",
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=ent,
                    evidence=EvidenceType.ENTROPY,
                    severity=_assess_severity(EvidenceType.ENTROPY, ent, "ENTROPY"),
                    confidence=_calculate_confidence(EvidenceType.ENTROPY, ent),
                    token=token,
                )
                findings.append(finding)

    return findings


def detect_text(
    text: str,
    threshold: float = 4.0,
) -> List[Finding]:
    """
    Detect secrets in a text blob.

    Args:
        text: Text content to scan
        threshold: Entropy threshold (currently unused but reserved)

    Returns:
        List of findings detected

    Example:
        >>> findings = detect_text("aws_key = AKIAIOSFODNN7EXAMPLE")
        >>> len(findings)
        1
        >>> findings[0].type
        'AWS_ACCESS_KEY_ID'
    """
    _ = threshold  # Reserved for future use
    findings: List[Finding] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        findings.extend(_scan_line(line, lineno, file=None))

    return findings


def detect_file(
    path: str | Path,
    threshold: float = 4.0,
    max_bytes: Optional[int] = None,
) -> List[Finding]:
    """
    Detect secrets in a file on disk.

    Uses safe text iteration that:
    - Skips binary files
    - Handles encoding errors gracefully
    - Respects max_bytes limit
    - Returns empty list for nonexistent files

    Args:
        path: Path to the file to scan
        threshold: Entropy threshold (reserved)
        max_bytes: Maximum bytes to read from file

    Returns:
        List of findings detected (empty list if file doesn't exist)

    Example:
        >>> findings = detect_file("secrets.env")
        >>> for f in findings:
        ...     print(f"{f.file}:{f.line} - {f.type}")
    """
    _ = threshold  # Reserved for future use
    path_obj = Path(path)

    # Return empty list for nonexistent files
    if not path_obj.exists():
        return []

    file_name = str(path_obj)
    findings: List[Finding] = []

    for lineno, line in iter_text_lines(path_obj, max_bytes=max_bytes):
        findings.extend(_scan_line(line, lineno, file=file_name))

    return findings


def _apply_ml_verification(
    findings: List[Finding],
    full_content: str,
    verifier: "Verifier",
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """
    Apply ML verification to filter false positives.

    Args:
        findings: List of findings to verify
        full_content: Full file/text content for context extraction
        verifier: ML verifier instance
        ml_threshold: Minimum combined confidence to keep finding

    Returns:
        Filtered list of findings with updated confidence/evidence
    """
    from Harpocrates.ml.context import extract_context

    verified_findings = []

    for finding in findings:
        # Extract context for this finding
        line_num = finding.line or 1
        context = extract_context(
            content=full_content,
            line_number=line_num,
            file_path=finding.file,
        )

        # Verify with ML model
        result = verifier.verify(finding, context)

        # Filter based on threshold and classification
        if result.is_secret and result.combined_confidence >= ml_threshold:
            # Update finding with ML results
            updated_finding = Finding(
                type=finding.type,
                file=finding.file,
                line=finding.line,
                snippet=finding.snippet,
                entropy=finding.entropy,
                evidence=EvidenceType.HYBRID,  # Now ML-verified
                severity=finding.severity,
                confidence=result.combined_confidence,
                token=finding.token,
            )
            verified_findings.append(updated_finding)

    return verified_findings


def detect_text_with_ml(
    text: str,
    verifier: "Verifier",
    threshold: float = 4.0,
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """
    Detect secrets in text with ML-based false positive filtering.

    Uses a three-phase approach:
    1. Regex pattern matching (high precision)
    2. Entropy-based detection (high recall)
    3. ML verification (context-aware filtering)

    Args:
        text: Text content to scan
        verifier: ML verifier instance
        threshold: Entropy threshold (reserved)
        ml_threshold: Minimum ML confidence to keep finding

    Returns:
        List of verified findings

    Example:
        >>> from Harpocrates.ml.verifier import XGBoostVerifier
        >>> verifier = XGBoostVerifier()
        >>> findings = detect_text_with_ml("secret = 'abc123'", verifier)
    """
    # Phase 1 & 2: Standard detection
    findings = detect_text(text, threshold)

    if not findings:
        return findings

    # Phase 3: ML verification
    return _apply_ml_verification(
        findings=findings,
        full_content=text,
        verifier=verifier,
        ml_threshold=ml_threshold,
    )


def detect_file_with_ml(
    path: str | Path,
    verifier: "Verifier",
    threshold: float = 4.0,
    max_bytes: Optional[int] = None,
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """
    Detect secrets in a file with ML-based false positive filtering.

    Uses a three-phase approach:
    1. Regex pattern matching (high precision)
    2. Entropy-based detection (high recall)
    3. ML verification (context-aware filtering)

    Args:
        path: Path to the file to scan
        verifier: ML verifier instance
        threshold: Entropy threshold (reserved)
        max_bytes: Maximum bytes to read from file
        ml_threshold: Minimum ML confidence to keep finding

    Returns:
        List of verified findings (empty list if file doesn't exist)

    Example:
        >>> from Harpocrates.ml.verifier import XGBoostVerifier
        >>> verifier = XGBoostVerifier()
        >>> findings = detect_file_with_ml("config.py", verifier)
    """
    path_obj = Path(path)

    if not path_obj.exists():
        return []

    # Phase 1 & 2: Standard detection
    findings = detect_file(path, threshold, max_bytes)

    if not findings:
        return findings

    # Read full content for context extraction
    try:
        full_content = path_obj.read_text(encoding="utf-8", errors="ignore")
    except (OSError, IOError):
        # If we can't read for context, return unverified findings
        return findings

    # Phase 3: ML verification
    return _apply_ml_verification(
        findings=findings,
        full_content=full_content,
        verifier=verifier,
        ml_threshold=ml_threshold,
    )


__all__ = [
    "detect_text",
    "detect_file",
    "detect_text_with_ml",
    "detect_file_with_ml",
    "Finding",
]
