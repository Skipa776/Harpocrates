"""
Core detection engine for Harpocrates.
Combines regex patterns and entropy analysis to detect secrets.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from Harpocrates.core.result import EvidenceType, Finding, Severity
from Harpocrates.detectors.entropy_detector import looks_like_secret, shannon_entropy
from Harpocrates.detectors.regex_patterns import SIGNATURES
from Harpocrates.utils.file_utils import iter_text_lines

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


__all__ = ["detect_text", "detect_file", "Finding"]
