"""
Core detection engine for Harpocrates.

Three-phase pipeline per line:
  1. CRITICAL regex  → Finding(CRITICAL, confidence=0.99) — skips ML
  2. HIGH regex      → Finding(HIGH,     confidence=0.95) — skips ML
  3. Entropy         → Finding(ENTROPY_CANDIDATE)         → ML verification

Regex hits bypass XGBoost entirely; only entropy candidates are forwarded
to the ML verifier. This keeps the fast path deterministic and CPU-free.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from Harpocrates.core.result import EvidenceType, Finding, Severity
from Harpocrates.detectors.entropy_detector import looks_like_secret, shannon_entropy
from Harpocrates.detectors.regex_patterns import CRITICAL_SIGNATURES, HIGH_SIGNATURES
from Harpocrates.utils.file_utils import iter_text_lines

if TYPE_CHECKING:
    from Harpocrates.ml.verifier import Verifier

# Tier 2: tightened token alphabet — drops '.' so dotted identifiers and URLs
# no longer form single long tokens. Retains '-' for UUID-style and hyphenated
# API keys (Azure SAS, some OAuth tokens). Minimum length matches the
# looks_like_secret() floor, eliminating a redundant per-token short check.
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")

# Tier 2: strip URL and data: URI runs before tokenizing so CDN/IDP/doc URLs
# and inline sourcemap data URIs don't generate entropy candidates.
_URL_RE = re.compile(r"(?:https?://|data:)\S+")

# Tier 1: pure base64 lines of PEM / X.509 certificate bodies.
# RFC 7468 specifies exactly 64 chars per non-terminal body line. Using {64}
# (not a range) avoids false-matching 60–63 or 65–76 char real-credential
# lines in non-PEM contexts (raw AES-256 keys, JWT segments, etc.).
_PEM_BODY_RE = re.compile(r"^[A-Za-z0-9+/]{64}$")

# Phase 2b: sensitive-variable assignment bypass — forwards low-entropy literals
# assigned to clearly credential-named variables directly to ML, skipping the
# entropy gate. Only fires when regex phases found nothing on the line.
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![a-zA-Z])(?:pass(?:word|wd|w)?|pwd|usr(?:name)?|user|host|conn(?:ection|str)?|secret|token|key|auth|cred)[a-z0-9_]*\s*[:=]\s*['\"]([^'\"]{3,100})['\"]"
)


def _calculate_entropy_confidence(entropy_val: Optional[float]) -> float:
    """Map entropy 4.0–5.5 linearly to confidence 0.6–0.8."""
    if entropy_val is None:
        return 0.6
    if entropy_val >= 5.5:
        return 0.8
    if entropy_val >= 4.0:
        return 0.6 + (entropy_val - 4.0) * (0.2 / 1.5)
    return 0.6


def _entropy_severity(entropy_val: Optional[float]) -> Severity:
    # Tier 1: entropy-only findings default to INFO so they don't trip the
    # default --fail-on=medium gate. ML-verified (evidence=hybrid) findings
    # keep this severity; users can opt in with --fail-on=info.
    _ = entropy_val
    return Severity.INFO


def _scan_line(line: str, lineno: int, file: Optional[str]) -> List[Finding]:
    """
    Scan one line through all three detection phases.

    Returns findings ordered: CRITICAL regex → HIGH regex → entropy.
    Entropy phase is skipped if any regex hit is found on the line.
    """
    findings: List[Finding] = []
    stripped = line.strip()

    if not stripped or stripped.startswith(("#", "/*#")):
        return findings

    # ------------------------------------------------------------------
    # Phase 1a: CRITICAL regex — deterministic, no ML needed.
    # ------------------------------------------------------------------
    for sig_name, pattern in CRITICAL_SIGNATURES.items():
        for match in pattern.finditer(stripped):
            token = match.group()
            findings.append(
                Finding(
                    type=sig_name,
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=shannon_entropy(token) if token else 0.0,
                    evidence=EvidenceType.REGEX,
                    severity=Severity.CRITICAL,
                    confidence=0.99,
                    token=token,
                )
            )

    # ------------------------------------------------------------------
    # Phase 1b: HIGH regex — also deterministic, also bypasses ML.
    # ------------------------------------------------------------------
    for sig_name, pattern in HIGH_SIGNATURES.items():
        for match in pattern.finditer(stripped):
            token = match.group()
            findings.append(
                Finding(
                    type=sig_name,
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=shannon_entropy(token) if token else 0.0,
                    evidence=EvidenceType.REGEX,
                    severity=Severity.HIGH,
                    confidence=0.95,
                    token=token,
                )
            )

    # ------------------------------------------------------------------
    # Phase 2: Entropy fallback — only when regex found nothing.
    # ------------------------------------------------------------------
    if not findings:
        # Tier 1: skip PEM/X.509 certificate body lines (pure base64, 60-76
        # chars). The BEGIN header was already caught by the regex tier above.
        if _PEM_BODY_RE.match(stripped):
            return findings

        # Tier 2: strip URL substrings before tokenizing so CDN/IDP/doc URLs
        # don't generate entropy candidates from their path components.
        scan_text = _URL_RE.sub(" ", stripped)

        for token in _TOKEN_RE.findall(scan_text):
            if looks_like_secret(token):
                ent = shannon_entropy(token)
                findings.append(
                    Finding(
                        type="ENTROPY_CANDIDATE",
                        file=file,
                        line=lineno,
                        snippet=stripped[:200],
                        entropy=ent,
                        evidence=EvidenceType.ENTROPY,
                        severity=_entropy_severity(ent),
                        confidence=_calculate_entropy_confidence(ent),
                        token=token,
                    )
                )

        found_tokens = {f.token for f in findings}
        for match in _SENSITIVE_ASSIGNMENT_RE.finditer(stripped):
            value = match.group(1)
            if value not in found_tokens:
                ent = shannon_entropy(value)
                findings.append(
                    Finding(
                        type="ML_CANDIDATE",
                        file=file,
                        line=lineno,
                        snippet=stripped[:200],
                        entropy=ent,
                        evidence=EvidenceType.ML,
                        severity=Severity.CRITICAL,
                        confidence=_calculate_entropy_confidence(ent),
                        token=value,
                    )
                )

    return findings


def detect_text(
    text: str,
    threshold: float = 4.0,
) -> List[Finding]:
    """
    Detect secrets in a text blob (regex + entropy, no ML).

    Args:
        text: Text content to scan
        threshold: Entropy threshold (reserved for future use)

    Returns:
        List of findings

    Example:
        >>> findings = detect_text("aws_key = AKIAIOSFODNN7EXAMPLE1234")
        >>> findings[0].type
        'AWS_ACCESS_KEY_ID'
    """
    _ = threshold
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
    Detect secrets in a file on disk (regex + entropy, no ML).

    Skips binary files, handles encoding errors gracefully, and
    returns an empty list for nonexistent files.

    Args:
        path: Path to the file to scan
        threshold: Entropy threshold (reserved)
        max_bytes: Maximum bytes to read

    Returns:
        List of findings
    """
    _ = threshold
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    file_name = str(path_obj)
    findings: List[Finding] = []
    for lineno, line in iter_text_lines(path_obj, max_bytes=max_bytes):
        findings.extend(_scan_line(line, lineno, file=file_name))
    return findings


# ---------------------------------------------------------------------------
# ML verification — applied only to entropy candidates, never regex hits.
# ---------------------------------------------------------------------------


def _apply_ml_verification(
    findings: List[Finding],
    full_content: str,
    verifier: "Verifier",
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """Filter entropy candidates through the ML verifier."""
    from Harpocrates.ml.context import extract_context

    verified: List[Finding] = []
    for finding in findings:
        line_num = finding.line or 1
        context = extract_context(
            content=full_content,
            line_number=line_num,
            file_path=finding.file,
        )
        result = verifier.verify(finding, context)
        if result.is_secret and result.combined_confidence >= ml_threshold:
            verified.append(
                Finding(
                    type=finding.type,
                    file=finding.file,
                    line=finding.line,
                    snippet=finding.snippet,
                    entropy=finding.entropy,
                    evidence=EvidenceType.HYBRID,
                    severity=finding.severity,
                    confidence=result.combined_confidence,
                    token=finding.token,
                )
            )
    return verified


def detect_text_with_ml(
    text: str,
    verifier: "Verifier",
    threshold: float = 4.0,
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """
    Detect secrets in text with ML verification for entropy candidates.

    Regex hits (CRITICAL/HIGH) are returned immediately — no ML call.
    Entropy candidates are forwarded to XGBoost for false-positive filtering.

    Args:
        text: Text content to scan
        verifier: ML verifier instance
        threshold: Entropy threshold (reserved)
        ml_threshold: Minimum ML confidence to keep an entropy finding

    Returns:
        List of findings
    """
    findings = detect_text(text, threshold)
    if not findings:
        return findings

    regex_findings = [f for f in findings if f.evidence == EvidenceType.REGEX]
    entropy_findings = [f for f in findings if f.evidence != EvidenceType.REGEX]

    if not entropy_findings:
        return regex_findings

    try:
        verified_entropy = _apply_ml_verification(
            findings=entropy_findings,
            full_content=text,
            verifier=verifier,
            ml_threshold=ml_threshold,
        )
    except Exception:
        verified_entropy = entropy_findings

    return regex_findings + verified_entropy


def detect_file_with_ml(
    path: str | Path,
    verifier: "Verifier",
    threshold: float = 4.0,
    max_bytes: Optional[int] = None,
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """
    Detect secrets in a file with ML verification for entropy candidates.

    Regex hits (CRITICAL/HIGH) are returned immediately — no ML call.
    Entropy candidates are forwarded to XGBoost for false-positive filtering.

    Args:
        path: Path to the file
        verifier: ML verifier instance
        threshold: Entropy threshold (reserved)
        max_bytes: Maximum bytes to read
        ml_threshold: Minimum ML confidence to keep an entropy finding

    Returns:
        List of findings
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    findings = detect_file(path, threshold, max_bytes)
    if not findings:
        return findings

    regex_findings = [f for f in findings if f.evidence == EvidenceType.REGEX]
    entropy_findings = [f for f in findings if f.evidence != EvidenceType.REGEX]

    if not entropy_findings:
        return regex_findings

    try:
        if max_bytes:
            with open(path_obj, "r", encoding="utf-8", errors="ignore") as f:
                full_content = f.read(max_bytes)
        else:
            full_content = path_obj.read_text(encoding="utf-8", errors="ignore")
    except (OSError, IOError):
        return findings

    try:
        verified_entropy = _apply_ml_verification(
            findings=entropy_findings,
            full_content=full_content,
            verifier=verifier,
            ml_threshold=ml_threshold,
        )
    except Exception:
        verified_entropy = entropy_findings

    return regex_findings + verified_entropy


__all__ = [
    "detect_text",
    "detect_file",
    "detect_text_with_ml",
    "detect_file_with_ml",
    "Finding",
]
