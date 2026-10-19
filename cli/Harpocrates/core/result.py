from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from Harpocrates.utils.redaction import mask_secrets


class EvidenceType(Enum):
    """
    Description: Represent the evidence source that triggered detection.
    Parameters: None
    Output: Enum members provide regex, entropy, ML, LLM, and hybrid evidence labels.
    Type of evidence that triggered the detection.
    """
    REGEX = "regex"
    ENTROPY = "entropy"
    ML = 'ml'
    LLM = 'llm'
    HYBRID = 'hybrid'

class Severity(Enum):
    """
    Description: Represent the severity assigned to a finding.
    Parameters: None
    Output: Enum members provide critical, high, medium, low, and informational levels.
    Severity level of the finding.
    """
    CRITICAL = 'critical'
    HIGH = "high"
    MEDIUM = 'medium'
    LOW = 'low'
    INFO = 'info'


@dataclass
class Finding:
    """
    Description: Represent a single secret detection with evidence, location, and optional
    classification metadata.
    Parameters:
        type: Type of detected secret, such as AWS_ACCESS_KEY or GITHUB_TOKEN.
        snippet: Source snippet containing the secret, potentially truncated.
        evidence: Evidence source that triggered detection.
        file: Optional source path; None for text scans.
        line: Optional source line number; None for text scans.
        entropy: Optional Shannon entropy of the token.
        severity: Finding severity, defaulting to medium.
        confidence: Optional model confidence from 0.0 to 1.0.
        token: Actual detected token; use with caution.
        token_start: Optional inclusive internal token offset.
        token_end: Optional exclusive internal token offset.
        in_comment: Optional flag for commented-out findings.
        category: Optional violation-category value.
        category_reason: Optional category inference trace.
    Output: Instances provide finding fields, redacted display, and serialization; raw token and
    snippet fields require care.
    Represents a single secret detection finding.
    Attributes: type: Type of secret detected (e.g., "AWS_ACCESS_KEY", "GITHUB_TOKEN") file: Path to
    the file where secret was found (None for text scans) line: Line number in the file (None for
    text scans) snippet: Code snippet containing the secret (truncated) entropy: Shannon entropy of
    the detected token (if applicable) evidence: How the secret was detected (regex, entropy, ml,
    etc.) severity: Severity level of the finding confidence: Confidence score (0.0-1.0) for
    ML-based detection token: The actual detected token (use with caution)
    """
    type: str
    snippet: str
    evidence: EvidenceType
    file: Optional[str] = None
    line: Optional[int] = None
    entropy: Optional[float] = None
    severity: Severity = Severity.MEDIUM
    confidence: Optional[float] = None
    token: Optional[str] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    in_comment: Optional[bool] = None
    # Violation classification — WHAT was leaked and WHY the category was inferred.
    # category: ViolationCategory.value string (e.g., "password", "api_token").
    # category_reason: structured debug trace (layer, matched pattern, confidence).
    # Both are intentionally additive / optional so existing callers remain unaffected.
    category: Optional[str] = None
    category_reason: Optional[str] = None

    @property
    def redacted_token(self) -> Optional[str]:
        """
        Description: Render the token through the shared redaction utility for consistent safe
        display.
        Parameters: None
        Output: Redacted token string or None for absent/blank input; lazily imports the shared
        utility.
        Returns a redacted version of the token for safe display.
        Delegates to :func:`Harpocrates.utils.redaction.redact_token` so that every surface (CLI,
        API, logs) shares a single redaction contract. Kept as a property for backward
        compatibility.
        """
        # Local import avoids a core <-> utils import cycle.
        from Harpocrates.utils.redaction import redact_token

        return redact_token(self.token)

    def to_dict(self) -> Dict[str, Any]:
        """
        Description: Serialize finding fields and enum values for programmatic consumers.
        Parameters: None
        Output: Dictionary retaining the raw token and snippet but omitting internal token offsets;
        retains in_comment.
        Convert finding to dictionary representation
        """
        result = asdict(self)
        result['evidence'] = self.evidence.value
        result['severity'] = self.severity.value
        # token_start/token_end are internal scaffolding for TokenMatch; strip from output.
        # in_comment is intentionally retained — callers and MCP consumers use it to
        # distinguish active leaks from commented-out ones for prioritisation and display.
        result.pop('token_start', None)
        result.pop('token_end', None)
        return result

    def to_json_dict(
        self, include_token: bool = False, also_mask: Iterable[Optional[str]] = ()
    ) -> Dict[str, Any]:
        """
        Description: Serialize a finding with optional raw-token inclusion for JSON consumers (SEC-01).
        Parameters:
            include_token: Whether to retain raw tokens and unredacted text in serialized findings.
            also_mask: Other secrets to hide from the text fields, such as the tokens of the other
                findings on the same line; ignored when include_token is true.
        Output: Dictionary excluding token by default. Unless include_token is true, snippet and
        category_reason carry each occurrence of the token (and of also_mask) in redacted form,
        so no field holds a plaintext secret. Field names are unchanged.
        """
        d = self.to_dict()
        if not include_token:
            d.pop('token', None)
            hidden = [self.token, *also_mask]
            d['snippet'] = mask_secrets(self.snippet, hidden)
            if self.category_reason:
                d['category_reason'] = mask_secrets(self.category_reason, hidden)
        return d

    def __str__(self) -> str:
        """
        Description: Format the finding location, category, severity, and evidence for
        human-readable output.
        Parameters: None
        Output: Summary string without the raw token or snippet.
        Human-readable string representation
        """
        loc = f"{self.file}:{self.line}" if self.file and self.line else "text"
        cat_str = f" [{self.category}]" if self.category else ""
        return (
            f"[{self.severity.value.upper()}] {self.type}{cat_str} "
            f"at {loc} (evidence: {self.evidence.value})"
        )

def findings_to_json(findings: List[Finding], include_token: bool = False) -> List[Dict[str, Any]]:
    """
    Description: Serialize findings for JSON output, hiding every finding's token from the text fields
        of all findings on the same line (SEC-01).
    Parameters:
        findings: Findings to serialize, in order.
        include_token: Whether to keep raw tokens and unredacted text (the --show-secrets opt-in).
    Output: One dictionary per finding, in the same order, as Finding.to_json_dict returns. Two
        findings on one line mask each other's secrets, so a snippet never shows a neighbour's value.
    """
    by_line: Dict[Any, List[Optional[str]]] = {}
    for f in findings:
        by_line.setdefault((f.file, f.line), []).append(f.token)
    return [f.to_json_dict(include_token=include_token, also_mask=by_line[(f.file, f.line)]) for f in findings]


@dataclass
class ScanResult:
    """
    Description: Represent all findings, scan statistics, and errors from a scan operation.
    Parameters:
        findings: List of detected findings.
        scanned_files: Number of scanned files.
        total_lines: Number of scanned source lines.
        duration_ms: Scan duration in milliseconds.
        errors: Optional list of scanning errors, normalized to an empty list if absent.
    Output: Instances provide findings, counts, errors, and serialization helpers.
    Represents the complete result of a scan operation.
    Attributes: findings: List of all findings detected scanned_files: Number of files scanned
    total_lines: Total lines of code scanned duration_ms: Scan duration in milliseconds errors: List
    of errors encountered during scanning
    """
    findings: List[Finding]
    scanned_files: int = 0
    total_lines: int = 0
    duration_ms: float = 0.0
    errors: Optional[list[str]] = None

    def __post_init__(self):
        """
        Description: Normalize missing scan errors to an independent list for each result.
        Parameters: None
        Output: None; replaces a None errors field with an empty list.
        """
        if self.errors is None:
            self.errors = []

    @property
    def found_secrets(self) -> bool:
        """
        Description: Check whether the scan produced any findings.
        Parameters: None
        Output: True for a nonempty findings list, otherwise False.
        Returns True if any findings were dectected
        """
        return len(self.findings) > 0

    @property
    def critical_count(self) -> int:
        """
        Description: Count critical findings for the scan summary.
        Parameters: None
        Output: Number of findings with Severity.CRITICAL.
        Count of critical severity findings were detected.
        """
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        """
        Description: Count high-severity findings for the scan summary.
        Parameters: None
        Output: Number of findings with Severity.HIGH.
        Count of high severity findings findings
        """
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    def to_dict(self, include_token: bool = False) -> Dict[str, Any]:
        """
        Description: Serialize findings and scan statistics with summary counts for consumers.
        Parameters:
            include_token: Whether to retain raw tokens in serialized findings.
        Output: Dictionary of findings, statistics, errors, and counts; raw tokens are excluded
        unless requested, and so is every plaintext secret in snippets (SEC-01).
        Convert scan result to a dictionary
        """
        return {
            "findings": findings_to_json(self.findings, include_token=include_token),
            "scanned_files": self.scanned_files,
            "total_lines": self.total_lines,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "summary": {
                "total_findings": len(self.findings),
                'critical': self.critical_count,
                'high': self.high_count,
            }
        }

    def __str__(self) -> str:
        """
        Description: Format scan counts and duration as a human-readable summary.
        Parameters: None
        Output: Summary string with finding, file, and line counts and duration in milliseconds.
        Human readable summary
        """
        return (
            f"Scan complete: {len(self.findings)} findings in "
            f"{self.scanned_files} files ({self.total_lines} lines) "
            f"[{self.duration_ms:.2f}ms]"
        )

__all__ = ["Finding", "ScanResult", "EvidenceType", "Severity", "findings_to_json"]
