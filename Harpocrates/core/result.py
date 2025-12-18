from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceType(Enum):
    """Type of evidence that triggered the detection."""
    REGEX = "regex"
    ENTROPY = "entropy"
    ML = 'ml'
    LLM = 'llm'
    HYBRID = 'hybrid'

class Severity(Enum):
    """Severity level of the finding."""
    CRITICAL = 'critical'
    HIGH = "high"
    MEDIUM = 'medium'
    LOW = 'low'
    INFO = 'info'

class Finding:
    """
    Represents a single secret detection finding.

    Attributes:
        type: Type of secret detected (e.g., "AWS_ACCESS_KEY", "GITHUB_TOKEN")
        file: Path to the file where secret was found (None for text scans)
        line: Line number in the file (None for text scans)
        snippet: Code snippet containing the secret (truncated)
        entropy: Shannon entropy of the detected token (if applicable)
        evidence: How the secret was detected (regex, entropy, ml, etc.)
        severity: Severity level of the finding
        confidence: Confidence score (0.0-1.0) for ML-based detection
        token: The actual detected token (use with caution)
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary representation"""
        result = asdict(self)
        result['evidence'] = self.evidence.value
        result['severity'] = self.severity.value
        return result

    def to_json_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop('token', None)
        return d

    def __str__(self) -> str:
        """Human-readable string representation"""
        loc = f"{self.file}:{self.line}" if self.file and self.line else "text"
        return (
            f"[{self.severity.value.upper()}] {self.type} "
            f"at {loc} (evidence: {self.evidence.value})"
        )

@dataclass
class ScanResult:
    """
    Represents the complete result of a scan operation.

    Attributes:
        findings: List of all findings detected
        scanned_files: Number of files scanned
        total_lines: Total lines of code scanned
        duration_ms: Scan duration in milliseconds
        errors: List of errors encountered during scanning
    """
    findings: List[Finding]
    scanned_files: int = 0
    total_lines: int = 0
    duration_ms: float = 0.0
    errors: Optional[list[str]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def found_secrets(self) -> bool:
        """Returns True if any findings were dectected"""
        return len(self.finigns) > 0

    @property
    def critical_count(self) -> int:
        """Count of critical severity finings were detected."""
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high severity findings findings"""
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to a dictionary"""
        return {
            "findings": [f.to_json_dict() for f in self.findings],
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
        """Human readable summary"""
        return (
            f"Scan complete: {len(self.findings)} findings in "
            f"{self.scanned_files} files ({self.total_lines} lines) "
            f"[{self.duration_ms:.2f}ms]"
        )

__all__ = ["Finding", "ScanResult", "EvidenceType", "Severity"]
