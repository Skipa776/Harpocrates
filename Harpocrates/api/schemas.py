"""
Pydantic schemas for API request and response models.

These schemas define the API contract and provide automatic validation.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# Enums (mirror core/result.py)
class Severity(str, Enum):
    """Severity level of a finding."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EvidenceType(str, Enum):
    """Type of evidence that triggered detection."""

    REGEX = "regex"
    ENTROPY = "entropy"
    ML = "ml"
    LLM = "llm"
    HYBRID = "hybrid"


# Request Models
class ScanRequest(BaseModel):
    """Request to scan code content for secrets."""

    content: str = Field(
        ...,
        description="Code content to scan for secrets",
        min_length=1,
    )
    filename: Optional[str] = Field(
        None,
        description="Optional filename for context (affects detection accuracy)",
        examples=["config.py", ".env", "settings.json"],
    )
    ml_verify: bool = Field(
        True,
        description="Apply ML verification to filter false positives",
    )


class BatchFileItem(BaseModel):
    """A single file in a batch scan request."""

    filename: str = Field(
        ...,
        description="Filename for this content",
    )
    content: str = Field(
        ...,
        description="File content to scan",
    )


class BatchScanRequest(BaseModel):
    """Request to scan multiple files."""

    files: List[BatchFileItem] = Field(
        ...,
        description="List of files to scan",
        min_length=1,
        max_length=100,
    )
    ml_verify: bool = Field(
        True,
        description="Apply ML verification to findings",
    )


class VerifyRequest(BaseModel):
    """Request to verify a potential secret with ML."""

    token: str = Field(
        ...,
        description="The potential secret token to verify",
        min_length=1,
    )
    context_before: str = Field(
        "",
        description="Code context before the token (for ML analysis)",
    )
    context_after: str = Field(
        "",
        description="Code context after the token (for ML analysis)",
    )
    variable_name: Optional[str] = Field(
        None,
        description="Variable name if the token is assigned to one",
        examples=["api_key", "AWS_SECRET", "password"],
    )
    filename: Optional[str] = Field(
        None,
        description="Source filename for context",
    )


# Response Models
class FindingResponse(BaseModel):
    """A detected secret finding."""

    secret_type: str = Field(
        ...,
        description="Type of secret detected",
        examples=["AWS_ACCESS_KEY", "GITHUB_TOKEN", "GENERIC_API_KEY"],
    )
    token_redacted: str = Field(
        ...,
        description="Redacted version of the token for safe display",
        examples=["AKIA...MPLE", "ghp_...xyz"],
    )
    line_number: Optional[int] = Field(
        None,
        description="Line number where the secret was found",
    )
    snippet: str = Field(
        ...,
        description="Code snippet containing the secret (redacted)",
    )
    severity: Severity = Field(
        ...,
        description="Severity level of this finding",
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="How the secret was detected",
    )
    entropy: Optional[float] = Field(
        None,
        description="Shannon entropy of the token",
    )
    ml_verified: Optional[bool] = Field(
        None,
        description="Whether ML verification was applied",
    )
    ml_confidence: Optional[float] = Field(
        None,
        description="ML model confidence score",
        ge=0.0,
        le=1.0,
    )


class ScanResponse(BaseModel):
    """Response from scanning content."""

    findings: List[FindingResponse] = Field(
        default_factory=list,
        description="List of detected secrets",
    )
    scan_time_ms: float = Field(
        ...,
        description="Time taken to scan in milliseconds",
    )
    ml_enabled: bool = Field(
        ...,
        description="Whether ML verification was enabled",
    )
    total_findings: int = Field(
        ...,
        description="Total number of findings",
    )
    high_confidence_findings: int = Field(
        ...,
        description="Number of findings with confidence > 0.8",
    )


class BatchScanFileResult(BaseModel):
    """Result for a single file in a batch scan."""

    findings: List[FindingResponse] = Field(
        default_factory=list,
        description="Findings for this file",
    )
    scan_time_ms: float = Field(
        ...,
        description="Time to scan this file",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if scanning failed",
    )


class BatchScanResponse(BaseModel):
    """Response from batch scanning."""

    results: Dict[str, BatchScanFileResult] = Field(
        ...,
        description="Scan results keyed by filename",
    )
    total_files: int = Field(
        ...,
        description="Total number of files scanned",
    )
    total_findings: int = Field(
        ...,
        description="Total findings across all files",
    )
    scan_time_ms: float = Field(
        ...,
        description="Total time for batch scan",
    )
    files_with_errors: int = Field(
        0,
        description="Number of files that had errors",
    )


class VerifyResponse(BaseModel):
    """Response from ML verification."""

    is_secret: bool = Field(
        ...,
        description="Whether the token is classified as a secret",
    )
    confidence: float = Field(
        ...,
        description="Overall confidence score",
        ge=0.0,
        le=1.0,
    )
    stage_a_score: Optional[float] = Field(
        None,
        description="Stage A (high-recall filter) score",
    )
    stage_b_score: Optional[float] = Field(
        None,
        description="Stage B (high-precision verifier) score",
    )
    decision_path: str = Field(
        ...,
        description="Which stage made the decision",
        examples=["stage_a_high", "stage_a_low", "stage_b"],
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        ...,
        description="Service status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    version: str = Field(
        ...,
        description="API version",
    )
    ml_loaded: bool = Field(
        ...,
        description="Whether ML models are loaded",
    )
    ml_mode: Optional[str] = Field(
        None,
        description="Active ML mode (two_stage or ensemble)",
    )
    uptime_seconds: float = Field(
        ...,
        description="Time since server started",
    )


class StageConfig(BaseModel):
    """Configuration for a single ML stage."""

    model_type: str
    feature_count: int
    threshold: Optional[float] = None
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None


class ConfigResponse(BaseModel):
    """ML configuration response."""

    ml_enabled: bool = Field(
        ...,
        description="Whether ML is enabled",
    )
    ml_mode: str = Field(
        ...,
        description="Active ML mode",
    )
    model_version: str = Field(
        ...,
        description="Model version string",
    )
    stage_a: Optional[StageConfig] = Field(
        None,
        description="Stage A configuration (two-stage mode)",
    )
    stage_b: Optional[StageConfig] = Field(
        None,
        description="Stage B configuration (two-stage mode)",
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Model performance metrics",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(
        ...,
        description="Error message",
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code",
    )


# LLM Schemas
class LLMVerifyRequest(BaseModel):
    """Request to verify a potential secret using LLM."""

    token: str = Field(
        ...,
        description="The potential secret token to verify",
        min_length=1,
    )
    context_before: str = Field(
        "",
        description="Code context before the token",
    )
    context_after: str = Field(
        "",
        description="Code context after the token",
    )
    variable_name: Optional[str] = Field(
        None,
        description="Variable name if the token is assigned to one",
    )
    filename: Optional[str] = Field(
        None,
        description="Source filename for context",
    )
    secret_type: Optional[str] = Field(
        None,
        description="Type of secret (if known from detection)",
    )


class LLMVerifyResponse(BaseModel):
    """Response from LLM verification."""

    is_secret: bool = Field(
        ...,
        description="Whether the token is classified as a secret",
    )
    confidence: float = Field(
        ...,
        description="LLM confidence score",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of the decision",
    )
    model: str = Field(
        ...,
        description="LLM model used",
    )
    provider: str = Field(
        ...,
        description="LLM provider (ollama, anthropic, openai)",
    )
    latency_ms: float = Field(
        ...,
        description="LLM inference latency in milliseconds",
    )


class LLMStatusResponse(BaseModel):
    """LLM status response."""

    enabled: bool = Field(
        ...,
        description="Whether LLM is enabled",
    )
    available: bool = Field(
        ...,
        description="Whether LLM provider is available",
    )
    provider: Optional[str] = Field(
        None,
        description="Active LLM provider",
    )
    model: Optional[str] = Field(
        None,
        description="Active LLM model",
    )
