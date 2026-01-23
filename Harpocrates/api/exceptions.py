"""
Custom API exceptions.

These exceptions are automatically converted to HTTP responses by FastAPI.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException


class HarpocratesAPIError(HTTPException):
    """Base exception for Harpocrates API errors."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class ContentTooLargeError(HarpocratesAPIError):
    """Raised when content exceeds maximum size limit."""

    def __init__(self, size: int, max_size: int):
        super().__init__(
            status_code=413,
            detail=f"Content size ({size} bytes) exceeds maximum ({max_size} bytes)",
        )


class BatchTooLargeError(HarpocratesAPIError):
    """Raised when batch size exceeds maximum limit."""

    def __init__(self, count: int, max_count: int):
        super().__init__(
            status_code=413,
            detail=f"Batch size ({count} files) exceeds maximum ({max_count} files)",
        )


class AuthenticationError(HarpocratesAPIError):
    """Raised when API key is missing or invalid."""

    def __init__(self, detail: str = "Invalid or missing API key"):
        super().__init__(
            status_code=401,
            detail=detail,
            headers={"WWW-Authenticate": "ApiKey"},
        )


class RateLimitExceededError(HarpocratesAPIError):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: int):
        super().__init__(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


class MLModelError(HarpocratesAPIError):
    """Raised when ML model fails to load or predict."""

    def __init__(self, detail: str = "ML model error"):
        super().__init__(
            status_code=503,
            detail=detail,
        )


class ScanError(HarpocratesAPIError):
    """Raised when scanning fails."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=f"Scan failed: {detail}",
        )
