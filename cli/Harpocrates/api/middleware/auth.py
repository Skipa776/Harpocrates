"""
API key authentication middleware.

Validates X-API-Key header when API key is configured.
If no API key is configured, authentication is disabled.
"""
from __future__ import annotations

import logging
import secrets
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API key in request headers.

    If api_key is None, authentication is disabled.
    Otherwise, requests must include the correct key in the configured header.
    """

    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
    }

    def __init__(
        self,
        app,
        api_key: Optional[str] = None,
        header_name: str = "X-API-Key",
    ):
        super().__init__(app)
        self.api_key = api_key
        self.header_name = header_name

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Validate API key for protected endpoints."""
        # Skip auth if no key configured
        if self.api_key is None:
            return await call_next(request)

        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Get API key from header
        provided_key = request.headers.get(self.header_name)

        if not provided_key:
            logger.warning(f"Missing API key for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Use constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(provided_key, self.api_key):
            logger.warning(f"Invalid API key for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)
