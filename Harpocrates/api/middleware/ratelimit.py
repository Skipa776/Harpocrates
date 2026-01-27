"""
Rate limiting middleware.

Simple in-memory rate limiter using sliding window algorithm.
For production, consider using Redis-backed rate limiting.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    In-memory rate limiter using sliding window algorithm.

    Tracks requests per client IP within a time window.
    Returns 429 Too Many Requests when limit is exceeded.
    """

    # Paths exempt from rate limiting
    EXEMPT_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        trust_proxy: bool = False,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.trust_proxy = trust_proxy
        # Client IP -> list of request timestamps
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, considering proxies."""
        # Only trust proxy headers when explicitly configured
        if self.trust_proxy:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()

            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip.strip()

        # Use direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _clean_old_requests(self, client_ip: str, now: float) -> None:
        """Remove requests outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if ts > cutoff
        ]

    def _is_rate_limited(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if client is rate limited.

        Returns:
            Tuple of (is_limited, retry_after_seconds)
        """
        now = time.time()
        self._clean_old_requests(client_ip, now)

        request_count = len(self._requests[client_ip])

        if request_count >= self.max_requests:
            # Calculate retry-after based on oldest request in window
            oldest = min(self._requests[client_ip]) if self._requests[client_ip] else now
            retry_after = int(oldest + self.window_seconds - now) + 1
            return True, max(1, retry_after)

        return False, 0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Check rate limit before processing request."""
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        is_limited, retry_after = self._is_rate_limited(client_ip)

        if is_limited:
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {request.url.path}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Retry after {retry_after} seconds."
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                },
            )

        # Record this request
        now = time.time()
        self._requests[client_ip].append(now)

        # Add rate limit headers to response
        response = await call_next(request)

        remaining = self.max_requests - len(self._requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))

        return response
