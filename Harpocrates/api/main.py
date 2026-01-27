"""
Harpocrates API - FastAPI application.

This module creates and configures the FastAPI application with:
- Lifespan management for ML model preloading
- CORS middleware
- Exception handlers
- Router registration
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from Harpocrates.api.config import settings

logger = logging.getLogger(__name__)

# Track server start time for uptime calculation
_start_time: float = 0.0


def get_uptime() -> float:
    """Get server uptime in seconds."""
    global _start_time
    if _start_time == 0:
        return 0.0
    return time.time() - _start_time


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifespan.

    Startup: Pre-load ML models for faster first request.
    Shutdown: Clean up resources.
    """
    global _start_time
    _start_time = time.time()

    # Startup
    logger.info("Starting Harpocrates API...")

    if settings.ml_enabled:
        logger.info(f"Pre-loading ML models (mode: {settings.ml_mode})...")
        try:
            from Harpocrates.ml.ensemble import get_verifier

            verifier = get_verifier(settings.ml_mode)
            if verifier:
                logger.info("ML models loaded successfully")
            else:
                logger.warning("ML verifier returned None")
        except Exception as e:
            logger.warning(f"Failed to pre-load ML models: {e}")
            logger.warning("ML verification will be attempted on first request")

    logger.info(f"Harpocrates API ready on {settings.host}:{settings.port}")

    yield

    # Shutdown
    logger.info("Shutting down Harpocrates API...")


# Create FastAPI application
app = FastAPI(
    title="Harpocrates API",
    description=(
        "ML-powered secrets detection API. "
        "Scan code for secrets with high precision and recall."
    ),
    version="2.4.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],  # Credentials require explicit origins per CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware (added before auth so rate limits apply to all requests)
if settings.rate_limit_enabled:
    from Harpocrates.api.middleware.ratelimit import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window,
    )

# API key authentication middleware
if settings.api_key:
    from Harpocrates.api.middleware.auth import APIKeyMiddleware

    app.add_middleware(
        APIKeyMiddleware,
        api_key=settings.api_key,
        header_name=settings.api_key_header,
    )


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Import and register routers
from Harpocrates.api.routers import health, llm, scan, verify

app.include_router(health.router)
app.include_router(scan.router)
app.include_router(verify.router)
app.include_router(llm.router)


def run() -> None:
    """Run the API server using uvicorn."""
    import uvicorn

    uvicorn.run(
        "Harpocrates.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
