"""
Harpocrates REST API.

FastAPI-based API for programmatic access to ML-powered secrets detection.

Endpoints:
- POST /scan - Scan code content for secrets
- POST /scan/batch - Batch scan multiple files
- POST /verify - ML-verify a single token
- GET /health - Health check
- GET /config - ML model configuration
"""
from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    """Create and configure the FastAPI application."""
    from Harpocrates.api.main import app
    return app
