"""
Health and configuration endpoints.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from Harpocrates.api.config import settings
from Harpocrates.api.main import get_uptime
from Harpocrates.api.schemas import ConfigResponse, HealthResponse, StageConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


def _load_model_config() -> Optional[dict]:
    """Load ML model configuration from disk."""
    config_path = Path(__file__).parent.parent.parent / "ml" / "models" / "two_stage_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")
    return None


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, version, and ML model availability.
    """
    ml_loaded = False
    ml_mode = None

    if settings.ml_enabled:
        try:
            from Harpocrates.ml.ensemble import get_verifier

            verifier = get_verifier(settings.ml_mode)
            ml_loaded = verifier is not None
            ml_mode = settings.ml_mode
        except Exception as e:
            logger.warning(f"ML health check failed: {e}")

    return HealthResponse(
        status="healthy" if ml_loaded or not settings.ml_enabled else "degraded",
        version="2.4.0",
        ml_loaded=ml_loaded,
        ml_mode=ml_mode,
        uptime_seconds=get_uptime(),
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Get ML model configuration.

    Returns model version, thresholds, and performance metrics.
    """
    config = _load_model_config()

    if config is None:
        return ConfigResponse(
            ml_enabled=settings.ml_enabled,
            ml_mode=settings.ml_mode,
            model_version="unknown",
            stage_a=None,
            stage_b=None,
            metrics={},
        )

    stage_a = None
    stage_b = None

    if "stage_a" in config:
        stage_a = StageConfig(
            model_type=config["stage_a"].get("model_type", "xgboost"),
            feature_count=config["stage_a"].get("feature_count", 23),
            threshold_low=config["stage_a"].get("threshold_low"),
            threshold_high=config["stage_a"].get("threshold_high"),
        )

    if "stage_b" in config:
        stage_b = StageConfig(
            model_type=config["stage_b"].get("model_type", "lightgbm"),
            feature_count=config["stage_b"].get("feature_count", 51),
            threshold=config["stage_b"].get("threshold"),
        )

    return ConfigResponse(
        ml_enabled=settings.ml_enabled,
        ml_mode=settings.ml_mode,
        model_version=config.get("version", "unknown"),
        stage_a=stage_a,
        stage_b=stage_b,
        metrics=config.get("combined_metrics", {}),
    )
