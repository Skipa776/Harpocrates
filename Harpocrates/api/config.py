"""
API configuration with environment variable support.

All settings can be configured via environment variables with HARPOCRATES_ prefix.
Example: HARPOCRATES_PORT=8080, HARPOCRATES_API_KEY=secret123
"""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="HARPOCRATES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False

    # Authentication
    api_key: Optional[str] = None  # If None, auth is disabled
    api_key_header: str = "X-API-Key"

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 60  # window in seconds

    # ML Configuration
    ml_enabled: bool = True
    ml_mode: str = "two_stage"  # "two_stage" or "ensemble"
    ml_threshold: float = 0.5

    # LLM Configuration
    llm_enabled: bool = False  # Disabled by default (requires setup)
    llm_provider: str = "auto"  # "auto", "ollama", "anthropic", "openai"
    llm_model: Optional[str] = None  # Provider-specific model name

    # Scanning limits
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    max_batch_size: int = 100  # max files per batch request

    # CORS
    cors_origins: str = "*"  # comma-separated origins or "*"


# Global settings instance
settings = APISettings()


def get_settings() -> APISettings:
    """Get the current API settings."""
    return settings
