"""
LLM provider implementations.

Supports multiple LLM backends:
- Ollama (local, free)
- Anthropic Claude (cloud)
- OpenAI GPT (cloud)
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name."""
        pass

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with generated content
        """
        pass

    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        return True


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM inference.

    Requires Ollama to be installed and running locally.
    See: https://ollama.ai
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid base_url scheme '{parsed.scheme}': only http/https allowed"
            )
        self._model = model
        self.base_url = base_url

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import urllib.request

            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate completion using Ollama API."""
        import time
        import urllib.request

        start = time.perf_counter()

        # Build request payload
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Make request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=result.get("response", ""),
            model=self._model,
            provider="ollama",
            tokens_used=result.get("eval_count"),
            latency_ms=elapsed_ms,
        )


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
    ):
        self._model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        import time
        import urllib.request

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        start = time.perf_counter()

        # Build request payload
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Make request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract content from response
        content = ""
        if result.get("content"):
            content = result["content"][0].get("text", "")

        tokens_used = None
        if result.get("usage"):
            tokens_used = result["usage"].get("output_tokens", 0)

        return LLMResponse(
            content=content,
            model=self._model,
            provider="anthropic",
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
        )


class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT provider.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self._model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        import time
        import urllib.request

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        start = time.perf_counter()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request payload
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        # Make request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract content from response
        content = ""
        if result.get("choices"):
            content = result["choices"][0].get("message", {}).get("content", "")

        tokens_used = None
        if result.get("usage"):
            tokens_used = result["usage"].get("completion_tokens", 0)

        return LLMResponse(
            content=content,
            model=self._model,
            provider="openai",
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
        )


def get_provider(
    provider_name: str = "auto",
    model: Optional[str] = None,
) -> LLMProvider:
    """
    Get an LLM provider by name.

    Args:
        provider_name: Provider name ("ollama", "anthropic", "openai", "auto")
        model: Optional model name override

    Returns:
        Configured LLM provider

    Raises:
        ValueError: If provider is not available
    """
    if provider_name == "auto":
        # Try providers in order of preference
        # 1. Ollama (free, local)
        ollama = OllamaProvider(model=model or "llama3.2")
        if ollama.is_available():
            return ollama

        # 2. Anthropic (if API key set)
        anthropic = AnthropicProvider(model=model or "claude-3-haiku-20240307")
        if anthropic.is_available():
            return anthropic

        # 3. OpenAI (if API key set)
        openai = OpenAIProvider(model=model or "gpt-4o-mini")
        if openai.is_available():
            return openai

        raise ValueError(
            "No LLM provider available. Install Ollama or set ANTHROPIC_API_KEY/OPENAI_API_KEY"
        )

    elif provider_name == "ollama":
        provider = OllamaProvider(model=model or "llama3.2")
        if not provider.is_available():
            raise ValueError("Ollama is not running. Start it with: ollama serve")
        return provider

    elif provider_name == "anthropic":
        provider = AnthropicProvider(model=model or "claude-3-haiku-20240307")
        if not provider.is_available():
            raise ValueError("ANTHROPIC_API_KEY not set")
        return provider

    elif provider_name == "openai":
        provider = OpenAIProvider(model=model or "gpt-4o-mini")
        if not provider.is_available():
            raise ValueError("OPENAI_API_KEY not set")
        return provider

    else:
        raise ValueError(f"Unknown provider: {provider_name}")
