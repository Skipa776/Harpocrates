"""
LLM-powered verification for Harpocrates.

This module provides context-aware secret verification using Large Language Models.
Supports both local models (via Ollama) and cloud models (Anthropic, OpenAI).

The LLM verifier can:
1. Analyze code context to determine if a token is a real secret
2. Generate human-readable explanations for findings
3. Reduce false positives by understanding code semantics
4. Handle edge cases that ML models might miss

Usage:
    from Harpocrates.llm import LLMVerifier, OllamaProvider

    provider = OllamaProvider(model="llama3.2")
    verifier = LLMVerifier(provider)

    result = verifier.verify(finding, context)
    print(f"Is secret: {result.is_secret}")
    print(f"Explanation: {result.explanation}")
"""
from __future__ import annotations

from Harpocrates.llm.verifier import LLMVerifier, LLMVerificationResult
from Harpocrates.llm.providers import (
    LLMProvider,
    OllamaProvider,
    AnthropicProvider,
    OpenAIProvider,
    get_provider,
)

__all__ = [
    "LLMVerifier",
    "LLMVerificationResult",
    "LLMProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "get_provider",
]
