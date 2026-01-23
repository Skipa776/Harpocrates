"""
LLM-based verification for secret findings.

Uses LLMs to analyze code context and determine if a detected token
is a real secret or a false positive.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from Harpocrates.llm.providers import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from Harpocrates.core.result import Finding
    from Harpocrates.ml.context import CodeContext

logger = logging.getLogger(__name__)


# System prompt for secret verification
VERIFICATION_SYSTEM_PROMPT = """You are a security expert analyzing code for potential secrets and credentials.

Your task is to determine if a detected token is a REAL secret (credential, API key, password, etc.) or a FALSE POSITIVE (test data, example, hash, UUID, etc.).

You must be VERY careful:
- FALSE POSITIVES waste developer time
- MISSED SECRETS are security vulnerabilities

Analyze the code context carefully. Consider:
1. Variable names (api_key, password = likely secret; sha, uuid, hash = likely not)
2. Code patterns (test files, example code, documentation)
3. Token format (known patterns like AKIA for AWS, ghp_ for GitHub)
4. Surrounding context (is this a commit SHA? A placeholder?)

Respond ONLY with valid JSON in this exact format:
{
    "is_secret": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your decision"
}"""


VERIFICATION_PROMPT_TEMPLATE = """Analyze this potential secret:

TOKEN: {token_redacted}
TYPE: {secret_type}
VARIABLE: {variable_name}
FILE: {filename}

CODE CONTEXT:
```
{context}
```

Is this a real secret or a false positive? Respond with JSON only."""


BATCH_VERIFICATION_PROMPT_TEMPLATE = """Analyze these {count} potential secrets. For each, determine if it's a real secret or false positive.

{findings_text}

Respond with a JSON array containing exactly {count} objects, one for each finding:
[
    {{"is_secret": true/false, "confidence": 0.0-1.0, "reasoning": "..."}},
    ...
]"""


@dataclass
class LLMVerificationResult:
    """Result of LLM verification."""

    is_secret: bool
    confidence: float
    reasoning: str
    model: str
    provider: str
    latency_ms: float
    tokens_used: Optional[int] = None

    # Raw response for debugging
    raw_response: Optional[str] = None


class LLMVerifier:
    """
    LLM-based verifier for secret findings.

    Uses an LLM to analyze code context and provide high-quality
    verification with explanations.
    """

    def __init__(
        self,
        provider: LLMProvider,
        temperature: float = 0.0,
        max_tokens: int = 300,
    ):
        """
        Initialize LLM verifier.

        Args:
            provider: LLM provider to use
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens per response
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    def verify(
        self,
        finding: "Finding",
        context: "CodeContext",
    ) -> LLMVerificationResult:
        """
        Verify a single finding using LLM.

        Args:
            finding: Finding to verify
            context: Code context around the finding

        Returns:
            LLMVerificationResult with classification and explanation
        """
        # Build prompt
        prompt = self._build_prompt(finding, context)

        # Call LLM
        try:
            response = self.provider.complete(
                prompt=prompt,
                system_prompt=VERIFICATION_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            # Return a conservative result on error
            return LLMVerificationResult(
                is_secret=True,  # Conservative: assume it's a secret
                confidence=0.5,
                reasoning=f"LLM verification failed: {e}",
                model=self.provider.model,
                provider=self.provider.name,
                latency_ms=0.0,
            )

        # Parse response
        return self._parse_response(response)

    def verify_batch(
        self,
        findings_with_context: List[Tuple["Finding", "CodeContext"]],
        batch_size: int = 5,
    ) -> List[LLMVerificationResult]:
        """
        Verify multiple findings efficiently.

        Uses batched prompts to reduce API calls.

        Args:
            findings_with_context: List of (finding, context) tuples
            batch_size: Number of findings per batch

        Returns:
            List of LLMVerificationResults in same order
        """
        if not findings_with_context:
            return []

        results = []

        # Process in batches
        for i in range(0, len(findings_with_context), batch_size):
            batch = findings_with_context[i : i + batch_size]

            if len(batch) == 1:
                # Single finding - use regular verify
                finding, context = batch[0]
                results.append(self.verify(finding, context))
            else:
                # Multiple findings - use batch prompt
                batch_results = self._verify_batch(batch)
                results.extend(batch_results)

        return results

    def _build_prompt(
        self,
        finding: "Finding",
        context: "CodeContext",
    ) -> str:
        """Build verification prompt for a single finding."""
        # Get redacted token
        token_redacted = finding.redacted_token or "***"

        # Extract variable name from context
        variable_name = self._extract_variable_name(context)

        # Build context string
        context_str = self._build_context_string(context)

        return VERIFICATION_PROMPT_TEMPLATE.format(
            token_redacted=token_redacted,
            secret_type=finding.type,
            variable_name=variable_name or "unknown",
            filename=context.file_path or "unknown",
            context=context_str,
        )

    def _build_context_string(self, context: "CodeContext") -> str:
        """Build a readable context string."""
        lines = []

        # Add lines before
        for line in context.lines_before[-3:]:
            lines.append(line)

        # Add current line (highlighted)
        lines.append(f">>> {context.line_content}")

        # Add lines after
        for line in context.lines_after[:3]:
            lines.append(line)

        return "\n".join(lines)

    def _extract_variable_name(self, context: "CodeContext") -> Optional[str]:
        """Extract variable name from code context."""
        line = context.line_content

        # Common patterns: var = "value", var: str = "value", const VAR = "value"
        patterns = [
            r"(\w+)\s*=\s*['\"]",  # var = "value"
            r"(\w+)\s*:\s*\w+\s*=\s*['\"]",  # var: str = "value"
            r"const\s+(\w+)\s*=",  # const VAR =
            r"let\s+(\w+)\s*=",  # let var =
            r"var\s+(\w+)\s*=",  # var var =
            r"(\w+)\s*:\s*['\"]",  # YAML: key: "value"
            r'"(\w+)":\s*["\']',  # JSON: "key": "value"
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)

        return None

    def _verify_batch(
        self,
        batch: List[Tuple["Finding", "CodeContext"]],
    ) -> List[LLMVerificationResult]:
        """Verify a batch of findings with a single LLM call."""
        # Build findings text
        findings_text_parts = []
        for i, (finding, context) in enumerate(batch, 1):
            token_redacted = finding.redacted_token or "***"
            variable_name = self._extract_variable_name(context)
            context_str = self._build_context_string(context)

            part = f"""
Finding {i}:
TOKEN: {token_redacted}
TYPE: {finding.type}
VARIABLE: {variable_name or "unknown"}
FILE: {context.file_path or "unknown"}
CONTEXT:
```
{context_str}
```
"""
            findings_text_parts.append(part)

        findings_text = "\n".join(findings_text_parts)

        prompt = BATCH_VERIFICATION_PROMPT_TEMPLATE.format(
            count=len(batch),
            findings_text=findings_text,
        )

        # Call LLM
        try:
            response = self.provider.complete(
                prompt=prompt,
                system_prompt=VERIFICATION_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens * len(batch),
            )
        except Exception as e:
            logger.error(f"Batch LLM verification failed: {e}")
            # Return conservative results on error
            return [
                LLMVerificationResult(
                    is_secret=True,
                    confidence=0.5,
                    reasoning=f"LLM verification failed: {e}",
                    model=self.provider.model,
                    provider=self.provider.name,
                    latency_ms=0.0,
                )
                for _ in batch
            ]

        # Parse batch response
        return self._parse_batch_response(response, len(batch))

    def _parse_response(self, response: LLMResponse) -> LLMVerificationResult:
        """Parse LLM response into verification result."""
        content = response.content.strip()

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            return LLMVerificationResult(
                is_secret=bool(data.get("is_secret", True)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "No explanation provided")),
                model=response.model,
                provider=response.provider,
                latency_ms=response.latency_ms or 0.0,
                tokens_used=response.tokens_used,
                raw_response=response.content,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

            # Try to extract boolean from text
            is_secret = True  # Conservative default
            if "false positive" in content.lower():
                is_secret = False
            elif "not a secret" in content.lower():
                is_secret = False
            elif "is a secret" in content.lower():
                is_secret = True

            return LLMVerificationResult(
                is_secret=is_secret,
                confidence=0.5,
                reasoning=content[:200] if content else "Failed to parse response",
                model=response.model,
                provider=response.provider,
                latency_ms=response.latency_ms or 0.0,
                tokens_used=response.tokens_used,
                raw_response=response.content,
            )

    def _parse_batch_response(
        self,
        response: LLMResponse,
        expected_count: int,
    ) -> List[LLMVerificationResult]:
        """Parse batch LLM response into list of results."""
        content = response.content.strip()
        latency_per_result = (response.latency_ms or 0.0) / expected_count

        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            if not isinstance(data, list):
                raise ValueError("Expected JSON array")

            results = []
            for i, item in enumerate(data):
                results.append(
                    LLMVerificationResult(
                        is_secret=bool(item.get("is_secret", True)),
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=str(item.get("reasoning", "No explanation")),
                        model=response.model,
                        provider=response.provider,
                        latency_ms=latency_per_result,
                        tokens_used=None,
                        raw_response=None,
                    )
                )

            # Pad with conservative results if needed
            while len(results) < expected_count:
                results.append(
                    LLMVerificationResult(
                        is_secret=True,
                        confidence=0.5,
                        reasoning="Missing from batch response",
                        model=response.model,
                        provider=response.provider,
                        latency_ms=latency_per_result,
                    )
                )

            return results[:expected_count]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse batch response: {e}")
            return [
                LLMVerificationResult(
                    is_secret=True,
                    confidence=0.5,
                    reasoning=f"Failed to parse batch response: {e}",
                    model=response.model,
                    provider=response.provider,
                    latency_ms=latency_per_result,
                    raw_response=response.content if i == 0 else None,
                )
                for i in range(expected_count)
            ]
