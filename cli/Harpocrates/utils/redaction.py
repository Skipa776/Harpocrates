"""
Secret redaction utilities.

Single source of truth for how Harpo renders secret tokens in any
user-facing output (CLI tables, JSON responses, API payloads, logs).
Keeping this logic in one place guarantees that every surface stays in
lockstep with the redaction contract.

The rules:
    - token is None, empty, or whitespace-only -> None
    - len(token) <  8                          -> "[REDACTED]"  (too short to show safely)
    - len(token) == 8                          -> "<first>...<last>"
    - 9 <= len(token) <= 10                    -> "<first2>...<last2>"
    - len(token) >= 11                         -> "<first4>...<last4>"
"""
from __future__ import annotations

from typing import Optional

_MIN_SAFE_LEN = 8
_SHORT_TOKEN_LEN = 8
_MEDIUM_TOKEN_LEN = 10
_ELLIPSIS = "..."
_REDACTED_PLACEHOLDER = "[REDACTED]"


def redact_token(token: Optional[str]) -> Optional[str]:
    """
    Description: Return a redacted display form of ``token``.
    Parameters:
        token: Optional token to render for display.
    Output: None for absent/blank input, the full redaction placeholder below eight characters, or
    an ellipsis with 1, 2, or 4 characters from each end at lengths 8, 9-10, or 11+. No side
    effects.
    Returns ``None`` when ``token`` is ``None``, empty, or whitespace-only so callers can cleanly
    distinguish "no token to show" from a redacted value. Tokens shorter than ``_MIN_SAFE_LEN`` are
    replaced entirely with ``"[REDACTED]"`` because slicing them would disclose a disproportionate
    fraction of the original (e.g. a 2-char token would leak both chars).
    Examples: >>> redact_token(None) >>> redact_token("") >>> redact_token(" ") >>>
    redact_token("ab") '[REDACTED]' >>> redact_token("abcdefgh") 'a...h' >>>
    redact_token("abcdefghij") 'ab...ij' >>> redact_token("abcdefghijklmnopqr") 'abcd...opqr'
    """
    if token is None or not token.strip():
        return None
    if len(token) < _MIN_SAFE_LEN:
        return _REDACTED_PLACEHOLDER
    if len(token) <= _SHORT_TOKEN_LEN:
        return f"{token[0]}{_ELLIPSIS}{token[-1]}"
    if len(token) <= _MEDIUM_TOKEN_LEN:
        return f"{token[:2]}{_ELLIPSIS}{token[-2:]}"
    return f"{token[:4]}{_ELLIPSIS}{token[-4:]}"


__all__ = ["redact_token"]
