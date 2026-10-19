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

Inside a source line (``mask_secrets``) a token under 16 characters is masked whole: the forms above show
8 of 11 characters, which is most of a short secret.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Optional

_MIN_SAFE_LEN = 8
_SHORT_TOKEN_LEN = 8
_MEDIUM_TOKEN_LEN = 10
_ELLIPSIS = "..."
_REDACTED_PLACEHOLDER = "[REDACTED]"
_MIN_LINE_FORM_LEN = 16  # shortest token whose redact_token form (8 characters shown) is at most half of it
_MIN_CUT_LEN = 4  # shortest piece of a token, cut off at the end of a snippet, that is still masked


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


def _line_form(token: str) -> str:
    """
    Description: Choose the text that replaces a token inside a source line.
    Parameters:
        token: Secret text, or the merged run of text that covers it.
    Output: ``[REDACTED]`` under 16 characters, otherwise the ``redact_token`` form.
    """
    if len(token) < _MIN_LINE_FORM_LEN:
        return _REDACTED_PLACEHOLDER
    return redact_token(token) or _REDACTED_PLACEHOLDER


def _spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """
    Description: Locate every token in text, including a token's start that a snippet cut off.
    Parameters:
        text: Source line to search.
        tokens: Secret values, longest first.
    Output: ``(start, end)`` spans of each occurrence, plus one span at the end of ``text`` when it ends
        with the first 4 or more characters of a token.
    """
    found = [m.span() for t in tokens for m in re.finditer(re.escape(t), text)]
    for t in tokens:
        for size in range(min(len(t) - 1, len(text)), _MIN_CUT_LEN - 1, -1):
            if text.endswith(t[:size]):
                found.append((len(text) - size, len(text)))
                break
    return found


def mask_secrets(text: str, tokens: Iterable[Optional[str]]) -> str:
    """
    Description: Replace every occurrence of each token in ``text`` with its redacted form so a source
        line can be shown without the secret.
    Parameters:
        text: Source line or other free text that may contain the tokens.
        tokens: Secret values to hide; ``None`` and blank entries are ignored.
    Output: ``text`` with each occurrence replaced by ``redact_token(token)``, or ``[REDACTED]`` when the
        token is under 16 characters. Overlapping or touching occurrences are merged and masked as one
        run, and the start of a token that a 200-character snippet cut off at the end of ``text`` is
        masked too (from 4 characters up). No side effects.
    """
    wanted = sorted({t for t in tokens if t and t.strip()}, key=len, reverse=True)
    merged: list[list[int]] = []
    for start, end in sorted(_spans(text, wanted)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces, at = [], 0
    for start, end in merged:
        pieces += [text[at:start], _line_form(text[start:end])]
        at = end
    return "".join(pieces) + text[at:]


__all__ = ["mask_secrets", "redact_token"]
