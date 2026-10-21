from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")


def shannon_entropy(s: str) -> float:
    """
    Description: Compute Shannon entropy (base-2) for a string, which is higher when its characters are spread more
        evenly.
    Parameters:
        s: The input string.
    Output: Base-2 Shannon entropy as a float, or 0.0 for an empty string.
    """
    if not s:
        return 0.0

    counts = Counter(s)
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent

def looks_like_secret(s: str, threshold: float = 4.0) -> bool:
    """
    Description: Apply length, character-class, and entropy filters to flag machine-generated candidates. They
        suppress config-like and English-like strings while still surfacing most generated tokens.
    Parameters:
        s: The input string.
        threshold: The entropy threshold. Defaults to 4.0.
    Output: True only when all filters pass: at least 20 characters, more than 3 unique characters, at least 2
        character classes among lowercase, uppercase, digits and special, Shannon entropy at or above threshold, and
        not alphabetic-only or lowercase-only without a digit; otherwise False.
    """
    if len(s) < 20:
        return False

    if len(set(s)) <= 3:
        return False

    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    has_digit = any(c.isdigit() for c in s)
    has_special = any(not c.isalnum() for c in s)

    if sum([has_upper, has_lower, has_digit, has_special]) < 2:
        return False

    # Tier 3: pure-alpha tokens (camelCase identifiers) are not machine-generated
    # secrets — all real credential schemes use at least one digit or special char.
    if not has_digit and not has_special:
        return False

    # Tier 3: lowercase-only + hyphen/slash but no digit = CSS class name, URL
    # slug, or file path component. UUID-style API keys always contain hex digits
    # (0-9a-f), so this rule never fires on them.
    if has_lower and not has_upper and not has_digit:
        return False

    return shannon_entropy(s) >= threshold

__all__ = ["shannon_entropy", "looks_like_secret"]
