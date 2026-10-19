from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")


def shannon_entropy(s: str) -> float:
    """
    Description: Compute Shannon entropy (base-2) for a string.
    Parameters:
        s: The input string.
    Output: Base-2 Shannon entropy, with 0.0 for empty strings. The older mixed-character example
    below has entropy log2(3), so its displayed >= 3.0 result is inaccurate.
    Notes: - Returns 0.0 for empty strings. - Uses O(n) counting via collections.Counter - Entropy
    is higher when character distribution is uniformly distributed
    Examples: >>> round(shannon_entropy("aaaaaaa"), 3) 0.0 >>> round(shannon_entropy("Aa1Aa1Aa1"),
    3) >= 3.0 True
    float: The Shannon entropy of the input string.
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
    Description: Apply length, character-class, and entropy filters to identify machine-generated
    candidates.
    Parameters:
        s: The input string.
        threshold: The entropy threshold. Defaults to 4.0.
    Output: True only when all implemented filters pass; otherwise False. The actual minimum length
    is 20, not the older >=2 note below. Pure alphabetic strings and lowercase strings without
    uppercase letters or digits are also rejected; the configuration-value example below remains
    False.
    Heuristic to decide if a string looks like a secret.
    Conditions (all must pass): - Length >=2 - Character diversity: > 3 unique characters - At least
    2 character classes among: [lowercase, uppercase, digits, special] - Shannon entropy >=
    threshold (default: 4.0)
    Rationale: These rules suppress many config-ish or English-like strings while still surfacing
    most machine-generated tokens.
    Examples: >>> looks_like_secret("just_a_normal_config_value", threshold=3.0) False
    The vendor-prefixed,
    20-character example passes at threshold=3.5; its credential-shaped value is omitted here. >>>
    looks_like_secret("aaaaaaaaaaaaaaaaaaaaaaaa", threshold=3.0) False
    bool: True if the string looks like a secret, False otherwise.
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
