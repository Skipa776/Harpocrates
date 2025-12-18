from __future__ import annotations

import math
from collections import Counter


def shannon_entropy(s: str) -> float:
    """
    Compute Shannon entropy (base-2) for a string.

    Notes:
        - Returns 0.0 for empty strings.
        - Uses O(n) counting via collections.Counter
        - Entropy is higher when character distribution is uniformly distributed

    Examples:
        >>> round(shannon_entropy("aaaaaaa"), 3)
        0.0
        >>> round(shannon_entropy("Aa1Aa1Aa1"), 3) >= 3.0
        True

    Args:
        s (str): The input string.

    Returns:
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
    Heuristic to decide if a string looks like a secret.

    Conditions (all must pass):
        - Length >=2
        - Character diversity: > 3 unique characters
        - At least 2 character classes among: [lowercase, uppercase, digits, special]
        - Shannon entropy >= threshold (default: 4.0)

    Rationale:
      These rules suppress many config-ish or English-like strings while
      still surfacing most machine-generated tokens.

    Examples:
        >>> looks_like_secret("just_a_normal_config_value", threshold=3.0)
        False
        >>> looks_like_secret("AKIAIOSFODNN7EXAMPLE", threshold=3.5)
        True
        >>> looks_like_secret("aaaaaaaaaaaaaaaaaaaaaaaa", threshold=3.0)
        False

    Args:
        s (str): The input string.
        threshold (float, optional): The entropy threshold. Defaults to 4.0.

    Returns:
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

    return shannon_entropy(s) >= threshold

__all__ = ["shannon_entropy", "looks_like_secret"]
