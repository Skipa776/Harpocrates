from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Any, List

from Harpocrates.scanner.base import BaseScanner
from Harpocrates.scanner.models import Finding

_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-.]{8,}")

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

__all__ = ["EntropyScanner", "looks_like_secret", "shannon_entropy"]

class EntropyScanner(BaseScanner):
    '''
    Scanner for entropy-based secret detection.
    '''
    def __init__(
        self,
        entropy_threshold: float = 4.0,
        base_confidence: float = 0.4,
    ) -> None:
        super().__init__(name="EntropyScanner")
        self.base_confidence = base_confidence
        self.entropy_threshold = entropy_threshold
        
    def scan(self, content: str, context: Dict[str, Any]) -> List[Finding]:
        file_path = str(context.get("file_path", ""))
        findings: List[Finding] = []
        
        for lineno, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            for token in _TOKEN_RE.findall(stripped):
                entropy_val = shannon_entropy(token)
                if not looks_like_secret(token, threshold=self.entropy_threshold):
                    continue
                
                confidence = min(
                    1.0,
                    self.base_confidence + max(0.0, entropy_val - self.entropy_threshold),
                )
                
                column = line.find(token)
                findings.append(
                    Finding(
                        scanner_name=self.name,
                        signature_name="ENTROPY_HEURISTIC",
                        file_path=file_path,
                        line_number=lineno,
                        column=column if column >= 0 else 0,
                        raw_text=token,
                        masked_text=self._mask(token),
                        confidence_score=confidence,
                        metadata={
                            "type": "ENTROPY_CANDIDATE",
                            "entropy": entropy_val,
                            "snippet": stripped[:200],
                            "evidence": "entropy",
                        },
                    )
                )
        return findings
    
    @staticmethod
    def _mask(value: str) -> str:
        """Mask helper that keeps the first/last two characters."""
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]