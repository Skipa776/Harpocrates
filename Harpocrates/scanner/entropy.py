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
    Calculate the Shannon entropy (base 2) of a string.
    
    Returns 0.0 for an empty string. Higher values indicate a more uniform character distribution.
    
    Parameters:
        s (str): The input string to measure.
    
    Returns:
        float: The Shannon entropy of `s` in bits.
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
    Determine whether a string resembles a secret token using length, character-class, and entropy heuristics.
    
    The string is considered a potential secret only if all of the following hold:
    - length is at least 20 characters;
    - contains more than 3 unique characters;
    - contains at least two of these character classes: lowercase, uppercase, digits, non-alphanumeric (special);
    - Shannon entropy (base-2) is greater than or equal to `threshold`.
    
    Parameters:
        s (str): Input string to evaluate.
        threshold (float, optional): Entropy cutoff; defaults to 4.0.
    
    Returns:
        bool: `True` if the string meets the heuristic criteria for a secret, `False` otherwise.
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

class EntropyScanner(BaseScanner):
    '''
    Scanner for entropy-based secret detection.
    '''
    def __init__(
        self,
        entropy_threshold: float = 4.0,
        base_confidence: float = 0.4,
    ) -> None:
        """
        Initialize the EntropyScanner.
        
        Parameters:
            entropy_threshold (float): Minimum Shannon entropy required for a token to be considered a potential secret.
            base_confidence (float): Baseline confidence score (0.0–1.0) used when computing each finding's confidence.
        """
        super().__init__(name="EntropyScanner")
        self.base_confidence = base_confidence
        self.entropy_threshold = entropy_threshold
        
    def scan(self, content: str, context: Dict[str, Any]) -> List[Finding]:
        """
        Scan the provided text for high-entropy tokens and produce findings for candidates that resemble secrets.
        
        Parameters:
            content (str): The text to scan, processed line-by-line.
            context (Dict[str, Any]): Additional context used when creating findings. If present, the key `"file_path"` will be included in each Finding.
        
        Returns:
            List[Finding]: A list of Finding objects representing tokens that passed the heuristic and entropy thresholds; each Finding includes location, masked text, confidence score, and metadata (including entropy and a snippet).
        """
        file_path = context.get("file_path")
        findings: List[Finding] = []
        
        for lineno, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            for token in _TOKEN_RE.findall(stripped):
                if not looks_like_secret(token, threshold=self.entropy_threshold):
                    continue
                entropy_val = shannon_entropy(token)
                if entropy_val < self.entropy_threshold:
                    continue
                
                confidence = min(
                    1.0,
                    self.base_confidence + max(0.0, entropy_val - (self.entropy_threshold * 0.05)),
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
        """
        Mask a string preserving the first two and last two characters.
        
        Parameters:
            value (str): The string to mask.
        
        Returns:
            str: The masked string with middle characters replaced by asterisks. If `value` length is 4 or less, returns a string of asterisks of the same length.
        """
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]