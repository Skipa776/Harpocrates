"""
Harpocrates detectors package.

Provides regex pattern matching and entropy-based secret detection.
"""
from __future__ import annotations

from Harpocrates.detectors.entropy_detector import looks_like_secret, shannon_entropy
from Harpocrates.detectors.regex_patterns import SIGNATURES

__all__ = [
    "SIGNATURES",
    "shannon_entropy",
    "looks_like_secret",
]
