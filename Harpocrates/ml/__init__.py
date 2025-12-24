"""
ML-based secrets verification module for Harpocrates.

This module provides context-aware detection using XGBoost to distinguish
between true secrets and false positives (e.g., Git SHAs, UUIDs).
"""
from __future__ import annotations

from Harpocrates.ml.context import CodeContext, extract_context
from Harpocrates.ml.features import FeatureVector, extract_features
from Harpocrates.ml.verifier import VerificationResult, XGBoostVerifier

__all__ = [
    "CodeContext",
    "extract_context",
    "FeatureVector",
    "extract_features",
    "VerificationResult",
    "XGBoostVerifier",
]
