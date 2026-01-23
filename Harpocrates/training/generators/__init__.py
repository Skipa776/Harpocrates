"""
Synthetic training data generators for Harpocrates.

Generates balanced datasets of true secrets and false positives
with realistic code context for ML model training.
"""
from __future__ import annotations

from Harpocrates.training.generators.context_templates import (
    generate_negative_context,
    generate_positive_context,
)
from Harpocrates.training.generators.generate_data import (
    generate_training_data,
    generate_training_record,
)
from Harpocrates.training.generators.secret_templates import (
    generate_aws_key,
    generate_github_token,
    generate_jwt_token,
    generate_openai_key,
    generate_random_secret,
    generate_stripe_key,
)

__all__ = [
    "generate_aws_key",
    "generate_github_token",
    "generate_jwt_token",
    "generate_openai_key",
    "generate_random_secret",
    "generate_stripe_key",
    "generate_positive_context",
    "generate_negative_context",
    "generate_training_data",
    "generate_training_record",
]
