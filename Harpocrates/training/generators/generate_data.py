"""
Synthetic training data generator for Harpocrates ML.

Generates balanced datasets of true secrets (positive) and
false positives (negative) with realistic code context.

This generator is designed to create ambiguous samples to train a robust
classifier, avoiding common shortcuts and label leakage.

Usage:
    python -m Harpocrates.training.generators.generate_data \
        --output training_data.jsonl \
        --count 10000 \
        --balance 0.5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Harpocrates.training.generators.context_templates import (
    SAFE_VAR_NAMES,
    SECRET_VAR_NAMES,
    TEST_VAR_NAMES,
    generate_context,
)
from Harpocrates.training.generators.secret_templates import (
    generate_aws_key,
    generate_base64_data,
    generate_base64_secret,
    generate_base64_telemetry_id,
    generate_checksum,
    generate_connection_string,
    generate_content_hash_with_context,
    generate_database_url,
    generate_documentation_example,
    generate_encoded_non_secret,
    generate_encoded_public_data,
    generate_fake_prefixed_token,
    generate_git_sha,
    generate_git_sha_with_context,
    generate_github_token,
    generate_hash_in_security_context,
    generate_hex_secret,
    generate_high_entropy_non_secret,
    generate_invalid_jwt,
    generate_jwt_token,
    generate_openai_key,
    generate_password,
    generate_pem_private_key,
    generate_random_secret,
    generate_revoked_token,
    generate_session_id_non_secret,
    generate_slack_token,
    generate_ssh_private_key,
    generate_stripe_key,
    generate_test_fixture_token,
    generate_uuid,
    generate_uuid_with_auth_context,
    generate_version_hash,
    generate_webhook_url_with_token,
    generate_certificate,
)

# --- Constants for Ambiguity Control ---

# 50% of positive samples use safe or neutral variable names (prevents shortcut)
SAFE_NAME_MIX_RATIO_POSITIVE = 0.5

# 50% of negative samples use secret-like variable names (prevents shortcut)
SECRET_NAME_MIX_RATIO_NEGATIVE = 0.5

# --- Data Generator Transformer Settings ---
# These settings implement context-token disentanglement per skill spec

# Label noise rate for training data (0 for test data)
LABEL_NOISE_RATE = 0.08

# Contrastive pair ratio (same token, different context)
CONTRASTIVE_RATIO = 0.3

# Vendor neutralization: ensure vendor strings appear in both classes
VENDOR_NEUTRALIZATION = True

# --- Token Generation ---

# Defines token types that are semantically secrets (label=1)
# Includes ambiguous formats (hex, base64) that look like non-secrets
POSITIVE_TOKEN_GENERATORS = {
    # Standard prefixed secrets
    "aws_access_key": lambda: generate_aws_key()[0],
    "aws_secret_key": lambda: generate_aws_key()[1],
    "github_token": generate_github_token,
    "stripe_key": lambda: generate_stripe_key(live=random.random() > 0.1),
    "openai_key": generate_openai_key,
    "jwt_token": generate_jwt_token,
    "slack_token": generate_slack_token,
    "password": lambda: generate_password(random.choice(["medium", "high"])),
    "generic_secret": generate_random_secret,
    # Ambiguous secrets that look like non-secrets (prevent format shortcuts)
    "hex_secret": generate_hex_secret,  # Looks like git SHA but IS a secret
    "base64_secret": generate_base64_secret,  # Looks like data but IS a secret
    # Embedded credentials (secrets in URLs/connection strings)
    "database_url": lambda: generate_database_url()[0],  # Full URL with embedded password
    "webhook_url": lambda: generate_webhook_url_with_token()[0],  # URL with token param
    "connection_string": lambda: generate_connection_string()[0],  # ADO/JDBC style
    # Multi-line secrets (PEM keys)
    "pem_private_key": lambda: generate_pem_private_key()[0],
    "ssh_private_key": lambda: generate_ssh_private_key()[0],
}

# Defines token types that are semantically non-secrets (label=0)
# Includes tokens with secret-like prefixes that are NOT secrets
NEGATIVE_TOKEN_GENERATORS = {
    # Standard non-secret formats
    "git_sha": generate_git_sha,
    "uuid": generate_uuid,
    "checksum": lambda: generate_checksum(random.choice(["md5", "sha1", "sha256"])),
    "base64_data": lambda: generate_base64_data(random.randint(24, 64)),
    # Tokens that look like secrets but are NOT (prevent prefix shortcuts)
    "fake_prefixed_token": generate_fake_prefixed_token,  # Has AKIA/ghp_/sk- prefix
    "fake_stripe_key": lambda: generate_stripe_key(live=False),
    "fake_github_token": lambda: generate_github_token(valid=False),
    "fake_aws_key": lambda: generate_aws_key(valid=False)[1],
    "high_entropy_string": lambda: generate_random_secret(20, 40),
    # --- HARD NEGATIVES (force context-based learning) ---
    # Documentation examples - well-known placeholders that should NEVER be flagged
    "doc_example": lambda: generate_documentation_example()[0],
    # Revoked tokens - valid format but appear in revocation contexts
    "revoked_token": lambda: generate_revoked_token()[0],
    # Test fixtures - valid format but clearly for testing
    "test_fixture": lambda: generate_test_fixture_token()[0],
    # Encoded non-secrets - base64 content that's not a secret
    "encoded_config": lambda: generate_encoded_non_secret()[0],
    # High entropy non-secrets - hashes, session IDs, cache keys
    "high_entropy_safe": generate_high_entropy_non_secret,
    # --- GIT SHA NEGATIVES (explicit context) ---
    # Git SHAs with explicit variable names like commit_sha, git_hash
    "git_sha_with_context": lambda: generate_git_sha_with_context()[0],
    # Content hashes with explicit variable names
    "content_hash": lambda: generate_content_hash_with_context()[0],
    # X.509 certificates (public, not secret)
    "certificate": lambda: generate_certificate()[0],
    # --- ENHANCED HARD NEGATIVES (precision boost) ---
    # UUIDs with auth-like variable names (api_key = 'uuid-here')
    "uuid_auth_context": lambda: generate_uuid_with_auth_context()[0],
    # Hashes in security-adjacent contexts (password_hash = 'hash-here')
    "hash_security_context": lambda: generate_hash_in_security_context()[0],
    # Invalid/expired JWT tokens
    "invalid_jwt": lambda: generate_invalid_jwt()[0],
    # Base64 telemetry/correlation IDs
    "telemetry_id": lambda: generate_base64_telemetry_id()[0],
    # Session IDs (identifiers, not secrets)
    "session_id": lambda: generate_session_id_non_secret()[0],
    # Encoded public data (configs, metadata)
    "encoded_public": lambda: generate_encoded_public_data()[0],
    # Version/content hashes for caching/integrity
    "version_hash": lambda: generate_version_hash()[0],
}


# --- Distribution Definitions ---
#
# IMPORTANT: These distributions are designed to PREVENT shortcut learning:
# - 40%+ of positive samples use hex/base64 format (indistinguishable from non-secrets)
# - 40%+ of negative samples have secret-like prefixes (AKIA, ghp_, sk-)
#
# This forces the model to learn from CONTEXT, not token format.

# High-level semantic distribution for positive samples.
POSITIVE_DISTRIBUTION = {
    "prefixed_api_key": 0.25,  # AWS, GitHub, Stripe, OpenAI, Slack
    "jwt": 0.08,
    "password": 0.08,
    "hex_secret": 0.15,  # Hex secrets that look like git SHAs
    "base64_secret": 0.10,  # Base64 secrets that look like data
    "generic": 0.08,
    # NEW: Embedded credentials (to train model to find secrets in URLs)
    "embedded_credential": 0.15,  # DATABASE_URL, webhook URLs, connection strings
    # NEW: Multi-line secrets (PEM keys)
    "pem_key": 0.11,  # Private keys in PEM format
}

# High-level semantic distribution for negative samples.
# Updated with ENHANCED HARD NEGATIVES to force context-based learning
# NOTE: Distribution rebalanced to target 90%+ precision while maintaining recall
NEGATIVE_DISTRIBUTION = {
    # Standard non-secrets - hex tokens (reduced to make room for hard negatives)
    "git_sha": 0.04,          # Pure hex (40 chars) - generic context
    "checksum": 0.03,         # Hex hashes
    "uuid": 0.02,
    "data": 0.02,
    # Fake prefixed tokens (existing)
    "fake_prefixed": 0.05,
    "ambiguous_key": 0.03,
    # --- HARD NEGATIVES (force context-based learning) ---
    "doc_example": 0.06,      # Documentation placeholders like AKIAIOSFODNN7EXAMPLE
    "revoked_token": 0.05,    # Valid format tokens in revocation contexts
    "test_fixture": 0.06,     # Test tokens like AKIATESTKEY123456789
    "encoded_config": 0.03,   # Base64 JSON configs, not secrets
    "high_entropy_safe": 0.05,  # Hashes, session IDs, cache keys
    # --- GIT SHA EXPLICIT CONTEXT (suppress false positives) ---
    "git_sha_explicit": 0.10,  # Git SHAs with commit_sha, git_hash var names
    "content_hash_explicit": 0.07,  # Content hashes with checksum, digest var names
    # --- CERTIFICATES (public, not secrets) ---
    "certificate": 0.02,  # X.509 certificates
    # --- ENHANCED HARD NEGATIVES (precision boost) ---
    # These are the critical additions for reaching 90% precision
    "uuid_auth_context": 0.10,  # UUIDs with api_key, auth_token var names
    "hash_security_context": 0.08,  # Hashes with password_hash, secret_hash var names
    "invalid_jwt": 0.05,  # JWT-shaped but invalid/expired tokens
    "telemetry_id": 0.05,  # Base64 trace_id, correlation_id, request_id
    "session_id": 0.04,  # Session IDs that look like secrets
    "encoded_public": 0.03,  # Base64 configs, metadata, feature flags
    "version_hash": 0.02,  # Asset hashes, etags, content hashes
}

# Mapping from high-level positive types to specific token generators
POSITIVE_TYPE_TO_GENERATOR = {
    "prefixed_api_key": [
        "aws_access_key", "aws_secret_key", "github_token",
        "stripe_key", "openai_key", "slack_token"
    ],
    "jwt": ["jwt_token"],
    "password": ["password"],
    "hex_secret": ["hex_secret"],  # Looks like SHA but IS a secret
    "base64_secret": ["base64_secret"],  # Looks like data but IS a secret
    "generic": ["generic_secret"],
    # NEW: Embedded credentials
    "embedded_credential": ["database_url", "webhook_url", "connection_string"],
    # NEW: PEM keys
    "pem_key": ["pem_private_key", "ssh_private_key"],
}

# Mapping from high-level negative types to specific token generators
NEGATIVE_TYPE_TO_GENERATOR = {
    # Standard non-secrets
    "git_sha": ["git_sha"],
    "uuid": ["uuid"],
    "checksum": ["checksum"],
    "data": ["base64_data"],
    # Fake prefixed tokens
    "fake_prefixed": ["fake_prefixed_token"],  # Has secret-like prefix but NOT a secret
    "ambiguous_key": ["fake_stripe_key", "fake_github_token", "fake_aws_key"],
    # --- HARD NEGATIVES ---
    "doc_example": ["doc_example"],           # AKIAIOSFODNN7EXAMPLE, ghp_xxx...
    "revoked_token": ["revoked_token"],       # Valid format in revocation context
    "test_fixture": ["test_fixture"],         # AKIATESTKEY, sk_test_*, etc.
    "encoded_config": ["encoded_config"],     # Base64 JSON configs
    "high_entropy_safe": ["high_entropy_safe", "high_entropy_string"],  # Hashes, IDs
    # --- GIT SHA EXPLICIT CONTEXT (suppress false positives) ---
    "git_sha_explicit": ["git_sha_with_context"],  # commit_sha, git_hash var names
    "content_hash_explicit": ["content_hash"],     # checksum, digest var names
    # --- CERTIFICATES ---
    "certificate": ["certificate"],  # X.509 public certs
    # --- ENHANCED HARD NEGATIVES (precision boost) ---
    "uuid_auth_context": ["uuid_auth_context"],  # UUIDs with api_key var names
    "hash_security_context": ["hash_security_context"],  # Hashes with password_hash var names
    "invalid_jwt": ["invalid_jwt"],  # JWT-shaped but invalid tokens
    "telemetry_id": ["telemetry_id"],  # trace_id, correlation_id
    "session_id": ["session_id"],  # Session IDs
    "encoded_public": ["encoded_public"],  # Config/metadata blobs
    "version_hash": ["version_hash"],  # Asset/content hashes
}

# Languages to use for context generation
LANGUAGES = ["python", "javascript", "yaml", "json", "shell", "go", "java"]

# Context types that can be applied to any sample
CONTEXT_TYPES = ["production", "test", "documentation", "configuration"]

# --- Context Symmetry Enforcement ---
# These lists ensure vendor names, auth keywords, and file patterns
# appear at similar frequencies in BOTH positive and negative classes.
# This prevents the model from learning superficial correlations.

# Vendor-related context strings (appear in comments, variable names, paths)
VENDOR_CONTEXT_STRINGS = [
    "aws", "amazon", "s3", "ec2", "lambda",
    "github", "gitlab", "bitbucket",
    "stripe", "payment", "checkout",
    "openai", "anthropic", "gpt",
    "slack", "discord", "teams",
    "google", "gcp", "firebase",
    "azure", "microsoft",
    "docker", "kubernetes", "k8s",
    "npm", "pypi", "pip",
]

# Auth-related keywords that could appear in either class
AUTH_CONTEXT_KEYWORDS = [
    "api", "key", "token", "secret", "password", "credential",
    "auth", "authentication", "authorization",
    "bearer", "oauth", "jwt", "session",
    "private", "public", "encryption", "signing",
]

# File path patterns for both classes
FILE_PATH_PATTERNS = {
    "config": ["config/", "settings/", ".env", "config.py", "settings.yaml"],
    "test": ["test/", "tests/", "spec/", "__tests__/", "_test.py", ".test.js"],
    "src": ["src/", "lib/", "app/", "pkg/", "internal/"],
    "deploy": ["deploy/", "infra/", "terraform/", "k8s/", ".github/"],
}


def _weighted_choice(distribution: Dict[str, float]) -> str:
    """Select a key based on weighted distribution."""
    items = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(items, weights=weights, k=1)[0]


def _get_var_name(positive: bool, token: str = "") -> str:
    """
    Get a variable name, mixing in confusing names for ambiguity.

    Args:
        positive: Whether this is a positive (secret) sample
        token: The token string - used to check for known secret prefixes

    Key insight: For negative samples with known secret prefixes (AKIA, ghp_, sk_),
    we should NOT use secret-like variable names. This creates impossible-to-classify
    samples that hurt model performance. Instead, use clearly safe/test names.
    """
    # Check if token has a known secret prefix
    known_secret_prefixes = ("AKIA", "ghp_", "gho_", "ghu_", "sk-", "sk_", "xoxb-", "xoxp-")
    has_secret_prefix = any(token.startswith(p) for p in known_secret_prefixes)

    # For positive samples, sometimes use a safe-looking name
    if positive and random.random() < SAFE_NAME_MIX_RATIO_POSITIVE:
        return random.choice(SAFE_VAR_NAMES + TEST_VAR_NAMES)

    # For negative samples with secret prefixes, ALWAYS use safe/test names
    # This avoids creating impossible-to-classify samples
    if not positive and has_secret_prefix:
        # Use names that clearly indicate this is not a real secret
        clear_non_secret_names = [
            "example_key", "mock_token", "test_api_key", "placeholder",
            "sample_token", "dummy_key", "fake_credential", "demo_token",
            "test_key", "mock_api_key", "example_token", "stub_key",
        ]
        return random.choice(clear_non_secret_names + TEST_VAR_NAMES)

    # For negative samples without secret prefixes, can use confusing names
    if not positive and random.random() < SECRET_NAME_MIX_RATIO_NEGATIVE:
        return random.choice(SECRET_VAR_NAMES)

    # Default case
    if positive:
        return random.choice(SECRET_VAR_NAMES)
    else:
        # Negative samples can have safe or test-like names
        return random.choice(SAFE_VAR_NAMES + TEST_VAR_NAMES)


def generate_training_record(positive: bool) -> Dict[str, Any]:
    """
    Generate a single, potentially ambiguous, training data record.

    Args:
        positive: If True, generate a true secret (label=1), otherwise a false positive.

    Returns:
        A dictionary containing the training record.
        The record contains ONLY: token, line_content, context_before,
        context_after, file_path, label.
    """
    # For specialized semantic types, use their specific var_name
    specialized_var_name = None

    if positive:
        # 1. Select a high-level positive type (e.g., "api_key")
        semantic_type = _weighted_choice(POSITIVE_DISTRIBUTION)
        # 2. Select a specific generator from that type (e.g., "github_token")
        generator_key = random.choice(POSITIVE_TYPE_TO_GENERATOR[semantic_type])
        # 3. Generate the token
        token = POSITIVE_TOKEN_GENERATORS[generator_key]()
        label = 1
    else:
        # 1. Select a high-level negative type (e.g., "ambiguous_key")
        semantic_type = _weighted_choice(NEGATIVE_DISTRIBUTION)
        # 2. Select a specific generator from that type (e.g., "fake_stripe_key")
        generator_key = random.choice(NEGATIVE_TYPE_TO_GENERATOR[semantic_type])

        # 3. Handle specialized generators that return tuples with var_name
        # These generators force specific variable names to create ambiguous contexts
        if semantic_type == "git_sha_explicit":
            token, specialized_var_name, _ = generate_git_sha_with_context()
        elif semantic_type == "content_hash_explicit":
            token, specialized_var_name, _ = generate_content_hash_with_context()
        elif semantic_type == "uuid_auth_context":
            # UUID with auth-like variable name (e.g., api_key = 'uuid-here')
            token, specialized_var_name, _ = generate_uuid_with_auth_context()
        elif semantic_type == "hash_security_context":
            # Hash with security-like variable name (e.g., password_hash = 'hash-here')
            token, specialized_var_name, _, _ = generate_hash_in_security_context()
        elif semantic_type == "telemetry_id":
            # Base64 telemetry ID with tracking variable name
            token, specialized_var_name, _ = generate_base64_telemetry_id()
        elif semantic_type == "session_id":
            # Session ID with session-like variable name
            token, specialized_var_name = generate_session_id_non_secret()
        elif semantic_type == "encoded_public":
            # Encoded public data with config-like variable name
            token, specialized_var_name, _ = generate_encoded_public_data()
        elif semantic_type == "version_hash":
            # Version/content hash with versioning variable name
            token, specialized_var_name, _ = generate_version_hash()
        else:
            token = NEGATIVE_TOKEN_GENERATORS[generator_key]()
        label = 0

    # 4. Choose a variable name - use specialized name if available
    if specialized_var_name:
        var_name = specialized_var_name
    else:
        var_name = _get_var_name(positive, token)

    # 5. Choose a random language and context type, independent of the label
    language = random.choice(LANGUAGES)
    context_type = random.choice(CONTEXT_TYPES)

    # 6. Generate the context using the unified context generator
    line_content, context_before, context_after, file_path = generate_context(
        token=token,
        var_name=var_name,
        language=language,
        context_type=context_type,
    )

    # 7. Return a clean record with no leaky fields
    return {
        "token": token,
        "line_content": line_content,
        "context_before": context_before,
        "context_after": context_after,
        "file_path": file_path,
        "label": label,
    }


def generate_training_data(
    count: int = 10000,
    balance: float = 0.5,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a balanced dataset of training examples.

    Args:
        count: Total number of examples to generate
        balance: Ratio of positive examples (0.5 = 50% positive, 50% negative)
        seed: Random seed for reproducibility

    Returns:
        List of training records
    """
    if seed is not None:
        random.seed(seed)

    positive_count = int(count * balance)
    negative_count = count - positive_count

    records = []

    # Generate positive examples
    for _ in range(positive_count):
        records.append(generate_training_record(positive=True))

    # Generate negative examples
    for _ in range(negative_count):
        records.append(generate_training_record(positive=False))

    # Shuffle the dataset
    random.shuffle(records)

    return records


def generate_deceptive_positive() -> Dict[str, Any]:
    """
    Generate a real secret with misleading context.

    This creates a SECRET that looks like a non-secret:
    - Uses hex format (looks like git SHA)
    - Uses safe-looking variable name (commit_sha, file_hash)
    - Appears in build/script context
    """
    # Real secret in hex format
    token = generate_hex_secret(40)

    # Use misleading variable name
    var_name = random.choice([
        "commit_hash", "file_checksum", "build_id", "request_id",
        "content_digest", "revision", "sha256_hash"
    ])

    language = random.choice(LANGUAGES)

    line_content, context_before, context_after, file_path = generate_context(
        token=token,
        var_name=var_name,
        language=language,
        context_type="production",
    )

    # Override file path to look like a build script
    file_path = random.choice([
        "scripts/build.py", "utils/git_info.py", "lib/hash.py",
        "scripts/deploy.sh", "build/version.go"
    ])

    return {
        "token": token,
        "line_content": line_content,
        "context_before": context_before,
        "context_after": context_after,
        "file_path": file_path,
        "label": 1,  # It IS a secret despite appearances
    }


def generate_deceptive_negative() -> Dict[str, Any]:
    """
    Generate a non-secret with misleading context.

    This creates a NON-SECRET that looks like a secret:
    - Uses secret-like prefix (AKIA, ghp_, sk-)
    - Uses secret-looking variable name (api_key, secret)
    - Appears in config context
    """
    # Non-secret with secret-like prefix
    token = generate_fake_prefixed_token()

    # Use secret-looking variable name
    var_name = random.choice([
        "api_key", "API_KEY", "secret_key", "SECRET_KEY",
        "auth_token", "password", "credentials"
    ])

    language = random.choice(LANGUAGES)

    line_content, context_before, context_after, file_path = generate_context(
        token=token,
        var_name=var_name,
        language=language,
        context_type="configuration",
    )

    # Override file path to look like config
    file_path = random.choice([
        "config/settings.py", "config/api.yaml", ".env",
        "src/config.ts", "app/settings.go"
    ])

    return {
        "token": token,
        "line_content": line_content,
        "context_before": context_before,
        "context_after": context_after,
        "file_path": file_path,
        "label": 0,  # It is NOT a secret despite appearances
    }


def generate_adversarial_test_data(
    count: int = 1000,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate adversarial test samples designed to expose shortcut learning.

    These samples are specifically crafted to be maximally confusing:
    - Secrets with safe-looking variable names in script contexts
    - Non-secrets with secret-looking variable names in config contexts
    - Hex secrets that look exactly like git SHAs
    - Fake API keys with real-looking prefixes

    A model that relies on shortcuts (prefix matching, variable names, file paths)
    will perform poorly on this data. A model that truly understands context
    should still achieve reasonable accuracy.

    Args:
        count: Total number of adversarial samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of adversarial training records
    """
    if seed is not None:
        random.seed(seed)

    records = []
    quarter = count // 4

    # Category 1: Secrets that look like non-secrets (25%)
    for _ in range(quarter):
        records.append(generate_deceptive_positive())

    # Category 2: Non-secrets that look like secrets (25%)
    for _ in range(quarter):
        records.append(generate_deceptive_negative())

    # Category 3: Standard ambiguous samples using new distributions (25%)
    for _ in range(quarter):
        records.append(generate_training_record(positive=random.random() > 0.5))

    # Category 4: More deceptive samples to fill remaining (25%)
    remaining = count - len(records)
    for _ in range(remaining):
        if random.random() > 0.5:
            records.append(generate_deceptive_positive())
        else:
            records.append(generate_deceptive_negative())

    random.shuffle(records)
    return records


def generate_contrastive_pair(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a contrastive example: same token, different context, opposite label.

    For secrets (label=1): Generate benign context → label=0
    For non-secrets (label=0): Generate secret-like context → label still 0

    This breaks token→label shortcuts by showing the same token can have different meanings.
    """
    token = record["token"]
    original_label = record["label"]

    if original_label == 1:
        # Secret token in benign context → now it's NOT a secret (revoked, example, etc.)
        var_name = random.choice([
            "example_key", "EXAMPLE_API_KEY", "placeholder_token",
            "test_api_key", "sample_secret", "revoked_token",
            "old_api_key", "deprecated_secret", "mock_credential",
        ])
        context_type = random.choice(["test", "documentation"])
        new_label = 0  # Same token, but context says it's not a real secret
    else:
        # Non-secret token in secret-like context → still NOT a secret
        # This tests if model can resist "secret context" pressure
        var_name = random.choice([
            "api_key", "API_SECRET", "auth_token", "private_key",
            "password", "credentials", "secret_key",
        ])
        context_type = random.choice(["production", "configuration"])
        new_label = 0  # Still not a secret despite context

    language = random.choice(LANGUAGES)
    line_content, context_before, context_after, file_path = generate_context(
        token=token,
        var_name=var_name,
        language=language,
        context_type=context_type,
    )

    return {
        "token": token,
        "line_content": line_content,
        "context_before": context_before,
        "context_after": context_after,
        "file_path": file_path,
        "label": new_label,
        "_contrastive": True,  # Mark for logging
    }


def apply_label_noise(
    records: List[Dict[str, Any]],
    noise_rate: float = 0.08,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Apply controlled label noise to training data.

    Only flips labels for ambiguous cases. Never flips:
    - High-entropy secrets with valid context
    - Explicit credential formats (AKIA*, ghp_*, etc.)

    Args:
        records: Training records
        noise_rate: Fraction of labels to flip
        seed: Random seed

    Returns:
        Tuple of (modified records, list of flipped indices)
    """
    if seed is not None:
        random.seed(seed)

    flipped_indices = []

    # Patterns that should NEVER have labels flipped
    protected_prefixes = ["AKIA", "ASIA", "ghp_", "gho_", "ghs_", "ghr_", "sk_live_", "rk_live_"]

    for i, record in enumerate(records):
        if random.random() > noise_rate:
            continue

        token = record.get("token", "")

        # Never flip protected patterns
        if any(token.startswith(p) for p in protected_prefixes):
            continue

        # Never flip high-confidence secrets (label=1 with secret var names)
        var_name_lower = record.get("line_content", "").lower()
        if record["label"] == 1 and any(kw in var_name_lower for kw in ["secret", "password", "credential"]):
            continue

        # Flip the label
        record["label"] = 1 - record["label"]
        record["_noisy"] = True  # Mark for logging
        flipped_indices.append(i)

    return records, flipped_indices


def validate_dataset_quality(
    records: List[Dict[str, Any]],
    max_feature_variance_explained: float = 0.30,
    max_stage_a_passthrough: float = 0.85,
) -> Dict[str, Any]:
    """
    Validate dataset quality to prevent shortcut learning.

    Checks:
    1. No single feature explains >30% of variance (prevents shortcut)
    2. Simulated Stage A routing sends <85% to Stage B (proper filtering)
    3. Token length distributions overlap between classes
    4. Context keywords appear in both classes

    Args:
        records: Training records to validate
        max_feature_variance_explained: Max allowed variance for single feature
        max_stage_a_passthrough: Max fraction sent to Stage B

    Returns:
        Validation report with pass/fail status and details
    """
    from Harpocrates.ml.features import extract_features_from_record

    report = {
        "passed": True,
        "warnings": [],
        "errors": [],
        "metrics": {},
    }

    if not records:
        report["passed"] = False
        report["errors"].append("No records to validate")
        return report

    # Extract features for analysis
    features_list = []
    labels = []
    for r in records:
        try:
            fv = extract_features_from_record(r)
            features_list.append(fv.to_array())
            labels.append(r["label"])
        except Exception as e:
            report["warnings"].append(f"Feature extraction failed: {e}")

    if len(features_list) < 100:
        report["warnings"].append(f"Only {len(features_list)} samples extracted")
        return report

    # Convert to arrays for analysis
    import numpy as np
    X = np.array(features_list)
    y = np.array(labels)

    # 1. Check feature variance correlation with labels
    feature_names = [
        "token_length", "token_entropy", "char_class_count", "digit_ratio",
        "uppercase_ratio", "special_char_ratio", "is_base64_like", "has_padding",
        "regex_match_type", "token_structure_score", "has_version_pattern",
        "normalized_entropy", "cryptographic_score", "vendor_prefix_boost",
        "token_span_offset", "token_in_multiline_block", "embedded_token_flag",
        "token_quote_type", "is_uuid_v4", "is_known_hash_length",
        "jwt_structure_valid", "entropy_charset_mismatch", "has_hash_prefix",
    ]  # Token features only for Stage A analysis

    feature_correlations = {}
    for i, name in enumerate(feature_names[:min(len(feature_names), X.shape[1])]):
        # Calculate point-biserial correlation
        feature_col = X[:, i]
        if np.std(feature_col) > 0:
            corr = np.corrcoef(feature_col, y)[0, 1]
            if not np.isnan(corr):
                feature_correlations[name] = abs(corr)

    # Check for dominant features
    max_corr = max(feature_correlations.values()) if feature_correlations else 0
    if max_corr > max_feature_variance_explained:
        dominant_features = [
            f for f, c in feature_correlations.items()
            if c > max_feature_variance_explained
        ]
        report["warnings"].append(
            f"High-correlation features detected: {dominant_features} "
            f"(max correlation: {max_corr:.2%})"
        )

    report["metrics"]["feature_correlations"] = feature_correlations
    report["metrics"]["max_feature_correlation"] = max_corr

    # 2. Simulate Stage A routing (high-recall filter)
    # Stage A should filter out obvious non-secrets, sending ~60-85% to Stage B
    # Token features: vendor_prefix_boost, token_entropy, normalized_entropy
    vendor_boost = X[:, 13]  # vendor_prefix_boost
    token_entropy = X[:, 1]  # token_entropy
    normalized_entropy = X[:, 11]  # normalized_entropy

    # Simple heuristic for Stage A: high vendor boost OR high entropy
    stage_a_pass = (
        (vendor_boost > 0.5) |
        ((token_entropy > 3.5) & (normalized_entropy > 0.7))
    )
    stage_a_passthrough_rate = np.mean(stage_a_pass)

    if stage_a_passthrough_rate > max_stage_a_passthrough:
        report["warnings"].append(
            f"Stage A passthrough rate too high: {stage_a_passthrough_rate:.1%} "
            f"(target: <{max_stage_a_passthrough:.0%})"
        )

    report["metrics"]["stage_a_passthrough_rate"] = float(stage_a_passthrough_rate)

    # 3. Token length overlap
    pos_lengths = [len(r["token"]) for r in records if r["label"] == 1]
    neg_lengths = [len(r["token"]) for r in records if r["label"] == 0]

    if pos_lengths and neg_lengths:
        pos_mean = np.mean(pos_lengths)
        neg_mean = np.mean(neg_lengths)
        length_diff = abs(pos_mean - neg_mean) / max(pos_mean, neg_mean)

        if length_diff > 0.3:
            report["warnings"].append(
                f"Token length divergence: pos_mean={pos_mean:.1f}, "
                f"neg_mean={neg_mean:.1f} (diff: {length_diff:.1%})"
            )

        report["metrics"]["token_length_divergence"] = float(length_diff)

    # 4. Context keyword symmetry
    auth_keywords = {"api", "key", "token", "secret", "password", "auth"}
    pos_auth_count = 0
    neg_auth_count = 0

    for r in records:
        context = (r.get("line_content", "") + " " +
                   " ".join(r.get("context_before", [])) + " " +
                   " ".join(r.get("context_after", ""))).lower()
        has_auth = any(kw in context for kw in auth_keywords)

        if r["label"] == 1 and has_auth:
            pos_auth_count += 1
        elif r["label"] == 0 and has_auth:
            neg_auth_count += 1

    pos_total = sum(1 for r in records if r["label"] == 1)
    neg_total = len(records) - pos_total

    if pos_total > 0 and neg_total > 0:
        pos_auth_rate = pos_auth_count / pos_total
        neg_auth_rate = neg_auth_count / neg_total
        auth_asymmetry = abs(pos_auth_rate - neg_auth_rate)

        if auth_asymmetry > 0.2:
            report["warnings"].append(
                f"Auth keyword asymmetry: pos={pos_auth_rate:.1%}, "
                f"neg={neg_auth_rate:.1%}"
            )

        report["metrics"]["auth_keyword_symmetry"] = {
            "positive_rate": float(pos_auth_rate),
            "negative_rate": float(neg_auth_rate),
            "asymmetry": float(auth_asymmetry),
        }

    return report


def generate_transformed_training_data(
    count: int = 10000,
    balance: float = 0.5,
    seed: Optional[int] = None,
    mode: str = "train",
    noise_rate: float = 0.08,
    contrastive_ratio: float = 0.3,
    vendor_neutralization: bool = True,
    validate: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate training data with full transformations per skill spec.

    Implements:
    1. Contrastive pair generation
    2. Context-token decoupling
    3. Vendor prefix neutralization
    4. Controlled label noise (train only)
    5. Hard negative injection
    6. Dataset quality validation

    Args:
        count: Total number of examples
        balance: Ratio of positive examples
        seed: Random seed
        mode: "train" or "test"
        noise_rate: Label noise rate (train only)
        contrastive_ratio: Fraction of samples to generate contrastive pairs for
        vendor_neutralization: Ensure vendor strings in both classes
        validate: Run dataset quality validation

    Returns:
        Tuple of (records, validation_report)
    """
    if seed is not None:
        random.seed(seed)

    # Start with base generation
    records = generate_training_data(count=count, balance=balance, seed=seed)

    # Track transformations for validation report
    report = {
        "mode": mode,
        "seed": seed,
        "original_count": len(records),
        "transformations": [],
    }

    # 1. Contrastive pair generation
    if contrastive_ratio > 0:
        contrastive_count = int(len(records) * contrastive_ratio)
        samples_for_contrastive = random.sample(records, min(contrastive_count, len(records)))
        contrastive_pairs = [generate_contrastive_pair(r) for r in samples_for_contrastive]
        records.extend(contrastive_pairs)
        report["transformations"].append({
            "type": "contrastive_pairs",
            "count": len(contrastive_pairs),
        })

    # 2. Label noise (train only)
    if mode == "train" and noise_rate > 0:
        records, flipped = apply_label_noise(records, noise_rate, seed)
        report["transformations"].append({
            "type": "label_noise",
            "flipped_count": len(flipped),
            "noise_rate": noise_rate,
        })
    elif mode == "test":
        report["transformations"].append({
            "type": "label_noise",
            "skipped": True,
            "reason": "Test mode - no noise allowed",
        })

    # Shuffle final dataset
    random.shuffle(records)

    # Generate validation statistics
    positive_count = sum(1 for r in records if r["label"] == 1)
    negative_count = len(records) - positive_count

    # Token length statistics
    pos_lengths = [len(r["token"]) for r in records if r["label"] == 1]
    neg_lengths = [len(r["token"]) for r in records if r["label"] == 0]

    report["final_count"] = len(records)
    report["positive_count"] = positive_count
    report["negative_count"] = negative_count
    report["positive_ratio"] = positive_count / len(records) if records else 0
    report["token_length_overlap"] = {
        "positive_mean": sum(pos_lengths) / len(pos_lengths) if pos_lengths else 0,
        "negative_mean": sum(neg_lengths) / len(neg_lengths) if neg_lengths else 0,
        "positive_range": (min(pos_lengths), max(pos_lengths)) if pos_lengths else (0, 0),
        "negative_range": (min(neg_lengths), max(neg_lengths)) if neg_lengths else (0, 0),
    }

    # Check for vendor neutralization
    vendor_prefixes = ["AKIA", "ghp_", "sk_", "xox"]
    vendor_in_positive = set()
    vendor_in_negative = set()
    for r in records:
        token = r.get("token", "")
        for vp in vendor_prefixes:
            if token.startswith(vp):
                if r["label"] == 1:
                    vendor_in_positive.add(vp)
                else:
                    vendor_in_negative.add(vp)

    report["vendor_neutralization"] = {
        "vendors_in_positive": list(vendor_in_positive),
        "vendors_in_negative": list(vendor_in_negative),
        "neutralized": list(vendor_in_positive & vendor_in_negative),
    }

    # 3. Dataset quality validation
    if validate:
        validation_result = validate_dataset_quality(records)
        report["validation"] = validation_result
        if validation_result["warnings"]:
            report["validation_warnings"] = validation_result["warnings"]

    return records, report


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Save records to JSONL file."""
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for Harpocrates ML"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("training_data.jsonl"),
        help="Output JSONL file path (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=10000,
        help="Number of examples to generate (default: 10000)",
    )
    parser.add_argument(
        "--balance",
        "-b",
        type=float,
        default=0.5,
        help="Ratio of positive examples (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test files (80/10/10)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Generation mode: train (with noise/augmentation) or test (clean)",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.08,
        help="Label noise rate for training (default: 0.08, ignored for test)",
    )
    parser.add_argument(
        "--contrastive-ratio",
        type=float,
        default=0.3,
        help="Ratio of contrastive pairs to generate (default: 0.3)",
    )
    parser.add_argument(
        "--transform",
        "-t",
        action="store_true",
        help="Apply full data transformations (contrastive, noise, etc.)",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=None,
        help="Output path for validation report JSON",
    )

    args = parser.parse_args()

    print(f"Generating {args.count} training examples...")
    print(f"Balance: {args.balance:.0%} positive, {1 - args.balance:.0%} negative")
    print(f"Mode: {args.mode}")

    if args.transform:
        print(f"Transformations: contrastive={args.contrastive_ratio}, noise={args.noise_rate if args.mode == 'train' else 0}")
        records, report = generate_transformed_training_data(
            count=args.count,
            balance=args.balance,
            seed=args.seed,
            mode=args.mode,
            noise_rate=args.noise_rate if args.mode == "train" else 0,
            contrastive_ratio=args.contrastive_ratio if args.mode == "train" else 0,
            vendor_neutralization=True,
        )
        # Save validation report if requested
        if args.report:
            with open(args.report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Validation report saved to {args.report}")
    else:
        records = generate_training_data(
            count=args.count,
            balance=args.balance,
            seed=args.seed,
        )

    if args.split:
        # Split into train/val/test
        train_count = int(len(records) * 0.8)
        val_count = int(len(records) * 0.1)

        train_records = records[:train_count]
        val_records = records[train_count : train_count + val_count]
        test_records = records[train_count + val_count :]

        base_path = args.output.parent
        stem = args.output.stem

        train_path = base_path / f"{stem}_train.jsonl"
        val_path = base_path / f"{stem}_val.jsonl"
        test_path = base_path / f"{stem}_test.jsonl"

        save_jsonl(train_records, train_path)
        save_jsonl(val_records, val_path)
        save_jsonl(test_records, test_path)

        print(f"Saved {len(train_records)} training examples to {train_path}")
        print(f"Saved {len(val_records)} validation examples to {val_path}")
        print(f"Saved {len(test_records)} test examples to {test_path}")
    else:
        save_jsonl(records, args.output)
        print(f"Saved {len(records)} examples to {args.output}")

    # Print distribution summary
    positive_count = sum(1 for r in records if r["label"] == 1)
    negative_count = len(records) - positive_count
    print(f"\nDistribution: {positive_count} positive, {negative_count} negative")

    return 0


if __name__ == "__main__":
    sys.exit(main())
