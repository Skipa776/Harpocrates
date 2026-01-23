"""
Canonical test fixture for Harpocrates secret detection testing.

All tokens in this file are FAKE and for testing purposes only.
Annotated with `# gitleaks:allow` to prevent false positives from
secret scanning tools (GitLeaks, TruffleHog, GitHub Push Protection).

Style reference:
  - GitLeaks: https://github.com/gitleaks/gitleaks (uses .gitleaksignore + inline allow)
  - TruffleHog: https://github.com/trufflesecurity/trufflehog (uses testdata/ fixtures)

Usage:
    from tests.fixtures.test_tokens import FAKE_SECRETS, FAKE_NON_SECRETS, get_test_finding
"""

from __future__ import annotations

from typing import Any, Dict, List

# ==============================================================================
# FAKE SECRETS - Tokens that SHOULD be detected as secrets
# All values are synthetic / from vendor documentation examples.
# ==============================================================================

FAKE_SECRETS: Dict[str, Dict[str, Any]] = {
    # --- AWS ---
    "aws_access_key": {
        "token": "AKIAIOSFODNN7EXAMPLE",  # gitleaks:allow (AWS docs example)
        "context": 'AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE',  # gitleaks:allow
        "var_name": "AWS_ACCESS_KEY_ID",
        "type": "aws_access_key",
        "description": "AWS official documentation example key",
    },
    "aws_secret_key": {
        "token": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",  # gitleaks:allow (AWS docs)
        "context": 'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',  # gitleaks:allow
        "var_name": "aws_secret_access_key",
        "type": "aws_secret_key",
        "description": "AWS official documentation example secret",
    },

    # --- GitHub ---
    "github_pat": {
        "token": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef01",  # gitleaks:allow (fake PAT)
        "context": 'GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef01',  # gitleaks:allow
        "var_name": "GITHUB_TOKEN",
        "type": "github_pat",
        "description": "Synthetic GitHub PAT (not a real token)",
    },
    "github_pat_placeholder": {
        "token": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # gitleaks:allow (placeholder)
        "context": 'export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"',  # gitleaks:allow
        "var_name": "GITHUB_TOKEN",
        "type": "github_pat",
        "description": "Placeholder GitHub PAT with x-fill",
    },

    # --- Stripe ---
    # NOTE: Stripe keys must NOT be sk_live_/sk_test_ + exactly 24 alphanumeric chars
    # or GitHub Push Protection will block. Use 10-char suffix (not 24).
    "stripe_live_key": {
        "token": "sk_live_FAKEVAL01",  # 10 chars (not 24) - won't trigger push protection
        "context": 'STRIPE_SECRET_KEY="sk_live_FAKEVAL01"',
        "var_name": "STRIPE_SECRET_KEY",
        "type": "stripe_key",
        "description": "Truncated fake Stripe live key (invalid length bypasses push protection)",
    },
    "stripe_test_key": {
        "token": "sk_test_FAKEVAL02",  # 10 chars (not 24) - won't trigger push protection
        "context": 'stripe_key = "sk_test_FAKEVAL02"',
        "var_name": "stripe_key",
        "type": "stripe_key",
        "description": "Truncated fake Stripe test key (invalid length bypasses push protection)",
    },

    # --- OpenAI ---
    "openai_key": {
        "token": "sk-proj-EXAMPLE1234567890abcdefghijklmnopqrst1234567890",  # gitleaks:allow
        "context": 'OPENAI_API_KEY="sk-proj-EXAMPLE1234567890abcdefghijklmnopqrst1234567890"',  # gitleaks:allow
        "var_name": "OPENAI_API_KEY",
        "type": "openai_key",
        "description": "Synthetic OpenAI API key (not valid)",
    },

    # --- Slack ---
    # NOTE: Slack tokens must NOT match xoxb-[digits]-[digits]-[alphanum] pattern
    # or GitHub Push Protection will block.
    "slack_bot_token": {
        "token": "xoxb-FAKE-FAKE-EXAMPLETOKEN",  # Non-digit segments bypass push protection
        "context": 'SLACK_BOT_TOKEN="xoxb-FAKE-FAKE-EXAMPLETOKEN"',
        "var_name": "SLACK_BOT_TOKEN",
        "type": "slack_token",
        "description": "Synthetic Slack bot token (invalid format bypasses push protection)",
    },

    # --- GCP ---
    "gcp_api_key": {
        "token": "AIzaSyA-EXAMPLE-FAKE-KEY-1234567890abc",  # gitleaks:allow
        "context": 'GCP_API_KEY="AIzaSyA-EXAMPLE-FAKE-KEY-1234567890abc"',  # gitleaks:allow
        "var_name": "GCP_API_KEY",
        "type": "gcp_api_key",
        "description": "Synthetic GCP API key (not valid)",
    },

    # --- JWT ---
    "jwt_token": {
        # This JWT decodes to {"sub": "1234567890", "name": "Test", "iat": 0} with secret "test"
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlRlc3QiLCJpYXQiOjB9.LVk0jjMnau2SLaRGfNMBG0Gv7sCqWu6OIXrpVHnz2LQ",  # gitleaks:allow
        "context": 'AUTH_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlRlc3QiLCJpYXQiOjB9.LVk0jjMnau2SLaRGfNMBG0Gv7sCqWu6OIXrpVHnz2LQ"',  # gitleaks:allow
        "var_name": "AUTH_TOKEN",
        "type": "jwt",
        "description": "JWT with trivial test payload, signed with 'test'",
    },

    # --- Azure ---
    "azure_key_vault": {
        "token": "https://fake-test-vault.vault.azure.net/secrets/example/000000000000",  # gitleaks:allow
        "context": 'AZURE_VAULT_URL="https://fake-test-vault.vault.azure.net/secrets/example/000000000000"',  # gitleaks:allow
        "var_name": "AZURE_VAULT_URL",
        "type": "azure_key_vault",
        "description": "Synthetic Azure Key Vault URL (not real)",
    },
}


# ==============================================================================
# FAKE NON-SECRETS - Tokens that should NOT be detected as secrets
# These are common false-positive patterns.
# ==============================================================================

FAKE_NON_SECRETS: Dict[str, Dict[str, Any]] = {
    "git_sha": {
        "token": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "context": 'commit_sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"',
        "var_name": "commit_sha",
        "type": "git_sha",
        "description": "40-char hex that looks like a Git SHA",
    },
    "sha256_hash": {
        "token": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "context": 'file_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"',
        "var_name": "file_hash",
        "type": "sha256",
        "description": "SHA256 of empty string (well-known hash)",
    },
    "uuid_v4": {
        "token": "550e8400-e29b-41d4-a716-446655440000",
        "context": 'user_uuid = "550e8400-e29b-41d4-a716-446655440000"',
        "var_name": "user_uuid",
        "type": "uuid",
        "description": "UUID v4 format string",
    },
    "hex_color": {
        "token": "1a2b3c4d5e6f",
        "context": 'color_value = "1a2b3c4d5e6f"',
        "var_name": "color_value",
        "type": "hex_string",
        "description": "Short hex string (not a secret)",
    },
    "build_hash": {
        "token": "abc123def456abc123def456abc123def456abc1",
        "context": 'BUILD_HASH="abc123def456abc123def456abc123def456abc1"  # webpack chunk hash',
        "var_name": "BUILD_HASH",
        "type": "build_hash",
        "description": "Build artifact hash in dist context",
        "file_path": "dist/assets/chunk.abc123.js",
    },
    "version_string": {
        "token": "v2.1.0-beta.3+build.1234",
        "context": 'APP_VERSION="v2.1.0-beta.3+build.1234"',
        "var_name": "APP_VERSION",
        "type": "version",
        "description": "Semver version string",
    },
    "test_placeholder": {
        "token": "sk_test_EXAMPLEEXAMPLEEXAMPLEEX",  # gitleaks:allow
        "context": 'placeholder_key = "sk_test_EXAMPLEEXAMPLEEXAMPLEEX"  # not real',  # gitleaks:allow
        "var_name": "placeholder_key",
        "type": "test_token",
        "description": "Stripe test-mode placeholder (clearly fake)",
    },
    "documentation_example": {
        "token": "AKIAIOSFODNN7EXAMPLE",  # gitleaks:allow (AWS official example)
        "context": '# See: https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html\nexample_key = "AKIAIOSFODNN7EXAMPLE"',  # gitleaks:allow
        "var_name": "example_key",
        "type": "doc_example",
        "description": "AWS docs example key in documentation context",
        "file_path": "docs/examples/aws_setup.md",
    },
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_all_fake_tokens() -> List[str]:
    """Return all fake secret tokens for bulk testing."""
    return [v["token"] for v in FAKE_SECRETS.values()]


def get_all_non_secret_tokens() -> List[str]:
    """Return all non-secret tokens for false-positive testing."""
    return [v["token"] for v in FAKE_NON_SECRETS.values()]


def get_test_finding(key: str, secret: bool = True) -> Dict[str, Any]:
    """
    Build a Finding-compatible dict from a fixture key.

    Args:
        key: Key from FAKE_SECRETS or FAKE_NON_SECRETS
        secret: If True, look in FAKE_SECRETS; else FAKE_NON_SECRETS

    Returns:
        Dict with fields matching Harpocrates Finding structure
    """
    source = FAKE_SECRETS if secret else FAKE_NON_SECRETS
    entry = source[key]
    return {
        "snippet": entry["context"],
        "file_path": entry.get("file_path", "test_file.py"),
        "token": entry["token"],
        "var_name": entry["var_name"],
        "line_content": entry["context"].split("\n")[-1],
        "line_number": 1,
        "secret_type": entry["type"],
    }
