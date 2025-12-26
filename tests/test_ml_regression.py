"""
ML regression test suite for Harpocrates.

Contains known samples that MUST be detected as secrets and known samples
that MUST NOT be detected as secrets. These tests ensure the ML model
doesn't regress on critical cases.

Note: These tests validate feature extraction behavior, not model predictions,
since the model may not be trained. The features should clearly differentiate
between secret and non-secret contexts.
"""
from __future__ import annotations

import pytest

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FeatureVector, extract_features


class TestKnownSecrets:
    """
    Tests for samples that MUST be detected as secrets.

    These are canonical examples of real secrets in realistic contexts.
    The feature extraction should produce features indicative of secrets.
    """

    def test_aws_key_in_config(self):
        """AWS access key with api_key assignment should look like a secret."""
        finding = Finding(
            type="AWS_ACCESS_KEY_ID",
            snippet='api_key = "AKIAEXAMPLEKEY12345"',
            evidence=EvidenceType.REGEX,
            token="AKIAEXAMPLEKEY12345",
        )
        context = CodeContext(
            line_content='api_key = "AKIAEXAMPLEKEY12345"',
            lines_before=["import boto3", "# AWS Configuration"],
            lines_after=["secret_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"],
            file_path="config/aws_config.py",
        )

        features = extract_features(finding, context)

        # Should have secret-indicating features
        assert features.var_contains_secret is True, "Should detect 'api_key' as secret variable"
        assert features.file_is_config is True, "Should detect config file"
        assert features.regex_match_type > 0, "Should have regex match"

    def test_jwt_in_config_file(self):
        """JWT token in config file should look like a secret."""
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        finding = Finding(
            type="JWT",
            snippet=f'AUTH_TOKEN = "{jwt_token}"',
            evidence=EvidenceType.REGEX,
            token=jwt_token,
        )
        context = CodeContext(
            line_content=f'AUTH_TOKEN = "{jwt_token}"',
            lines_before=["# Authentication configuration", ""],
            lines_after=["", "API_URL = 'https://api.example.com'"],
            file_path="config/settings.yaml",
        )

        features = extract_features(finding, context)

        assert features.is_base64_like is True, "JWT should be detected as base64-like"
        assert features.file_is_config is True, "Should detect config file"
        assert features.token_entropy > 3.5, "JWT should have high entropy"

    def test_github_token_in_env_file(self):
        """GitHub token in .env file should look like a secret."""
        finding = Finding(
            type="GITHUB_TOKEN",
            snippet="GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            evidence=EvidenceType.REGEX,
            token="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        )
        context = CodeContext(
            line_content="GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            lines_before=["# Environment variables", "DATABASE_URL=postgres://..."],
            lines_after=["API_KEY=sk_test_xxxxx"],
            file_path=".env",
        )

        features = extract_features(finding, context)

        # .env files are detected as config files (Path(".env").suffix returns "")
        assert features.file_is_config is True, ".env should be config file"
        assert features.regex_match_type > 0, "Should have regex match"
        assert features.var_contains_secret is True, "GITHUB_TOKEN contains 'TOKEN'"

    def test_stripe_live_key(self):
        """Stripe live key should look like a secret."""
        finding = Finding(
            type="STRIPE_SECRET_KEY",
            snippet='stripe.api_key = "sk_test_FAKE_PLACEHOLDER"',
            evidence=EvidenceType.REGEX,
            token="sk_test_FAKE_PLACEHOLDER",
        )
        context = CodeContext(
            line_content='stripe.api_key = "sk_test_FAKE_PLACEHOLDER"',
            lines_before=["import stripe", ""],
            lines_after=["", "def charge_customer():"],
            file_path="payments/stripe_handler.py",
        )

        features = extract_features(finding, context)

        assert features.var_contains_secret is True, "api_key should be detected"
        assert features.regex_match_type > 0, "Should have regex match"

    def test_openai_api_key(self):
        """OpenAI API key should look like a secret."""
        finding = Finding(
            type="OPENAI_API_KEY",
            snippet='OPENAI_API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"',
            evidence=EvidenceType.REGEX,
            token="sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        )
        context = CodeContext(
            line_content='OPENAI_API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"',
            lines_before=["# OpenAI Configuration"],
            lines_after=["MODEL = 'gpt-4'"],
            file_path="config.py",
        )

        features = extract_features(finding, context)

        assert features.var_contains_secret is True
        assert features.file_is_config is True

    def test_private_key_pem(self):
        """Private key in .pem file should be high risk."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet="MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC7",
            evidence=EvidenceType.ENTROPY,
            token="MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC7",
        )
        context = CodeContext(
            line_content="MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC7",
            lines_before=["-----BEGIN PRIVATE KEY-----"],
            lines_after=["...more base64..."],
            file_path="certs/private.pem",
        )

        features = extract_features(finding, context)

        assert features.file_extension_risk == 2, ".pem should be high risk"
        assert features.is_base64_like is True


class TestKnownNonSecrets:
    """
    Tests for samples that MUST NOT be detected as secrets.

    These are canonical examples of false positives that the model
    should learn to ignore. Features should indicate non-secret context.
    """

    def test_aws_example_key(self):
        """AKIAIOSFODNN7EXAMPLE is AWS's official documentation key."""
        finding = Finding(
            type="AWS_ACCESS_KEY_ID",
            snippet='# Example: aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"',
            evidence=EvidenceType.REGEX,
            token="AKIAIOSFODNN7EXAMPLE",
        )
        context = CodeContext(
            line_content='# Example: aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"',
            lines_before=["# AWS Configuration Example", "# Replace with your actual credentials"],
            lines_after=["# aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"],
            file_path="docs/aws_setup.md",
            in_comment=True,  # Line starts with #
        )

        features = extract_features(finding, context)

        # Should have non-secret indicators
        assert features.line_is_comment is True, "Should detect comment"
        assert features.context_mentions_test is True, "Should detect 'example' in context"
        # token_structure_score should indicate structured token (contains 'EXAMPLE')
        assert features.token_structure_score > 0, "EXAMPLE pattern should be detected"
        # Semantic context should be negative (safe) due to 'example' mentions
        assert features.semantic_context_score < 0, "Example context should be safe"

    def test_git_sha_in_commit_context(self):
        """Git SHA in commit context should not be a secret."""
        sha = "a1b2c3d4e5f6789012345678901234567890abcd"
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'commit_sha = "{sha}"',
            evidence=EvidenceType.ENTROPY,
            token=sha,
        )
        context = CodeContext(
            line_content=f'commit_sha = "{sha}"',
            lines_before=["# Get the current git commit", "git_log = subprocess.run(['git', 'log'])"],
            lines_after=["print(f'Building from commit: {commit_sha}')"],
            file_path="scripts/build.py",
        )

        features = extract_features(finding, context)

        assert features.var_contains_safe is True, "'sha' should be detected as safe"
        assert features.context_mentions_git is True, "Should detect git context"
        assert features.token_length == 40, "Git SHA is 40 chars"

    def test_uuid_as_identifier(self):
        """UUID used as identifier should not be a secret."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'user_id = "{uuid}"',
            evidence=EvidenceType.ENTROPY,
            token=uuid,
        )
        context = CodeContext(
            line_content=f'user_id = "{uuid}"',
            lines_before=["def get_user():", "    # Fetch user by ID"],
            lines_after=["    return db.users.find(user_id)"],
            file_path="services/user_service.py",
        )

        features = extract_features(finding, context)

        # UUID pattern should be detected
        assert features.var_contains_safe is True, "'id' in variable name should be safe"

    def test_checksum_hash(self):
        """MD5/SHA checksum should not be a secret."""
        checksum = "d41d8cd98f00b204e9800998ecf8427e"
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'file_hash = "{checksum}"',
            evidence=EvidenceType.ENTROPY,
            token=checksum,
        )
        context = CodeContext(
            line_content=f'file_hash = "{checksum}"',
            lines_before=["import hashlib", "# Verify file integrity"],
            lines_after=["if calculated_hash != file_hash:", "    raise IntegrityError()"],
            file_path="utils/file_utils.py",
        )

        features = extract_features(finding, context)

        assert features.var_contains_safe is True, "'hash' should be detected as safe"
        assert features.context_mentions_hash is True, "Should detect hash context"

    def test_mock_token_in_test_file(self):
        """Mock/fake token in test file should not be a secret."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='fake_api_key = "test_key_1234567890abcdef"',
            evidence=EvidenceType.ENTROPY,
            token="test_key_1234567890abcdef",
        )
        context = CodeContext(
            line_content='fake_api_key = "test_key_1234567890abcdef"',
            lines_before=["import pytest", "", "def test_api_authentication():"],
            lines_after=["    mock_client = MockClient(api_key=fake_api_key)"],
            file_path="tests/test_api.py",
            in_test_file=True,
        )

        features = extract_features(finding, context)

        assert features.file_is_test is True, "Should detect test file"
        assert features.context_mentions_test is True, "Should detect test context"

    def test_placeholder_token(self):
        """Placeholder token with obvious pattern should not be a secret."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='api_key = "YOUR_API_KEY_HERE"',
            evidence=EvidenceType.ENTROPY,
            token="YOUR_API_KEY_HERE",
        )
        context = CodeContext(
            line_content='api_key = "YOUR_API_KEY_HERE"',
            lines_before=["# Configuration template", "# Replace with your actual key"],
            lines_after=[""],
            file_path="config.example.py",
        )

        features = extract_features(finding, context)

        # Should detect structured/placeholder pattern
        assert features.token_structure_score > 0, "Placeholder should have structure"

    def test_stripe_test_key(self):
        """Stripe test key (sk_test_) should be lower risk than live."""
        finding = Finding(
            type="STRIPE_SECRET_KEY",
            snippet='stripe.api_key = "sk_test_PLACEHOLDER_NOT_REAL"',
            evidence=EvidenceType.REGEX,
            token="sk_test_PLACEHOLDER_NOT_REAL",
        )
        context = CodeContext(
            line_content='stripe.api_key = "sk_test_PLACEHOLDER_NOT_REAL"',
            lines_before=["# Test configuration", "import stripe"],
            lines_after=[""],
            file_path="tests/test_payments.py",
            in_test_file=True,
        )

        features = extract_features(finding, context)

        assert features.file_is_test is True
        assert features.context_mentions_test is True

    def test_version_string(self):
        """Version string should not be a secret."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='VERSION = "v2.1.3-beta.4"',
            evidence=EvidenceType.ENTROPY,
            token="v2.1.3-beta.4",
        )
        context = CodeContext(
            line_content='VERSION = "v2.1.3-beta.4"',
            lines_before=["# Package metadata"],
            lines_after=["AUTHOR = 'Jane Doe'"],
            file_path="__version__.py",
        )

        features = extract_features(finding, context)

        assert features.has_version_pattern is True, "Should detect version pattern"


class TestContextDifferentiation:
    """
    Tests verifying that identical tokens are differentiated by context.

    The same token should produce different features based on surrounding code.
    """

    def test_same_token_different_variable_names(self):
        """Same token should be treated differently based on variable name."""
        token = "a1b2c3d4e5f6789012345678901234567890abcd"

        # Secret context
        secret_finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'api_secret = "{token}"',
            evidence=EvidenceType.ENTROPY,
            token=token,
        )
        secret_context = CodeContext(
            line_content=f'api_secret = "{token}"',
            lines_before=["# Production credentials"],
            lines_after=[""],
            file_path="config/production.py",
        )

        # Non-secret context
        safe_finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'commit_hash = "{token}"',
            evidence=EvidenceType.ENTROPY,
            token=token,
        )
        safe_context = CodeContext(
            line_content=f'commit_hash = "{token}"',
            lines_before=["# Git information"],
            lines_after=[""],
            file_path="build/version.py",
        )

        secret_features = extract_features(secret_finding, secret_context)
        safe_features = extract_features(safe_finding, safe_context)

        # Variable name features should differ
        assert secret_features.var_contains_secret is True
        assert safe_features.var_contains_secret is False
        assert safe_features.var_contains_safe is True

    def test_same_token_different_file_types(self):
        """Same token should be treated differently based on file type."""
        token = "AKIAIOSFODNN7EXAMPLE"

        # Config file (higher risk)
        config_finding = Finding(
            type="AWS_ACCESS_KEY_ID",
            snippet=f'key = "{token}"',
            evidence=EvidenceType.REGEX,
            token=token,
        )
        config_context = CodeContext(
            line_content=f'key = "{token}"',
            file_path="config/aws.yaml",
        )

        # Test file (lower risk)
        test_finding = Finding(
            type="AWS_ACCESS_KEY_ID",
            snippet=f'key = "{token}"',
            evidence=EvidenceType.REGEX,
            token=token,
        )
        test_context = CodeContext(
            line_content=f'key = "{token}"',
            file_path="tests/test_aws.py",
            in_test_file=True,
        )

        config_features = extract_features(config_finding, config_context)
        test_features = extract_features(test_finding, test_context)

        assert config_features.file_is_config is True
        assert test_features.file_is_test is True

    def test_semantic_context_score_differentiation(self):
        """Semantic context score should differentiate safe vs risky contexts."""
        token = "sometoken123456789"

        # Risky context
        risky_finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'production_secret = "{token}"',
            evidence=EvidenceType.ENTROPY,
            token=token,
        )
        risky_context = CodeContext(
            line_content=f'production_secret = "{token}"',
            lines_before=["# Production credentials - DO NOT SHARE"],
            lines_after=["# Use this for API authentication"],
            file_path="config/prod.env",
        )

        # Safe context
        safe_finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'test_placeholder = "{token}"',
            evidence=EvidenceType.ENTROPY,
            token=token,
        )
        safe_context = CodeContext(
            line_content=f'test_placeholder = "{token}"',
            lines_before=["# Example/mock data for testing"],
            lines_after=["# This is a dummy value"],
            file_path="tests/fixtures.py",
            in_test_file=True,
        )

        risky_features = extract_features(risky_finding, risky_context)
        safe_features = extract_features(safe_finding, safe_context)

        # Semantic context score should differ
        assert risky_features.semantic_context_score > safe_features.semantic_context_score


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_token(self):
        """Empty token should not crash."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='key = ""',
            evidence=EvidenceType.ENTROPY,
            token="",
        )
        context = CodeContext(line_content='key = ""')

        features = extract_features(finding, context)
        assert features.token_length == 0
        assert len(features.to_array()) == 51

    def test_very_long_token(self):
        """Very long token should not crash."""
        token = "x" * 10000
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'data = "{token[:100]}..."',
            evidence=EvidenceType.ENTROPY,
            token=token,
        )
        context = CodeContext(line_content=f'data = "{token[:100]}..."')

        features = extract_features(finding, context)
        assert features.token_length == 10000
        assert len(features.to_array()) == 51

    def test_special_characters_in_token(self):
        """Token with special characters should not crash."""
        token = "abc!@#$%^&*()_+-=[]{}|;':\",./<>?"
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'weird = "{token}"',
            evidence=EvidenceType.ENTROPY,
            token=token,
        )
        context = CodeContext(line_content=f'weird = "{token}"')

        features = extract_features(finding, context)
        assert len(features.to_array()) == 51
        assert features.special_char_ratio > 0

    def test_binary_like_content(self):
        """Binary-like content should not crash."""
        token = "\x00\x01\x02\x03"
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f"binary = {repr(token)}",
            evidence=EvidenceType.ENTROPY,
            token=token,
        )
        context = CodeContext(line_content=f"binary = {repr(token)}")

        features = extract_features(finding, context)
        assert len(features.to_array()) == 51
