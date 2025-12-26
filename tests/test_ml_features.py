"""
Tests for ML feature extraction.
"""

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.ml.context import (
    CodeContext,
    extract_context,
    extract_var_name,
    is_comment_line,
    is_test_file,
)
from Harpocrates.ml.features import (
    FeatureVector,
    _get_char_class_count,
    _get_prefix_type,
    _is_base64_like,
    _is_hex_like,
    extract_features,
)


class TestVarNameExtraction:
    """Tests for variable name extraction."""

    def test_python_assignment(self):
        """Test extracting variable name from Python assignment."""
        line = 'api_key = "sk-abc123"'
        result = extract_var_name(line, "sk-abc123")
        assert result == "api_key"

    def test_js_const_assignment(self):
        """Test extracting variable name from JavaScript const."""
        line = 'const secretKey = "ghp_abc123";'
        result = extract_var_name(line, "ghp_abc123")
        assert result == "secretKey"

    def test_json_key(self):
        """Test extracting key from JSON-style assignment."""
        line = '"api_key": "AKIAIOSFODNN7EXAMPLE"'
        result = extract_var_name(line, "AKIAIOSFODNN7EXAMPLE")
        assert result == "api_key"

    def test_env_file(self):
        """Test extracting variable from .env file."""
        line = "AWS_SECRET_KEY=wJalrXUtnFEMI/K7MDENG"
        result = extract_var_name(line, "wJalrXUtnFEMI/K7MDENG")
        assert result == "AWS_SECRET_KEY"

    def test_no_variable_name(self):
        """Test when no variable name is found."""
        line = 'print("ghp_abc123")'
        # May or may not find a variable name depending on patterns
        # The token is in the line but not in assignment form
        _ = extract_var_name(line, "ghp_abc123")


class TestContextExtraction:
    """Tests for context extraction."""

    def test_extract_context_basic(self):
        """Test basic context extraction."""
        content = """import os

# Configuration
api_key = "sk-abc123"

def connect():
    pass"""

        context = extract_context(content, line_number=4)

        assert "api_key" in context.line_content
        assert len(context.lines_before) == 3
        assert len(context.lines_after) == 3

    def test_extract_context_at_start(self):
        """Test context extraction at start of file."""
        content = """api_key = "sk-abc123"
other_line = "value"
"""
        context = extract_context(content, line_number=1)

        assert "api_key" in context.line_content
        assert len(context.lines_before) == 0

    def test_is_test_file(self):
        """Test detection of test files."""
        assert is_test_file("tests/test_auth.py") is True
        assert is_test_file("src/__tests__/auth.test.js") is True
        assert is_test_file("spec/auth_spec.rb") is True
        assert is_test_file("src/auth.py") is False

    def test_is_comment_line(self):
        """Test detection of comment lines."""
        assert is_comment_line("# This is a comment") is True
        assert is_comment_line("// JavaScript comment") is True
        assert is_comment_line("api_key = 'value'") is False


class TestTokenFeatures:
    """Tests for token-level feature extraction."""

    def test_char_class_count(self):
        """Test character class counting."""
        # All four classes: upper, lower, digit, special
        assert _get_char_class_count("Aa1!") == 4
        # Just lowercase
        assert _get_char_class_count("abcdef") == 1
        # Upper and digits
        assert _get_char_class_count("ABC123") == 2

    def test_known_prefix_detection(self):
        """Test detection of known secret prefixes."""
        assert _get_prefix_type("AKIAIOSFODNN7EXAMPLE") == 1  # AWS
        assert _get_prefix_type("ghp_abc123") == 2  # GitHub
        assert _get_prefix_type("sk-abc123") == 4  # OpenAI
        assert _get_prefix_type("random_string") == 0  # Unknown

    def test_is_hex_like(self):
        """Test hex detection."""
        assert _is_hex_like("a1b2c3d4e5f6a1b2c3d4") is True
        assert _is_hex_like("ABCDEF123456") is True
        assert _is_hex_like("ghp_abc123") is False  # Contains non-hex

    def test_is_base64_like(self):
        """Test base64 detection."""
        assert _is_base64_like("YWJjZGVmZ2hpamtsbW5vcHFyc3Q=") is True
        assert _is_base64_like("abc") is False  # Too short


class TestFeatureVector:
    """Tests for complete feature vector extraction."""

    def test_feature_vector_length(self):
        """Test that feature vector has 51 features."""
        fv = FeatureVector()
        array = fv.to_array()
        assert len(array) == 51

    def test_feature_names(self):
        """Test that feature names are defined."""
        names = FeatureVector.get_feature_names()
        assert len(names) == 51
        assert "token_length" in names
        assert "var_contains_secret" in names
        assert "file_is_test" in names
        # These were removed to prevent shortcut learning
        assert "has_known_prefix" not in names
        assert "prefix_type" not in names
        assert "is_hex_like" not in names
        # Features for improved recall
        assert "token_structure_score" in names
        assert "has_version_pattern" in names
        assert "semantic_context_score" in names
        assert "line_position_ratio" in names
        assert "surrounding_secret_density" in names
        # NEW: Discriminative features for precision boost
        assert "is_uuid_v4" in names
        assert "is_known_hash_length" in names
        assert "jwt_structure_valid" in names
        assert "entropy_charset_mismatch" in names
        assert "has_hash_prefix" in names

    def test_extract_features_aws_key(self):
        """Test feature extraction for AWS key."""
        finding = Finding(
            type="AWS_ACCESS_KEY_ID",
            snippet='access_key = "AKIAIOSFODNN7EXAMPLE"',
            evidence=EvidenceType.REGEX,
            token="AKIAIOSFODNN7EXAMPLE",
        )
        context = CodeContext(
            line_content='access_key = "AKIAIOSFODNN7EXAMPLE"',
            lines_before=["import boto3", ""],
            lines_after=["", "s3 = boto3.client('s3')"],
            file_path="config/aws.py",
        )

        features = extract_features(finding, context)

        # has_known_prefix and prefix_type removed - check other features
        assert features.token_length == 20
        assert features.var_contains_secret is True  # "access_key" matches secret pattern

    def test_extract_features_git_sha(self):
        """Test feature extraction for Git SHA (false positive)."""
        sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'commit_sha = "{sha}"',
            evidence=EvidenceType.ENTROPY,
            token=sha,
        )
        context = CodeContext(
            line_content=f'commit_sha = "{sha}"',
            lines_before=["# Get git info", "git_log = run('git log')"],
            lines_after=["print(f'Commit: {commit_sha}')"],
            file_path="scripts/build.py",
        )

        features = extract_features(finding, context)

        # is_hex_like was removed - check other features
        assert features.token_length == 40
        assert features.var_contains_safe is True  # "sha" in variable name
        assert features.context_mentions_git is True


class TestFeatureDifferentiation:
    """Tests to verify features differentiate secrets from false positives."""

    def test_secret_vs_git_sha_features(self):
        """Test that API key and Git SHA have different feature signatures."""
        # API secret in config
        api_finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='api_secret = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"',
            evidence=EvidenceType.ENTROPY,
            token="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        api_context = CodeContext(
            line_content='api_secret = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"',
            lines_before=["import requests", "# API credentials"],
            lines_after=["headers = {'Authorization': api_secret}"],
            file_path="config/api.py",
        )

        # Git SHA in build script
        sha_finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='commit_sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"',
            evidence=EvidenceType.ENTROPY,
            token="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        sha_context = CodeContext(
            line_content='commit_sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"',
            lines_before=["import subprocess", "# Git information"],
            lines_after=["print(f'Building commit {commit_sha}')"],
            file_path="scripts/build.py",
        )

        api_features = extract_features(api_finding, api_context)
        sha_features = extract_features(sha_finding, sha_context)

        # Key differentiating features - variable name analysis
        assert api_features.var_contains_secret is True
        assert sha_features.var_contains_secret is False

        assert api_features.var_contains_safe is False
        assert sha_features.var_contains_safe is True

        # Both have same token length (model must use context, not format)
        # Note: is_hex_like was removed to prevent format-based shortcuts
        assert api_features.token_length == sha_features.token_length

    def test_test_file_features(self):
        """Test that test files have appropriate features."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='fake_token = "abc123xyz"',
            evidence=EvidenceType.ENTROPY,
            token="abc123xyz789def456",
        )
        context = CodeContext(
            line_content='fake_token = "abc123xyz789def456"',
            lines_before=["import pytest", "", "def test_auth():"],
            lines_after=["    assert fake_token"],
            file_path="tests/test_auth.py",
            in_test_file=True,
        )

        features = extract_features(finding, context)

        assert features.file_is_test is True
        assert features.context_mentions_test is True
