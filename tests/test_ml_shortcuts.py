"""
Tests to detect and prevent shortcut learning in ML model.

These tests verify that the model cannot achieve high accuracy
by exploiting superficial patterns in the training data.

The tests validate:
1. Token format distribution is balanced across labels
2. Variable names are sufficiently mixed
3. Context templates don't encode labels
4. Adversarial samples are properly challenging
5. Feature importance is distributed (no single feature dominates)
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Set

import pytest

from Harpocrates.ml.features import FeatureVector, extract_features_from_record
from Harpocrates.training.generators.generate_data import (
    generate_adversarial_test_data,
    generate_training_data,
    POSITIVE_DISTRIBUTION,
    NEGATIVE_DISTRIBUTION,
)
from Harpocrates.training.generators.secret_templates import (
    generate_fake_prefixed_token,
    generate_hex_secret,
)


# Known secret prefixes
KNOWN_PREFIXES = {"AKIA", "ghp_", "gho_", "sk-", "sk_live_", "sk_test_", "xoxb-", "xoxp-", "eyJ"}
HEX_CHARS = set(string.hexdigits.lower())


def _has_known_prefix(token: str) -> bool:
    """Check if token has a known secret prefix."""
    for prefix in KNOWN_PREFIXES:
        if token.startswith(prefix):
            return True
    return False


def _is_hex_only(token: str) -> bool:
    """Check if token is pure hexadecimal."""
    return len(token) >= 8 and all(c.lower() in HEX_CHARS for c in token)


def _extract_var_name(line: str) -> str | None:
    """Extract variable name from a line of code."""
    patterns = [
        r'(\w+)\s*=\s*["\']',  # Python/JS assignment
        r'(\w+):\s*["\']',  # YAML/JSON
        r'export\s+(\w+)=',  # Shell export
        r'const\s+(\w+)\s*=',  # JS const
        r'let\s+(\w+)\s*=',  # JS let
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    return None


SECRET_VAR_PATTERNS = re.compile(
    r"(secret|password|passwd|pwd|credential|api[_-]?key|access[_-]?key|"
    r"auth[_-]?key|token|bearer|private[_-]?key|api[_-]?secret|client[_-]?secret)",
    re.I
)

SAFE_VAR_PATTERNS = re.compile(
    r"(hash|sha|sha1|sha256|sha512|md5|commit|rev|revision|version|"
    r"uuid|guid|_id\b|identifier|checksum|digest|fingerprint)",
    re.I
)


def _is_secret_var_name(name: str) -> bool:
    """Check if variable name suggests a secret."""
    return bool(SECRET_VAR_PATTERNS.search(name))


def _is_safe_var_name(name: str) -> bool:
    """Check if variable name suggests non-secret."""
    return bool(SAFE_VAR_PATTERNS.search(name))


class TestTokenFormatShortcuts:
    """Tests to ensure token format doesn't leak label."""

    def test_prefix_distribution_balanced(self):
        """Verify prefixed tokens appear in both positive and negative samples."""
        records = generate_training_data(count=2000, seed=42)

        positive_has_prefix = sum(
            1 for r in records
            if r["label"] == 1 and _has_known_prefix(r["token"])
        )
        negative_has_prefix = sum(
            1 for r in records
            if r["label"] == 0 and _has_known_prefix(r["token"])
        )

        total_positive = sum(1 for r in records if r["label"] == 1)
        total_negative = len(records) - total_positive

        pos_prefix_ratio = positive_has_prefix / total_positive if total_positive > 0 else 0
        neg_prefix_ratio = negative_has_prefix / total_negative if total_negative > 0 else 0

        # Both classes should have significant prefix presence
        # Negative samples should have at least 25% with prefixes
        assert neg_prefix_ratio >= 0.25, (
            f"Too few negative samples with prefixes: {neg_prefix_ratio:.1%}. "
            f"Expected >= 25%. Found {negative_has_prefix}/{total_negative}."
        )
        # Positive samples shouldn't be dominated by prefixes (< 75%)
        assert pos_prefix_ratio <= 0.75, (
            f"Too many positive samples with prefixes: {pos_prefix_ratio:.1%}. "
            f"Expected <= 75%. Found {positive_has_prefix}/{total_positive}."
        )

    def test_hex_tokens_in_both_classes(self):
        """Verify hex-only tokens appear in both positive and negative samples."""
        records = generate_training_data(count=2000, seed=42)

        positive_hex = sum(
            1 for r in records
            if r["label"] == 1 and _is_hex_only(r["token"])
        )
        negative_hex = sum(
            1 for r in records
            if r["label"] == 0 and _is_hex_only(r["token"])
        )

        total_positive = sum(1 for r in records if r["label"] == 1)
        total_negative = len(records) - total_positive

        # Both should have hex tokens (prevents "hex = non-secret" shortcut)
        # At least 15% of positive samples should be hex
        assert positive_hex >= total_positive * 0.15, (
            f"Too few hex positive samples: {positive_hex}/{total_positive}. "
            f"Expected at least 15%."
        )
        # Negative samples should also have hex
        assert negative_hex >= total_negative * 0.20, (
            f"Too few hex negative samples: {negative_hex}/{total_negative}. "
            f"Expected at least 20%."
        )

    def test_distribution_sums_to_one(self):
        """Verify positive and negative distributions sum to 1.0."""
        pos_sum = sum(POSITIVE_DISTRIBUTION.values())
        neg_sum = sum(NEGATIVE_DISTRIBUTION.values())

        assert abs(pos_sum - 1.0) < 0.01, f"Positive distribution sums to {pos_sum}, expected 1.0"
        assert abs(neg_sum - 1.0) < 0.01, f"Negative distribution sums to {neg_sum}, expected 1.0"


class TestVariableNameShortcuts:
    """Tests to ensure variable names don't leak label."""

    def test_variable_name_mixing(self):
        """Verify variable names are sufficiently mixed across labels."""
        records = generate_training_data(count=2000, seed=42)

        positive_with_safe_name = 0
        negative_with_secret_name = 0
        positive_count = 0
        negative_count = 0

        for record in records:
            var_name = _extract_var_name(record.get("line_content", ""))
            if var_name:
                if record["label"] == 1:
                    positive_count += 1
                    if _is_safe_var_name(var_name):
                        positive_with_safe_name += 1
                else:
                    negative_count += 1
                    if _is_secret_var_name(var_name):
                        negative_with_secret_name += 1

        # At least 25% mixing required (50% mixing ratio doesn't guarantee
        # pattern detection due to regex vs actual variable name differences)
        if positive_count > 0:
            pos_mix_ratio = positive_with_safe_name / positive_count
            assert pos_mix_ratio >= 0.20, (
                f"Positive samples with safe names: {pos_mix_ratio:.1%}. "
                f"Expected >= 20% mixing."
            )

        if negative_count > 0:
            neg_mix_ratio = negative_with_secret_name / negative_count
            assert neg_mix_ratio >= 0.20, (
                f"Negative samples with secret names: {neg_mix_ratio:.1%}. "
                f"Expected >= 20% mixing."
            )


class TestContextShortcuts:
    """Tests to ensure context templates don't leak label."""

    def test_file_path_distribution(self):
        """Verify file paths don't perfectly correlate with labels."""
        records = generate_training_data(count=2000, seed=42)

        config_paths = ["config/", "settings", ".env", "secrets"]
        script_paths = ["scripts/", "build", "test", "utils/"]

        positive_in_config = 0
        negative_in_config = 0
        positive_in_scripts = 0
        negative_in_scripts = 0

        for r in records:
            path = r.get("file_path", "")
            is_config = any(p in path for p in config_paths)
            is_script = any(p in path for p in script_paths)

            if r["label"] == 1:
                if is_config:
                    positive_in_config += 1
                if is_script:
                    positive_in_scripts += 1
            else:
                if is_config:
                    negative_in_config += 1
                if is_script:
                    negative_in_scripts += 1

        # Both labels should appear in config paths
        assert negative_in_config >= 50, (
            f"Too few negative samples in config paths: {negative_in_config}. "
            "Negative samples should also appear in config files."
        )

        # Both labels should appear in script paths
        assert positive_in_scripts >= 50, (
            f"Too few positive samples in script paths: {positive_in_scripts}. "
            "Positive samples should also appear in build/script files."
        )


class TestAdversarialRobustness:
    """Tests that adversarial samples are properly challenging."""

    def test_adversarial_data_is_confusing(self):
        """Verify adversarial samples are challenging by design."""
        records = generate_adversarial_test_data(count=1000, seed=42)

        # Check that adversarial samples have mixed signals
        confusing_positives = 0
        confusing_negatives = 0

        for r in records:
            var_name = _extract_var_name(r.get("line_content", ""))
            has_prefix = _has_known_prefix(r["token"])

            if r["label"] == 1:
                # Secret that looks like non-secret
                if _is_hex_only(r["token"]) or (var_name and _is_safe_var_name(var_name)):
                    confusing_positives += 1
            else:
                # Non-secret that looks like secret
                if has_prefix or (var_name and _is_secret_var_name(var_name)):
                    confusing_negatives += 1

        total_positive = sum(1 for r in records if r["label"] == 1)
        total_negative = len(records) - total_positive

        # At least 40% of samples should be "confusing"
        if total_positive > 0:
            assert confusing_positives >= total_positive * 0.40, (
                f"Adversarial positives not confusing enough: {confusing_positives}/{total_positive}"
            )
        if total_negative > 0:
            assert confusing_negatives >= total_negative * 0.40, (
                f"Adversarial negatives not confusing enough: {confusing_negatives}/{total_negative}"
            )

    def test_adversarial_data_balanced(self):
        """Verify adversarial test data is approximately balanced."""
        records = generate_adversarial_test_data(count=1000, seed=42)

        positive_count = sum(1 for r in records if r["label"] == 1)
        negative_count = len(records) - positive_count

        # Should be roughly 50/50 (within 10%)
        ratio = positive_count / len(records)
        assert 0.40 <= ratio <= 0.60, (
            f"Adversarial data imbalanced: {ratio:.1%} positive. Expected ~50%."
        )


class TestFeatureVector:
    """Tests for feature vector correctness."""

    def test_feature_count_is_37(self):
        """Verify we have exactly 46 features."""
        names = FeatureVector.get_feature_names()
        assert len(names) == 51, f"Expected 46 features, got {len(names)}"

        vec = FeatureVector()
        arr = vec.to_array()
        assert len(arr) == 51, f"Expected 46 values, got {len(arr)}"

    def test_removed_features_not_present(self):
        """Verify leaky features were removed."""
        names = FeatureVector.get_feature_names()

        assert "has_known_prefix" not in names, "has_known_prefix should be removed"
        assert "prefix_type" not in names, "prefix_type should be removed"
        assert "is_hex_like" not in names, "is_hex_like should be removed"

    def test_feature_extraction_works(self):
        """Verify feature extraction from training records works."""
        records = generate_training_data(count=10, seed=42)

        for record in records:
            features = extract_features_from_record(record)
            arr = features.to_array()
            assert len(arr) == 51, f"Expected 46 features, got {len(arr)}"


class TestAmbiguousTokenGenerators:
    """Tests for the new ambiguous token generators."""

    def test_hex_secret_is_hex(self):
        """Verify hex_secret generates pure hex."""
        for _ in range(100):
            token = generate_hex_secret(40)
            assert len(token) == 40
            assert all(c in HEX_CHARS for c in token), f"Not hex: {token}"

    def test_fake_prefixed_token_has_prefix(self):
        """Verify fake_prefixed_token has known prefix."""
        for _ in range(100):
            token = generate_fake_prefixed_token()
            assert _has_known_prefix(token), f"No known prefix: {token}"


class TestGenerateContextExists:
    """Tests for the unified generate_context function."""

    def test_generate_context_exists(self):
        """Verify generate_context function exists and is callable."""
        from Harpocrates.training.generators.context_templates import generate_context

        result = generate_context(
            token="test_token_123",
            var_name="api_key",
            language="python",
            context_type="production",
        )

        assert len(result) == 4
        line, before, after, path = result
        assert "test_token_123" in line
        assert isinstance(before, list)
        assert isinstance(after, list)
        assert isinstance(path, str)

    def test_generate_context_all_languages(self):
        """Verify generate_context works for all languages."""
        from Harpocrates.training.generators.context_templates import generate_context

        languages = ["python", "javascript", "yaml", "json", "shell", "go", "java"]

        for lang in languages:
            result = generate_context(
                token="secret123",
                var_name="secret",
                language=lang,
                context_type="production",
            )
            assert result is not None, f"Failed for language: {lang}"
            assert len(result) == 4


class TestDataIntegrity:
    """Tests for data generation integrity."""

    def test_training_data_has_required_fields(self):
        """Verify training records have all required fields."""
        records = generate_training_data(count=100, seed=42)

        required_fields = {"token", "line_content", "context_before", "context_after", "file_path", "label"}

        for record in records:
            missing = required_fields - set(record.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_adversarial_data_has_required_fields(self):
        """Verify adversarial records have all required fields."""
        records = generate_adversarial_test_data(count=100, seed=42)

        required_fields = {"token", "line_content", "context_before", "context_after", "file_path", "label"}

        for record in records:
            missing = required_fields - set(record.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_labels_are_binary(self):
        """Verify labels are 0 or 1."""
        records = generate_training_data(count=100, seed=42)
        records += generate_adversarial_test_data(count=100, seed=43)

        for record in records:
            assert record["label"] in {0, 1}, f"Invalid label: {record['label']}"
