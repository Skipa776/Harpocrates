"""Tests for the entropy detector module."""
from __future__ import annotations

from Harpocrates.detectors.entropy_detector import looks_like_secret, shannon_entropy


def test_shannon_entropy_empty_string() -> None:
    """Test entropy of empty string is 0."""
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_single_char() -> None:
    """Test entropy of single repeated character is 0."""
    assert shannon_entropy("aaaaaaaaaaaaaaaaaaaaaaaa") == 0.0


def test_shannon_entropy_low_vs_high() -> None:
    """Test that high entropy strings have higher entropy than low entropy strings."""
    low = "aaaaaaaaaaaaaaaaaaaaaaaa"
    high = "AKIAIOSFODNN7EXAMPLE"
    assert shannon_entropy(low) < shannon_entropy(high)


def test_shannon_entropy_typical_secret() -> None:
    """Test that typical secret has high entropy."""
    secret = "aB3dE5fG7hI9jK1lM3nO5pQ7rS9t"
    entropy = shannon_entropy(secret)
    # Typical secrets should have entropy > 4.0
    assert entropy > 4.0


def test_looks_like_secret_true_positive() -> None:
    """Test that random-looking strings are detected as secrets."""
    secret = "a9fK3LmP8QzX1cV7bW2Yxyz"  # Mixed case, digits, 23 chars
    assert looks_like_secret(secret)


def test_looks_like_secret_false_for_words() -> None:
    """Test that normal words are not detected as secrets."""
    not_secret = "thisisjustanormalconfigurationvalue"  # All lowercase
    assert not looks_like_secret(not_secret)


def test_looks_like_secret_short_string() -> None:
    """Test that short strings are not detected as secrets."""
    short = "abc123"  # Too short (< 20 chars)
    assert not looks_like_secret(short)


def test_looks_like_secret_low_diversity() -> None:
    """Test that strings with low character diversity are not secrets."""
    low_diversity = "aaaaaaaaaaaaaaaaaaaaaaaa"  # Only one unique character
    assert not looks_like_secret(low_diversity)


def test_looks_like_secret_threshold() -> None:
    """Test custom threshold parameter."""
    # A string that passes with low threshold but fails with high threshold
    borderline = "AKIAIOSFODNN7EXAMPLE"  # Entropy around 3.6

    # With default threshold (4.0), may or may not pass depending on exact entropy
    # With very low threshold, should pass
    assert looks_like_secret(borderline, threshold=3.0)


def test_looks_like_secret_single_char_class() -> None:
    """Test that strings with only one character class are not secrets."""
    only_lower = "abcdefghijklmnopqrstuvwxyz"
    only_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    only_digits = "12345678901234567890"

    # These have only one character class, should not be detected
    assert not looks_like_secret(only_lower)
    assert not looks_like_secret(only_upper)
    assert not looks_like_secret(only_digits)
