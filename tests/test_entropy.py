from __future__ import annotations

from Harpocrates.scanner.entropy import looks_like_secret, shannon_entropy


def test_shannon_entropy_low_vs_high() -> None:
    low = "aaaaaaaaaaaaaaaaaaaaaaaa"
    high = "AKIAIOSFODNN7EXAMPLE"
    assert shannon_entropy(low) < shannon_entropy(high)


def test_looks_like_secret_basic() -> None:
    secret = "a9fK3LmP8QzX1cV7bW2Y"
    not_secret = "thisisjustanormalconfigurationvalue"  # all lowercase letters only

    assert looks_like_secret(secret)
    assert not looks_like_secret(not_secret)
