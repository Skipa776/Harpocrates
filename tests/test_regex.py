"""Tests for the regex patterns module."""
from __future__ import annotations

from Harpocrates.detectors.regex_patterns import SIGNATURES


def test_aws_access_key_id() -> None:
    """Test AWS access key ID pattern."""
    pattern = SIGNATURES["AWS_ACCESS_KEY_ID"]
    # Valid format: AKIA + 16 uppercase alphanumeric
    assert pattern.search("key=AKIAIOSFODNN7EXAMPLE")
    assert pattern.search("AKIA1234567890ABCD12")
    # Invalid: wrong prefix
    assert not pattern.search("AKIB1234567890ABCD12")
    # Invalid: too short
    assert not pattern.search("AKIA12345")


def test_aws_secret_access_key() -> None:
    """Test AWS secret access key pattern."""
    pattern = SIGNATURES["AWS_SECRET_ACCESS_KEY"]
    # Valid: 40 base64-like characters
    secret = "A" * 40
    assert pattern.search(f"aws_secret={secret}")
    # Invalid: too short
    not_secret = "A" * 39
    assert not pattern.search(f"aws_secret={not_secret}")


def test_github_token() -> None:
    """Test GitHub personal access token pattern."""
    pattern = SIGNATURES["GITHUB_TOKEN"]
    # Valid: ghp_ + 36 alphanumeric
    token = "ghp_" + "a" * 36
    assert pattern.search(f"token={token}")
    # Invalid: wrong prefix
    not_token = "ghq_" + "a" * 36
    assert not pattern.search(f"token={not_token}")


def test_gcp_api_key() -> None:
    """Test GCP API key pattern."""
    pattern = SIGNATURES["GCP_API_KEY"]
    # Valid: AIza + 35 alphanumeric/dash/underscore
    key = "AIza" + "A" * 35
    assert pattern.search(f"gcp_key={key}")
    # Invalid: wrong prefix (lowercase)
    not_key = "Aiza" + "A" * 35
    assert not pattern.search(f"gcp_key={not_key}")


def test_azure_vault_url() -> None:
    """Test Azure Key Vault URL pattern."""
    pattern = SIGNATURES["AZURE_KEY_VAULT_URL"]
    # Valid: https://xxx.vault.azure.net
    url = "https://myvault.vault.azure.net/secrets"
    assert pattern.search(url)
    # Invalid: http (not https) - actually the pattern uses IGNORECASE so http works
    # Let's test a non-vault URL
    not_url = "https://example.com/vault"
    assert not pattern.search(not_url)


def test_jwt_signature() -> None:
    """Test JWT pattern."""
    pattern = SIGNATURES["JWT"]
    # Valid JWT format: header.payload.signature
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    assert pattern.search(jwt)
    # Invalid: not a JWT
    not_jwt = "this_is_not_a_jwt"
    assert not pattern.search(not_jwt)


def test_slack_token() -> None:
    """Test Slack token pattern."""
    pattern = SIGNATURES["SLACK_TOKEN"]
    # Valid: xoxb, xoxa, xoxp, xoxr, xoxs prefix
    token_bot = "xoxb-1234567890-ABCDEFG"
    token_app = "xoxa-1234567890-ABCDEFG"
    assert pattern.search(token_bot)
    assert pattern.search(token_app)
    # Invalid: wrong prefix
    not_token = "xoxc-1234567890-ABCDEFG"
    assert not pattern.search(not_token)


def test_stripe_secret_key() -> None:
    """Test Stripe secret key pattern."""
    pattern = SIGNATURES["STRIPE_SECRET_KEY"]
    # Valid: sk_live_ + 24 alphanumeric
    key = "sk_live_" + "x" * 24
    assert pattern.search(f"stripe={key}")
    # Invalid: test key (sk_test_) - different from live
    not_key = "sk_test_" + "x" * 24
    assert not pattern.search(f"stripe={not_key}")


def test_openai_api_key() -> None:
    """Test OpenAI API key pattern."""
    pattern = SIGNATURES["OPENAI_API_KEY"]
    # Valid: sk- + 48 alphanumeric
    key = "sk-" + "x" * 48
    assert pattern.search(f"openai={key}")
    # Invalid: wrong prefix
    not_key = "sx-" + "x" * 48
    assert not pattern.search(f"openai={not_key}")


def test_all_signatures_are_compiled() -> None:
    """Test that all signatures are compiled regex patterns."""
    import re
    for name, pattern in SIGNATURES.items():
        assert isinstance(pattern, re.Pattern), f"{name} is not a compiled pattern"
