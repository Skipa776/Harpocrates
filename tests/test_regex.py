"""Tests for the regex patterns module."""
from __future__ import annotations

import re

from Harpocrates.detectors.regex_patterns import (
    CRITICAL_SIGNATURES,
    HIGH_SIGNATURES,
    SIGNATURES,
)

# ---------------------------------------------------------------------------
# CRITICAL patterns
# ---------------------------------------------------------------------------


def test_aws_access_key_id() -> None:
    pattern = CRITICAL_SIGNATURES["AWS_ACCESS_KEY_ID"]
    assert pattern.search("AWS_KEY=AKIAIOSFODNN7EXAMPLE")
    assert pattern.search("ASIA1234567890ABCDEF")  # STS temp cred prefix
    assert not pattern.search("AKIB1234567890ABCD12")  # wrong prefix family
    assert not pattern.search("AKIA12345")              # too short


def test_github_pat() -> None:
    pattern = CRITICAL_SIGNATURES["GITHUB_PAT"]
    # Classic PAT
    assert pattern.search("ghp_" + "a" * 36)
    # OAuth token
    assert pattern.search("gho_" + "a" * 36)
    # Fine-grained PAT: github_pat_ + 22 alphanum + _ + 59 alphanum
    fine = "github_pat_" + "A" * 22 + "_" + "B" * 59
    assert pattern.search(fine)
    # Invalid prefix
    assert not pattern.search("ghq_" + "a" * 36)


def test_slack_token() -> None:
    pattern = CRITICAL_SIGNATURES["SLACK_TOKEN"]
    # Valid: xoxb/xoxp/xoxo/xoxa/xoxr + 10-13 digit workspace ID + 24-34 char secret
    token = "xoxb-1234567890123-" + "A" * 24
    assert pattern.search(token)
    token_app = "xoxo-12345678901-" + "B" * 30
    assert pattern.search(token_app)
    # Invalid prefix (xoxc not in [pboar])
    assert not pattern.search("xoxc-1234567890123-" + "A" * 24)


def test_stripe_key() -> None:
    pattern = CRITICAL_SIGNATURES["STRIPE_KEY"]
    # Live secret, test secret, live restricted, test restricted
    assert pattern.search("sk_live_" + "x" * 24)
    assert pattern.search("sk_test_" + "x" * 24)  # test keys are equally sensitive
    assert pattern.search("rk_live_" + "x" * 24)
    assert pattern.search("rk_test_" + "x" * 24)
    # Invalid: wrong prefix letter
    assert not pattern.search("pk_live_" + "x" * 24)


def test_openai_api_key() -> None:
    pattern = CRITICAL_SIGNATURES["OPENAI_API_KEY"]
    # Legacy format: sk- + 48 alphanum
    assert pattern.search("sk-" + "x" * 48)
    # Project format: sk-proj- + 20 + T3 + 20+
    assert pattern.search("sk-proj-" + "a" * 20 + "T3" + "b" * 20)
    # Invalid prefix
    assert not pattern.search("sx-" + "x" * 48)


def test_anthropic_api_key() -> None:
    pattern = CRITICAL_SIGNATURES["ANTHROPIC_API_KEY"]
    assert pattern.search("sk-ant-api03-" + "a" * 93)
    assert not pattern.search("sk-ant-api03-" + "a" * 10)  # too short


def test_gcp_api_key() -> None:
    pattern = CRITICAL_SIGNATURES["GCP_API_KEY"]
    assert pattern.search("key=AIza" + "A" * 35)
    assert not pattern.search("key=Aiza" + "A" * 35)  # wrong prefix case


def test_npm_token() -> None:
    pattern = CRITICAL_SIGNATURES["NPM_TOKEN"]
    assert pattern.search("NPM_TOKEN=npm_" + "a" * 36)
    assert not pattern.search("npm_" + "a" * 10)  # too short


def test_pypi_token() -> None:
    pattern = CRITICAL_SIGNATURES["PYPI_TOKEN"]
    assert pattern.search("pypi-" + "a" * 50)
    assert not pattern.search("pypi-" + "a" * 10)  # too short


# ---------------------------------------------------------------------------
# HIGH patterns
# ---------------------------------------------------------------------------


def test_private_key_header() -> None:
    pattern = HIGH_SIGNATURES["PRIVATE_KEY"]
    assert pattern.search("-----BEGIN RSA PRIVATE KEY-----")
    assert pattern.search("-----BEGIN EC PRIVATE KEY-----")
    assert pattern.search("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert pattern.search("-----BEGIN PRIVATE KEY-----")  # PKCS#8
    assert not pattern.search("-----BEGIN CERTIFICATE-----")


def test_slack_webhook() -> None:
    pattern = HIGH_SIGNATURES["SLACK_WEBHOOK"]
    url = "https://hooks.slack.com/services/T12345678/B123456789/" + "a" * 24
    assert pattern.search(url)
    assert not pattern.search("https://example.com/services/T12345678/B12345678/" + "a" * 24)


def test_discord_webhook() -> None:
    pattern = HIGH_SIGNATURES["DISCORD_WEBHOOK"]
    url = "https://discord.com/api/webhooks/123456789012345678/" + "a" * 68
    assert pattern.search(url)
    assert not pattern.search("https://discord.com/api/webhooks/123/" + "a" * 68)  # short ID


def test_sendgrid_api_key() -> None:
    pattern = HIGH_SIGNATURES["SENDGRID_API_KEY"]
    key = "SG." + "a" * 22 + "." + "b" * 43
    assert pattern.search(key)
    assert not pattern.search("SG." + "a" * 5 + "." + "b" * 43)  # too short


def test_twilio_api_key() -> None:
    pattern = HIGH_SIGNATURES["TWILIO_API_KEY"]
    assert pattern.search("SK" + "a1b2c3d4" * 4)
    assert not pattern.search("SK" + "g" * 32)  # 'g' not in [0-9a-fA-F]


def test_databricks_token() -> None:
    pattern = HIGH_SIGNATURES["DATABRICKS_TOKEN"]
    assert pattern.search("dapi" + "a1b2" * 8)
    assert not pattern.search("dapi" + "z" * 32)  # 'z' not in [a-f0-9]


def test_hashicorp_vault_token() -> None:
    pattern = HIGH_SIGNATURES["HASHICORP_VAULT_TOKEN"]
    assert pattern.search("hvs." + "a" * 90)
    assert not pattern.search("hvs." + "a" * 10)  # too short


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 6.4: OpenAI legacy key
# ---------------------------------------------------------------------------


def test_openai_legacy_key_matches_short_form() -> None:
    """sk-RQMJj8ELDjv7TRc-dS9sSw style keys with hyphens in the body."""
    pattern = HIGH_SIGNATURES["OPENAI_API_KEY_LEGACY"]
    assert pattern.search("sk-RQMJj8ELDjv7TRc-dS9sSw")
    assert pattern.search("sk-" + "a" * 16)              # minimum 16 chars
    assert pattern.search("sk-" + "a-b_C" * 5)           # hyphens and underscores


def test_openai_legacy_key_requires_minimum_length() -> None:
    pattern = HIGH_SIGNATURES["OPENAI_API_KEY_LEGACY"]
    assert not pattern.search("sk-abc")                   # too short (< 16 chars)
    assert not pattern.search("sk-" + "a" * 15)          # exactly 15: no match


def test_openai_legacy_key_does_not_match_stripe() -> None:
    """Stripe sk_live_/sk_test_ use underscores — no collision with sk-."""
    pattern = HIGH_SIGNATURES["OPENAI_API_KEY_LEGACY"]
    assert not pattern.search("sk_live_" + "x" * 24)
    assert not pattern.search("sk_test_" + "x" * 24)


def test_openai_strict_still_matches_48_char_form() -> None:
    """CRITICAL pattern (48-char sk-) still fires on fully valid keys."""
    pattern = CRITICAL_SIGNATURES["OPENAI_API_KEY"]
    assert pattern.search("sk-" + "x" * 48)


def test_openai_legacy_does_not_double_fire_on_48_char_key() -> None:
    """CRITICAL already owns sk-[a-zA-Z0-9]{48} — LEGACY must not also match."""
    pattern = HIGH_SIGNATURES["OPENAI_API_KEY_LEGACY"]
    assert not pattern.search("sk-" + "x" * 48)


def test_openai_legacy_does_not_fire_on_proj_key() -> None:
    """CRITICAL already owns sk-proj-... — LEGACY must not also match."""
    pattern = HIGH_SIGNATURES["OPENAI_API_KEY_LEGACY"]
    assert not pattern.search("sk-proj-" + "a" * 20 + "T3" + "b" * 20)


def test_openai_legacy_fires_on_47_char_key() -> None:
    """47-char pure-alphanum key is NOT owned by CRITICAL — LEGACY should catch it."""
    pattern = HIGH_SIGNATURES["OPENAI_API_KEY_LEGACY"]
    assert pattern.search("sk-" + "x" * 47)


def test_openai_legacy_detected_at_high_severity() -> None:
    """End-to-end: short OpenAI key → HIGH (not CRITICAL)."""
    from Harpocrates.core.detector import detect_text
    from Harpocrates.core.result import EvidenceType, Severity

    key = "sk-RQMJj8ELDjv7TRc-dS9sSw"
    findings = detect_text(f'OPENAI_API_KEY="{key}"\n')
    legacy = [f for f in findings if f.type == "OPENAI_API_KEY_LEGACY"]
    assert legacy, "Expected OPENAI_API_KEY_LEGACY finding"
    assert all(f.evidence == EvidenceType.REGEX for f in legacy)
    assert all(f.severity == Severity.HIGH for f in legacy)
    assert all(f.category == "openai_key" for f in legacy)


def test_all_signatures_are_compiled() -> None:
    for name, pattern in SIGNATURES.items():
        assert isinstance(pattern, re.Pattern), f"{name} is not a compiled pattern"


def test_signatures_merged_view_is_complete() -> None:
    """SIGNATURES must contain every pattern from both tier dicts."""
    for name in CRITICAL_SIGNATURES:
        assert name in SIGNATURES
    for name in HIGH_SIGNATURES:
        assert name in SIGNATURES
