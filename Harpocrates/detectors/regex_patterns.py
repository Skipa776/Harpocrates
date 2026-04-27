from __future__ import annotations

import re
from re import Pattern
from typing import Dict

# ---------------------------------------------------------------------------
# Fast-path regex layer — "Fat ML, Thin Regex" architecture.
#
# Only structurally-anchored patterns with effectively zero false-positive
# rate live here. Everything else falls through to XGBoost.
#
# Patterns are split into two tiers so detector.py can assign severity
# without inspecting individual pattern names.
#
# All patterns are compiled at module load for zero per-call overhead.
# ---------------------------------------------------------------------------

# CRITICAL: cloud/SaaS credentials — deterministic prefixes, fixed lengths.
CRITICAL_SIGNATURES: Dict[str, Pattern] = {
    # AWS key ID — all IAM principal types share this prefix family.
    "AWS_ACCESS_KEY_ID": re.compile(
        r"\b(?:AKIA|A3T|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b"
    ),
    # GitHub PAT — classic (ghp_/gho_/ghu_/ghs_/ghr_) and fine-grained.
    "GITHUB_PAT": re.compile(
        r"\b(?:gh[pousr]_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59})\b"
    ),
    # Slack bot/app/user/workspace/refresh tokens.
    "SLACK_TOKEN": re.compile(
        r"\bxox[pboar]-[0-9]{10,13}-[a-zA-Z0-9\-]{24,34}\b"
    ),
    # Stripe standard and restricted keys (live and test).
    "STRIPE_KEY": re.compile(
        r"\b[rs]k_(?:live|test)_[a-zA-Z0-9]{24,99}\b"
    ),
    # OpenAI legacy key and current project-scoped key.
    "OPENAI_API_KEY": re.compile(
        r"\b(?:sk-[a-zA-Z0-9]{48}|sk-proj-[a-zA-Z0-9]{20}T3[a-zA-Z0-9]{20,})\b"
    ),
    # Anthropic Claude API key.
    "ANTHROPIC_API_KEY": re.compile(
        r"\bsk-ant-api03-[a-zA-Z0-9\-_]{93,}\b"
    ),
    # Google Cloud Platform API key.
    "GCP_API_KEY": re.compile(
        r"\bAIza[0-9A-Za-z\-_]{35}\b"
    ),
    # NPM automation/publish token.
    "NPM_TOKEN": re.compile(
        r"\bnpm_[a-zA-Z0-9]{36}\b"
    ),
    # PyPI API token.
    "PYPI_TOKEN": re.compile(
        r"\bpypi-[a-zA-Z0-9\-_]{50,}\b"
    ),
}

# HIGH: infrastructure credentials — anchored but slightly broader scope.
HIGH_SIGNATURES: Dict[str, Pattern] = {
    # PEM private key header — RSA/EC/OPENSSH typed or PKCS#8 untyped.
    "PRIVATE_KEY": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    # Slack incoming webhook URL.
    "SLACK_WEBHOOK": re.compile(
        r"\bhttps://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8,10}/B[a-zA-Z0-9_]{8,10}/[a-zA-Z0-9_]{24}\b"
    ),
    # Discord webhook URL.
    "DISCORD_WEBHOOK": re.compile(
        r"\bhttps://discord\.com/api/webhooks/[0-9]{17,19}/[a-zA-Z0-9\-_]{68}\b"
    ),
    # SendGrid API key.
    "SENDGRID_API_KEY": re.compile(
        r"\bSG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}\b"
    ),
    # Twilio API key SID.
    "TWILIO_API_KEY": re.compile(
        r"\bSK[0-9a-fA-F]{32}\b"
    ),
    # Databricks personal access token.
    "DATABRICKS_TOKEN": re.compile(
        r"\bdapi[a-f0-9]{32}\b"
    ),
    # HashiCorp Vault service token (v2 format).
    "HASHICORP_VAULT_TOKEN": re.compile(
        r"\bhvs\.[a-zA-Z0-9\-_]{90,}\b"
    ),
}

# Merged view preserved for any external callers that import SIGNATURES directly.
SIGNATURES: Dict[str, Pattern] = {**CRITICAL_SIGNATURES, **HIGH_SIGNATURES}

__all__ = ["CRITICAL_SIGNATURES", "HIGH_SIGNATURES", "SIGNATURES"]
