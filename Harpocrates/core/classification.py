"""
Heuristic violation classification for Harpocrates findings.

Provides ViolationCategory (WHAT was leaked) and CategoryInference (WHY the
category was chosen), distinct from EvidenceType (HOW it was found).

Architecture:
  Three inference layers, applied in priority order:
  1. Regex signature mapping (confidence 1.0) — exact match via sig name.
  2. Variable-name lexicon (confidence 0.7–0.9) — left-hand-side var name.
  3. Value structure analysis (confidence 0.6–0.95) — token shape/prefix.

The module is intentionally self-contained: it imports only from the standard
library. Callers (core/detector.py) pass in pre-extracted signals rather than
raw Finding objects to avoid circular imports.

category_reason format (structured for grepping/debugging):
  "layer=<layer> matched=<pattern_or_signal> var_name='<x>' confidence=<f>"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ViolationCategory(Enum):
    # Specific — 1:1 with regex signature names in regex_patterns.py
    AWS_KEY = "aws_key"
    GITHUB_TOKEN = "github_token"
    SLACK_TOKEN = "slack_token"
    STRIPE_KEY = "stripe_key"
    OPENAI_KEY = "openai_key"
    ANTHROPIC_KEY = "anthropic_key"
    GCP_KEY = "gcp_key"
    NPM_TOKEN = "npm_token"
    PYPI_TOKEN = "pypi_token"
    PRIVATE_KEY = "private_key"
    SLACK_WEBHOOK = "slack_webhook"
    DISCORD_WEBHOOK = "discord_webhook"
    SENDGRID_KEY = "sendgrid_key"
    TWILIO_KEY = "twilio_key"
    DATABRICKS_TOKEN = "databricks_token"
    VAULT_TOKEN = "vault_token"

    # Inferred — heuristic classification for entropy/ML path
    PASSWORD = "password"
    API_TOKEN = "api_token"
    CONNECTION_STRING = "connection_string"
    OAUTH_SECRET = "oauth_secret"
    WEBHOOK_URL = "webhook_url"
    CRYPTO_KEY = "crypto_key"
    SESSION_TOKEN = "session_token"
    JWT = "jwt"
    SSH_KEY = "ssh_key"

    # Fallback
    GENERIC_SECRET = "generic_secret"


# Structured reason string — stable format for log parsing.
# Format: "layer=<l> matched=<m> [var_name='<v>'] confidence=<c>"
def _reason(layer: str, matched: str, confidence: float,
            var_name: Optional[str] = None) -> str:
    vn = f" var_name='{var_name}'" if var_name else ""
    return f"layer={layer} matched={matched}{vn} confidence={confidence:.2f}"


@dataclass(frozen=True)
class CategoryInference:
    """
    Result of heuristic classification.

    `reason` is a structured developer-debuggable string:
      "layer=signature matched=GITHUB_PAT confidence=1.00"
      "layer=var_name_lexicon matched=/password/ var_name='DB_PASSWORD' confidence=0.90"
      "layer=value_structure matched=jwt_three_segment confidence=0.95"
      "layer=fallback matched=no_signal confidence=0.30"

    `confidence` is 0.0–1.0. Low confidence (<0.5) → hedged language in
    display ("potential password"). High confidence → asserted.
    """
    category: ViolationCategory
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# Layer 1: regex signature → category mapping
# ---------------------------------------------------------------------------

_SIGNATURE_TO_CATEGORY: dict[str, ViolationCategory] = {
    "AWS_ACCESS_KEY_ID": ViolationCategory.AWS_KEY,
    "GITHUB_PAT":        ViolationCategory.GITHUB_TOKEN,
    "SLACK_TOKEN":       ViolationCategory.SLACK_TOKEN,
    "STRIPE_KEY":        ViolationCategory.STRIPE_KEY,
    "OPENAI_API_KEY":    ViolationCategory.OPENAI_KEY,
    "ANTHROPIC_API_KEY": ViolationCategory.ANTHROPIC_KEY,
    "GCP_API_KEY":       ViolationCategory.GCP_KEY,
    "NPM_TOKEN":         ViolationCategory.NPM_TOKEN,
    "PYPI_TOKEN":        ViolationCategory.PYPI_TOKEN,
    "PRIVATE_KEY":       ViolationCategory.PRIVATE_KEY,
    "SLACK_WEBHOOK":     ViolationCategory.SLACK_WEBHOOK,
    "DISCORD_WEBHOOK":   ViolationCategory.DISCORD_WEBHOOK,
    "SENDGRID_API_KEY":  ViolationCategory.SENDGRID_KEY,
    "TWILIO_API_KEY":    ViolationCategory.TWILIO_KEY,
    "DATABRICKS_TOKEN":  ViolationCategory.DATABRICKS_TOKEN,
    "HASHICORP_VAULT_TOKEN": ViolationCategory.VAULT_TOKEN,
    "OPENAI_API_KEY_LEGACY": ViolationCategory.OPENAI_KEY,
}

# ---------------------------------------------------------------------------
# Layer 2: variable-name lexicon (most-specific first)
# ---------------------------------------------------------------------------

# Each entry: (compiled_pattern, category, confidence)
# No \b word boundaries — var names are snake_case/camelCase identifiers and
# `_` is `\w`, so \b does NOT fire between "_" and a letter. Substring matching
# via re.search() is safe here because the input is a single programming
# identifier, not free prose.
_VAR_NAME_LEXICON: Tuple[Tuple[re.Pattern, ViolationCategory, float], ...] = (
    (re.compile(r"(?i)(?:private|signing|encryption|hmac|aes|rsa|ec)_?key"),
     ViolationCategory.CRYPTO_KEY, 0.85),
    (re.compile(r"(?i)(?:db|database|pg|mysql|mongo|redis|amqp|mq)_?(?:url|conn(?:ection)?|str(?:ing)?)"),
     ViolationCategory.CONNECTION_STRING, 0.90),
    (re.compile(r"(?i)(?:connection|conn)_?(?:url|string|str)"),
     ViolationCategory.CONNECTION_STRING, 0.85),
    (re.compile(r"(?i)(?:jwt|bearer)_?(?:token|secret)?"),
     ViolationCategory.JWT, 0.85),
    (re.compile(r"(?i)client_?secret"),
     ViolationCategory.OAUTH_SECRET, 0.85),
    (re.compile(r"(?i)(?:refresh|access|id)_?token"),
     ViolationCategory.OAUTH_SECRET, 0.80),
    (re.compile(r"(?i)session(?:_id|_token|_key|id)?"),
     ViolationCategory.SESSION_TOKEN, 0.75),
    (re.compile(r"(?i)webhook(?:_(?:url|secret|key))?"),
     ViolationCategory.WEBHOOK_URL, 0.80),
    (re.compile(r"(?i)ssh_?(?:key|priv(?:ate)?|id)"),
     ViolationCategory.SSH_KEY, 0.85),
    (re.compile(r"(?i)pass(?:word|wd|w)?"),
     ViolationCategory.PASSWORD, 0.90),
    (re.compile(r"(?i)api_?(?:key|token|secret)"),
     ViolationCategory.API_TOKEN, 0.85),
    (re.compile(r"(?i)auth(?:_(?:key|token|secret))?"),
     ViolationCategory.API_TOKEN, 0.70),
    # Suffix-style identifiers — catches *_KEY, *_SECRET, *_TOKEN patterns
    # where the prefix doesn't match a more-specific category above.
    # All use `$` (end-of-string only) to avoid matching key_id, key_type,
    # token_endpoint etc. that share the word but aren't credentials.
    # Compound suffixes (_API_KEY, _SECRET_KEY) MUST come before bare _KEY.
    # Note: _password and _private_key are intentionally absent — the earlier
    # `pass(?:word|wd|w)?` and `(?:private|...)_?key` entries catch those first.
    (re.compile(r"(?i)(?:^|_)(?:api|client|access|refresh|bearer)_(?:secret_)?key$"),
     ViolationCategory.API_TOKEN, 0.80),
    (re.compile(r"(?i)(?:^|_)secret(?:_key)?$"),
     ViolationCategory.API_TOKEN, 0.80),
    (re.compile(r"(?i)(?:^|_)token$"),
     ViolationCategory.API_TOKEN, 0.75),
    (re.compile(r"(?i)(?:^|_)credentials?$"),
     ViolationCategory.API_TOKEN, 0.75),
    # Bare _KEY suffix — GENERIC_SECRET at 0.65 (INFO band). Django/SQLAlchemy
    # primary_key, cache_key, partition_key, foreign_key are extremely common and
    # not credentials. Using `$` anchor prevents matching key_id, key_type, etc.
    (re.compile(r"(?i)(?:^|_)key$"),
     ViolationCategory.GENERIC_SECRET, 0.65),
)

# ---------------------------------------------------------------------------
# Layer 3: value structure analysis
# ---------------------------------------------------------------------------

_DB_URI_RE = re.compile(
    r"^(?:postgres(?:ql)?|mysql|redis|mongodb(?:\+srv)?|amqps?|rabbitmq)://[^:]*:[^@]+@",
    re.IGNORECASE,
)
_WEBHOOK_URL_RE = re.compile(
    r"^https?://\S+[?&](?:token|key|secret|sig(?:nature)?)=[^&\s]+",
    re.IGNORECASE,
)
_HEX32_RE = re.compile(r"^[a-f0-9]{32}$")
_HEX40_RE = re.compile(r"^[a-f0-9]{40}$")
_HEX64_RE = re.compile(r"^[a-f0-9]{64}$")


def _classify_by_value(token: str) -> Optional[CategoryInference]:
    """Layer 3: structural signal from the token value itself."""
    # JWT — three base64url segments separated by dots
    if token.startswith("eyJ") and token.count(".") == 2:
        return CategoryInference(
            ViolationCategory.JWT,
            _reason("value_structure", "jwt_three_segment", 0.95),
            0.95,
        )
    # PEM/OpenSSH private key header
    if "BEGIN" in token and any(k in token for k in ("PRIVATE KEY", "RSA", "OPENSSH")):
        return CategoryInference(
            ViolationCategory.PRIVATE_KEY,
            _reason("value_structure", "pem_private_key_header", 0.99),
            0.99,
        )
    # Database / AMQP URI with embedded credentials
    if _DB_URI_RE.match(token):
        return CategoryInference(
            ViolationCategory.CONNECTION_STRING,
            _reason("value_structure", "db_uri_with_credentials", 0.95),
            0.95,
        )
    # Webhook URL with token-like query parameter
    if _WEBHOOK_URL_RE.match(token):
        return CategoryInference(
            ViolationCategory.WEBHOOK_URL,
            _reason("value_structure", "webhook_url_with_token_param", 0.85),
            0.85,
        )
    # Hex strings — low-confidence crypto key indicators
    if _HEX64_RE.match(token):
        return CategoryInference(
            ViolationCategory.CRYPTO_KEY,
            _reason("value_structure", "hex_64_chars_possible_aes256", 0.60),
            0.60,
        )
    if _HEX32_RE.match(token):
        return CategoryInference(
            ViolationCategory.CRYPTO_KEY,
            _reason("value_structure", "hex_32_chars_possible_aes128", 0.55),
            0.55,
        )
    if _HEX40_RE.match(token):
        return CategoryInference(
            ViolationCategory.CRYPTO_KEY,
            _reason("value_structure", "hex_40_chars_possible_sha1_key", 0.50),
            0.50,
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_category(
    *,
    signature_name: Optional[str],
    var_name: Optional[str],
    token: str,
) -> CategoryInference:
    """
    Infer violation category from available signals (three-layer heuristic).

    Args:
        signature_name: Regex signature name from CRITICAL_SIGNATURES /
            HIGH_SIGNATURES if this finding came from the regex tier, else None.
        var_name: The left-hand-side variable name (e.g., 'database_password').
            None when the token has no associated LHS variable.
        token: The raw detected token value.

    Returns:
        CategoryInference with category, reason, and confidence.
    """
    # Layer 1: regex signature is authoritative.
    if signature_name and signature_name in _SIGNATURE_TO_CATEGORY:
        return CategoryInference(
            category=_SIGNATURE_TO_CATEGORY[signature_name],
            reason=_reason("signature", f"signature_name={signature_name}", 1.0),
            confidence=1.0,
        )

    # Layer 2: variable name lexicon.
    var_match: Optional[CategoryInference] = None
    if var_name:
        for pattern, category, conf in _VAR_NAME_LEXICON:
            if pattern.search(var_name):
                var_match = CategoryInference(
                    category=category,
                    reason=_reason("var_name_lexicon", f"/{pattern.pattern}/",
                                   conf, var_name=var_name),
                    confidence=conf,
                )
                break

    # Layer 3: value structure.
    value_match = _classify_by_value(token)

    # Combine: value structure wins decisively only when confidence delta > 0.1.
    if var_match and value_match:
        if value_match.confidence > var_match.confidence + 0.1:
            return value_match
        # Both fired — report the var_name category with combined reason.
        combined_reason = (
            f"{var_match.reason}; also layer=value_structure "
            f"matched={value_match.category.value}"
        )
        return CategoryInference(
            category=var_match.category,
            reason=combined_reason,
            confidence=min(0.95, var_match.confidence + 0.05),
        )
    if var_match:
        return var_match
    if value_match:
        return value_match

    return CategoryInference(
        category=ViolationCategory.GENERIC_SECRET,
        reason=_reason("fallback", "no_signal", 0.30),
        confidence=0.30,
    )


def extract_var_name(line: str, token: str) -> Optional[str]:
    """
    Pull the LHS variable name from a line of the form `var = 'token'`.

    Looks left of the token for `identifier =` or `identifier:` patterns.
    Returns None if no clear LHS variable is found (multi-assignment, tuple
    unpacking, dict literals, etc.) — callers fall through to value_structure.
    """
    idx = line.find(token)
    if idx <= 0:
        return None
    lhs = line[:idx]
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*['\"]?$", lhs)
    return m.group(1) if m else None


__all__ = ["ViolationCategory", "CategoryInference", "infer_category", "extract_var_name"]
