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
    AZURE_KEY = "azure_key"
    NPM_TOKEN = "npm_token"
    PYPI_TOKEN = "pypi_token"
    PRIVATE_KEY = "private_key"
    SLACK_WEBHOOK = "slack_webhook"
    DISCORD_WEBHOOK = "discord_webhook"
    SENDGRID_KEY = "sendgrid_key"
    TWILIO_KEY = "twilio_key"
    DATABRICKS_TOKEN = "databricks_token"
    VAULT_TOKEN = "vault_token"
    TELEGRAM_TOKEN = "telegram_token"

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


# FR-CORE-02: the six secret types every layer reports (placeholders, logs, blocks).
SECRET_TYPES = ("cloud_key", "api_token", "db_credential", "private_key", "password", "generic")
_SECRET_TYPE_BY_CATEGORY = {
    "aws_key": "cloud_key", "gcp_key": "cloud_key", "azure_key": "cloud_key", "databricks_token": "cloud_key",
    "connection_string": "db_credential",
    "private_key": "private_key", "ssh_key": "private_key",
    "crypto_key": "generic",  # symmetric or hex key material: not an asymmetric private key
    "password": "password",
    "generic_secret": "generic",
}


def secret_type(category: Optional[str]) -> str:
    """Coarse FR-CORE-02 type for a ViolationCategory value; unknown or None -> generic.
    Every provider token, webhook, JWT, OAuth and session secret is an api_token."""
    if category is None:
        return "generic"
    if category in _SECRET_TYPE_BY_CATEGORY:
        return _SECRET_TYPE_BY_CATEGORY[category]
    return "api_token" if category in {c.value for c in ViolationCategory} else "generic"


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
    "TELEGRAM_BOT_TOKEN": ViolationCategory.TELEGRAM_TOKEN,
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
    (re.compile(r"(?i)(?:private|rsa)_?key"),
     ViolationCategory.PRIVATE_KEY, 0.85),
    (re.compile(r"(?i)(?:signing|encryption|hmac|aes|ec)_?key"),
     ViolationCategory.CRYPTO_KEY, 0.85),
    (re.compile(r"(?i)(?:db|database|pg|mysql|mongo|redis|amqp|mq)_?(?:url|conn(?:ection)?|str(?:ing)?)"),
     ViolationCategory.CONNECTION_STRING, 0.90),
    (re.compile(r"(?i)(?:connection|conn)_?(?:url|string|str)"),
     ViolationCategory.CONNECTION_STRING, 0.85),
    (re.compile(r"(?i)(?:jwt|bearer)_?(?:token|secret)?"),
     ViolationCategory.JWT, 0.85),
    (re.compile(r"(?i)client_?secret"),
     ViolationCategory.OAUTH_SECRET, 0.85),
    # session must come before the refresh/access/id_token pattern: `sessionIdToken`
    # contains `idToken` which the oauth pattern would greedily match otherwise.
    (re.compile(r"(?i)session(?:_id|_token|_key|id)?"),
     ViolationCategory.SESSION_TOKEN, 0.75),
    (re.compile(r"(?i)(?:refresh|access)_?token"),
     ViolationCategory.OAUTH_SECRET, 0.80),
    # `id_token` (standalone, not inside sessionIdToken) is an OAuth 2.0 term.
    # Requires ^ or _ boundary so it doesn't match mid-identifier (sessionIdToken).
    (re.compile(r"(?i)(?:^|_)id_?token"),
     ViolationCategory.OAUTH_SECRET, 0.80),
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
    # Endpoint / URL variable suffix — covers OPENAI_BASE_URL, SSO_CALLBACK_URL,
    # APIM_TOKEN_URLS and similar. INFO band (cat_conf < 0.5) only — these are
    # operational metadata, not secrets. Must come last — three higher-priority
    # patterns intercept before reaching here: credentialed DB/AMQP URLs
    # (DATABASE_URL, MONGO_URL, REDIS_URL) via the connection_string entries at
    # 0.85-0.90, and WEBHOOK_URL via the webhook entry at 0.80.
    (re.compile(r"(?i)(?:^|_)urls?$"),
     ViolationCategory.GENERIC_SECRET, 0.45),
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


# Public provider formats with no detection regex of their own (the ML layer catches them). A match is
# decisive: 0.99 beats any var name, so `password = "github_pat_..."` still reports a GitHub token.
_PROVIDER_FORMATS: Tuple[Tuple[re.Pattern, ViolationCategory, str], ...] = (
    (re.compile(r"^github_pat_\w{22,}$"), ViolationCategory.GITHUB_TOKEN, "github_fine_grained_pat"),
    (re.compile(r"^do[opr]_v1_[a-f0-9]{64}$"), ViolationCategory.API_TOKEN, "digitalocean_token"),
    (re.compile(r"^hv[sbr]\.[\w-]{20,}$"), ViolationCategory.VAULT_TOKEN, "vault_token"),
    (re.compile(r"^(?:AC|SK)[a-f0-9]{32}$"), ViolationCategory.TWILIO_KEY, "twilio_sid"),
    (re.compile(r"^[MNO][\w-]{23,25}\.[\w-]{6}\.[\w-]{27,38}$"), ViolationCategory.API_TOKEN, "discord_bot_token"),
)


def _classify_by_value(token: str) -> Optional[CategoryInference]:
    """Layer 3: structural signal from the token value itself."""
    for pattern, category, name in _PROVIDER_FORMATS:
        if pattern.match(token):
            return CategoryInference(category, _reason("value_structure", name, 0.99), 0.99)
    # JWT — three base64url segments separated by dots (full token)
    if token.startswith("eyJ") and token.count(".") == 2:
        return CategoryInference(
            ViolationCategory.JWT,
            _reason("value_structure", "jwt_three_segment", 0.95),
            0.95,
        )
    # JWT header/payload segment — `eyJ` is base64url for `{"` and is highly
    # distinctive. Confidence 0.86 is intentionally above var-name + 0.10 so
    # this wins over API_TOKEN 0.75 from a bare `token` var name (0.86 > 0.85).
    if token.startswith("eyJ"):
        return CategoryInference(
            ViolationCategory.JWT,
            _reason("value_structure", "jwt_eyj_prefix_segment", 0.86),
            0.86,
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
# Layer 4: the token is a piece of a connection string on its line
# ---------------------------------------------------------------------------

_AZURE_CONN_RE = re.compile(r"(?i)(?:AccountKey|SharedAccessKey|DefaultEndpointsProtocol)\s*=")
_DB_URI_SCHEME_RE = re.compile(
    r"(?i)jdbc:|(?:postgres(?:ql)?|mysql|mariadb|rediss?|mongodb(?:\+srv)?|amqps?|rabbitmq|mssql|sqlserver)://"
)
_ADO_PASSWORD_RE = re.compile(r"(?i)(?:password|pwd)\s*=")
_ADO_SERVER_RE = re.compile(r"(?i)(?:server|data source|host|database|initial catalog)\s*=")
_SEGMENT_CAP = 512  # chars either side of the token; connection strings are far shorter


def _segment(line: str, token: str, i: int) -> str:
    """The string literal holding the token when matching quotes enclose it, else its whitespace-delimited
    word. ponytail: an escaped quote inside a literal cuts it short; a real string parser if the per-type
    report shows misses from it."""
    j = i + len(token)
    lo, hi = max(0, i - _SEGMENT_CAP), min(len(line), j + _SEGMENT_CAP)
    left = max(line.rfind(q, lo, i) for q in "\"'`")
    if left >= 0:
        right = line.find(line[left], j, hi)
        if right >= 0:
            return line[left + 1:right]
    start = max(line.rfind(" ", lo, i), line.rfind("\t", lo, i)) + 1
    stop = min((k for k in (line.find(" ", j, hi), line.find("\t", j, hi)) if k >= 0), default=hi)
    return line[max(start, lo):stop]


def _classify_by_line(line: str, token: str) -> Optional[CategoryInference]:
    """Layer 4: the string holding the token is a connection string (a password inside a JDBC URL, an
    AccountKey inside an Azure storage string). Linear time: the read tool scans minified files."""
    i = line.find(token)
    if i < 0:
        return None
    seg = _segment(line, token, i)
    if _AZURE_CONN_RE.search(seg):
        return CategoryInference(ViolationCategory.AZURE_KEY,
                                 _reason("line_context", "azure_connection_string", 0.90), 0.90)
    if _DB_URI_SCHEME_RE.search(seg) or (";" in seg and _ADO_PASSWORD_RE.search(seg)
                                         and _ADO_SERVER_RE.search(seg)):
        return CategoryInference(ViolationCategory.CONNECTION_STRING,
                                 _reason("line_context", "inside_connection_string", 0.90), 0.90)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_category(
    *,
    signature_name: Optional[str],
    var_name: Optional[str],
    token: str,
    line: Optional[str] = None,
) -> CategoryInference:
    """
    Infer violation category from available signals (three-layer heuristic).

    Args:
        signature_name: Regex signature name from CRITICAL_SIGNATURES /
            HIGH_SIGNATURES if this finding came from the regex tier, else None.
        var_name: The left-hand-side variable name (e.g., 'database_password').
            None when the token has no associated LHS variable.
        token: The raw detected token value.
        line: The source line, when known. A token inside a connection string on it is reported
            as that connection string's type unless its own value is decisive (FR-CORE-02).

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

    inference = _infer_from_name_and_value(var_name, token)
    if line and inference.confidence < 0.99:
        return _classify_by_line(line, token) or inference
    return inference


def _infer_from_name_and_value(var_name: Optional[str], token: str) -> CategoryInference:
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
        # A provider format or PEM header (0.99) is decisive whatever the variable is called.
        if value_match.confidence >= 0.99 or value_match.confidence > var_match.confidence + 0.1:
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


_VAR_NAME_WINDOW = 256  # chars left of a token searched for `name =`; identifiers are far shorter


def extract_var_name(line: str, token: str) -> Optional[str]:
    """
    Pull the LHS variable name from a line of the form `var = 'token'`.

    Looks left of the token for `identifier =` or `identifier:` patterns.
    Also handles dotenv/shell KEY=VALUE format where the token starts at
    position 0 and the key is embedded as `KEY=value` in the token itself.

    Returns None if no clear LHS variable is found (multi-assignment, tuple
    unpacking, dict literals, etc.) — callers fall through to value_structure.
    """
    idx = line.find(token)
    if idx < 0:
        return None
    if idx == 0:
        # Token starts at line beginning — dotenv/shell KEY=VALUE or KEY:VALUE.
        # Extract the identifier before the first `=` or `:` separator.
        eq = token.find("=")
        colon = token.find(":")
        sep = min((p for p in (eq, colon) if p > 0), default=-1)
        if sep > 0:
            candidate = token[:sep].strip()
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)$", candidate)
            return m.group(1) if m else None
        return None
    # Only the text just left of the token can hold `name =`. Searching the whole prefix was
    # quadratic in word-character runs (minified JS, inline base64) and hung on 100k+ char lines.
    lhs = line[max(0, idx - _VAR_NAME_WINDOW):idx]
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*['\"]?$", lhs)
    return m.group(1) if m else None


__all__ = ["ViolationCategory", "CategoryInference", "infer_category", "extract_var_name"]
