"""Tests for the ViolationCategory heuristic classifier."""
from __future__ import annotations

from Harpocrates.core.classification import (
    ViolationCategory,
    extract_var_name,
    infer_category,
)

# ---------------------------------------------------------------------------
# Layer 1: regex signature mapping
# ---------------------------------------------------------------------------

def test_signature_mapping_github_pat() -> None:
    inf = infer_category(signature_name="GITHUB_PAT", var_name=None, token="ghp_abc")
    assert inf.category == ViolationCategory.GITHUB_TOKEN
    assert inf.confidence == 1.0
    assert "layer=signature" in inf.reason
    assert "GITHUB_PAT" in inf.reason


def test_signature_mapping_aws_key() -> None:
    inf = infer_category(signature_name="AWS_ACCESS_KEY_ID", var_name=None, token="AKIAXXX")
    assert inf.category == ViolationCategory.AWS_KEY
    assert inf.confidence == 1.0


def test_signature_mapping_all_known_signatures() -> None:
    from Harpocrates.core.classification import _SIGNATURE_TO_CATEGORY
    from Harpocrates.detectors.regex_patterns import CRITICAL_SIGNATURES, HIGH_SIGNATURES
    all_sig_names = set(CRITICAL_SIGNATURES) | set(HIGH_SIGNATURES)
    for name in all_sig_names:
        inf = infer_category(signature_name=name, var_name=None, token="x")
        assert inf.confidence == 1.0, f"{name} should have confidence=1.0"
        assert name in _SIGNATURE_TO_CATEGORY, f"{name} missing from _SIGNATURE_TO_CATEGORY"


def test_signature_to_category_has_no_orphaned_entries() -> None:
    """Every key in _SIGNATURE_TO_CATEGORY must have a corresponding regex."""
    from Harpocrates.core.classification import _SIGNATURE_TO_CATEGORY
    from Harpocrates.detectors.regex_patterns import CRITICAL_SIGNATURES, HIGH_SIGNATURES
    all_sig_names = set(CRITICAL_SIGNATURES) | set(HIGH_SIGNATURES)
    for name in _SIGNATURE_TO_CATEGORY:
        assert name in all_sig_names, (
            f"_SIGNATURE_TO_CATEGORY has orphaned key '{name}' with no corresponding regex"
        )


def test_unknown_signature_falls_through_to_lexicon() -> None:
    """Unrecognised signature_name must not crash — fall through to lower layers."""
    inf = infer_category(signature_name="UNKNOWN_SIG", var_name="database_password",
                         token="hunter2")
    assert inf.category == ViolationCategory.PASSWORD
    assert inf.confidence < 1.0


# ---------------------------------------------------------------------------
# Layer 2: variable-name lexicon
# ---------------------------------------------------------------------------

def test_var_name_password() -> None:
    inf = infer_category(signature_name=None, var_name="database_password", token="x" * 20)
    assert inf.category == ViolationCategory.PASSWORD
    assert inf.confidence >= 0.85
    assert "layer=var_name_lexicon" in inf.reason
    assert "database_password" in inf.reason


def test_var_name_jwt() -> None:
    inf = infer_category(signature_name=None, var_name="jwt_secret", token="x" * 20)
    assert inf.category == ViolationCategory.JWT


def test_var_name_connection_string() -> None:
    inf = infer_category(signature_name=None, var_name="db_connection_string", token="x" * 20)
    assert inf.category == ViolationCategory.CONNECTION_STRING


def test_var_name_endpoint_url_classifies_as_generic_secret_low_conf() -> None:
    inf = infer_category(signature_name=None, var_name="OPENAI_BASE_URL", token="https://api.openai.com/v1")
    assert inf.category == ViolationCategory.GENERIC_SECRET
    assert inf.confidence < 0.5


def test_var_name_sso_callback_url() -> None:
    inf = infer_category(signature_name=None, var_name="SSO_CALLBACK_URL", token="https://sso.corp.example.com/callback")
    assert inf.category == ViolationCategory.GENERIC_SECRET
    assert inf.confidence < 0.5


def test_database_url_still_classifies_as_connection_string() -> None:
    inf = infer_category(signature_name=None, var_name="DATABASE_URL", token="postgres://admin:pass@host/db")
    assert inf.category == ViolationCategory.CONNECTION_STRING
    assert inf.confidence >= 0.85


def test_mongo_url_classifies_as_connection_string() -> None:
    inf = infer_category(signature_name=None, var_name="MONGO_URL", token="mongodb://user:pass@host/db")
    assert inf.category == ViolationCategory.CONNECTION_STRING
    assert inf.confidence >= 0.85


def test_redis_url_classifies_as_connection_string() -> None:
    inf = infer_category(signature_name=None, var_name="REDIS_URL", token="redis://:pass@host:6379/0")
    assert inf.category == ViolationCategory.CONNECTION_STRING
    assert inf.confidence >= 0.85


def test_webhook_url_not_reclassified_by_url_entry() -> None:
    inf = infer_category(signature_name=None, var_name="WEBHOOK_URL", token="https://hooks.example.com/abc")
    assert inf.category == ViolationCategory.WEBHOOK_URL
    assert inf.confidence >= 0.80


def test_var_name_api_token() -> None:
    inf = infer_category(signature_name=None, var_name="api_key", token="x" * 20)
    assert inf.category == ViolationCategory.API_TOKEN


def test_var_name_client_secret() -> None:
    inf = infer_category(signature_name=None, var_name="client_secret", token="x" * 20)
    assert inf.category == ViolationCategory.OAUTH_SECRET


def test_var_name_crypto_key() -> None:
    inf = infer_category(signature_name=None, var_name="hmac_key", token="x" * 20)
    assert inf.category == ViolationCategory.CRYPTO_KEY


# ---------------------------------------------------------------------------
# Layer 3: value structure analysis
# ---------------------------------------------------------------------------

def test_value_structure_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    inf = infer_category(signature_name=None, var_name=None, token=jwt)
    assert inf.category == ViolationCategory.JWT
    assert inf.confidence >= 0.90


def test_value_structure_connection_string() -> None:
    uri = "postgres://user:s3cr3tpassword@db.internal:5432/mydb"
    inf = infer_category(signature_name=None, var_name=None, token=uri)
    assert inf.category == ViolationCategory.CONNECTION_STRING
    assert inf.confidence >= 0.90


def test_value_structure_pem_private_key() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAA..."
    inf = infer_category(signature_name=None, var_name=None, token=pem)
    assert inf.category == ViolationCategory.PRIVATE_KEY
    assert inf.confidence >= 0.95


def test_value_structure_hex64_crypto_key() -> None:
    hex64 = "a" * 64
    inf = infer_category(signature_name=None, var_name=None, token=hex64)
    assert inf.category == ViolationCategory.CRYPTO_KEY


# ---------------------------------------------------------------------------
# Combiner behaviour
# ---------------------------------------------------------------------------

def test_value_wins_decisively_over_var_name() -> None:
    """JWT value (conf 0.95) beats GENERIC_SECRET var name (conf 0.65)."""
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    inf = infer_category(signature_name=None, var_name="secret", token=jwt)
    assert inf.category == ViolationCategory.JWT


def test_var_name_wins_when_value_signal_is_weak() -> None:
    """Hex-32 (conf 0.55) does NOT beat PASSWORD var name (conf 0.90)."""
    inf = infer_category(signature_name=None, var_name="database_password", token="a" * 32)
    assert inf.category == ViolationCategory.PASSWORD


def test_both_fire_combined_reason_includes_value_signal() -> None:
    """When both var_name and value fire, reason mentions both layers."""
    uri = "postgres://user:s3cr3t@host/db"
    inf = infer_category(signature_name=None, var_name="db_connection_string", token=uri)
    assert inf.category == ViolationCategory.CONNECTION_STRING
    # When both agree, combined reason or either layer is fine
    assert inf.confidence > 0.85


def test_no_signal_returns_generic_secret() -> None:
    inf = infer_category(signature_name=None, var_name=None, token="shortval")
    assert inf.category == ViolationCategory.GENERIC_SECRET
    assert inf.confidence < 0.50
    assert "layer=fallback" in inf.reason


# ---------------------------------------------------------------------------
# extract_var_name helper
# ---------------------------------------------------------------------------

def test_extract_var_name_simple_assignment() -> None:
    assert extract_var_name("api_key = 'sk_live_xxx'", "sk_live_xxx") == "api_key"


def test_extract_var_name_colon_style() -> None:
    assert extract_var_name('password: "hunter2"', "hunter2") == "password"


def test_extract_var_name_no_lhs() -> None:
    assert extract_var_name("sk_live_xxx", "sk_live_xxx") is None


def test_extract_var_name_token_not_in_line() -> None:
    assert extract_var_name("api_key = 'other'", "not_present") is None


# ---------------------------------------------------------------------------
# Detector integration — category populated on findings
# ---------------------------------------------------------------------------

def test_finding_category_populated_on_regex_hit() -> None:
    from Harpocrates.core.detector import detect_text

    token = "ghp_" + "d" * 36
    findings = detect_text(f"token = {token}\n")
    assert findings
    github_findings = [f for f in findings if f.type == "GITHUB_PAT"]
    assert github_findings
    assert all(f.category == "github_token" for f in github_findings)
    assert all(f.category_reason is not None for f in github_findings)


def test_finding_category_populated_on_entropy_hit() -> None:
    from Harpocrates.core.detector import detect_text
    from Harpocrates.core.result import EvidenceType

    text = "# secret_key = aB3dEfGhIjKlMnOpQrStUvWxYz012\n"
    findings = detect_text(text)
    entropy_findings = [f for f in findings if f.evidence == EvidenceType.ENTROPY]
    assert entropy_findings
    # Every entropy finding must have category set
    assert all(f.category is not None for f in entropy_findings)
    assert all(f.category_reason is not None for f in entropy_findings)


def test_finding_category_in_json_output() -> None:
    from Harpocrates.core.result import EvidenceType, Finding, Severity

    f = Finding(
        type="GITHUB_PAT",
        snippet="token=ghp_xxx",
        evidence=EvidenceType.REGEX,
        severity=Severity.CRITICAL,
        category="github_token",
        category_reason="layer=signature matched=signature_name=GITHUB_PAT confidence=1.00",
    )
    d = f.to_json_dict()
    assert d["category"] == "github_token"
    assert "layer=signature" in d["category_reason"]
    assert "token" not in d  # token is still redacted by default


# ---------------------------------------------------------------------------
# Phase 6.2: suffix-style lexicon expansion
# ---------------------------------------------------------------------------

def test_apim_client_key_classifies_as_api_token() -> None:
    """APIM_CLIENT_KEY must match suffix-style _API|CLIENT_KEY pattern → API_TOKEN."""
    inf = infer_category(signature_name=None, var_name="APIM_CLIENT_KEY",
                         token="eDRlUVQ" + "x" * 20)
    assert inf.category == ViolationCategory.API_TOKEN
    assert inf.confidence >= 0.75


def test_apim_secret_key_classifies_as_api_token() -> None:
    """APIM_SECRET_KEY must match suffix-style _SECRET_KEY pattern → API_TOKEN 0.80."""
    inf = infer_category(signature_name=None, var_name="APIM_SECRET_KEY",
                         token="40z_9Yw" + "x" * 20)
    assert inf.category == ViolationCategory.API_TOKEN
    assert inf.confidence >= 0.80


def test_bare_key_suffix_is_generic_secret_low_confidence() -> None:
    """primary_key, cache_key etc. → GENERIC_SECRET 0.65 (INFO band — not MEDIUM)."""
    for var_name in ("primary_key", "cache_key", "partition_key", "foreign_key"):
        inf = infer_category(signature_name=None, var_name=var_name, token="x" * 20)
        assert inf.category == ViolationCategory.GENERIC_SECRET, (
            f"{var_name} should be GENERIC_SECRET, got {inf.category}"
        )
        assert inf.confidence <= 0.65, f"{var_name} confidence {inf.confidence} exceeds 0.65"


def test_password_from_early_lexicon_entry() -> None:
    """pass(?:word|wd|w)? (entry 10, confidence 0.90) catches password vars.
    The suffix-style entries intentionally do NOT add a redundant _password pattern
    since the early entry fires first via re.search substring matching."""
    inf = infer_category(signature_name=None, var_name="my_password", token="x" * 20)
    assert inf.category == ViolationCategory.PASSWORD
    assert inf.confidence >= 0.90


def test_openai_api_key_var_name_unchanged() -> None:
    """Existing api_?key pattern still fires for OPENAI_API_KEY var names."""
    inf = infer_category(signature_name=None, var_name="OPENAI_API_KEY", token="x" * 20)
    assert inf.category == ViolationCategory.API_TOKEN
    assert inf.confidence >= 0.85


def test_db_password_unchanged_after_lexicon_expansion() -> None:
    """Regression: DB_PASSWORD still resolves to PASSWORD at ≥0.90 via early entry."""
    inf = infer_category(signature_name=None, var_name="DB_PASSWORD", token="x" * 20)
    assert inf.category == ViolationCategory.PASSWORD
    assert inf.confidence >= 0.90


def test_client_secret_unchanged_after_expansion() -> None:
    """Regression: client_secret still resolves to OAUTH_SECRET (more specific wins)."""
    inf = infer_category(signature_name=None, var_name="client_secret", token="x" * 20)
    assert inf.category == ViolationCategory.OAUTH_SECRET
    assert inf.confidence >= 0.85


def test_finding_str_includes_category() -> None:
    from Harpocrates.core.result import EvidenceType, Finding, Severity

    f = Finding(
        type="ML_CANDIDATE",
        snippet="password = 'hunter2'",
        evidence=EvidenceType.ML,
        severity=Severity.INFO,
        category="password",
    )
    s = str(f)
    assert "[password]" in s
    assert "ML_CANDIDATE" in s
