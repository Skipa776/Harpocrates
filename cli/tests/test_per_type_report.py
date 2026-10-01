"""FR-CORE-02: every labeled secret in the eval sets maps to one of the six reported types."""
from __future__ import annotations

from bench.per_type_report import TRUE_TYPE, summarize
from Harpocrates.core.classification import SECRET_TYPES

EVAL_POSITIVE_TYPES = {
    "connection_uri", "generic_random", "sendgrid_key", "npm_token", "jdbc_url", "azure_connection_string",
    "slack_token", "discord_token", "ado_connection_string", "gcp_api_key", "aws_secret_key", "password",
    "openai_key", "pypi_token", "github_fine_grained", "token_url", "vault_token", "aws_access_key",
    "telegram_token", "twilio_sid", "jwt", "digitalocean_token", "stripe_key", "github_token",
}


def test_fr_core_02_every_eval_type_maps_to_a_secret_type() -> None:
    assert EVAL_POSITIVE_TYPES <= set(TRUE_TYPE)
    assert set(TRUE_TYPE.values()) <= set(SECRET_TYPES)


def test_fr_core_02_summarize_counts_per_type() -> None:
    # (true type or None for a non-secret record, [(predicted type, overlaps the secret)])
    rows = [("cloud_key", [("cloud_key", True)]), ("cloud_key", []),
            ("password", [("generic", True)]), (None, [("password", False)])]
    out = summarize(rows)
    assert out["cloud_key"] == {"secrets": 2, "recall": 0.5, "findings": 1, "precision": 1.0}
    assert out["password"]["recall"] == 1.0 and out["password"]["precision"] == 0.0
    assert out["generic"]["precision"] == 1.0
    assert out["_type_agreement"] == 0.5  # 1 of 2 caught secrets got its true type
