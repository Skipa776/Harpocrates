from __future__ import annotations

from Harpocrates.scanner.regex_signatures import SIGNATURES


def test_aws_access_key_id() -> None:
    pattern = SIGNATURES["AWS_ACCESS_KEY_ID"]
    assert pattern.search("key=AKIA1234567890ABCD12")
    assert not pattern.search("AKIB1234567890ABCD12")


def test_aws_secret_access_key() -> None:
    pattern = SIGNATURES["AWS_SECRET_ACCESS_KEY"]
    secret = "A" * 40
    not_secret = "A" * 39
    assert pattern.search(f"aws_secret={secret}")
    assert not pattern.search(f"aws_secret={not_secret}")


def test_github_token() -> None:
    pattern = SIGNATURES["GITHUB_TOKEN"]
    token = "ghp_" + "a" * 36
    not_token = "ghq_" + "a" * 36
    assert pattern.search(f"token={token}")
    assert not pattern.search(f"token={not_token}")


def test_gcp_api_key() -> None:
    pattern = SIGNATURES["GCP_API_KEY"]
    key = "AIza" + "A" * 35
    not_key = "Aiza" + "A" * 35
    assert pattern.search(f"gcp_key={key}")
    assert not pattern.search(f"gcp_key={not_key}")


def test_azure_vault_url() -> None:
    pattern = SIGNATURES["AZURE_KEY_VAULT_URL"]
    url = "https://myvault.vault.azure.net/secrets"
    not_url = "http://myvault.vault.azure.net/secrets"
    assert pattern.search(url)
    assert not pattern.search(not_url)


def test_jwt_signature() -> None:
    pattern = SIGNATURES["JWT"]
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    not_jwt = "this_is_not_a_jwt"
    assert pattern.search(jwt)
    assert not pattern.search(not_jwt)


def test_slack_token() -> None:
    pattern = SIGNATURES["SLACK_TOKEN"]
    token = "xoxb-1234567890-ABCDEFG"
    not_token = "xoxc-1234567890-ABCDEFG"
    assert pattern.search(token)
    assert not pattern.search(not_token)


def test_stripe_secret_key() -> None:
    pattern = SIGNATURES["STRIPE_SECRET_KEY"]
    key = "sk_live_" + "x" * 24
    not_key = "sk_test_" + "x" * 24
    assert pattern.search(f"stripe={key}")
    assert not pattern.search(f"stripe={not_key}")

def test_open_ai_key() -> None:
    pattern = SIGNATURES["OPENAI_API_KEY"]
    key = "sk-" + "x" * 48
    not_key = "sx-" + "x" * 48
    assert pattern.search(f"openai={key}")
    assert not pattern.search(f"openai={not_key}")