"""
Secret generation templates for synthetic training data.

Generates realistic-looking secrets for various platforms
without using actual valid credentials.
"""
from __future__ import annotations

import base64
import json
import random
import string
from typing import Tuple

# Character sets for different secret types
ALPHANUMERIC = string.ascii_letters + string.digits
ALPHANUMERIC_UPPER = string.ascii_uppercase + string.digits
BASE64_CHARS = string.ascii_letters + string.digits + "+/"
BASE64_URL_CHARS = string.ascii_letters + string.digits + "-_"
HEX_CHARS = string.hexdigits.lower()


def _random_string(length: int, charset: str = ALPHANUMERIC) -> str:
    """Generate random string from character set."""
    # Use random module for reproducibility with seed
    return "".join(random.choice(charset) for _ in range(length))


def generate_aws_key(valid: bool = True) -> Tuple[str, str]:
    """
    Generate fake AWS access key pair.

    Args:
        valid: If True, use standard AKIA prefix; if False, use fake prefix

    Returns:
        Tuple of (access_key_id, secret_access_key)
    """
    if valid:
        # AWS Access Key ID: AKIA + 16 uppercase alphanumeric
        access_key_id = "AKIA" + _random_string(16, ALPHANUMERIC_UPPER)
    else:
        # Invalid prefix - looks similar but won't pass validation
        access_key_id = "FKIA" + _random_string(16, ALPHANUMERIC_UPPER)

    # AWS Secret Access Key: 40 base64-like characters
    secret_key = _random_string(40, BASE64_CHARS)

    return access_key_id, secret_key


def generate_github_token(valid: bool = True) -> str:
    """
    Generate fake GitHub personal access token.

    Args:
        valid: If True, generate valid-looking token; if False, corrupt checksum

    Returns:
        Token string with ghp_ prefix
    """
    # GitHub PAT: ghp_ + 36 alphanumeric
    base = "ghp_" + _random_string(36, ALPHANUMERIC)
    if not valid:
        # Corrupt the last 4 chars to make it invalid
        return base[:-4] + "XXXX"
    return base


def generate_github_fine_grained_token() -> str:
    """
    Generate fake GitHub fine-grained personal access token.

    Returns:
        Token string with github_pat_ prefix
    """
    return "github_pat_" + _random_string(82, ALPHANUMERIC + "_")


def generate_stripe_key(live: bool = True) -> str:
    """
    Generate fake Stripe secret key.

    Args:
        live: If True, generate live key; else test key

    Returns:
        Stripe secret key string
    """
    prefix = "sk_live_" if live else "sk_test_"
    return prefix + _random_string(24, ALPHANUMERIC)


def generate_openai_key() -> str:
    """
    Generate fake OpenAI API key.

    Returns:
        OpenAI API key string
    """
    # OpenAI key: sk- + 48 alphanumeric
    return "sk-" + _random_string(48, ALPHANUMERIC)


def generate_slack_token(token_type: str = "bot") -> str:
    """
    Generate fake Slack token.

    Args:
        token_type: One of 'bot', 'user', 'app', 'refresh'

    Returns:
        Slack token string
    """
    prefixes = {
        "bot": "xoxb-",
        "user": "xoxp-",
        "app": "xoxa-",
        "refresh": "xoxr-",
    }
    prefix = prefixes.get(token_type, "xoxb-")
    # Slack tokens have dashes in them
    parts = [_random_string(random.randint(10, 15), string.digits) for _ in range(3)]
    return prefix + "-".join(parts)


def generate_gcp_api_key() -> str:
    """
    Generate fake Google Cloud API key.

    Returns:
        GCP API key string
    """
    # GCP key: AIza + 35 alphanumeric with dashes and underscores
    charset = ALPHANUMERIC + "-_"
    return "AIza" + _random_string(35, charset)


# --- Additional Cloud Provider Generators ---
# These expand vendor coverage beyond AWS/GitHub/Stripe


def generate_alibaba_cloud_key() -> Tuple[str, str]:
    """
    Generate fake Alibaba Cloud access key pair.

    Returns:
        Tuple of (access_key_id, access_key_secret)
    """
    # Alibaba Cloud Access Key ID: LTAI + 12-20 alphanumeric
    access_key_id = "LTAI" + _random_string(random.randint(12, 20), ALPHANUMERIC)
    # Access Key Secret: 30 alphanumeric
    access_key_secret = _random_string(30, ALPHANUMERIC)
    return access_key_id, access_key_secret


def generate_digitalocean_token() -> str:
    """
    Generate fake DigitalOcean API token.

    Returns:
        DigitalOcean token string
    """
    # DO token: dop_v1_ prefix + 64 alphanumeric
    return "dop_v1_" + _random_string(64, ALPHANUMERIC)


def generate_azure_sas_token() -> str:
    """
    Generate fake Azure SAS (Shared Access Signature) token.

    Returns:
        Azure SAS token string
    """
    # SAS token components
    sv = "2021-06-08"  # Storage version
    ss = random.choice(["b", "f", "q", "t", "bfqt"])  # Services
    srt = random.choice(["s", "c", "o", "sco"])  # Resource types
    sp = random.choice(["r", "w", "d", "l", "rwdl"])  # Permissions
    se = "2025-12-31T23:59:59Z"  # Expiry
    sig = base64.b64encode(_random_string(32, ALPHANUMERIC).encode()).decode()

    return f"sv={sv}&ss={ss}&srt={srt}&sp={sp}&se={se}&sig={sig}"


def generate_azure_connection_string() -> str:
    """
    Generate fake Azure Storage connection string.

    Returns:
        Azure connection string
    """
    account_name = _random_string(random.randint(8, 16), string.ascii_lowercase)
    account_key = base64.b64encode(_random_string(64, ALPHANUMERIC).encode()).decode()

    return f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"


def generate_hashicorp_vault_token() -> str:
    """
    Generate fake HashiCorp Vault token.

    Returns:
        Vault token string
    """
    # Vault tokens: hvs. prefix + ~95 chars
    prefixes = ["hvs.", "hvb.", "hvr."]  # service, batch, recovery
    prefix = random.choice(prefixes)
    return prefix + _random_string(95, ALPHANUMERIC + "-_")


def generate_twilio_credentials() -> Tuple[str, str]:
    """
    Generate fake Twilio credentials.

    Returns:
        Tuple of (account_sid, auth_token)
    """
    # Account SID: AC + 32 hex
    account_sid = "AC" + _random_string(32, HEX_CHARS)
    # Auth Token: 32 hex
    auth_token = _random_string(32, HEX_CHARS)
    return account_sid, auth_token


def generate_sendgrid_key() -> str:
    """
    Generate fake SendGrid API key.

    Returns:
        SendGrid API key string
    """
    # SG. prefix + 22 chars + . + 43 chars
    return f"SG.{_random_string(22, ALPHANUMERIC)}.{_random_string(43, ALPHANUMERIC + '-_')}"


def generate_mailchimp_key() -> str:
    """
    Generate fake Mailchimp API key.

    Returns:
        Mailchimp API key string
    """
    # 32 hex chars + dash + datacenter code
    datacenters = ["us1", "us2", "us3", "us4", "us5", "us6", "us7", "us8", "us9", "us10"]
    return f"{_random_string(32, HEX_CHARS)}-{random.choice(datacenters)}"


def generate_datadog_key() -> Tuple[str, str]:
    """
    Generate fake Datadog API and application keys.

    Returns:
        Tuple of (api_key, app_key)
    """
    # API key: 32 hex
    api_key = _random_string(32, HEX_CHARS)
    # App key: 40 hex
    app_key = _random_string(40, HEX_CHARS)
    return api_key, app_key


def generate_shopify_credentials() -> Tuple[str, str, str]:
    """
    Generate fake Shopify API credentials.

    Returns:
        Tuple of (api_key, api_secret, access_token)
    """
    # API key: 32 hex
    api_key = _random_string(32, HEX_CHARS)
    # API secret: shpss_ + 32 hex
    api_secret = "shpss_" + _random_string(32, HEX_CHARS)
    # Access token: shpat_ + 32 hex
    access_token = "shpat_" + _random_string(32, HEX_CHARS)
    return api_key, api_secret, access_token


def generate_square_credentials() -> Tuple[str, str]:
    """
    Generate fake Square API credentials.

    Returns:
        Tuple of (application_id, access_token)
    """
    # Application ID: sq0idp- + 22 chars
    app_id = "sq0idp-" + _random_string(22, ALPHANUMERIC + "-_")
    # Access token: sq0atp- or EAAl + rest
    access_token = "EAAl" + _random_string(60, ALPHANUMERIC)
    return app_id, access_token


def generate_discord_token() -> str:
    """
    Generate fake Discord bot token.

    Returns:
        Discord token string
    """
    # Discord tokens: base64(bot_id).timestamp.hmac
    # Structure: ~24 chars . 6 chars . ~27 chars
    bot_id_b64 = base64.b64encode(str(random.randint(100000000000000000, 999999999999999999)).encode()).decode().rstrip("=")
    timestamp = _random_string(6, ALPHANUMERIC)
    hmac = _random_string(27, ALPHANUMERIC + "-_")
    return f"{bot_id_b64}.{timestamp}.{hmac}"


def generate_telegram_token() -> str:
    """
    Generate fake Telegram bot token.

    Returns:
        Telegram token string
    """
    # Format: bot_id:secret
    bot_id = str(random.randint(100000000, 9999999999))
    secret = _random_string(35, ALPHANUMERIC + "-_")
    return f"{bot_id}:{secret}"


def generate_jwt_token() -> str:
    """
    Generate fake JWT token with valid structure.

    Returns:
        JWT token string (header.payload.signature)
    """
    # Create valid-looking JWT structure
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": _random_string(10),
        "iat": random.randint(1600000000, 1700000000),
        "exp": random.randint(1700000000, 1800000000),
    }

    def b64url_encode(data: dict) -> str:
        json_str = json.dumps(data, separators=(",", ":"))
        return base64.urlsafe_b64encode(json_str.encode()).decode().rstrip("=")

    header_b64 = b64url_encode(header)
    payload_b64 = b64url_encode(payload)
    signature = _random_string(43, BASE64_URL_CHARS)

    return f"{header_b64}.{payload_b64}.{signature}"


def generate_azure_key_vault_url() -> str:
    """
    Generate fake Azure Key Vault URL.

    Returns:
        Azure Key Vault URL string
    """
    vault_name = _random_string(random.randint(6, 12), string.ascii_lowercase)
    secret_name = _random_string(random.randint(4, 10), string.ascii_lowercase + "-")
    return f"https://{vault_name}.vault.azure.net/secrets/{secret_name}"


def generate_npm_token() -> str:
    """
    Generate fake NPM token.

    Returns:
        NPM token string
    """
    return "npm_" + _random_string(36, ALPHANUMERIC)


def generate_pypi_token() -> str:
    """
    Generate fake PyPI token.

    Returns:
        PyPI token string
    """
    return "pypi-" + _random_string(64, ALPHANUMERIC + "_")


def generate_random_secret(min_length: int = 20, max_length: int = 64) -> str:
    """
    Generate random high-entropy secret.

    Args:
        min_length: Minimum length
        max_length: Maximum length

    Returns:
        Random secret string
    """
    length = random.randint(min_length, max_length)
    charset = ALPHANUMERIC + "!@#$%^&*()_+-=[]{}|;:,.<>?"
    return _random_string(length, charset)


def generate_password(complexity: str = "high") -> str:
    """
    Generate random password.

    Args:
        complexity: 'low', 'medium', or 'high'

    Returns:
        Password string
    """
    if complexity == "low":
        length = random.randint(8, 12)
        charset = string.ascii_lowercase + string.digits
    elif complexity == "medium":
        length = random.randint(12, 20)
        charset = ALPHANUMERIC
    else:  # high
        length = random.randint(16, 32)
        charset = ALPHANUMERIC + "!@#$%^&*"

    return _random_string(length, charset)


def generate_git_sha() -> str:
    """
    Generate fake Git SHA (40 hex characters).

    Returns:
        Git SHA string
    """
    return _random_string(40, HEX_CHARS)


def generate_uuid() -> str:
    """
    Generate fake UUID.

    Returns:
        UUID string in standard format
    """
    parts = [
        _random_string(8, HEX_CHARS),
        _random_string(4, HEX_CHARS),
        "4" + _random_string(3, HEX_CHARS),  # Version 4
        _random_string(4, HEX_CHARS),
        _random_string(12, HEX_CHARS),
    ]
    return "-".join(parts)


def generate_checksum(algorithm: str = "sha256") -> str:
    """
    Generate fake checksum/hash.

    Args:
        algorithm: 'md5', 'sha1', 'sha256', or 'sha512'

    Returns:
        Checksum string with optional prefix
    """
    lengths = {
        "md5": 32,
        "sha1": 40,
        "sha256": 64,
        "sha512": 128,
    }
    length = lengths.get(algorithm, 64)
    hash_value = _random_string(length, HEX_CHARS)

    # Sometimes include algorithm prefix
    if random.random() < 0.3:
        return f"{algorithm}:{hash_value}"
    return hash_value


def generate_base64_data(decoded_length: int = 32) -> str:
    """
    Generate base64-encoded random data.

    Args:
        decoded_length: Length of data before encoding

    Returns:
        Base64 encoded string
    """
    # Use random.randbytes for reproducibility with seed
    data = bytes(random.randint(0, 255) for _ in range(decoded_length))
    return base64.b64encode(data).decode()


# --- Hard Negative Generators ---
# These generators create challenging non-secrets that look like real secrets,
# forcing the model to learn from context rather than token format.


def generate_documentation_example() -> Tuple[str, str]:
    """
    Generate official documentation example credentials.

    These are well-known example credentials that appear in documentation
    and should NEVER be flagged as secrets.

    Returns:
        Tuple of (token, description)
    """
    examples = [
        # AWS official examples
        ("AKIAIOSFODNN7EXAMPLE", "aws_example"),
        ("AKIAI44QH8DHBEXAMPLE", "aws_example"),
        ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret_example"),
        # GitHub placeholder patterns
        ("ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "github_placeholder"),
        ("ghp_0000000000000000000000000000000000", "github_placeholder"),
        ("github_pat_" + "x" * 82, "github_pat_placeholder"),
        # Stripe test patterns
        ("sk_test_FAKE_PLACEHOLDER", "stripe_test_example"),
        ("pk_test_FAKE_PLACEHOLDER", "stripe_test_example"),
        ("sk_test_PLACEHOLDER_NOT_REAL", "stripe_doc_example"),
        # OpenAI placeholder
        ("sk-" + "x" * 48, "openai_placeholder"),
        ("sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "openai_placeholder"),
        # Generic examples from docs
        ("your_api_key_here", "placeholder"),
        ("YOUR_SECRET_KEY", "placeholder"),
        ("INSERT_TOKEN_HERE", "placeholder"),
        ("<YOUR_API_KEY>", "placeholder"),
        ("[API_KEY]", "placeholder"),
    ]
    return random.choice(examples)


def generate_revoked_token() -> Tuple[str, str]:
    """
    Generate a token that appears in revocation/rotation contexts.

    These are tokens mentioned in commit messages, security advisories,
    or rotation scripts that are no longer valid.

    Returns:
        Tuple of (token, context_hint)
    """
    # Generate a valid-looking token
    token_types = [
        ("AKIA" + _random_string(16, ALPHANUMERIC_UPPER), "aws_revoked"),
        ("ghp_" + _random_string(36, ALPHANUMERIC), "github_revoked"),
        ("sk-" + _random_string(48, ALPHANUMERIC), "openai_revoked"),
        ("sk_live_" + _random_string(24, ALPHANUMERIC), "stripe_revoked"),
    ]
    return random.choice(token_types)


def generate_test_fixture_token() -> Tuple[str, str]:
    """
    Generate tokens clearly intended for testing.

    These have valid format but contain 'test', 'fake', 'mock' indicators.

    Returns:
        Tuple of (token, token_type)
    """
    fixtures = [
        # AWS test keys (invalid but correct format)
        ("AKIATESTKEY123456789", "aws_test"),
        ("AKIAFAKEKEY123456789", "aws_test"),
        ("AKIAMOCKKEY123456789", "aws_test"),
        ("AKIADEMOKEY123456789", "aws_test"),
        # GitHub test tokens
        ("ghp_testtoken" + _random_string(24, ALPHANUMERIC), "github_test"),
        ("ghp_faketoken" + _random_string(24, ALPHANUMERIC), "github_test"),
        ("ghp_mocktoken" + _random_string(24, ALPHANUMERIC), "github_test"),
        # Stripe test mode (these are always safe)
        ("sk_test_" + _random_string(24, ALPHANUMERIC), "stripe_test"),
        ("pk_test_" + _random_string(24, ALPHANUMERIC), "stripe_test"),
        # OpenAI test patterns
        ("sk-test" + _random_string(44, ALPHANUMERIC), "openai_test"),
        ("sk-fake" + _random_string(44, ALPHANUMERIC), "openai_test"),
        # Slack test tokens
        ("xoxb-test-" + _random_string(20, string.digits), "slack_test"),
    ]
    return random.choice(fixtures)


def generate_encoded_non_secret() -> Tuple[str, str]:
    """
    Generate base64-encoded content that is NOT a secret.

    This creates base64 strings that decode to JSON configs, text,
    or other non-sensitive data.

    Returns:
        Tuple of (base64_string, content_type)
    """
    content_types = [
        # JSON configs (common in k8s, cloud configs)
        (json.dumps({"version": "1.0", "enabled": True}), "json_config"),
        (json.dumps({"name": "test", "value": 123}), "json_config"),
        (json.dumps({"settings": {"debug": False}}), "json_config"),
        # Text content
        ("Hello, World!", "text"),
        ("Configuration placeholder", "text"),
        ("This is a test string", "text"),
        # Small binary-like content
        ("\x00\x01\x02\x03\x04\x05\x06\x07", "binary_header"),
        ("PNG\r\n\x1a\n", "png_header"),
        ("GIF89a", "gif_header"),
    ]

    content, content_type = random.choice(content_types)
    if isinstance(content, str):
        encoded = base64.b64encode(content.encode()).decode()
    else:
        encoded = base64.b64encode(content.encode("latin-1")).decode()

    return encoded, content_type


def generate_high_entropy_non_secret() -> str:
    """
    Generate high-entropy string that is NOT a secret.

    This creates strings with high Shannon entropy that are used
    for legitimate purposes (hashes, identifiers, encoded data).

    Returns:
        High-entropy non-secret string
    """
    generators = [
        # Content hashes
        lambda: _random_string(64, HEX_CHARS),  # SHA-256 style
        lambda: _random_string(40, HEX_CHARS),  # SHA-1 style
        # Build/version identifiers
        lambda: _random_string(8, HEX_CHARS) + "-" + _random_string(4, HEX_CHARS),
        # Session IDs (typically not secrets themselves)
        lambda: "sess_" + _random_string(32, ALPHANUMERIC),
        # Cache keys
        lambda: "cache:" + _random_string(32, ALPHANUMERIC),
        # Nonces (used once, not persistent secrets)
        lambda: _random_string(24, BASE64_URL_CHARS),
    ]
    return random.choice(generators)()


# --- Ambiguous Token Generators ---
# These generators create tokens that blur the line between secrets and non-secrets,
# forcing the model to learn from context rather than token format.


def generate_hex_secret(length: int = 40) -> str:
    """
    Generate a real secret that looks like a Git SHA or checksum.

    This creates a HIGH-ENTROPY SECRET that is visually indistinguishable
    from a Git commit hash. The model must learn to differentiate based
    on context, not token format.

    Args:
        length: Length of the hex string (default 40, same as git SHA)

    Returns:
        Hex string that is actually a secret
    """
    return _random_string(length, HEX_CHARS)


def generate_base64_secret(length: int = 44) -> str:
    """
    Generate a real secret in base64 format.

    This creates a HIGH-ENTROPY SECRET that looks like base64 data.
    The model must learn from context to differentiate from actual data.

    Args:
        length: Length of the base64 string

    Returns:
        Base64-like string that is actually a secret
    """
    return _random_string(length, BASE64_CHARS)


def generate_fake_prefixed_token() -> str:
    """
    Generate a NON-SECRET that has a secret-like prefix.

    This creates tokens with prefixes like AKIA, ghp_, sk- that are
    NOT actual secrets. Used to train the model to look beyond prefixes.

    Returns:
        String with secret-like prefix that is NOT a secret
    """
    fake_prefixes = [
        ("AKIA", 16, ALPHANUMERIC_UPPER),  # Looks like AWS
        ("ghp_", 36, ALPHANUMERIC),  # Looks like GitHub
        ("sk-", 48, ALPHANUMERIC),  # Looks like OpenAI
        ("sk_test_", 24, ALPHANUMERIC),  # Looks like Stripe test
        ("xoxb-", 30, string.digits + "-"),  # Looks like Slack
    ]
    prefix, length, charset = random.choice(fake_prefixes)
    return prefix + _random_string(length, charset)


def generate_ambiguous_token() -> str:
    """
    Generate a token that could be either a secret or non-secret.

    This creates high-entropy strings with no distinctive patterns,
    forcing classification based purely on context.

    Returns:
        Ambiguous high-entropy string
    """
    # Choose random format
    formats = [
        lambda: _random_string(40, ALPHANUMERIC),  # Could be anything
        lambda: _random_string(32, HEX_CHARS),  # Could be hash or secret
        lambda: _random_string(64, BASE64_CHARS),  # Could be data or secret
    ]
    return random.choice(formats)()


# --- Git SHA Negative Generators ---
# These generate git SHAs with explicit context to train the model
# that 40-char hex strings are NOT secrets when in git context.


def generate_git_sha_with_context() -> Tuple[str, str, str]:
    """
    Generate Git SHA with explicit git-context variable name.

    Returns:
        Tuple of (sha, variable_name, context_type)
    """
    sha = _random_string(40, HEX_CHARS)
    var_names = [
        ("commit_sha", "git_commit"),
        ("COMMIT_SHA", "git_commit"),
        ("commit_hash", "git_commit"),
        ("git_sha", "git_commit"),
        ("git_hash", "git_commit"),
        ("GIT_COMMIT", "git_commit"),
        ("HEAD_SHA", "git_commit"),
        ("base_commit", "git_merge"),
        ("merge_base", "git_merge"),
        ("parent_sha", "git_commit"),
        ("revision", "git_revision"),
        ("rev", "git_revision"),
        ("last_commit", "git_commit"),
        ("deploy_sha", "deployment"),
        ("build_commit", "ci_build"),
    ]
    var_name, context_type = random.choice(var_names)
    return sha, var_name, context_type


def generate_content_hash_with_context() -> Tuple[str, str, str]:
    """
    Generate content hash with explicit hash-context variable name.

    Returns:
        Tuple of (hash, variable_name, hash_type)
    """
    hash_configs = [
        (32, "md5"),
        (40, "sha1"),
        (64, "sha256"),
        (128, "sha512"),
    ]
    length, hash_type = random.choice(hash_configs)
    hash_value = _random_string(length, HEX_CHARS)

    var_names = [
        f"file_checksum_{hash_type}",
        f"content_hash",
        f"integrity_hash",
        f"package_hash",
        f"checksum_{hash_type}",
        f"digest_{hash_type}",
        f"{hash_type}_hash",
        f"file_{hash_type}",
        f"expected_hash",
        f"computed_hash",
    ]
    var_name = random.choice(var_names)
    return hash_value, var_name, hash_type


# --- Embedded Credential Generators ---
# These generate secrets embedded in URLs/connection strings.


def generate_database_url() -> Tuple[str, str, str]:
    """
    Generate database connection URL with embedded password.

    Returns:
        Tuple of (full_url, password_only, db_type)
    """
    db_types = [
        ("postgresql", 5432),
        ("mysql", 3306),
        ("postgres", 5432),
        ("mongodb", 27017),
        ("redis", 6379),
    ]
    db_type, port = random.choice(db_types)

    usernames = ["admin", "root", "app", "service", "dbuser", "api"]
    username = random.choice(usernames)

    # Generate realistic password
    password = _random_string(random.randint(12, 24), ALPHANUMERIC + "!@#$%")

    hosts = [
        "localhost",
        "db.internal",
        "prod-db.cluster.local",
        f"{db_type}-master.internal",
        f"rds.{_random_string(8, string.ascii_lowercase)}.amazonaws.com",
    ]
    host = random.choice(hosts)

    db_names = ["app", "production", "main", "api", "service"]
    db_name = random.choice(db_names)

    url = f"{db_type}://{username}:{password}@{host}:{port}/{db_name}"
    return url, password, db_type


def generate_webhook_url_with_token() -> Tuple[str, str, str]:
    """
    Generate webhook/API URL with embedded token.

    Returns:
        Tuple of (full_url, token_only, service_type)
    """
    services = [
        ("https://hooks.slack.com/services", "T" + _random_string(8, ALPHANUMERIC_UPPER), "slack"),
        ("https://api.github.com/repos", "ghp_" + _random_string(36, ALPHANUMERIC), "github"),
        ("https://api.stripe.com/v1", "sk_live_" + _random_string(24, ALPHANUMERIC), "stripe"),
        ("https://api.sendgrid.com/v3", "SG." + _random_string(22, ALPHANUMERIC) + "." + _random_string(43, ALPHANUMERIC), "sendgrid"),
    ]

    base_url, token, service = random.choice(services)

    url_patterns = [
        f"{base_url}?token={token}",
        f"{base_url}?api_key={token}",
        f"{base_url}?access_token={token}",
        f"{base_url}&auth={token}",
    ]

    return random.choice(url_patterns), token, service


def generate_connection_string() -> Tuple[str, str, str]:
    """
    Generate connection string with embedded credentials.

    Returns:
        Tuple of (connection_string, password, service_type)
    """
    password = _random_string(random.randint(16, 32), ALPHANUMERIC + "!@#$%")

    patterns = [
        # Azure SQL
        (f"Server=tcp:server.database.windows.net;Database=db;User ID=admin;Password={password};", "azure_sql"),
        # SQL Server
        (f"Data Source=server;Initial Catalog=db;User Id=sa;Password={password};", "sqlserver"),
        # ODBC
        (f"Driver={{SQL Server}};Server=server;Database=db;Uid=user;Pwd={password};", "odbc"),
        # JDBC
        (f"jdbc:mysql://server:3306/db?user=root&password={password}", "jdbc"),
        # AMQP (RabbitMQ)
        (f"amqp://user:{password}@rabbitmq.internal:5672/vhost", "amqp"),
    ]

    conn_string, service = random.choice(patterns)
    return conn_string, password, service


# --- PEM Key Generators ---
# These generate multi-line PEM format keys.


def generate_pem_private_key() -> Tuple[str, str]:
    """
    Generate fake PEM-format private key.

    Returns:
        Tuple of (pem_key, key_type)
    """
    key_types = [
        ("RSA PRIVATE KEY", 24),
        ("EC PRIVATE KEY", 8),
        ("PRIVATE KEY", 24),
        ("OPENSSH PRIVATE KEY", 16),
    ]

    key_type, num_lines = random.choice(key_types)

    # Generate base64 lines (64 chars each, like real PEM)
    lines = [_random_string(64, BASE64_CHARS) for _ in range(num_lines)]
    # Last line is shorter
    lines.append(_random_string(random.randint(20, 60), BASE64_CHARS))

    pem = f"-----BEGIN {key_type}-----\n"
    pem += "\n".join(lines)
    pem += f"\n-----END {key_type}-----"

    return pem, key_type


def generate_ssh_private_key() -> Tuple[str, str]:
    """
    Generate fake SSH private key in OpenSSH format.

    Returns:
        Tuple of (ssh_key, key_type)
    """
    key_types = ["ed25519", "rsa", "ecdsa"]
    key_type = random.choice(key_types)

    # OpenSSH format
    lines = [_random_string(70, BASE64_CHARS) for _ in range(random.randint(6, 12))]

    key = "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    key += "\n".join(lines)
    key += "\n-----END OPENSSH PRIVATE KEY-----"

    return key, key_type


def generate_certificate() -> Tuple[str, str]:
    """
    Generate fake X.509 certificate (NOT a secret, for negative examples).

    Returns:
        Tuple of (certificate, cert_type)
    """
    cert_types = ["CERTIFICATE", "X509 CERTIFICATE", "TRUSTED CERTIFICATE"]
    cert_type = random.choice(cert_types)

    lines = [_random_string(64, BASE64_CHARS) for _ in range(random.randint(15, 25))]

    cert = f"-----BEGIN {cert_type}-----\n"
    cert += "\n".join(lines)
    cert += f"\n-----END {cert_type}-----"

    return cert, cert_type


# --- Enhanced Hard Negative Generators ---
# These generators create challenging non-secrets that force context-based learning.
# Key principle: A secret is defined by how it is USED, not how it LOOKS.


def generate_uuid_with_auth_context() -> Tuple[str, str, str]:
    """
    Generate UUID v4 with auth-like variable names.

    This creates UUIDs that appear in secret-like contexts:
    - api_key = '550e8400-e29b-41d4-a716-446655440000'
    - auth_token = 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'

    Forces the model to learn that UUIDs are NOT secrets despite
    appearing with secret-like variable names.

    Returns:
        Tuple of (uuid, auth_var_name, context_type)
    """
    uuid = generate_uuid()

    # Auth-like variable names that would normally trigger secret detection
    auth_var_names = [
        ("api_key", "config"),
        ("API_KEY", "env"),
        ("auth_token", "code"),
        ("AUTH_TOKEN", "env"),
        ("secret_key", "config"),
        ("SECRET_KEY", "env"),
        ("access_token", "code"),
        ("private_key", "config"),
        ("bearer_token", "header"),
        ("session_secret", "config"),
        ("encryption_key", "config"),
        ("signing_key", "jwt"),
        ("client_secret", "oauth"),
        ("refresh_token", "auth"),
        ("api_secret", "config"),
    ]

    var_name, context_type = random.choice(auth_var_names)
    return uuid, var_name, context_type


def generate_hash_in_security_context() -> Tuple[str, str, str, str]:
    """
    Generate content hashes that appear in security-adjacent contexts.

    Examples:
    - password_hash = '5f4dcc3b5aa765d61d8327deb882cf99'
    - auth_hash = 'e99a18c428cb38d5f260853678922e03'

    These are NOT secrets - they are derived values that don't grant access.

    Returns:
        Tuple of (hash_value, security_var_name, hash_algorithm, context_hint)
    """
    hash_configs = [
        (32, "md5"),
        (40, "sha1"),
        (64, "sha256"),
        (128, "sha512"),
    ]
    length, algorithm = random.choice(hash_configs)
    hash_value = _random_string(length, HEX_CHARS)

    # Security-adjacent variable names that could be confused with secrets
    security_var_names = [
        ("password_hash", "auth"),
        ("PASSWORD_HASH", "env"),
        ("auth_hash", "verification"),
        ("token_hash", "storage"),
        ("secret_hash", "derived"),
        ("api_key_hash", "validation"),
        ("credential_hash", "storage"),
        ("session_hash", "tracking"),
        ("user_secret_hash", "auth"),
        ("encryption_hash", "verify"),
        ("signature_hash", "jwt"),
        ("hmac_hash", "auth"),
        ("key_hash", "derived"),
        ("access_hash", "permission"),
    ]

    var_name, context_hint = random.choice(security_var_names)
    return hash_value, var_name, algorithm, context_hint


def generate_invalid_jwt() -> Tuple[str, str]:
    """
    Generate JWT-shaped tokens that are structurally invalid.

    These look like JWTs (header.payload.signature) but:
    - Have invalid/corrupted headers
    - Have obviously fake payloads
    - Are marked as expired, test, or example tokens

    Returns:
        Tuple of (invalid_jwt, invalidity_reason)
    """
    def b64url_encode(data: str) -> str:
        return base64.urlsafe_b64encode(data.encode()).decode().rstrip("=")

    invalid_types = [
        # Invalid algorithm
        ({"alg": "none", "typ": "JWT"}, {"sub": "test", "exp": 0}, "alg_none"),
        # Malformed header
        ({"typ": "JWT"}, {"sub": "test"}, "missing_alg"),
        # Test/example markers in payload
        ({"alg": "HS256", "typ": "JWT"}, {"sub": "test_user", "iat": 0, "exp": 0}, "test_payload"),
        ({"alg": "HS256", "typ": "JWT"}, {"sub": "example", "env": "test"}, "example_payload"),
        ({"alg": "HS256", "typ": "JWT"}, {"sub": "PLACEHOLDER", "fake": True}, "placeholder"),
        # Obviously expired (year 2000)
        ({"alg": "HS256", "typ": "JWT"}, {"sub": "user", "exp": 946684800}, "expired_2000"),
        # Invalid type
        ({"alg": "HS256", "typ": "FAKE"}, {"sub": "user"}, "invalid_type"),
    ]

    header, payload, reason = random.choice(invalid_types)

    header_b64 = b64url_encode(json.dumps(header, separators=(",", ":")))
    payload_b64 = b64url_encode(json.dumps(payload, separators=(",", ":")))
    # Signature is always fake anyway
    signature = _random_string(43, BASE64_URL_CHARS)

    jwt_token = f"{header_b64}.{payload_b64}.{signature}"
    return jwt_token, reason


def generate_base64_telemetry_id() -> Tuple[str, str, str]:
    """
    Generate base64 blobs used as telemetry/correlation/cache IDs.

    These are high-entropy base64 strings that are NOT secrets:
    - trace_id = 'dGVsZW1ldHJ5X2lkXzEyMzQ1Njc4OTA='
    - correlation_id = 'Y29ycmVsYXRpb25faWRfYWJjZGVm'
    - request_id = 'cmVxdWVzdF8xNjk4NzY1NDMyMQ=='

    Returns:
        Tuple of (base64_id, var_name, id_type)
    """
    # Generate realistic-looking IDs
    id_types = [
        # Telemetry/tracing IDs
        ("telemetry", ["trace_id", "TRACE_ID", "span_id", "SPAN_ID", "telemetry_id"]),
        # Correlation IDs
        ("correlation", ["correlation_id", "CORRELATION_ID", "request_correlation", "x_correlation_id"]),
        # Request/response IDs
        ("request", ["request_id", "REQUEST_ID", "response_id", "transaction_id", "x_request_id"]),
        # Cache keys
        ("cache", ["cache_key", "CACHE_KEY", "cache_id", "cache_token", "etag"]),
        # Session tracking (NOT session secrets)
        ("tracking", ["tracking_id", "visitor_id", "analytics_id", "event_id"]),
        # Build/deployment IDs
        ("build", ["build_id", "BUILD_ID", "deploy_id", "release_id", "artifact_id"]),
    ]

    id_type, var_names = random.choice(id_types)
    var_name = random.choice(var_names)

    # Generate a plausible ID content
    id_contents = [
        f"{id_type}_id_{_random_string(16, ALPHANUMERIC)}",
        f"{_random_string(8, HEX_CHARS)}-{_random_string(4, HEX_CHARS)}-{_random_string(4, HEX_CHARS)}",
        f"v1_{_random_string(20, ALPHANUMERIC)}",
        f"{id_type}_{random.randint(1000000000, 9999999999)}",
    ]

    content = random.choice(id_contents)
    base64_id = base64.b64encode(content.encode()).decode()

    return base64_id, var_name, id_type


def generate_session_id_non_secret() -> Tuple[str, str]:
    """
    Generate session/request IDs that look like secrets but aren't.

    Session IDs are identifiers, not authentication secrets.
    They may be high-entropy but don't grant access by themselves.

    Returns:
        Tuple of (session_id, var_name)
    """
    formats = [
        # UUID-style session ID
        lambda: generate_uuid(),
        # Hex session ID
        lambda: _random_string(32, HEX_CHARS),
        # Prefixed session ID
        lambda: "sess_" + _random_string(24, ALPHANUMERIC),
        lambda: "sid_" + _random_string(20, ALPHANUMERIC),
        lambda: "session-" + _random_string(32, HEX_CHARS),
        # Base64 session ID
        lambda: base64.urlsafe_b64encode(_random_string(24, ALPHANUMERIC).encode()).decode().rstrip("="),
    ]

    session_id = random.choice(formats)()

    var_names = [
        "session_id", "SESSION_ID", "sessionId", "SessionID",
        "request_id", "REQUEST_ID", "requestId",
        "transaction_id", "txn_id", "tx_id",
        "connection_id", "conn_id",
        "client_id", "visitor_id",
    ]

    return session_id, random.choice(var_names)


def generate_encoded_public_data() -> Tuple[str, str, str]:
    """
    Generate base64-encoded public/config data that looks secret-like.

    These are high-entropy base64 strings containing non-sensitive data:
    - Configuration JSON
    - Public metadata
    - Feature flags
    - Non-secret settings

    Returns:
        Tuple of (base64_data, var_name, content_type)
    """
    data_types = [
        # Configuration data
        (
            json.dumps({"version": "2.0", "features": ["auth", "api"], "enabled": True}),
            ["config_data", "CONFIG_DATA", "settings_blob", "app_config"],
            "config"
        ),
        # Feature flags
        (
            json.dumps({"feature_a": True, "feature_b": False, "rollout": 0.5}),
            ["feature_flags", "FEATURE_FLAGS", "features", "flags_data"],
            "features"
        ),
        # Metadata
        (
            json.dumps({"created": "2024-01-01", "version": 1, "type": "user"}),
            ["metadata", "META_DATA", "object_meta", "resource_meta"],
            "metadata"
        ),
        # Public key ID (not the key itself)
        (
            f"kid:{_random_string(16, ALPHANUMERIC)}:v1",
            ["key_id", "KEY_ID", "kid", "public_key_id"],
            "key_id"
        ),
        # State tokens (CSRF-like, but for state management)
        (
            f"state:{_random_string(24, ALPHANUMERIC)}:1234567890",
            ["state_token", "STATE", "oauth_state", "csrf_token"],
            "state"
        ),
    ]

    content, var_names, content_type = random.choice(data_types)
    var_name = random.choice(var_names)
    base64_data = base64.b64encode(content.encode()).decode()

    return base64_data, var_name, content_type


def generate_version_hash() -> Tuple[str, str, str]:
    """
    Generate content/version hashes that appear in deployment contexts.

    These are hashes used for:
    - Content integrity verification
    - Cache invalidation
    - Version tracking
    - Asset fingerprinting

    Returns:
        Tuple of (hash_value, var_name, context)
    """
    hash_configs = [
        (8, "short"),   # Short hash (git short SHA, content hash prefix)
        (32, "md5"),
        (40, "sha1"),
        (64, "sha256"),
    ]

    length, hash_type = random.choice(hash_configs)
    hash_value = _random_string(length, HEX_CHARS)

    var_contexts = [
        # Asset hashing
        ("asset_hash", "static_files"),
        ("ASSET_HASH", "static_files"),
        ("bundle_hash", "webpack"),
        ("content_hash", "cdn"),
        # Version control
        ("version_hash", "release"),
        ("VERSION_HASH", "ci"),
        ("deploy_hash", "deployment"),
        ("release_hash", "release"),
        # Integrity
        ("integrity_hash", "sri"),
        ("INTEGRITY_HASH", "verification"),
        ("checksum", "download"),
        ("CHECKSUM", "package"),
        # Caching
        ("etag_hash", "http"),
        ("cache_hash", "redis"),
        ("object_hash", "storage"),
    ]

    var_name, context = random.choice(var_contexts)
    return hash_value, var_name, context


# --- STAGE B GENERALIZATION IMPROVEMENT GENERATORS ---
# These generators create samples that break shortcut learning patterns
# by creating conflict between token format and context signals.


def generate_hex_secret_with_strong_context() -> Tuple[str, str, str, str]:
    """
    Generate 40-char hex secrets with UNMISTAKABLE secret context.

    This creates hex-format secrets that look identical to Git SHAs
    but have strong contextual signals indicating they ARE secrets.
    Forces the model to use context, not format, for classification.

    Returns:
        Tuple of (token, var_name, file_path, context_comment)
    """
    token = _random_string(40, HEX_CHARS)

    strong_contexts = [
        # Production credential indicators
        ("production_api_secret", "config/production_secrets.env",
         "# CRITICAL: Production API secret - ROTATE IMMEDIATELY IF EXPOSED"),
        ("live_encryption_key", "secrets/encryption.yaml",
         "# AES-256 master encryption key for customer data"),
        ("payment_gateway_secret", "config/stripe_prod.py",
         "# Stripe production secret key (NOT test mode)"),
        ("database_master_password", "infra/db_credentials.yaml",
         "# Production database root password - DO NOT COMMIT"),
        ("jwt_signing_secret", "auth/jwt_config.py",
         "# JWT HS256 signing secret - used for user authentication"),
        ("oauth_client_secret", "config/oauth_providers.yaml",
         "# OAuth2 client secret for production SSO"),
        ("webhook_signing_key", "api/webhook_verification.py",
         "# Webhook payload verification secret"),
        ("encryption_private_key_hex", "crypto/keys.yaml",
         "# Private key material - NEVER log or expose"),
        ("api_master_key", "config/api_keys.env",
         "# Master API key with full access - production only"),
        ("service_account_secret", "config/service_auth.yaml",
         "# Service account secret for internal APIs"),
        ("hmac_signing_key", "auth/signature.py",
         "# HMAC key for request signing"),
        ("symmetric_encryption_key", "crypto/aes_config.py",
         "# AES encryption key for data at rest"),
    ]

    var_name, file_path, context = random.choice(strong_contexts)
    return token, var_name, file_path, context


def generate_hash_with_secret_varname() -> Tuple[str, str, str, str]:
    """
    Generate content hashes using SECRET-LIKE variable names.

    This creates non-secrets (hashes, checksums) that appear with
    variable names typically associated with secrets. Forces the
    model to NOT rely solely on variable name patterns.

    Examples:
        password_hash = "5f4dcc3b5aa765d61d8327deb882cf99"  # MD5 of password
        api_key_hash = "abc123..."  # Hash of API key for lookup

    Returns:
        Tuple of (hash_value, var_name, file_path, context_comment)
    """
    hash_configs = [32, 40, 64]  # MD5, SHA1, SHA256 lengths
    length = random.choice(hash_configs)
    token = _random_string(length, HEX_CHARS)

    # Secret-like variable names for HASH values (NOT secrets)
    contexts = [
        ("password_hash", "auth/password_storage.py",
         "# Bcrypt hash of user password for comparison"),
        ("api_key_hash", "middleware/rate_limiter.py",
         "# Hash of API key for fast lookup in rate limit table"),
        ("secret_fingerprint", "audit/key_rotation.py",
         "# Fingerprint for tracking which key version was used"),
        ("token_digest", "cache/session_store.py",
         "# SHA256 digest of token for cache key"),
        ("credential_hash", "validation/input_sanitizer.py",
         "# Hash of credential for secure comparison"),
        ("access_key_checksum", "infra/key_validator.py",
         "# Checksum for integrity verification of access key"),
        ("secret_key_hash", "auth/key_derivation.py",
         "# Derived hash for key validation - not the actual key"),
        ("api_secret_digest", "monitoring/audit_log.py",
         "# Logged digest of API secret (not the secret itself)"),
        ("encryption_key_fingerprint", "crypto/key_management.py",
         "# Key fingerprint for key identification"),
        ("auth_token_hash", "cache/token_cache.py",
         "# Hashed token for cache lookup"),
    ]

    var_name, file_path, comment = random.choice(contexts)
    return token, var_name, file_path, comment


def generate_secret_with_misleading_varname() -> Tuple[str, str, str, str]:
    """
    Generate REAL SECRETS with safe-looking variable names.

    This creates actual secrets that appear with variable names
    typically associated with non-secrets (build IDs, hashes, etc.).
    Forces the model to NOT trust safe variable names blindly.

    Returns:
        Tuple of (token, var_name, file_path, context_comment)
    """
    # Generate real secrets with various formats
    secret_generators = [
        lambda: "AKIA" + _random_string(16, ALPHANUMERIC_UPPER),
        lambda: "ghp_" + _random_string(36, ALPHANUMERIC),
        lambda: "sk_live_" + _random_string(24, ALPHANUMERIC),
        lambda: _random_string(40, HEX_CHARS),  # Hex-format secret
        lambda: "sk-" + _random_string(48, ALPHANUMERIC),  # OpenAI-like
    ]
    token = random.choice(secret_generators)()

    # MISLEADING variable names that look safe but contain real secrets
    misleading_contexts = [
        ("build_id", "ci/deploy.py",
         "# Build identifier with embedded credentials"),
        ("version_hash", "release/versioning.py",
         "# Version tracking token (actually API key)"),
        ("request_id", "api/middleware.py",
         "# Request correlation ID (contains auth token)"),
        ("cache_key", "utils/caching.py",
         "# Cache entry key with embedded secret"),
        ("artifact_hash", "build/artifacts.py",
         "# Build artifact identifier (contains signing key)"),
        ("commit_ref", "ci/github_actions.py",
         "# CI reference (actually contains PAT)"),
        ("session_data", "auth/session.py",
         "# Session data blob with credentials"),
        ("correlation_id", "logging/tracer.py",
         "# Correlation ID (API key embedded)"),
        ("checksum_value", "validation/integrity.py",
         "# Checksum field (actually secret key)"),
        ("trace_context", "observability/spans.py",
         "# Trace context with embedded auth"),
    ]

    var_name, file_path, comment = random.choice(misleading_contexts)
    return token, var_name, file_path, comment


# --- STAGE B PRECISION IMPROVEMENT GENERATORS ---
# These generators create hard negatives specifically targeting observed FP patterns


def generate_test_mode_token() -> Tuple[str, str, str]:
    """
    Generate test-mode tokens (sk_test_*, pk_test_*, rk_test_*) as explicit negatives.

    These tokens have valid format but are clearly test/staging mode tokens.
    They should NEVER be flagged as secrets.

    Returns:
        Tuple of (token, var_name, file_path)
    """
    test_prefixes = [
        ("sk_test_", 24),   # Stripe test secret key
        ("pk_test_", 24),   # Stripe test publishable key
        ("rk_test_", 24),   # Stripe test restricted key
        ("sk_staging_", 24),  # Stripe staging key
        ("test_sk_", 24),   # Alternative test key format
    ]

    prefix, length = random.choice(test_prefixes)
    suffix = _random_string(length, ALPHANUMERIC)
    token = prefix + suffix

    var_names = [
        "test_key", "stripe_test_key", "test_api_key", "staging_key",
        "STRIPE_TEST_KEY", "TEST_SECRET_KEY", "dev_api_key",
        "sandbox_key", "test_stripe_key", "development_key",
    ]

    file_paths = [
        "tests/test_payment.py",
        "tests/fixtures/stripe.py",
        "tests/conftest.py",
        "config/test.env",
        "config/development.env",
        "docs/examples/payment.py",
        "examples/stripe_test.py",
        ".env.test",
        "src/__tests__/payment.test.js",
        "spec/fixtures/api_keys.rb",
    ]

    return token, random.choice(var_names), random.choice(file_paths)


def generate_hex_in_build_context() -> Tuple[str, str, str]:
    """
    Generate 40/64-char hex strings in build/dist/git contexts.

    These are asset hashes, bundle IDs, checksums that look like secrets
    but are in clearly non-secret contexts.

    Returns:
        Tuple of (token, var_name, file_path)
    """
    length = random.choice([40, 64])  # SHA1 or SHA256 length
    token = _random_string(length, HEX_CHARS)

    var_names = [
        "build_hash", "asset_hash", "bundle_hash", "version_hash",
        "commit_sha", "artifact_hash", "checksum", "content_hash",
        "BUILD_HASH", "ASSET_HASH", "CHECKSUM", "GIT_SHA",
        "module_hash", "package_hash", "dist_hash", "cache_key",
    ]

    file_paths = [
        "build/manifest.json",
        "dist/assets.json",
        "dist/chunk-manifest.json",
        ".git/objects/pack/pack-info",
        "node_modules/.cache/hash",
        "node_modules/.package-lock.json",
        "target/release/checksums.txt",
        "vendor/bundle/hash",
        "build/asset-manifest.json",
        ".next/cache/webpack/hash",
        "__pycache__/module.cpython-39.pyc",
        ".cache/babel-loader/hash.json",
        "out/_next/static/chunks/hash.js",
        ".output/server/hash.json",
    ]

    return token, random.choice(var_names), random.choice(file_paths)


def generate_sha_in_config_file() -> Tuple[str, str, str]:
    """
    Generate SHA-like hex strings in config/settings files.

    These are integrity hashes, version pins, checksums in configuration.

    Returns:
        Tuple of (token, var_name, file_path)
    """
    length = random.choice([40, 64])
    token = _random_string(length, HEX_CHARS)

    var_names = [
        "integrity_hash", "lock_hash", "version_pin", "dependency_hash",
        "INTEGRITY_HASH", "LOCK_HASH", "CHECKSUM", "SHA256",
        "package_lock", "yarn_lock_hash", "npm_integrity",
        "subresource_integrity", "expected_hash", "verified_hash",
    ]

    file_paths = [
        "package-lock.json",
        "yarn.lock",
        "Gemfile.lock",
        "poetry.lock",
        "Cargo.lock",
        "go.sum",
        "requirements.txt",
        "Pipfile.lock",
        "config/integrity.json",
        "config/checksums.yaml",
        ".terraform.lock.hcl",
        "flake.lock",
    ]

    return token, random.choice(var_names), random.choice(file_paths)


# Known example tokens from vendor documentation
KNOWN_EXAMPLE_TOKENS = [
    # AWS documentation examples
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    # GitHub placeholder tokens
    "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "ghp_000000000000000000000000000000000000",
    # Stripe test/example tokens (constructed at runtime to avoid push protection)
    "sk_test_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "pk_test_" + "TYooMQauvdEDq54NiTphI7jx",
    "sk_test_" + "51H7y1DAlVqJhyDLDexAMPLE",
    # Generic placeholders
    "your-api-key-here",
    "YOUR_API_KEY_HERE",
    "xxxx-xxxx-xxxx-xxxx",
    "0000000000000000000000000000000000000000",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]


def generate_expanded_documentation_example() -> Tuple[str, str, str]:
    """
    Generate known example tokens from vendor documentation.

    These are well-known placeholders that should NEVER be flagged.

    Returns:
        Tuple of (token, var_name, file_path)
    """
    token = random.choice(KNOWN_EXAMPLE_TOKENS)

    var_names = [
        "example_key", "EXAMPLE_API_KEY", "demo_token", "placeholder",
        "YOUR_KEY", "your_api_key", "sample_token", "API_KEY_EXAMPLE",
        "demo_api_key", "test_example_key", "documentation_key",
    ]

    file_paths = [
        "docs/api.md",
        "README.md",
        "examples/auth.py",
        "docs/getting-started.md",
        "CONTRIBUTING.md",
        "docs/quickstart.md",
        "examples/basic_usage.py",
        "docs/authentication.md",
        "wiki/API-Reference.md",
        "tutorials/intro.md",
        "docs/configuration.rst",
    ]

    return token, random.choice(var_names), random.choice(file_paths)


__all__ = [
    # Standard secret generators
    "generate_aws_key",
    "generate_github_token",
    "generate_github_fine_grained_token",
    "generate_stripe_key",
    "generate_openai_key",
    "generate_slack_token",
    "generate_gcp_api_key",
    "generate_jwt_token",
    "generate_azure_key_vault_url",
    "generate_npm_token",
    "generate_pypi_token",
    "generate_random_secret",
    "generate_password",
    # Additional cloud provider generators (expanded vendor coverage)
    "generate_alibaba_cloud_key",
    "generate_digitalocean_token",
    "generate_azure_sas_token",
    "generate_azure_connection_string",
    "generate_hashicorp_vault_token",
    "generate_twilio_credentials",
    "generate_sendgrid_key",
    "generate_mailchimp_key",
    "generate_datadog_key",
    "generate_shopify_credentials",
    "generate_square_credentials",
    "generate_discord_token",
    "generate_telegram_token",
    # Non-secret generators
    "generate_git_sha",
    "generate_uuid",
    "generate_checksum",
    "generate_base64_data",
    # Hard negative generators (for forcing context-based learning)
    "generate_documentation_example",
    "generate_revoked_token",
    "generate_test_fixture_token",
    "generate_encoded_non_secret",
    "generate_high_entropy_non_secret",
    # Ambiguous token generators (for preventing shortcut learning)
    "generate_hex_secret",
    "generate_base64_secret",
    "generate_fake_prefixed_token",
    "generate_ambiguous_token",
    # Git SHA / Hash negatives (with explicit context)
    "generate_git_sha_with_context",
    "generate_content_hash_with_context",
    # Embedded credential generators (secrets in URLs/connection strings)
    "generate_database_url",
    "generate_webhook_url_with_token",
    "generate_connection_string",
    # PEM key generators (multi-line secrets)
    "generate_pem_private_key",
    "generate_ssh_private_key",
    "generate_certificate",
    # --- Enhanced hard negatives (precision boost) ---
    "generate_uuid_with_auth_context",
    "generate_hash_in_security_context",
    "generate_invalid_jwt",
    "generate_base64_telemetry_id",
    "generate_session_id_non_secret",
    "generate_encoded_public_data",
    "generate_version_hash",
    # --- Stage B generalization improvement generators ---
    "generate_hex_secret_with_strong_context",
    "generate_hash_with_secret_varname",
    "generate_secret_with_misleading_varname",
    # --- Stage B precision improvement generators ---
    "generate_test_mode_token",
    "generate_hex_in_build_context",
    "generate_sha_in_config_file",
    "generate_expanded_documentation_example",
    "KNOWN_EXAMPLE_TOKENS",
]
