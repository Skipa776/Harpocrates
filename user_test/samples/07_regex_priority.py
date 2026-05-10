# Python config — regex-tier priority demo
# Run: harpocrates scan user_test/samples/07_regex_priority.py
#
# Harpocrates runs three detection passes in priority order:
#   1. CRITICAL_SIGNATURES regex  — fires first, no ML involved
#   2. HIGH_SIGNATURES regex      — fires second, no ML involved
#   3. Shannon entropy + XGBoost  — fires last when regex does not match
#
# Every finding in this file should carry:
#   evidence: "regex"           (never "entropy" or "ml")
#   severity: "critical"        (CRITICAL tier)
#          or "severity: "high" (HIGH tier)
#
# No --ml flag needed — the regex pass runs unconditionally on every scan.
# This file also has NO ML findings by design: all tokens match a hard
# regex pattern before the entropy gate is ever reached.
#
# EXPECTED FINDINGS (no --ml needed):
#   Line 28  AWS_ACCESS_KEY_ID   → CRITICAL, category: aws_key
#   Line 31  GITHUB_PAT          → CRITICAL, category: github_token
#   Line 34  SLACK_BOT_TOKEN     → CRITICAL, category: slack_token
#   Line 37  STRIPE_LIVE_KEY     → CRITICAL, category: stripe_key
#   Line 40  OPENAI_STRICT_KEY   → CRITICAL, category: openai_key
#   Line 43  ANTHROPIC_API_KEY   → CRITICAL, category: anthropic_key
#   Line 46  GCP_API_KEY         → CRITICAL, category: gcp_key
#   Line 49  NPM_TOKEN           → CRITICAL, category: npm_token
#   Line 52  PYPI_API_TOKEN      → CRITICAL, category: pypi_token
#   Line 58  private_key_pem     → HIGH,     category: private_key
#   Line 61  SENDGRID_API_KEY    → HIGH,     category: sendgrid_key
#   Line 64  TWILIO_API_KEY      → HIGH,     category: twilio_key
#   Line 67  DATABRICKS_TOKEN    → HIGH,     category: databricks_token
#   Line 70  VAULT_SERVICE_TOKEN → HIGH,     category: vault_token

# ============================================================
# CRITICAL_SIGNATURES — deterministic format + fixed length
# ============================================================

# AWS IAM Access Key ID (AKIA family prefix + exactly 16 uppercase alphanumeric)
AWS_ACCESS_KEY_ID = "AKIATESTEXAMPLEKEY01"

# GitHub PAT classic (ghp_ + exactly 36 alphanumeric chars)
GITHUB_TOKEN = "ghp_xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3"

# Slack bot token (xoxb- + 10-13 digits + - + 24-34 alphanumeric)
SLACK_BOT_TOKEN = "xoxb-1234567890-xK9mN3pQ7rS2tV8wY1zA5cE0g"

# Stripe live key (sk_live_ + 24-99 alphanumeric)
STRIPE_LIVE_KEY = "sk_live_xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7"

# OpenAI strict format (sk- + exactly 48 pure-alphanumeric chars, no hyphens in body)
OPENAI_STRICT_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL"

# Anthropic Claude API key (sk-ant-api03- + 93+ alphanumeric chars)
ANTHROPIC_API_KEY = "sk-ant-api03-xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8mN1pR4sU7vX0yC3eG6iM9oQ2tA5wD8zB1eF4hK7mN0pR3qT"

# Google Cloud Platform API key (AIza + 35 alphanumeric/dash/underscore chars)
GCP_API_KEY = "AIzaSyD1234567890xK9mN3pQ7rS2tV8wY1zA5c"

# NPM automation token (npm_ + exactly 36 alphanumeric chars)
NPM_TOKEN = "npm_xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3"

# PyPI API token (pypi- + 50+ alphanumeric/dash/underscore chars)
PYPI_API_TOKEN = "pypi-xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8mN"

# ============================================================
# HIGH_SIGNATURES — anchored patterns, slightly broader scope
# ============================================================

# PEM private key header (RSA / EC / OPENSSH variants all match)
private_key_pem = "-----BEGIN RSA PRIVATE KEY-----"

# SendGrid API key (SG. + 22-char body . 43-char body)
SENDGRID_API_KEY = "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"

# Twilio API key SID (SK + exactly 32 hex chars)
TWILIO_API_KEY = "SK1234567890abcdef1234567890abcdef"

# Databricks personal access token (dapi + exactly 32 lowercase hex chars)
DATABRICKS_TOKEN = "dapi1234567890abcdef1234567890abcdef"

# HashiCorp Vault service token v2 (hvs. + 90+ alphanumeric chars)
VAULT_SERVICE_TOKEN = "hvs.xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8mN1pR4sU7vX0yC3eG6iM9oQ2tA5wD8zB1eF4hK7mN0pR3qT"
