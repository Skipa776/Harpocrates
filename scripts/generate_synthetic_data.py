#!/usr/bin/env python3
"""
Synthetic training data generator for Harpocrates v0.2.0.

Strategy (ML overhaul plan):
  Positive class — Mixed generation:
    ~60% LLM-generated code with hardcoded credentials (via Dynamic Prompt Matrix)
    ~40% Script-generated positives with realistic secret patterns
  Negative class — Mixed generation:
    ~60% Script-generated safe high-entropy strings
    ~40% LLM-generated code with safe high-entropy tokens (via Dynamic Prompt Matrix)

Target leakage prevention: Both classes use both generation methods so the model
cannot learn "LLM-generated = positive" as a confounding signal.

Dynamic Prompt Matrix: Instead of static prompt lists, each LLM call assembles a
unique prompt by sampling one element from each dimension (language, file type,
industry, secret type, etc). Combinatorial explosion prevents semantic collapse.

Usage:
    # Full run (requires LM Studio + loaded model)
    python scripts/generate_synthetic_data.py --output data/synthetic_v2.jsonl --count 40000

    # Negative-class only (no LLM required)
    python scripts/generate_synthetic_data.py --no-llm --count 1000

    # Custom concurrency for async batching
    python scripts/generate_synthetic_data.py --max-concurrent 8

Requirements:
    pip install harpocrates[ml] aiohttp tqdm
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Harpocrates.ml.features import extract_features_from_record

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FIELD_LENGTH: int = 2048
_DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
_DEFAULT_MODEL = "gemma-4-e2b"
_DEFAULT_MAX_CONCURRENT = 4

_SYSTEM_MESSAGE = (
    "You are a code generator. Write realistic, production-quality code files. "
    "Use realistic-looking credential values (random strings, not placeholders). "
    "Never use words like YOUR_, CHANGEME, EXAMPLE, INSERT, PLACEHOLDER, or TODO "
    "in credential values. Generate values that look like real API keys, passwords, "
    "and connection strings."
)

_SYSTEM_MESSAGE_NEGATIVE = (
    "You are a code generator. Write realistic, production-quality code files. "
    "All high-entropy values must be clearly non-secret: content hashes, public keys, "
    "mock tokens, nonces, SRI hashes, or test fixtures. Never include real API keys, "
    "passwords, private keys, or any actual secrets."
)

# Targeted AI placeholder guard — blocks unambiguous LLM artifacts only.
# Does NOT block test, demo, sample (can be real short passwords in legacy systems).
_AI_PLACEHOLDER_RE = re.compile(
    r"YOUR_|_HERE\b|CHANGEME|REPLACE_ME|INSERT_|EXAMPLE_|PLACEHOLDER|X{7,}",
    re.IGNORECASE,
)

# Token extraction regex patterns
_TOKEN_PATTERNS = [
    re.compile(r'["\']([A-Za-z0-9+/=_\-\.]{6,})["\']'),
    re.compile(r'=\s*([A-Za-z0-9+/=_\-\.]{6,})'),
    re.compile(r':\s*([A-Za-z0-9+/=_\-\.]{6,})'),
]


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _min_entropy_for_length(length: int) -> float:
    """Dynamic entropy floor: short tokens need higher entropy to pass."""
    if length <= 6:
        return 3.2
    if length >= 20:
        return 2.0
    return 3.2 - (length - 6) * (1.2 / 14)


# ---------------------------------------------------------------------------
# Token extraction with position tracking (Amendment #8)
# ---------------------------------------------------------------------------

class TokenMatch(NamedTuple):
    token: str
    start: int
    end: int


def _extract_token_from_line(line: str) -> Optional[TokenMatch]:
    """Extract the highest-entropy token from a line of code.

    Returns a TokenMatch with the token string AND its exact character
    position in the line (from the regex match). Position info is required
    by downstream features like is_env_fallback_value (Amendment #8).
    """
    best: Optional[TokenMatch] = None
    best_entropy = 0.0

    for pat in _TOKEN_PATTERNS:
        for m in pat.finditer(line):
            candidate = m.group(1)
            if len(candidate) < 6:
                continue
            ent = _shannon_entropy(candidate)
            if ent < _min_entropy_for_length(len(candidate)):
                continue
            if _AI_PLACEHOLDER_RE.search(candidate):
                continue
            if ent > best_entropy:
                best = TokenMatch(candidate, m.start(1), m.end(1))
                best_entropy = ent

    return best


# ---------------------------------------------------------------------------
# Random data helpers
# ---------------------------------------------------------------------------

def _rand_hex(n: int) -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _rand_b64(n_bytes: int) -> str:
    import base64, os
    return base64.b64encode(os.urandom(n_bytes)).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------
# Dynamic Prompt Matrix — Positive (Amendment #7)
# ---------------------------------------------------------------------------

_POS_LANGUAGES = [
    "Python", "JavaScript (Node.js)", "Go", "Java", "Ruby", "PHP",
    "C#/.NET", "Swift", "Perl", "Rust", "TypeScript", "Kotlin",
    "Scala", "Bash/Shell script", "R",
]

_POS_FILE_TYPES = [
    "module with 3+ classes", "settings/config file", "test suite with 5+ test cases",
    "CI/CD workflow (GitHub Actions / GitLab CI)", "deployment script",
    "database migration", "initializer/bootstrap", "CLI tool entrypoint",
    "REST API client library", "background job/worker", "Makefile or Taskfile",
    "infrastructure-as-code (Terraform/Ansible/K8s)", "Docker Compose file",
]

_POS_INDUSTRIES = [
    "fintech payment processor", "healthcare patient portal", "e-commerce marketplace",
    "SaaS analytics platform", "gaming leaderboard service", "media streaming backend",
    "enterprise HR system", "startup MVP", "government compliance portal",
    "education LMS", "IoT device management", "logistics/shipping tracker",
]

_POS_SECRET_TYPES = [
    "API key (with realistic prefix like sk_live_, AKIA, ghp_, xoxb-)",
    "database connection string with embedded password",
    "RSA/EC private key (PEM format, 5+ lines)",
    "JWT signing secret", "webhook signing secret (HMAC)",
    "SMTP/email credentials (username + app password)",
    "OAuth client secret", "cloud provider access key + secret key pair",
    "encryption/signing key (AES-256 or HMAC key)",
    "service account token or bearer token",
    "Redis/cache AUTH password in connection URL",
]

_POS_ASSIGNMENT_PATTERNS = [
    "hardcoded as a module-level constant",
    "set as a class/struct property with default value",
    "passed as a default parameter in a constructor",
    "embedded in a configuration dictionary/map",
    "used as a fallback in os.getenv()/process.env || pattern",
    "written inline in a function call argument",
    "stored in a YAML/JSON value field",
    "assigned in a ${VAR:-hardcoded_default} Docker pattern",
    "set via ENV.fetch('KEY') { 'hardcoded_fallback' } Ruby pattern",
]

_POS_NOISE_ELEMENTS = [
    "also include 3+ SHA-256 content verification hashes as constants",
    "also include 2+ UUID v4 correlation IDs as constants",
    "surround with 30+ lines of error handling, retry logic, and logging",
    "include SRI integrity hashes for CDN script tags",
    "mix in bcrypt password hashes (derived values, not secrets)",
    "include extensive inline comments explaining each config section",
    "add type annotations/hints and docstrings throughout",
    "include 20+ unrelated configuration settings around the secrets",
    "add feature flags, timeouts, and non-secret env vars alongside",
    "include base64-encoded non-secret data (images, fixtures) nearby",
]

_POS_FILE_PATHS: Dict[str, List[str]] = {
    "Python":             ["config.py", "settings.py", "src/api_client.py", "services/email.py"],
    "JavaScript (Node.js)": ["config.js", "src/auth.js", "lib/api-client.js"],
    "Go":                 ["cmd/main.go", "internal/config/config.go", "pkg/db/connection.go"],
    "Java":               ["src/main/java/config/AppConfig.java", "src/main/java/auth/LdapAuth.java"],
    "Ruby":               ["config/initializers/secrets.rb", "config/database.yml", "lib/api_client.rb"],
    "PHP":                ["config/database.php", "src/Service/ApiClient.php", "lib/FtpUploader.php"],
    "C#/.NET":            ["appsettings.json", "Data/ConnectionFactory.cs", "Services/AuthService.cs"],
    "Swift":              ["AppDelegate.swift", "Config/Secrets.swift"],
    "Perl":               ["cgi-bin/report.pl", "lib/DBConfig.pm"],
    "Rust":               ["src/config.rs", "src/auth.rs"],
    "TypeScript":         ["src/config.ts", "src/api-client.ts", "tests/auth.test.ts"],
    "Kotlin":             ["src/main/kotlin/Config.kt", "src/main/kotlin/ApiClient.kt"],
    "Scala":              ["src/main/scala/Config.scala"],
    "Bash/Shell script":  ["deploy.sh", "scripts/setup.sh", "Makefile"],
    "R":                  ["config.R", "scripts/analysis.R"],
}


def _build_positive_prompt() -> Tuple[str, str]:
    """Build a unique positive LLM prompt by sampling one element per dimension."""
    lang = random.choice(_POS_LANGUAGES)
    file_type = random.choice(_POS_FILE_TYPES)
    industry = random.choice(_POS_INDUSTRIES)
    secret_type = random.choice(_POS_SECRET_TYPES)
    assignment = random.choice(_POS_ASSIGNMENT_PATTERNS)
    noise = random.choice(_POS_NOISE_ELEMENTS)
    file_path = random.choice(_POS_FILE_PATHS.get(lang, ["config.txt"]))

    prompt = (
        f"Write a {lang} {file_type} for a {industry}. "
        f"The file must contain a hardcoded {secret_type}, "
        f"{assignment}. "
        f"Use realistic-looking credential values (random alphanumeric strings with "
        f"realistic prefixes), NOT placeholder text like YOUR_KEY or CHANGEME. "
        f"{noise}. "
        f"The file should be 40+ lines with realistic variable names, imports, "
        f"and error handling. Output ONLY the code."
    )
    return prompt, file_path


# ---------------------------------------------------------------------------
# Dynamic Prompt Matrix — Negative (Amendment #7)
# ---------------------------------------------------------------------------

_NEG_LANGUAGES = _POS_LANGUAGES

_NEG_FILE_TYPES = [
    "test suite with 5+ test cases", "verification/validation module",
    "data pipeline with deduplication", "integrity checking tool",
    "content-addressable storage library", "cryptographic verification module",
    "CI/CD configuration file", "build system file",
    "infrastructure template (ConfigMap only, no Secrets)",
    "benchmark/profiling harness", "schema migration with checksums",
]

_NEG_HIGH_ENTROPY_TYPES = [
    "SHA-256 content digests for file deduplication",
    "SHA-384 SRI (Subresource Integrity) hashes for CDN script tags",
    "bcrypt password hashes ($2b$ format) as test fixtures — DERIVED values, not passwords",
    "HMAC-SHA256 webhook signature values for verification tests",
    "CSP nonces and per-request security tokens",
    "Ed25519/X25519 PUBLIC keys (explicitly labeled, NOT private keys)",
    "mock/fake bearer tokens clearly labeled as test fixtures",
    "base64-encoded image/SVG data URIs",
    "Docker layer SHA-256 digests from manifest files",
    "UUID v4 correlation IDs and trace identifiers",
    "IPFS CID content identifiers",
    "git tree/blob object hashes",
]

_NEG_CONTEXT_HINTS = [
    "all high-entropy values must be clearly labeled as mock/fake/test data",
    "include comments explicitly stating these are NOT real credentials",
    "use variable names like mock_*, fake_*, test_*, fixture_*, stub_*",
    "wrap values in a verification/validation function context",
    "label all keys as PUBLIC and all hashes as DERIVED/COMPUTED",
    "include assertions that verify hash correctness, not authenticate",
    "use in a deduplication or integrity-checking context",
    "embed in test fixture setup/teardown methods",
    "include in a content-addressable or cache-key context",
]

_NEG_FILE_PATHS: Dict[str, List[str]] = {
    "Python":             ["tests/test_auth.py", "tests/conftest.py", "utils/integrity.py", "lib/content_hash.py"],
    "JavaScript (Node.js)": ["tests/auth.test.js", "lib/verify.js", "utils/hashing.js"],
    "Go":                 ["pkg/verify/verify.go", "internal/hash/content.go", "cmd/verify/main.go"],
    "Java":               ["src/test/java/AuthTest.java", "src/main/java/util/HashVerifier.java"],
    "Ruby":               ["spec/auth_spec.rb", "lib/integrity.rb", "test/fixtures/auth_helpers.rb"],
    "PHP":                ["tests/VerifyTest.php", "src/Service/HashVerifier.php"],
    "C#/.NET":            ["Tests/AuthTests.cs", "Utils/IntegrityChecker.cs"],
    "Swift":              ["Tests/AuthTests.swift", "Utils/HashVerifier.swift"],
    "Perl":               ["t/verify.t", "lib/Integrity.pm"],
    "Rust":               ["src/verify.rs", "tests/crypto_test.rs"],
    "TypeScript":         ["tests/auth.test.ts", "src/utils/verify.ts", "test/fixtures/tokens.ts"],
    "Kotlin":             ["src/test/kotlin/AuthTest.kt", "src/main/kotlin/HashVerifier.kt"],
    "Scala":              ["src/test/scala/VerifySpec.scala"],
    "Bash/Shell script":  ["tests/verify.sh", "scripts/check_integrity.sh"],
    "R":                  ["tests/test_hashing.R"],
}


def _build_negative_prompt() -> Tuple[str, str]:
    """Build a unique negative LLM prompt — produces code with high-entropy
    non-secret tokens that LOOK like secrets but ARE NOT."""
    lang = random.choice(_NEG_LANGUAGES)
    file_type = random.choice(_NEG_FILE_TYPES)
    entropy_type = random.choice(_NEG_HIGH_ENTROPY_TYPES)
    context_hint = random.choice(_NEG_CONTEXT_HINTS)
    file_path = random.choice(_NEG_FILE_PATHS.get(lang, ["tests/test_verify.py"]))

    prompt = (
        f"Write a {lang} {file_type}. "
        f"Include 5+ hardcoded {entropy_type}. "
        f"{context_hint}. "
        f"The file should be 30+ lines with realistic structure. "
        f"Do NOT include any real API keys, passwords, private keys, or secrets. "
        f"All high-entropy strings must be derived/public/mock values. "
        f"Output ONLY the code."
    )
    return prompt, file_path


# ---------------------------------------------------------------------------
# Script-generated negative generators (existing 8 + 8 new)
# ---------------------------------------------------------------------------

def _neg_lock_file_hash() -> Dict[str, Any]:
    pkg = random.choice(["lodash", "react", "axios", "express", "webpack", "vue", "svelte"])
    ver = f"{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)}"
    h = _rand_b64(48)
    line = f'    "integrity": "sha512-{h}==",'
    return {
        "token": h,
        "line_content": line,
        "context_before": [f'  "{pkg}@{ver}": {{', f'    "version": "{ver}",'],
        "context_after":  ['    "resolved": "https://registry.npmjs.org/..."', "  },"],
        "file_path": "package-lock.json",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_css_hex_color() -> Dict[str, Any]:
    color = _rand_hex(6)
    prop = random.choice(["background-color", "color", "border-color", "fill", "outline-color"])
    sel = random.choice([".btn", ".header", "#main", "body", "a:hover", ".card"])
    return {
        "token": color,
        "line_content": f"  {prop}: #{color};",
        "context_before": [f"{sel} {{"],
        "context_after":  ["}"],
        "file_path": "styles/main.css",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_git_sha() -> Dict[str, Any]:
    sha = _rand_hex(40)
    msg = random.choice(["Fix auth bug", "Update deps", "Refactor auth", "Add tests"])
    return {
        "token": sha,
        "line_content": f"# commit: {sha}  # {msg}",
        "context_before": [f"# Revert commit {sha[:8]}"],
        "context_after":  ["# Author: Dev <dev@example.com>"],
        "file_path": "scripts/deploy.sh",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_uuid() -> Dict[str, Any]:
    import uuid
    uid = str(uuid.uuid4())
    var = random.choice(["user_id", "session_id", "request_id", "trace_id", "correlation_id"])
    return {
        "token": uid,
        "line_content": f'    "{var}": "{uid}",',
        "context_before": ['  "user": {'],
        "context_after":  ['    "name": "Alice"', "  },"],
        "file_path": "tests/fixtures/user_fixture.json",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_gpg_pubkey_line() -> Dict[str, Any]:
    data = _rand_b64(60)
    return {
        "token": data,
        "line_content": data,
        "context_before": ["-----BEGIN PGP PUBLIC KEY BLOCK-----", ""],
        "context_after":  ["-----END PGP PUBLIC KEY BLOCK-----"],
        "file_path": "keys/developer.pub",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_docker_digest() -> Dict[str, Any]:
    sha = _rand_hex(64)
    img = random.choice(["nginx", "postgres", "redis", "node", "python", "alpine"])
    return {
        "token": sha,
        "line_content": f"FROM {img}@sha256:{sha}",
        "context_before": ["# Production image"],
        "context_after":  ["RUN apt-get update -y"],
        "file_path": "Dockerfile",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_yarn_lock_sha() -> Dict[str, Any]:
    sha = _rand_hex(64)
    pkg = random.choice(["typescript", "eslint", "prettier", "jest", "babel-core"])
    ver = f"{random.randint(4,8)}.{random.randint(0,9)}.{random.randint(0,9)}"
    return {
        "token": sha,
        "line_content": f'  "{pkg}" "{ver}" sha256:{sha}',
        "context_before": ["__metadata:", "  version: 6"],
        "context_after":  [""],
        "file_path": "yarn.lock",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_asset_manifest() -> Dict[str, Any]:
    sha = _rand_hex(40)
    artifact = random.choice(["app.bundle.js", "main.css", "vendor.js", "runtime.js"])
    return {
        "token": sha,
        "line_content": f'  "{artifact}": "{sha}",',
        "context_before": ["{", '  "version": 1,', '  "files": {'],
        "context_after":  ["},"],
        "file_path": "dist/asset-manifest.json",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_bcrypt_hash() -> Dict[str, Any]:
    rounds = random.choice([10, 12, 14])
    salt = _rand_b64(16)
    hash_body = _rand_b64(23)
    full_hash = f"$2b${rounds}${salt}{hash_body}"
    var = random.choice(["password_hash", "hashed_password", "user_pwd_hash",
                         "stored_hash", "auth_hash"])
    return {
        "token": full_hash,
        "line_content": f'    "{var}": "{full_hash}",',
        "context_before": ['  "user": {', f'    "email": "alice@example.com",'],
        "context_after":  ['    "created_at": "2024-01-15T10:30:00Z"', "  },"],
        "file_path": random.choice(["db/seeds.json", "tests/fixtures/users.json",
                                     "data/user_dump.json"]),
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_base64_image_data() -> Dict[str, Any]:
    data = _rand_b64(random.randint(40, 80))
    fmt = random.choice(["png", "gif", "webp", "jpeg"])
    var = random.choice(["icon_data", "logo_base64", "thumbnail", "sprite_data",
                         "favicon_b64"])
    return {
        "token": data,
        "line_content": f'{var} = "data:image/{fmt};base64,{data}"',
        "context_before": [f"# Inline {fmt.upper()} asset (avoid network request)",
                          "import base64"],
        "context_after":  ["ICON_SIZE = (16, 16)", ""],
        "file_path": random.choice(["src/assets.py", "lib/icons.ts",
                                     "static/inline_images.js"]),
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_hmac_webhook_signature() -> Dict[str, Any]:
    sig = _rand_hex(64)
    provider = random.choice(["stripe", "github", "shopify", "twilio", "slack"])
    header = random.choice([
        f"X-{provider.title()}-Signature",
        "X-Hub-Signature-256",
        "X-Webhook-Signature",
    ])
    return {
        "token": sig,
        "line_content": f'    "{header}": "sha256={sig}",',
        "context_before": ['headers = {',
                          '    "Content-Type": "application/json",'],
        "context_after":  ['}', 'response = requests.post(webhook_url, headers=headers)'],
        "file_path": random.choice(["tests/test_webhooks.py",
                                     "tests/webhook_fixtures.json",
                                     "tests/integration/test_stripe.py"]),
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_sri_hash() -> Dict[str, Any]:
    algo = random.choice(["sha256", "sha384", "sha512"])
    length = {"sha256": 32, "sha384": 48, "sha512": 64}[algo]
    digest = _rand_b64(length)
    lib = random.choice(["react", "vue", "jquery", "lodash", "bootstrap", "d3"])
    ver = f"{random.randint(1,18)}.{random.randint(0,9)}.{random.randint(0,9)}"
    return {
        "token": digest,
        "line_content": (f'<script src="https://cdn.example.com/{lib}@{ver}/dist/{lib}.min.js" '
                        f'integrity="{algo}-{digest}" crossorigin="anonymous"></script>'),
        "context_before": ["<head>", '  <meta charset="UTF-8">'],
        "context_after":  ["</head>"],
        "file_path": random.choice(["templates/base.html", "public/index.html",
                                     "views/layout.ejs"]),
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_test_fixture_mock_token() -> Dict[str, Any]:
    token = _rand_b64(random.randint(20, 40))
    var = random.choice(["mock_api_key", "fake_token", "test_bearer_token",
                         "stub_secret_key", "fixture_auth_token", "dummy_api_secret"])
    framework = random.choice(["pytest", "jest", "rspec", "junit"])
    return {
        "token": token,
        "line_content": f'{var} = "{token}"',
        "context_before": [
            random.choice([
                f"# {framework} fixture — NOT a real credential",
                "// Test helper — fake credentials for mocking",
                "@pytest.fixture",
                "describe('authentication', () => {",
            ]),
            random.choice([
                "def setup_auth_mocks():",
                "const mockAuth = {",
                "class FakeAuthProvider:",
            ]),
        ],
        "context_after": [
            random.choice([
                "    mock_expiry = datetime.now() + timedelta(hours=1)",
                "    expected_status = 401  # Should fail with fake token",
                "    assert response.status_code == 200",
            ]),
            "",
        ],
        "file_path": random.choice(["tests/test_auth.py", "tests/auth.test.js",
                                     "spec/auth_spec.rb", "tests/conftest.py",
                                     "test/fixtures/auth_helpers.ts"]),
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_csprng_nonce() -> Dict[str, Any]:
    nonce = _rand_hex(random.choice([24, 32, 48]))
    var = random.choice(["nonce", "csrf_nonce", "request_nonce", "encryption_nonce",
                         "iv", "initialization_vector"])
    return {
        "token": nonce,
        "line_content": f'{var} = "{nonce}"',
        "context_before": [
            random.choice([
                "# Generate per-request nonce for CSP header",
                "// Nonce for AES-GCM encryption (NOT a key)",
                "# CSRF nonce — changes every request",
            ]),
            random.choice([
                "import secrets",
                "from cryptography.hazmat.primitives.ciphers import Cipher",
                "const crypto = require('crypto');",
            ]),
        ],
        "context_after": [
            random.choice([
                f"response.headers['X-Nonce'] = {var}",
                f"cipher = Cipher(algorithm, modes.GCM({var}))",
                f"html = f'<script nonce=\"{{{var}}}\">'",
            ]),
            "",
        ],
        "file_path": random.choice(["middleware/csp.py", "lib/encryption.py",
                                     "src/middleware/csrf.ts", "utils/crypto.go"]),
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_content_addressable_hash() -> Dict[str, Any]:
    hash_type = random.choice(["ipfs", "git_tree", "git_blob", "docker_layer"])
    if hash_type == "ipfs":
        token = "bafy" + _rand_b64(32).lower().replace("+", "a").replace("/", "b")
        line = f'"hash": "{token}"'
        ctx_before = ['  "pins": [', '    {']
        ctx_after = ['      "name": "model-weights-v3"', '    },']
        fp = "ipfs/pin_list.json"
    elif hash_type in ("git_tree", "git_blob"):
        token = _rand_hex(40)
        obj = random.choice(["tree", "blob"])
        line = f"{token} {obj} {random.randint(100,9999)}    src/{random.choice(['main', 'lib', 'utils'])}"
        ctx_before = ["# git ls-tree output", f"{_rand_hex(40)} commit"]
        ctx_after = [f"{_rand_hex(40)} blob {random.randint(100,999)}    README.md"]
        fp = "scripts/tree_snapshot.txt"
    else:
        token = _rand_hex(64)
        line = f'"digest": "sha256:{token}"'
        ctx_before = ['"layers": [', '  {']
        ctx_after = [f'    "size": {random.randint(1000000,9999999)}', '  },']
        fp = "docker/manifest.json"

    return {
        "token": token,
        "line_content": line,
        "context_before": ctx_before,
        "context_after": ctx_after,
        "file_path": fp,
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_env_loaded_secret() -> Dict[str, Any]:
    """Secret loaded from environment — variable name says 'secret' but pattern is safe."""
    var = random.choice(["API_KEY", "SECRET_KEY", "DATABASE_PASSWORD",
                         "AUTH_TOKEN", "STRIPE_KEY", "JWT_SECRET"])
    loader = random.choice([
        f'{var} = os.getenv("{var}")',
        f'{var} = os.environ["{var}"]',
        f'{var} = os.environ.get("{var}")',
        f'const {var.lower()} = process.env.{var}',
        f'{var.lower()} = ENV["{var}"]',
        f'{var.lower()} = ENV.fetch("{var}")',
    ])
    token = var
    return {
        "token": token,
        "line_content": loader,
        "context_before": [
            random.choice([
                "from dotenv import load_dotenv",
                "import os",
                "require('dotenv').config()",
                "load_dotenv()",
            ]),
            random.choice([
                "",
                "# Configuration loaded from environment",
                "// Load secrets from .env file",
            ]),
        ],
        "context_after": [
            random.choice([
                f'if not {var}: raise ValueError("Missing {var}")',
                f"assert {var.lower()}, '{var} must be set'",
                f"logger.info('Loaded {var} from environment')",
            ]),
            "",
        ],
        "file_path": random.choice(["config/settings.py", "src/config.ts",
                                     "config/application.rb", "cmd/main.go"]),
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


_NEGATIVE_GENERATORS = [
    _neg_lock_file_hash,
    _neg_css_hex_color,
    _neg_git_sha,
    _neg_uuid,
    _neg_gpg_pubkey_line,
    _neg_docker_digest,
    _neg_yarn_lock_sha,
    _neg_asset_manifest,
    _neg_bcrypt_hash,
    _neg_base64_image_data,
    _neg_hmac_webhook_signature,
    _neg_sri_hash,
    _neg_test_fixture_mock_token,
    _neg_csprng_nonce,
    _neg_content_addressable_hash,
    _neg_env_loaded_secret,
]


# ---------------------------------------------------------------------------
# Script-generated positive generators (Amendment #1 — target leakage fix)
# ---------------------------------------------------------------------------

def _pos_hardcoded_api_key() -> Dict[str, Any]:
    prefix = random.choice(["sk_live_", "ak_", "AKIA", "ghp_", "xoxb-", "rk_live_"])
    key_body = _rand_b64(random.randint(20, 40))
    token = f"{prefix}{key_body}"
    var = random.choice(["API_KEY", "SECRET_KEY", "AUTH_TOKEN",
                         "api_key", "secret_key", "access_token"])
    return {
        "token": token,
        "line_content": f'{var} = "{token}"',
        "context_before": [
            random.choice(["import requests", "from flask import Flask",
                          "import boto3", "from stripe import Stripe"]),
            random.choice(["", "# Application configuration",
                          "class Config:", "def get_client():"]),
        ],
        "context_after": [
            random.choice([
                f"client = Client(api_key={var})",
                f"headers = {{'Authorization': f'Bearer {{{var}}}'}}",
                f"session.headers.update({{'X-API-Key': {var}}})",
            ]),
            "",
        ],
        "file_path": random.choice(["config.py", "settings.py",
                                     "src/api_client.py", "lib/auth.ts"]),
        "label": 1,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _pos_connection_string() -> Dict[str, Any]:
    user = random.choice(["admin", "app_user", "dbadmin", "root", "postgres"])
    password = _rand_b64(random.randint(12, 24))
    host = random.choice(["db.internal", "rds.amazonaws.com", "postgres.railway.app",
                          "cluster0.mongodb.net", "127.0.0.1"])
    db = random.choice(["myapp", "production", "main", "users", "analytics"])
    proto = random.choice(["postgresql", "mysql", "mongodb+srv", "redis"])
    port = {"postgresql": 5432, "mysql": 3306, "mongodb+srv": 27017, "redis": 6379}[proto]
    token = f"{proto}://{user}:{password}@{host}:{port}/{db}"
    var = random.choice(["DATABASE_URL", "DB_URI", "SQLALCHEMY_DATABASE_URI",
                         "MONGO_URI", "connection_string"])
    return {
        "token": token,
        "line_content": f'{var} = "{token}"',
        "context_before": [
            random.choice(["from sqlalchemy import create_engine",
                          "import psycopg2", "from pymongo import MongoClient"]),
            "",
        ],
        "context_after": [
            random.choice([
                f"engine = create_engine({var})",
                f"conn = psycopg2.connect({var})",
                f"db = MongoClient({var})",
            ]),
            "",
        ],
        "file_path": random.choice(["config/database.py", "settings/prod.py",
                                     "src/db.ts", "config/database.yml"]),
        "label": 1,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _pos_private_key_pem() -> Dict[str, Any]:
    key_data = _rand_b64(random.randint(40, 80))
    token = key_data
    key_type = random.choice(["RSA PRIVATE KEY", "EC PRIVATE KEY",
                              "PRIVATE KEY", "OPENSSH PRIVATE KEY"])
    var = random.choice(["PRIVATE_KEY", "signing_key", "ssh_key",
                         "tls_key", "JWT_PRIVATE_KEY"])
    return {
        "token": token,
        "line_content": f'{var} = "-----BEGIN {key_type}-----\\n{key_data}\\n-----END {key_type}-----"',
        "context_before": [
            random.choice(["import jwt", "from cryptography.hazmat.primitives import serialization",
                          "const crypto = require('crypto');"]),
            "",
        ],
        "context_after": [
            random.choice([
                f"token = jwt.encode(payload, {var}, algorithm='RS256')",
                f"signer = PKCS1v15.new(RSA.import_key({var}))",
                "const signer = crypto.createSign('SHA256');",
            ]),
            "",
        ],
        "file_path": random.choice(["auth/keys.py", "config/certs.py",
                                     "src/signing.ts", "lib/jwt_config.rb"]),
        "label": 1,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _pos_env_fallback_secret() -> Dict[str, Any]:
    prefix = random.choice(["sk_live_", "AKIA", "ghp_", "whsec_", ""])
    secret_value = prefix + _rand_b64(random.randint(16, 32))
    var = random.choice(["SECRET_KEY", "API_KEY", "DATABASE_PASSWORD",
                         "STRIPE_SECRET_KEY", "JWT_SECRET"])
    pattern = random.choice([
        f'{var} = os.getenv("{var}", "{secret_value}")',
        f'{var} = os.environ.get("{var}", "{secret_value}")',
        f'const {var.lower()} = process.env.{var} || "{secret_value}"',
        f'{var.lower()} = ENV.fetch("{var}") {{ "{secret_value}" }}',
    ])
    return {
        "token": secret_value,
        "line_content": pattern,
        "context_before": [
            random.choice(["from dotenv import load_dotenv", "import os",
                          "require('dotenv').config()", "load_dotenv()"]),
            "",
        ],
        "context_after": [
            random.choice([
                f"app.config['{var}'] = {var}",
                f"client = Client({var.lower()})",
                "# WARNING: fallback should only be used in development",
            ]),
            "",
        ],
        "file_path": random.choice(["config/settings.py", "src/config.ts",
                                     "config/application.rb"]),
        "label": 1,
        "secret_type": "ENTROPY_CANDIDATE",
    }


_POSITIVE_GENERATORS = [
    _pos_hardcoded_api_key,
    _pos_connection_string,
    _pos_private_key_pem,
    _pos_env_fallback_secret,
]


def generate_script_positive_samples(count: int) -> List[Dict[str, Any]]:
    samples = [random.choice(_POSITIVE_GENERATORS)() for _ in range(count)]
    for s in samples:
        s["source"] = "script"
    return samples


def generate_script_negative_samples(count: int) -> List[Dict[str, Any]]:
    samples = [random.choice(_NEGATIVE_GENERATORS)() for _ in range(count)]
    for s in samples:
        s["source"] = "script"
    return samples


# ---------------------------------------------------------------------------
# LLM generation — async with concurrency control (Amendment #5)
# ---------------------------------------------------------------------------

def _validate_lm_studio_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"--lm-studio-url must use http:// or https://, got: {url!r}. "
            "file://, ftp://, and other schemes are not permitted."
        )
    if not parsed.hostname:
        raise ValueError(f"--lm-studio-url has no hostname: {url!r}")
    return url


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```\w*\n?", "", text)


async def _call_lmstudio_async(
    session: Any,
    prompt: str,
    system_msg: str,
    model: str,
    base_url: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """POST to LM Studio's OpenAI-compatible endpoint with concurrency control."""
    import aiohttp

    async with semaphore:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": random.uniform(0.5, 1.0),
            "max_tokens": 1500,
            "stream": False,
        }

        try:
            async with session.post(
                f"{url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"  [WARN] LM Studio request failed: {e}", file=sys.stderr)
            return None


def _extract_samples_from_response(
    response: str, file_path: str, label: int
) -> List[Dict[str, Any]]:
    """Extract token samples from an LLM response."""
    lines = _strip_code_fences(response).splitlines()
    lines = [ln for ln in lines if ln.strip()]
    samples: List[Dict[str, Any]] = []

    for i, line in enumerate(lines):
        match = _extract_token_from_line(line)
        if match is None:
            continue

        samples.append({
            "token": match.token,
            "token_start": match.start,
            "token_end": match.end,
            "line_content": line,
            "context_before": lines[max(0, i - 2):i],
            "context_after": lines[i + 1: i + 3],
            "file_path": file_path,
            "label": label,
            "secret_type": "ENTROPY_CANDIDATE",
            "source": "llm",
        })

    return samples


async def generate_llm_samples_async(
    count: int,
    model: str,
    lm_studio_url: str,
    max_concurrent: int,
    label: int,
    progress_prefix: str = "",
) -> List[Dict[str, Any]]:
    """Generate samples via LLM with async batching."""
    import aiohttp

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    semaphore = asyncio.Semaphore(max_concurrent)
    samples: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 4

    build_prompt = _build_positive_prompt if label == 1 else _build_negative_prompt
    system_msg = _SYSTEM_MESSAGE if label == 1 else _SYSTEM_MESSAGE_NEGATIVE

    label_name = "pos" if label == 1 else "neg"
    pbar = tqdm(total=count, desc=f"{progress_prefix}LLM ({label_name})",
                disable=not has_tqdm)

    async with aiohttp.ClientSession() as session:
        batch_size = max_concurrent * 2

        while len(samples) < count and attempts < max_attempts:
            prompts_and_paths = [build_prompt() for _ in range(min(batch_size, max_attempts - attempts))]
            attempts += len(prompts_and_paths)

            tasks = [
                _call_lmstudio_async(session, prompt, system_msg, model, lm_studio_url, semaphore)
                for prompt, _ in prompts_and_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (_, file_path), result in zip(prompts_and_paths, results):
                if isinstance(result, Exception) or result is None:
                    continue
                new_samples = _extract_samples_from_response(result, file_path, label)
                samples.extend(new_samples)
                pbar.update(len(new_samples))

            if attempts % 100 == 0:
                print(f"  [{progress_prefix}] {len(samples)}/{count} samples after {attempts} calls",
                      file=sys.stderr)

    pbar.close()

    if len(samples) < count:
        print(
            f"[WARN] Only generated {len(samples)}/{count} {'positive' if label == 1 else 'negative'} "
            f"LLM samples after {attempts} calls.",
            file=sys.stderr,
        )

    return samples[:count]


# ---------------------------------------------------------------------------
# Feature extraction wrapper
# ---------------------------------------------------------------------------

def _sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    for field in ("token", "line_content", "file_path"):
        v = record.get(field)
        if isinstance(v, str) and len(v) > _MAX_FIELD_LENGTH:
            record[field] = v[:_MAX_FIELD_LENGTH]
    for field in ("context_before", "context_after"):
        v = record.get(field)
        if isinstance(v, list):
            record[field] = [
                line[:_MAX_FIELD_LENGTH] if isinstance(line, str) else line
                for line in v
            ]
    return record


def _attach_features(record: Dict[str, Any]) -> Dict[str, Any]:
    record = _sanitize_record(record)
    try:
        fv = extract_features_from_record(record)
        record["features_65"] = fv.to_array()
    except Exception as e:
        print(f"  [WARN] Feature extraction failed: {e}", file=sys.stderr)
        record["features_65"] = None
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data (v0.2.0)")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_v2.jsonl"))
    parser.add_argument("--count", type=int, default=2000, help="Total samples to generate")
    parser.add_argument("--balance", type=float, default=0.5,
                        help="Fraction of samples that are positive class (default: 0.5)")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                        help=f"LM Studio model name (default: {_DEFAULT_MODEL})")
    parser.add_argument("--lm-studio-url", type=str, default=_DEFAULT_LM_STUDIO_URL,
                        dest="lm_studio_url")
    parser.add_argument("--max-concurrent", type=int, default=_DEFAULT_MAX_CONCURRENT,
                        help=f"Max concurrent LLM requests (default: {_DEFAULT_MAX_CONCURRENT})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM generation — script-generated samples only")
    parser.add_argument("--no-features", action="store_true",
                        help="Skip feature extraction (faster, no feature columns)")
    # Legacy alias for backward compatibility
    parser.add_argument("--ollama-url", type=str, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ollama_url and args.lm_studio_url == _DEFAULT_LM_STUDIO_URL:
        args.lm_studio_url = args.ollama_url

    n_positive = int(args.count * args.balance)
    n_negative = args.count - n_positive

    # Target: 60% LLM, 40% script for each class (breaks target leakage)
    n_pos_llm = int(n_positive * 0.6)
    n_pos_script = n_positive - n_pos_llm
    n_neg_llm = int(n_negative * 0.4)
    n_neg_script = n_negative - n_neg_llm

    print(f"Target: {n_positive} positive + {n_negative} negative → {args.output}")
    print(f"  Positive: {n_pos_llm} LLM + {n_pos_script} script")
    print(f"  Negative: {n_neg_llm} LLM + {n_neg_script} script")

    all_samples: List[Dict[str, Any]] = []

    # Script-generated samples (no LLM needed)
    print(f"\n[+] Generating {n_pos_script} script-generated positives...")
    all_samples.extend(generate_script_positive_samples(n_pos_script))

    print(f"[-] Generating {n_neg_script} script-generated negatives...")
    all_samples.extend(generate_script_negative_samples(n_neg_script))

    # LLM-generated samples
    if not args.no_llm:
        try:
            _validate_lm_studio_url(args.lm_studio_url)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"\n[LLM] Generating {n_pos_llm} positive + {n_neg_llm} negative via LLM "
              f"({args.model} @ {args.lm_studio_url}, concurrency={args.max_concurrent})...")

        async def _run_llm() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            pos = await generate_llm_samples_async(
                n_pos_llm, args.model, args.lm_studio_url, args.max_concurrent,
                label=1, progress_prefix="+ ",
            )
            neg = await generate_llm_samples_async(
                n_neg_llm, args.model, args.lm_studio_url, args.max_concurrent,
                label=0, progress_prefix="- ",
            )
            return pos, neg

        llm_pos, llm_neg = asyncio.run(_run_llm())
        all_samples.extend(llm_pos)
        all_samples.extend(llm_neg)
    else:
        print("Skipping LLM generation (--no-llm).")

    random.shuffle(all_samples)

    if not args.no_features:
        print(f"\n[F] Extracting features from {len(all_samples)} records...")
        all_samples = [_attach_features(s) for s in all_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for record in all_samples:
            f.write(json.dumps(record) + "\n")

    pos_n = sum(1 for r in all_samples if r.get("label") == 1)
    neg_n = sum(1 for r in all_samples if r.get("label") == 0)
    print(f"\nWrote {len(all_samples)} records → {args.output}")
    print(f"  positive (secret): {pos_n}")
    print(f"  negative (safe):   {neg_n}")


if __name__ == "__main__":
    main()
