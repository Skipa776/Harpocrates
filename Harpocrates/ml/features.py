"""
Feature engineering for ML-based secrets verification.

Extracts 65 features from tokens and their context to enable
context-aware classification of potential secrets.

Features are organized into three categories:
- Token features (23): Properties of the token itself including advanced
  entropy analysis, vendor prefix detection, and discriminative features
  (is_uuid_v4, is_known_hash_length, jwt_structure_valid, entropy_charset_mismatch, has_hash_prefix)
- Variable name features (10): Properties of the variable/key name including
  N-gram scoring for secret and safe patterns
- Context features (18): Properties of surrounding code including semantic
  analysis and secret density
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from Harpocrates.detectors.entropy_detector import shannon_entropy
from Harpocrates.ml.context import (
    CodeContext,
    extract_var_name,
)

if TYPE_CHECKING:
    from Harpocrates.core.result import Finding

# Known secret prefixes and their categories
KNOWN_PREFIXES: Dict[str, int] = {
    "AKIA": 1,  # AWS Access Key
    "ASIA": 1,  # AWS STS
    "ghp_": 2,  # GitHub PAT
    "gho_": 2,  # GitHub OAuth
    "ghu_": 2,  # GitHub User-to-server
    "ghs_": 2,  # GitHub Server-to-server
    "ghr_": 2,  # GitHub Refresh token
    "github_pat_": 2,  # GitHub Fine-grained PAT
    "sk_live_": 3,  # Stripe Live
    "sk_test_": 3,  # Stripe Test
    "pk_live_": 3,  # Stripe Publishable Live
    "pk_test_": 3,  # Stripe Publishable Test
    "sk-": 4,  # OpenAI
    "xox": 5,  # Slack (xoxb, xoxp, xoxa, xoxr, xoxs)
    "AIza": 6,  # Google API
    "eyJ": 7,  # JWT (base64 encoded JSON)
    "npm_": 8,  # NPM token
    "pypi-": 8,  # PyPI token
}

# Variable name patterns indicating secrets
SECRET_VAR_PATTERNS = [
    re.compile(r"(secret|password|passwd|pwd|credential)", re.I),
    re.compile(r"(api[_-]?key|access[_-]?key|auth[_-]?key)", re.I),
    re.compile(r"(token|bearer|private[_-]?key)", re.I),
    re.compile(r"(api[_-]?secret|client[_-]?secret)", re.I),
]

# Variable name patterns indicating safe (non-secret) content
SAFE_VAR_PATTERNS = [
    re.compile(r"(hash|sha|sha1|sha256|sha512|md5)", re.I),
    re.compile(r"(commit|rev|revision|version)", re.I),
    re.compile(r"(uuid|guid|identifier)", re.I),
    # Match _id suffix but NOT when preceded by key/secret/api (e.g., user_id yes, key_id no)
    re.compile(r"(?<!key)(?<!secret)(?<!api)_id\b", re.I),
    re.compile(r"(checksum|digest|fingerprint)", re.I),
]

# Base64 alphabet
BASE64_CHARS = set(string.ascii_letters + string.digits + "+/=")
BASE64_URL_CHARS = set(string.ascii_letters + string.digits + "-_=")
HEX_CHARS = set(string.hexdigits)

# UUID v4 pattern (8-4-4-4-12 with version 4 marker)
UUID_V4_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)

# Known hash length patterns
KNOWN_HASH_LENGTHS = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}

# Maximum possible entropy per character class
MAX_ENTROPY_ALPHANUMERIC = 5.954  # log2(62) for a-zA-Z0-9
MAX_ENTROPY_BASE64 = 6.0  # log2(64)
MAX_ENTROPY_HEX = 4.0  # log2(16)

# v0.2.0: AI-generated placeholder detection (design doc §2 "AI Leftover" Paradigm).
# Word-boundary anchors are intentionally omitted for underscore-joined patterns like
# dummy_token, DUMMY_SECRET, FAKE_API_KEY which are the primary LLM leftovers.
_AI_PLACEHOLDER_RE = re.compile(
    r"YOUR_|_HERE|dummy|CHANGEME|PLACEHOLDER|<[A-Z_]+>|TODO|REPLACE[_\-]|FAKE|TEST[_\-]",
    re.IGNORECASE,
)

# v0.2.0: Connection-string and credential-context keywords
_CONNECTION_KEYWORDS = [
    "jdbc:", "bearer", "authorization:", "database", "password",
    "credentials", "connection_string", "dsn", "api_key", "access_key",
    "secret_key", "private_key", "auth_token", "client_secret",
]

# Variable name N-gram weights for secret detection
# STAGE B GENERALIZATION: Weights scaled by 0.4 (max was 1.0 → now 0.40)
# This prevents N-gram scores from dominating predictions and forces
# the model to learn from multiple signals instead of a single feature.
SECRET_NGRAMS: Dict[str, float] = {
    # Highest confidence secret indicators (scaled: 1.0 → 0.40)
    "api_key": 0.40, "apikey": 0.40, "api-key": 0.40,
    "secret": 0.36, "password": 0.36, "passwd": 0.36, "pwd": 0.34,
    "private_key": 0.40, "privatekey": 0.40,
    "access_key": 0.38, "accesskey": 0.38,
    "auth_token": 0.38, "authtoken": 0.38,
    "bearer": 0.36, "credential": 0.36,
    # Medium confidence (scaled: 0.7 → 0.28, 0.5 → 0.20)
    "token": 0.28, "key": 0.20,  # Generic, needs context
    "client_secret": 0.38, "clientsecret": 0.38,
    "signing_key": 0.36, "signingkey": 0.36,
    "encryption_key": 0.36, "encryptionkey": 0.36,
}

# Variable name N-gram weights for safe (non-secret) detection
# STAGE B GENERALIZATION: Weights scaled by 0.4 (max was 1.0 → now 0.40)
SAFE_NGRAMS: Dict[str, float] = {
    # Highest confidence safe indicators (scaled: 1.0 → 0.40)
    "commit": 0.36, "commit_sha": 0.40, "commitsha": 0.40,
    "sha": 0.32, "sha1": 0.36, "sha256": 0.38, "sha512": 0.38,
    "hash": 0.34, "checksum": 0.38, "digest": 0.36,
    "uuid": 0.38, "guid": 0.38, "identifier": 0.28,
    "version": 0.32, "revision": 0.34, "rev": 0.28,
    # Context-dependent safe indicators (scaled)
    "example": 0.36, "sample": 0.34, "placeholder": 0.38,
    "test": 0.28, "mock": 0.34, "fake": 0.36, "dummy": 0.36,
    "fingerprint": 0.36, "session_id": 0.28, "trace_id": 0.32,
}


@dataclass
class FeatureVector:
    """
    63 extracted features for ML classification.

    Organized into five categories:
    - Token features (23): Properties of the token itself
    - Variable name features (10): Properties of the variable/key name
    - Context features (18): Properties of surrounding code
    - Stage B precision features (7): Targeted FP reduction features
    - Stage B generalization features (5): Hex disambiguation features

    NOTE: The following features were REMOVED to prevent shortcut learning:
    - has_known_prefix: Directly encodes token type based on prefix
    - prefix_type: Maps prefixes to secret categories
    - is_hex_like: Strongly correlated with non-secrets (git SHAs)

    DISCRIMINATIVE FEATURES (for precision boost):
    - is_uuid_v4: Detects UUID v4 format (strong non-secret indicator)
    - is_known_hash_length: Token length matches MD5/SHA1/SHA256/SHA512
    - jwt_structure_valid: JWT has valid base64-encoded JSON header
    - entropy_charset_mismatch: High entropy but low charset diversity (suspicious)
    - has_hash_prefix: Starts with hash algorithm prefix (sha256:, md5:)

    STAGE B PRECISION FEATURES (targeting observed FP patterns):
    - is_hex_len_40: Token is exactly 40 hex chars (git SHA/SHA1)
    - is_hex_len_64: Token is exactly 64 hex chars (SHA256)
    - is_test_token: Has _test_, _staging_, test_ prefix
    - contains_example_keyword: Contains EXAMPLE, xxxx, demo, placeholder
    - file_is_git_related: Path contains .git/, hash, commit
    - file_is_build: Path contains build/, dist/, node_modules/
    - file_is_example: Path contains example/, docs/, demo/

    STAGE B GENERALIZATION FEATURES (hex disambiguation):
    - hex_context_git_keywords: Context contains git, commit, merge keywords
    - hex_context_crypto_keywords: Context contains encrypt, sign, key keywords
    - hex_adjacent_assignment_pattern: Assignment pattern (env_var, config, func_arg)
    - hex_in_url_or_dsn: Token is embedded in URL/DSN
    - hex_file_suggests_secret: File path suggests secrets (secrets/, .env)

    The model should learn from CONTEXT, not token format.
    """

    # Token features (23) - includes advanced entropy analysis + position/structure
    token_length: int = 0
    token_entropy: float = 0.0
    char_class_count: int = 0  # Number of character classes present
    digit_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    special_char_ratio: float = 0.0
    is_base64_like: bool = False
    has_padding: bool = False  # Ends with = or ==
    regex_match_type: int = 0  # 0 = entropy, 1+ = regex pattern
    token_structure_score: float = 0.0  # 0=random, 1=structured (words, separators)
    has_version_pattern: bool = False  # Contains v1.2.3, 1.0.0 patterns
    # Advanced entropy features
    normalized_entropy: float = 0.0  # entropy / max_possible (0-1 scale)
    cryptographic_score: float = 0.0  # 0=structured, 1=cryptographically random
    vendor_prefix_boost: float = 0.0  # Boost for known vendor prefixes (AKIA, ghp_, etc.)
    # Position and structural features
    token_span_offset: float = 0.0  # Position within line (0=start, 1=end)
    token_in_multiline_block: bool = False  # Part of PEM/SSH block
    embedded_token_flag: bool = False  # Extracted from URL/DSN
    token_quote_type: int = 0  # 0=none, 1=single, 2=double, 3=backtick
    # NEW: Discriminative features for precision boost
    is_uuid_v4: bool = False  # Matches UUID v4 format (strong non-secret signal)
    is_known_hash_length: bool = False  # Length matches MD5/SHA1/SHA256/SHA512
    jwt_structure_valid: bool = False  # JWT has valid JSON header
    entropy_charset_mismatch: float = 0.0  # High entropy but low charset diversity
    has_hash_prefix: bool = False  # Starts with sha256:, md5:, etc.

    # Variable name features (10) - includes N-gram scoring
    var_name_extracted: bool = False
    var_contains_secret: bool = False
    var_contains_safe: bool = False
    var_name_length: int = 0
    var_is_uppercase: bool = False  # CONSTANT_STYLE
    var_is_camelcase: bool = False
    assignment_type: int = 0  # 0=unknown, 1==, 2=:, 3=:=, 4=func_arg
    in_string_literal: bool = True
    # N-gram weighted scores
    var_ngram_secret_score: float = 0.0  # Weighted sum of secret N-gram matches
    var_ngram_safe_score: float = 0.0  # Weighted sum of safe N-gram matches

    # Context features (18)
    line_is_comment: bool = False
    context_mentions_test: bool = False
    context_mentions_git: bool = False
    context_mentions_hash: bool = False
    context_has_import: bool = False
    context_has_function_def: bool = False
    file_is_test: bool = False
    file_is_config: bool = False
    file_extension_risk: int = 0  # 0=low, 1=medium, 2=high
    surrounding_entropy_avg: float = 0.0
    semantic_context_score: float = 0.0  # -1=safe, 0=neutral, 1=risky
    line_position_ratio: float = 0.0  # 0=top, 1=bottom (secrets often at top)
    surrounding_secret_density: float = 0.0  # Ratio of nearby secret-like tokens
    # NEW: Advanced context features
    surrounding_token_count: int = 0  # Count of high-entropy tokens on same line
    key_value_distance: int = -1  # Distance between var name and token (-1 if no var)
    json_path_hint: int = 0  # Depth in JSON/YAML structure (0=root)
    adjacency_ngram_score: float = 0.0  # Sum of N-gram scores in nearby var names
    cross_line_entropy: float = 0.0  # Average entropy across ±3 lines

    # STAGE B PRECISION IMPROVEMENT: Features targeting observed FP patterns
    is_hex_len_40: bool = False  # Token is exactly 40 hex chars (git SHA/SHA1)
    is_hex_len_64: bool = False  # Token is exactly 64 hex chars (SHA256)
    is_test_token: bool = False  # Has _test_, _staging_, test_ prefix
    contains_example_keyword: bool = False  # Contains EXAMPLE, xxxx, demo, placeholder
    file_is_git_related: bool = False  # Path contains .git/, hash, commit
    file_is_build: bool = False  # Path contains build/, dist/, node_modules/
    file_is_example: bool = False  # Path contains example/, docs/, demo/

    # STAGE B GENERALIZATION IMPROVEMENT: Hex disambiguation features
    # These help distinguish hex secrets from git SHAs/checksums
    hex_context_git_keywords: bool = False  # Context contains git, commit, merge, branch
    hex_context_crypto_keywords: bool = False  # Context contains encrypt, sign, key, secret
    hex_adjacent_assignment_pattern: int = 0  # 0=unknown, 1=env_var, 2=config, 3=func_arg
    hex_in_url_or_dsn: bool = False  # Token is embedded in URL/DSN
    hex_file_suggests_secret: bool = False  # Path suggests secrets (secrets/, .env, credentials/)

    # Env-loading awareness features (v0.2.0 — 63 → 65)
    env_loader_in_context: bool = False  # Context contains env-loading pattern
    is_env_fallback_value: bool = False  # Token is the fallback/default arg in env-loader call

    def to_array(self) -> List[float]:
        """Convert to numpy-compatible array of 65 floats."""
        return [
            # Token features (23)
            float(self.token_length),
            self.token_entropy,
            float(self.char_class_count),
            self.digit_ratio,
            self.uppercase_ratio,
            self.special_char_ratio,
            float(self.is_base64_like),
            float(self.has_padding),
            float(self.regex_match_type),
            self.token_structure_score,
            float(self.has_version_pattern),
            self.normalized_entropy,
            self.cryptographic_score,
            self.vendor_prefix_boost,
            # Token position/structure features
            self.token_span_offset,
            float(self.token_in_multiline_block),
            float(self.embedded_token_flag),
            float(self.token_quote_type),
            # NEW: Discriminative features for precision boost
            float(self.is_uuid_v4),
            float(self.is_known_hash_length),
            float(self.jwt_structure_valid),
            self.entropy_charset_mismatch,
            float(self.has_hash_prefix),
            # Variable name features (10)
            float(self.var_name_extracted),
            float(self.var_contains_secret),
            float(self.var_contains_safe),
            float(self.var_name_length),
            float(self.var_is_uppercase),
            float(self.var_is_camelcase),
            float(self.assignment_type),
            float(self.in_string_literal),
            self.var_ngram_secret_score,
            self.var_ngram_safe_score,
            # Context features (18)
            float(self.line_is_comment),
            float(self.context_mentions_test),
            float(self.context_mentions_git),
            float(self.context_mentions_hash),
            float(self.context_has_import),
            float(self.context_has_function_def),
            float(self.file_is_test),
            float(self.file_is_config),
            float(self.file_extension_risk),
            self.surrounding_entropy_avg,
            self.semantic_context_score,
            self.line_position_ratio,
            self.surrounding_secret_density,
            # Advanced context features
            float(self.surrounding_token_count),
            float(self.key_value_distance),
            float(self.json_path_hint),
            self.adjacency_ngram_score,
            self.cross_line_entropy,
            # Stage B precision improvement features (7)
            float(self.is_hex_len_40),
            float(self.is_hex_len_64),
            float(self.is_test_token),
            float(self.contains_example_keyword),
            float(self.file_is_git_related),
            float(self.file_is_build),
            float(self.file_is_example),
            # Stage B generalization improvement features (5)
            float(self.hex_context_git_keywords),
            float(self.hex_context_crypto_keywords),
            float(self.hex_adjacent_assignment_pattern),
            float(self.hex_in_url_or_dsn),
            float(self.hex_file_suggests_secret),
            # Env-loading awareness features (2)
            float(self.env_loader_in_context),
            float(self.is_env_fallback_value),
        ]

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get ordered list of 65 feature names."""
        return [
            # Token features (23)
            "token_length",
            "token_entropy",
            "char_class_count",
            "digit_ratio",
            "uppercase_ratio",
            "special_char_ratio",
            "is_base64_like",
            "has_padding",
            "regex_match_type",
            "token_structure_score",
            "has_version_pattern",
            "normalized_entropy",
            "cryptographic_score",
            "vendor_prefix_boost",
            "token_span_offset",
            "token_in_multiline_block",
            "embedded_token_flag",
            "token_quote_type",
            # NEW: Discriminative features
            "is_uuid_v4",
            "is_known_hash_length",
            "jwt_structure_valid",
            "entropy_charset_mismatch",
            "has_hash_prefix",
            # Variable name features (10)
            "var_name_extracted",
            "var_contains_secret",
            "var_contains_safe",
            "var_name_length",
            "var_is_uppercase",
            "var_is_camelcase",
            "assignment_type",
            "in_string_literal",
            "var_ngram_secret_score",
            "var_ngram_safe_score",
            # Context features (18)
            "line_is_comment",
            "context_mentions_test",
            "context_mentions_git",
            "context_mentions_hash",
            "context_has_import",
            "context_has_function_def",
            "file_is_test",
            "file_is_config",
            "file_extension_risk",
            "surrounding_entropy_avg",
            "semantic_context_score",
            "line_position_ratio",
            "surrounding_secret_density",
            "surrounding_token_count",
            "key_value_distance",
            "json_path_hint",
            "adjacency_ngram_score",
            "cross_line_entropy",
            # Stage B precision improvement features (7)
            "is_hex_len_40",
            "is_hex_len_64",
            "is_test_token",
            "contains_example_keyword",
            "file_is_git_related",
            "file_is_build",
            "file_is_example",
            # Stage B generalization improvement features (5)
            "hex_context_git_keywords",
            "hex_context_crypto_keywords",
            "hex_adjacent_assignment_pattern",
            "hex_in_url_or_dsn",
            "hex_file_suggests_secret",
            # Env-loading awareness features (2)
            "env_loader_in_context",
            "is_env_fallback_value",
        ]

def _get_char_class_count(token: str) -> int:
    """Count distinct character classes in token."""
    classes = 0
    if any(c.islower() for c in token):
        classes += 1
    if any(c.isupper() for c in token):
        classes += 1
    if any(c.isdigit() for c in token):
        classes += 1
    if any(not c.isalnum() for c in token):
        classes += 1
    return classes


def _get_prefix_type(token: str) -> int:
    """Get encoded prefix type for known secret formats."""
    for prefix, prefix_type in KNOWN_PREFIXES.items():
        if token.startswith(prefix):
            return prefix_type
    return 0


def _is_base64_like(token: str) -> bool:
    """Check if token looks like base64 encoded data."""
    if len(token) < 4:
        return False
    # Check if mostly base64 characters
    base64_chars = sum(1 for c in token if c in BASE64_CHARS)
    url_safe_chars = sum(1 for c in token if c in BASE64_URL_CHARS)
    return max(base64_chars, url_safe_chars) / len(token) >= 0.95


def _is_hex_like(token: str) -> bool:
    """Check if token is pure hexadecimal."""
    if len(token) < 8:
        return False
    return all(c in HEX_CHARS for c in token)


def _is_camelcase(name: str) -> bool:
    """Check if variable name uses camelCase."""
    if not name or name.isupper() or name.islower():
        return False
    # Has mix of upper and lower with lower start
    return name[0].islower() and any(c.isupper() for c in name[1:])


def _calculate_token_structure_score(token: str) -> float:
    """
    Calculate how structured (non-random) a token appears.

    Returns:
        0.0 = completely random (likely secret)
        1.0 = highly structured (likely not secret)
    """
    if not token or len(token) < 4:
        return 0.0

    score = 0.0

    # Check for repeated characters (aaaa, xxxx) - indicates placeholder
    if re.search(r'(.)\1{3,}', token):
        score += 0.25

    # Check for word-like patterns (sequences of lowercase letters)
    word_matches = re.findall(r'[a-z]{4,}', token, re.I)
    if word_matches:
        # More word-like patterns = more structured
        score += min(0.3, len(word_matches) * 0.1)

    # Check for structural separators (multiple dashes, underscores, dots)
    if re.search(r'[-_./]{2,}', token):
        score += 0.15

    # Check for version-like patterns
    if re.search(r'v?\d+\.\d+', token):
        score += 0.2

    # Check for common placeholder patterns
    placeholder_patterns = [
        r'example', r'sample', r'test', r'demo', r'fake',
        r'xxxx', r'0000', r'placeholder', r'your[_-]?',
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, token, re.I):
            score += 0.1
            break

    return min(score, 1.0)


def _has_version_pattern(token: str) -> bool:
    """Check if token contains version-like patterns."""
    if not token:
        return False
    # Match patterns like v1.2.3, 1.0.0, v2.0, etc.
    version_patterns = [
        re.compile(r'v\d+\.\d+(\.\d+)?', re.I),  # v1.2.3
        re.compile(r'\d+\.\d+\.\d+'),  # 1.2.3
        re.compile(r'version[_-]?\d+', re.I),  # version1, version_1
    ]
    return any(p.search(token) for p in version_patterns)


def _calculate_normalized_entropy(token: str) -> float:
    """
    Calculate entropy normalized by the maximum possible entropy for the character set.

    Returns:
        0.0-1.0: Normalized entropy (1.0 = maximum randomness for char set)
    """
    if not token or len(token) < 4:
        return 0.0

    raw_entropy = shannon_entropy(token)

    # Determine the character set used
    has_special = any(not c.isalnum() for c in token)

    # Determine max entropy based on character set
    if not has_special:
        if _is_hex_like(token):
            max_entropy = MAX_ENTROPY_HEX
        elif _is_base64_like(token):
            max_entropy = MAX_ENTROPY_BASE64
        else:
            max_entropy = MAX_ENTROPY_ALPHANUMERIC
    else:
        # Include special characters - assume ~90 printable ASCII chars
        max_entropy = 6.5  # log2(90) ≈ 6.5

    return min(raw_entropy / max_entropy, 1.0)


def _calculate_cryptographic_score(token: str) -> float:
    """
    Score how cryptographically random a token appears.

    This uses multiple heuristics to distinguish cryptographic secrets from
    structured data like UUIDs, version strings, etc.

    Returns:
        0.0 = structured/predictable
        1.0 = cryptographically random
    """
    if not token or len(token) < 8:
        return 0.0

    score = 0.0

    # High normalized entropy is a strong indicator
    norm_entropy = _calculate_normalized_entropy(token)
    if norm_entropy > 0.9:
        score += 0.4
    elif norm_entropy > 0.7:
        score += 0.2

    # Check for lack of structure
    structure_score = _calculate_token_structure_score(token)
    if structure_score < 0.1:
        score += 0.3
    elif structure_score < 0.3:
        score += 0.15

    # Check character distribution uniformity
    # Cryptographic secrets have more uniform character distribution
    char_counts = {}
    for c in token:
        char_counts[c] = char_counts.get(c, 0) + 1

    if char_counts:
        avg_count = len(token) / len(char_counts)
        max_count = max(char_counts.values())
        # Low max deviation from average = more uniform
        uniformity = 1 - (max_count - avg_count) / len(token) if len(token) > 0 else 0
        if uniformity > 0.8:
            score += 0.2
        elif uniformity > 0.6:
            score += 0.1

    # Penalize known non-cryptographic patterns
    # UUIDs have predictable structure
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', token, re.I):
        score -= 0.4

    # Git SHA (40 hex chars) - common false positive
    if len(token) == 40 and all(c in HEX_CHARS for c in token):
        score -= 0.3

    return max(0.0, min(1.0, score))


def _get_vendor_prefix_boost(token: str) -> float:
    """
    Return a boost score for tokens with known vendor secret prefixes.

    Known prefixes indicate high probability of being a real secret.

    Returns:
        0.0 = no known prefix
        0.5-1.0 = known vendor prefix with confidence level
    """
    if not token:
        return 0.0

    # Highest confidence prefixes (1.0)
    high_confidence_prefixes = [
        "AKIA",  # AWS Access Key ID (exactly 20 chars)
        "ASIA",  # AWS STS (exactly 20 chars)
        "ghp_",  # GitHub PAT
        "gho_",  # GitHub OAuth
        "ghu_",  # GitHub User-to-server
        "ghs_",  # GitHub Server-to-server
        "ghr_",  # GitHub Refresh
        "github_pat_",  # GitHub Fine-grained PAT
        "sk_live_",  # Stripe Live
        "rk_live_",  # Stripe Restricted Key Live
    ]

    for prefix in high_confidence_prefixes:
        if token.startswith(prefix):
            return 1.0

    # Medium confidence prefixes (0.8)
    medium_confidence_prefixes = [
        "sk_test_",  # Stripe Test
        "rk_test_",  # Stripe Restricted Key Test
        "pk_live_",  # Stripe Publishable Live
        "pk_test_",  # Stripe Publishable Test
        "xoxb-",  # Slack Bot
        "xoxp-",  # Slack User
        "xoxa-",  # Slack App
        "xoxr-",  # Slack Refresh
        "npm_",  # NPM token
        "pypi-",  # PyPI token
    ]

    for prefix in medium_confidence_prefixes:
        if token.startswith(prefix):
            return 0.8

    # Lower confidence prefixes (0.6) - more common/generic
    lower_confidence_prefixes = [
        "sk-",  # OpenAI (but could be other things)
        "AIza",  # Google API
        "eyJ",  # JWT (but needs verification)
        "Bearer ",  # Bearer token (often not the actual secret)
    ]

    for prefix in lower_confidence_prefixes:
        if token.startswith(prefix):
            return 0.6

    return 0.0


# --- NEW DISCRIMINATIVE FEATURES ---
# These features help distinguish real secrets from look-alikes


def _is_uuid_v4(token: str) -> bool:
    """
    Check if token matches UUID v4 format.

    UUID v4 has the format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    where x is a hex digit and y is one of 8, 9, a, or b.

    UUIDs are strong non-secret indicators - they are identifiers, not credentials.

    Returns:
        True if token is a valid UUID v4 format
    """
    if not token or len(token) != 36:
        return False
    return bool(UUID_V4_PATTERN.match(token))


def _is_known_hash_length(token: str) -> bool:
    """
    Check if token length matches known hash algorithm output lengths.

    Common hash lengths:
    - MD5: 32 hex characters
    - SHA-1: 40 hex characters (also Git SHA)
    - SHA-256: 64 hex characters
    - SHA-512: 128 hex characters

    Returns:
        True if token length matches a known hash length AND is hex-like
    """
    if not token:
        return False

    # Must be all hex characters
    if not all(c in HEX_CHARS for c in token):
        return False

    return len(token) in KNOWN_HASH_LENGTHS


def _is_jwt_structure_valid(token: str) -> bool:
    """
    Check if a JWT-like token has valid structure.

    A valid JWT has:
    1. Three parts separated by dots
    2. First part (header) is valid base64-encoded JSON with 'alg' field
    3. Second part (payload) is valid base64-encoded JSON

    Invalid/test JWTs often have:
    - alg: "none"
    - Obviously fake/test payloads
    - Expired timestamps from the past

    Returns:
        True if JWT structure appears valid (could be a real secret)
        False if JWT is invalid/test/expired (likely not a real secret)
    """
    if not token or "." not in token:
        return False

    parts = token.split(".")
    if len(parts) != 3:
        return False

    try:
        import base64
        import json

        # Decode header
        header_b64 = parts[0]
        # Add padding if needed
        padding = 4 - len(header_b64) % 4
        if padding != 4:
            header_b64 += "=" * padding

        header_json = base64.urlsafe_b64decode(header_b64).decode('utf-8')
        header = json.loads(header_json)

        # Check for invalid algorithm
        if header.get("alg") == "none":
            return False

        # Check for invalid type
        if header.get("typ") not in (None, "JWT", "at+jwt"):
            return False

        # Decode payload
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload_json = base64.urlsafe_b64decode(payload_b64).decode('utf-8')
        payload = json.loads(payload_json)

        # Check for test/example markers
        sub = str(payload.get("sub", "")).lower()
        if any(marker in sub for marker in ["test", "example", "fake", "placeholder", "demo"]):
            return False

        # Check for obviously expired tokens (before 2020)
        exp = payload.get("exp")
        if exp is not None and isinstance(exp, (int, float)):
            # Timestamp before 2020-01-01
            if exp < 1577836800:
                return False

        return True

    except Exception:
        # If we can't decode, it's not a valid JWT
        return False


def _calculate_entropy_charset_mismatch(token: str) -> float:
    """
    Calculate mismatch between entropy and charset diversity.

    A secret-like token should have:
    - High entropy AND high charset diversity (uses many character types)

    A non-secret often has:
    - High entropy but low charset diversity (e.g., hex hashes, UUIDs)
    - Low entropy with high charset diversity (e.g., words with special chars)

    Returns:
        0.0 = entropy matches charset expectations (likely real secret or benign)
        0.3-0.6 = moderate mismatch (suspicious)
        0.6-1.0 = high mismatch (likely not a secret)
    """
    if not token or len(token) < 8:
        return 0.0

    entropy = shannon_entropy(token)
    char_classes = _get_char_class_count(token)

    # If we have high entropy but low charset diversity, it's suspicious
    # This catches hex hashes and UUIDs which have high entropy but only hex chars
    # Threshold: entropy > 3.5 (typical for random hex) and only 1-2 char classes
    if entropy > 3.5 and char_classes <= 2:
        # High entropy with limited charset - likely hash/UUID
        # Scale from 0.3 at entropy=3.5 to 0.8 at entropy=4.5+
        mismatch = 0.3 + min(0.5, (entropy - 3.5) / 2.0)
        return min(1.0, mismatch)

    # If we have low entropy with high charset, it's structured text (not concerning)
    if entropy < 3.0 and char_classes >= 3:
        return 0.2  # Slightly indicative but not definitive

    return 0.0


def _has_hash_prefix(token: str) -> bool:
    """
    Check if token starts with a hash algorithm prefix.

    Common patterns:
    - sha256:xxxx
    - sha1:xxxx
    - md5:xxxx
    - sha512:xxxx

    These prefixes strongly indicate the value is a hash, not a secret.

    Returns:
        True if token has a hash algorithm prefix
    """
    if not token:
        return False

    hash_prefixes = [
        "sha256:", "SHA256:",
        "sha1:", "SHA1:",
        "sha512:", "SHA512:",
        "md5:", "MD5:",
        "sha384:", "SHA384:",
        "blake2:", "BLAKE2:",
    ]

    return any(token.startswith(prefix) for prefix in hash_prefixes)


# --- STAGE B PRECISION IMPROVEMENT FEATURES ---
# These features target specific false positive patterns observed in Stage B


def _is_hex_exact_length(token: str, length: int) -> bool:
    """
    Check if token is exactly `length` hex characters.

    Args:
        token: The token to check
        length: Expected length (40 for SHA1/git, 64 for SHA256)

    Returns:
        True if token is exactly `length` hex characters
    """
    if not token or len(token) != length:
        return False
    return all(c in HEX_CHARS for c in token)


def _is_test_token(token: str) -> bool:
    """
    Check if token has test/staging prefix patterns.

    Targets FP pattern: sk_test_*, pk_test_*, rk_test_* tokens that are
    NOT real secrets but test/sandbox credentials.

    Returns:
        True if token contains test/staging indicators
    """
    if not token:
        return False

    test_patterns = [
        '_test_', '_staging_', 'test_', 'TEST_', '_demo_',
        'sk_test_', 'pk_test_', 'rk_test_',
    ]
    return any(p in token for p in test_patterns)


def _contains_example_keyword(token: str) -> bool:
    """
    Check if token contains known example/placeholder patterns.

    Targets FP pattern: Documentation example tokens like AKIAIOSFODNN7EXAMPLE
    or placeholder patterns with 'xxxx', 'demo', etc.

    Returns:
        True if token contains example/placeholder keywords
    """
    if not token:
        return False

    patterns = [
        'EXAMPLE', 'example', 'xxxx', 'XXXX', 'demo', 'DEMO',
        'placeholder', 'your_', 'YOUR_', '0000', 'sample', 'SAMPLE',
        'fake', 'FAKE', 'dummy', 'DUMMY', 'mock', 'MOCK',
    ]
    return any(p in token for p in patterns)


def _is_git_related_path(file_path: Optional[str]) -> bool:
    """
    Check if file path is git-related.

    Targets FP pattern: 40-char hex tokens in .git/ directories or
    files with git-related names that contain commit SHAs.

    Returns:
        True if path indicates git-related content
    """
    if not file_path:
        return False

    path_lower = file_path.lower()
    patterns = [
        '.git/', '/hash', 'commit', '/objects/', 'refs/',
        'git-sha', 'gitsha', 'commit_sha', 'commitsha',
    ]
    return any(p in path_lower for p in patterns)


def _is_build_path(file_path: Optional[str]) -> bool:
    """
    Check if file path is in build/dist directories.

    Targets FP pattern: Hex tokens in build artifacts, node_modules,
    vendor directories that are checksums or asset hashes.

    Returns:
        True if path is a build/artifact directory
    """
    if not file_path:
        return False

    path_lower = file_path.lower()
    patterns = [
        'build/', 'dist/', 'node_modules/', 'vendor/',
        'target/', '.output/', '__pycache__/', '.cache/',
        'pkg/', 'bin/', 'obj/', '.next/', '.nuxt/',
    ]
    return any(p in path_lower for p in patterns)


def _is_example_path(file_path: Optional[str]) -> bool:
    """
    Check if file path is in example/docs directories.

    Targets FP pattern: Tokens in documentation, examples, or demo
    directories that are often placeholder/example credentials.

    Returns:
        True if path indicates example/documentation content
    """
    if not file_path:
        return False

    # Case-sensitive for README, CONTRIBUTING
    patterns_case_sensitive = ['README', 'CONTRIBUTING']
    if any(p in file_path for p in patterns_case_sensitive):
        return True

    path_lower = file_path.lower()
    patterns = [
        'example', 'docs/', 'demo/', 'samples/',
        'tutorial/', 'fixtures/', 'mock/', 'stub/',
    ]
    return any(p in path_lower for p in patterns)


def _calculate_ngram_score(var_name: str, ngram_dict: Dict[str, float]) -> float:
    """
    Calculate weighted N-gram score for a variable name.

    STAGE B GENERALIZATION: Changed from MAX to SUM calculation.
    Using sum requires multiple signals to accumulate a high score,
    preventing single-feature domination. The 0.3 scale factor ensures
    that a variable with multiple matches doesn't exceed 1.0 too easily.

    Args:
        var_name: The variable/key name to analyze
        ngram_dict: Dictionary mapping N-grams to weights

    Returns:
        0.0-1.0: Weighted sum of matches (capped at 1.0)
    """
    if not var_name:
        return 0.0

    var_lower = var_name.lower()
    total_score = 0.0

    for ngram, weight in ngram_dict.items():
        if ngram in var_lower:
            total_score += weight * 0.5  # Scale factor to prevent score explosion

    return min(total_score, 1.0)  # Cap at 1.0


def _calculate_semantic_context_score(context_text: str) -> float:
    """
    Calculate a weighted semantic context score.

    Returns:
        -1.0 = strongly suggests safe (non-secret)
         0.0 = neutral
        +1.0 = strongly suggests secret
    """
    if not context_text:
        return 0.0

    score = 0.0

    # Safe indicators (negative score)
    safe_patterns = [
        (r'\b(test|mock|fake|example|sample|demo)\b', -0.3),
        (r'\b(commit|revision|sha|hash|checksum|digest)\b', -0.25),
        (r'\b(uuid|guid|identifier|id)\b', -0.2),
        (r'\b(placeholder|dummy|stub)\b', -0.3),
        (r'#.*example', -0.2),  # Comments mentioning example
        (r'//.*test', -0.2),  # Comments mentioning test
    ]

    # Risky indicators (positive score)
    risky_patterns = [
        (r'\b(secret|password|credential|auth)\b', 0.3),
        (r'\b(api[_-]?key|access[_-]?key|private[_-]?key)\b', 0.35),
        (r'\b(production|prod|live)\b', 0.2),
        (r'\.(env|secret|key|pem)\b', 0.25),
        (r'\b(export|set|define)\s+\w*(secret|key|token)', 0.3),
    ]

    for pattern, weight in safe_patterns:
        if re.search(pattern, context_text, re.I):
            score += weight

    for pattern, weight in risky_patterns:
        if re.search(pattern, context_text, re.I):
            score += weight

    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, score))


def _calculate_surrounding_secret_density(context_text: str) -> float:
    """
    Calculate the density of secret-like tokens in surrounding context.

    Returns:
        0.0-1.0: Ratio of high-entropy tokens that look like secrets
    """
    if not context_text:
        return 0.0

    # Find all high-entropy token candidates
    token_pattern = re.compile(r'[A-Za-z0-9+/=_\-.]{16,}')
    tokens = token_pattern.findall(context_text)

    if not tokens:
        return 0.0

    secret_like_count = 0
    for token in tokens[:20]:  # Limit to first 20 tokens
        # Check if token looks like a secret
        entropy = shannon_entropy(token)
        if entropy > 4.0:
            # High entropy
            if len(token) >= 20:
                secret_like_count += 1
        # Check for known secret prefixes
        for prefix in KNOWN_PREFIXES:
            if token.startswith(prefix):
                secret_like_count += 1
                break

    return secret_like_count / len(tokens[:20])


def _calculate_token_span_offset(line: str, token: str) -> float:
    """
    Calculate the position of the token within the line.

    Returns:
        0.0-1.0: Position ratio (0=start, 1=end, 0.5=middle)
    """
    if not token or not line:
        return 0.5  # Default to middle

    token_pos = line.find(token)
    if token_pos == -1:
        return 0.5

    # Calculate position as ratio of line length
    line_len = len(line.strip())
    if line_len == 0:
        return 0.5

    # Use the center of the token
    token_center = token_pos + len(token) / 2
    offset = token_center / line_len

    # Clamp to [0, 1]
    return max(0.0, min(1.0, offset))


def _detect_multiline_block(context_text: str, token: str) -> bool:
    """
    Detect if token is part of a multi-line secret block (PEM, SSH keys).

    Returns:
        True if token appears in a PEM/SSH block
    """
    if not context_text or not token:
        return False

    # Check for PEM block markers
    pem_patterns = [
        r'-----BEGIN [A-Z ]+-----',
        r'-----END [A-Z ]+-----',
        r'BEGIN RSA PRIVATE KEY',
        r'BEGIN PRIVATE KEY',
        r'BEGIN OPENSSH PRIVATE KEY',
        r'BEGIN EC PRIVATE KEY',
    ]

    for pattern in pem_patterns:
        if re.search(pattern, context_text):
            # Token is in context with PEM markers
            return True

    return False


def _detect_embedded_token(line: str, token: str) -> bool:
    """
    Detect if token was extracted from a URL/DSN/connection string.

    Returns:
        True if token appears embedded in a URL-like structure
    """
    if not line or not token:
        return False

    # Check if token is part of a URL or DSN pattern
    url_patterns = [
        r'(https?|ftp|jdbc|postgresql|mysql|mongodb)://',
        r'://[^@]*@',  # user:pass@host pattern
        r'[?&][a-z_]+=' + re.escape(token),  # Query parameter
    ]

    for pattern in url_patterns:
        if re.search(pattern, line):
            # Token is in a line with URL-like patterns
            return True

    return False


def _detect_quote_type(line: str, token: str) -> int:
    """
    Detect the type of quotes wrapping the token.

    Returns:
        0: No quotes
        1: Single quotes (')
        2: Double quotes (")
        3: Backticks (`)
    """
    if not token or not line:
        return 0

    # Find the position of the token in the line
    token_pos = line.find(token)
    if token_pos == -1:
        return 0

    token_end = token_pos + len(token)

    # Check for quotes immediately before and after the token
    if token_pos > 0 and token_end < len(line):
        char_before = line[token_pos - 1]
        char_after = line[token_end]

        if char_before == "'" and char_after == "'":
            return 1
        if char_before == '"' and char_after == '"':
            return 2
        if char_before == '`' and char_after == '`':
            return 3

    # Check if token is inside a longer quoted string
    quote_chars = [('"', 2), ("'", 1), ('`', 3)]

    for quote, code in quote_chars:
        # Find all quote positions
        quotes = [i for i, c in enumerate(line) if c == quote]
        if len(quotes) >= 2:
            # Check if token is between any pair of quotes
            for i in range(0, len(quotes) - 1, 2):
                if quotes[i] < token_pos and token_end < quotes[i + 1]:
                    return code

    return 0


def _count_surrounding_tokens(line: str) -> int:
    """
    Count the number of high-entropy tokens on the same line.

    Returns:
        Count of tokens with entropy > 3.5
    """
    if not line:
        return 0

    # Find all alphanumeric tokens of reasonable length
    token_pattern = re.compile(r'[A-Za-z0-9+/=_\-.]{8,}')
    tokens = token_pattern.findall(line)

    high_entropy_count = 0
    for token in tokens:
        if shannon_entropy(token) > 3.5:
            high_entropy_count += 1

    return high_entropy_count


def _calculate_key_value_distance(line: str, token: str, var_name: Optional[str]) -> int:
    """
    Calculate the distance (in characters) between variable name and token.

    Returns:
        -1 if no variable name
        Distance in characters otherwise
    """
    if not var_name or not token or not line:
        return -1

    var_pos = line.find(var_name)
    token_pos = line.find(token)

    if var_pos == -1 or token_pos == -1:
        return -1

    # Distance from end of var name to start of token
    return abs(token_pos - (var_pos + len(var_name)))


def _calculate_json_path_hint(line: str) -> int:
    """
    Estimate the depth in JSON/YAML structure.

    Returns:
        0: Root level
        N: Nested N levels deep (based on leading whitespace or braces)
    """
    if not line:
        return 0

    # Count leading whitespace (YAML-style)
    leading_spaces = len(line) - len(line.lstrip())
    yaml_depth = leading_spaces // 2  # Assume 2-space indents

    # Count opening braces/brackets before the line content (JSON-style)
    json_depth = line.count('{') + line.count('[')

    # Return the maximum depth hint
    return max(yaml_depth, json_depth)


def _calculate_adjacency_ngram_score(context_text: str, token: str) -> float:
    """
    Calculate N-gram score for variable names near the token.

    Returns:
        Sum of secret N-gram scores for neighboring variables
    """
    if not context_text or not token:
        return 0.0

    # Find the token position in context
    token_pos = context_text.find(token)
    if token_pos == -1:
        return 0.0

    # Extract surrounding lines (±2 lines)
    lines = context_text.split('\n')
    total_score = 0.0

    for line in lines:
        if token in line:
            continue  # Skip the line containing the token itself

        # Find variable names in adjacent lines
        var_pattern = re.compile(r'\b([a-z_][a-z0-9_]{2,})\b', re.I)
        var_names = var_pattern.findall(line)

        for var_name in var_names:
            # Calculate N-gram score for this variable
            score = _calculate_ngram_score(var_name, SECRET_NGRAMS)
            total_score += score

    return total_score


def _calculate_cross_line_entropy(context_text: str, token: str) -> float:
    """
    Calculate average entropy across ±3 lines from the token.

    Returns:
        Average Shannon entropy of nearby lines
    """
    if not context_text or not token:
        return 0.0

    lines = context_text.split('\n')

    # Find the line containing the token
    token_line_idx = -1
    for i, line in enumerate(lines):
        if token in line:
            token_line_idx = i
            break

    if token_line_idx == -1:
        return 0.0

    # Get ±3 lines (excluding the token line itself)
    start_idx = max(0, token_line_idx - 3)
    end_idx = min(len(lines), token_line_idx + 4)

    nearby_lines = []
    for i in range(start_idx, end_idx):
        if i != token_line_idx:  # Exclude the token line
            nearby_lines.append(lines[i])

    if not nearby_lines:
        return 0.0

    # Calculate average entropy
    entropies = [shannon_entropy(line) for line in nearby_lines if line.strip()]

    if not entropies:
        return 0.0

    return sum(entropies) / len(entropies)


def _detect_assignment_type(line: str, token: str) -> int:
    """Detect the type of assignment operator used."""
    token_pos = line.find(token)
    if token_pos == -1:
        return 0

    prefix = line[:token_pos]

    if ":=" in prefix:
        return 3  # Go-style
    if "=>" in prefix:
        return 4  # Arrow function or map
    if re.search(r":\s*[\"']?$", prefix):
        return 2  # JSON/YAML/Dict style
    if re.search(r"=\s*[\"']?$", prefix):
        return 1  # Standard assignment

    return 0


def _is_token_in_string_literal(line: str, token: str) -> bool:
    """
    Check if token is actually wrapped in quotes (string literal).

    This properly detects if the token itself is inside quotes, not just
    whether quotes exist anywhere in the line.

    Args:
        line: The line of code containing the token
        token: The token to check

    Returns:
        True if token is wrapped in quotes (single, double, or backtick)

    Examples:
        >>> _is_token_in_string_literal('api_key = "sk_live_xxx"', 'sk_live_xxx')
        True
        >>> _is_token_in_string_literal('some_var = "hello"; sk_live_xxx', 'sk_live_xxx')
        False
        >>> _is_token_in_string_literal("token = 'ghp_xxxx'", 'ghp_xxxx')
        True
    """
    if not token or not line:
        return False

    # Find the position of the token in the line
    token_pos = line.find(token)
    if token_pos == -1:
        return False

    token_end = token_pos + len(token)

    # Check for quotes immediately before and after the token
    # Handle single quotes, double quotes, and backticks
    quote_chars = ['"', "'", '`']

    for quote in quote_chars:
        # Check if token is wrapped: <quote>token<quote>
        if token_pos > 0 and token_end < len(line):
            char_before = line[token_pos - 1]
            char_after = line[token_end]

            if char_before == quote and char_after == quote:
                return True

    # Also check for partial matches where token is inside a longer quoted string
    # Walk backwards to find opening quote
    for i in range(token_pos - 1, -1, -1):
        if line[i] in quote_chars:
            opening_quote = line[i]
            # Walk forward to find closing quote
            for j in range(token_end, len(line)):
                if line[j] == opening_quote:
                    # Check if this is escaped
                    if j > 0 and line[j - 1] != '\\':
                        # Token is between opening_quote at i and closing_quote at j
                        return True
            break
        # Stop if we hit an assignment operator (token is not in a string)
        if line[i] in ('=', ':') and (i == 0 or line[i - 1] != '\\'):
            break

    return False


def _extract_token_features(
    token: str,
    regex_match_type: int = 0,
    line: str = "",
    context_text: str = "",
) -> dict:
    """Extract token-level features (23 features).

    NOTE: The following features are intentionally NOT included to prevent
    shortcut learning:
    - has_known_prefix: Would allow model to trivially match AKIA*, ghp_*, sk-*
    - prefix_type: Would directly encode secret type from prefix
    - is_hex_like: Would strongly correlate with non-secrets (git SHAs)

    Instead, we use advanced entropy analysis and vendor prefix boost which
    provide the model with useful signal without enabling trivial shortcuts.

    NEW DISCRIMINATIVE FEATURES (for precision boost):
    - is_uuid_v4: UUIDs are identifiers, not secrets
    - is_known_hash_length: Hex tokens matching hash lengths are likely hashes
    - jwt_structure_valid: Invalid JWTs are not real secrets
    - entropy_charset_mismatch: High entropy + low charset = likely hash/UUID
    - has_hash_prefix: sha256:xxx prefixes indicate hashes
    """
    length = len(token)

    return {
        "token_length": length,
        "token_entropy": shannon_entropy(token) if token else 0.0,
        "char_class_count": _get_char_class_count(token),
        "digit_ratio": sum(1 for c in token if c.isdigit()) / max(length, 1),
        "uppercase_ratio": sum(1 for c in token if c.isupper()) / max(length, 1),
        "special_char_ratio": sum(1 for c in token if not c.isalnum()) / max(length, 1),
        "is_base64_like": _is_base64_like(token),
        "has_padding": token.endswith("=") or token.endswith("=="),
        "regex_match_type": regex_match_type,
        "token_structure_score": _calculate_token_structure_score(token),
        "has_version_pattern": _has_version_pattern(token),
        # Advanced entropy features
        "normalized_entropy": _calculate_normalized_entropy(token),
        "cryptographic_score": _calculate_cryptographic_score(token),
        "vendor_prefix_boost": _get_vendor_prefix_boost(token),
        # Token position/structure features
        "token_span_offset": _calculate_token_span_offset(line, token),
        "token_in_multiline_block": _detect_multiline_block(context_text, token),
        "embedded_token_flag": _detect_embedded_token(line, token),
        "token_quote_type": _detect_quote_type(line, token),
        # NEW: Discriminative features for precision boost
        "is_uuid_v4": _is_uuid_v4(token),
        "is_known_hash_length": _is_known_hash_length(token),
        "jwt_structure_valid": _is_jwt_structure_valid(token),
        "entropy_charset_mismatch": _calculate_entropy_charset_mismatch(token),
        "has_hash_prefix": _has_hash_prefix(token),
    }


def _extract_var_features(var_name: Optional[str], line: str, token: str) -> dict:
    """Extract variable name features (10 features)."""
    if not var_name:
        return {
            "var_name_extracted": False,
            "var_contains_secret": False,
            "var_contains_safe": False,
            "var_name_length": 0,
            "var_is_uppercase": False,
            "var_is_camelcase": False,
            "assignment_type": _detect_assignment_type(line, token),
            "in_string_literal": _is_token_in_string_literal(line, token),
            # N-gram scores default to 0 when no var name
            "var_ngram_secret_score": 0.0,
            "var_ngram_safe_score": 0.0,
        }

    return {
        "var_name_extracted": True,
        "var_contains_secret": any(p.search(var_name) for p in SECRET_VAR_PATTERNS),
        "var_contains_safe": any(p.search(var_name) for p in SAFE_VAR_PATTERNS),
        "var_name_length": len(var_name),
        "var_is_uppercase": var_name.isupper() and "_" in var_name,
        "var_is_camelcase": _is_camelcase(var_name),
        "assignment_type": _detect_assignment_type(line, token),
        "in_string_literal": _is_token_in_string_literal(line, token),
        # N-gram weighted scores for secret/safe variable name patterns
        "var_ngram_secret_score": _calculate_ngram_score(var_name, SECRET_NGRAMS),
        "var_ngram_safe_score": _calculate_ngram_score(var_name, SAFE_NGRAMS),
    }


def _extract_context_features(
    context: CodeContext,
    token: str = "",
    var_name: Optional[str] = None,
) -> dict:
    """Extract context-level features (18 features including advanced context)."""
    full_text = context.full_context
    line = context.line_content

    # Check for import statements
    import_patterns = [
        re.compile(r"^\s*import\s+", re.M),
        re.compile(r"^\s*from\s+\w+\s+import", re.M),
        re.compile(r'\brequire\s*\(["\']', re.M),
    ]
    has_import = any(p.search(full_text) for p in import_patterns)

    # Check for function definitions
    func_patterns = [
        re.compile(r"^\s*def\s+\w+\s*\(", re.M),  # Python
        re.compile(r"^\s*function\s+\w+\s*\(", re.M),  # JavaScript
        re.compile(r"^\s*func\s+\w+\s*\(", re.M),  # Go
        re.compile(r"^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(", re.M),  # Java
    ]
    has_func_def = any(p.search(full_text) for p in func_patterns)

    # Check for test/git/hash mentions in safe patterns
    test_pattern = re.compile(r"(test|mock|fake|example|sample|dummy|pytest|unittest)", re.I)
    git_pattern = re.compile(r"(git|commit|sha|rev|revision)", re.I)
    hash_pattern = re.compile(r"(hash|digest|checksum|md5|sha256)", re.I)

    # Determine file extension risk
    ext_risk = 0
    if context.is_high_risk_file:
        ext_risk = 2
    elif context.is_config_file:
        ext_risk = 1

    # Calculate average entropy of nearby tokens
    token_pattern = re.compile(r"[A-Za-z0-9+/=_\-.]{16,}")
    nearby_tokens = token_pattern.findall(full_text)
    avg_entropy = 0.0
    if nearby_tokens:
        entropies = [shannon_entropy(t) for t in nearby_tokens[:10]]
        avg_entropy = sum(entropies) / len(entropies)

    # NEW: Calculate line position ratio (0=top, 1=bottom)
    # Secrets are often defined at the top of files (config sections)
    line_position_ratio = 0.5  # Default to middle
    if context.line_number is not None and context.total_lines is not None:
        if context.total_lines > 0:
            line_position_ratio = context.line_number / context.total_lines

    return {
        "line_is_comment": context.in_comment,
        "context_mentions_test": bool(test_pattern.search(full_text)),
        "context_mentions_git": bool(git_pattern.search(full_text)),
        "context_mentions_hash": bool(hash_pattern.search(full_text)),
        "context_has_import": has_import,
        "context_has_function_def": has_func_def,
        "file_is_test": context.in_test_file,
        "file_is_config": context.is_config_file,
        "file_extension_risk": ext_risk,
        "surrounding_entropy_avg": avg_entropy,
        # NEW: Enhanced context analysis features
        "semantic_context_score": _calculate_semantic_context_score(full_text),
        "line_position_ratio": line_position_ratio,
        "surrounding_secret_density": _calculate_surrounding_secret_density(full_text),
        # NEW: Advanced context features
        "surrounding_token_count": _count_surrounding_tokens(line),
        "key_value_distance": _calculate_key_value_distance(line, token, var_name),
        "json_path_hint": _calculate_json_path_hint(line),
        "adjacency_ngram_score": _calculate_adjacency_ngram_score(full_text, token),
        "cross_line_entropy": _calculate_cross_line_entropy(full_text, token),
    }


def _extract_stage_b_precision_features(token: str, file_path: Optional[str]) -> dict:
    """
    Extract Stage B precision improvement features (7 features).

    These features specifically target observed false positive patterns:
    - 40/64-char hex tokens (git SHAs, SHA256 hashes)
    - Test-mode tokens (sk_test_*, pk_test_*)
    - Example/placeholder tokens
    - Git-related, build, and example file paths

    Args:
        token: The token to analyze
        file_path: The file path for context

    Returns:
        Dict with 7 Stage B precision features
    """
    return {
        "is_hex_len_40": _is_hex_exact_length(token, 40),
        "is_hex_len_64": _is_hex_exact_length(token, 64),
        "is_test_token": _is_test_token(token),
        "contains_example_keyword": _contains_example_keyword(token),
        "file_is_git_related": _is_git_related_path(file_path),
        "file_is_build": _is_build_path(file_path),
        "file_is_example": _is_example_path(file_path),
    }


def _extract_hex_disambiguation_features(
    token: str,
    context_text: str,
    file_path: Optional[str],
    line_content: str,
    embedded_in_url: bool = False,
) -> dict:
    """
    Extract Stage B generalization features for hex token disambiguation (5 features).

    These features help distinguish hex secrets from git SHAs/checksums by
    analyzing the surrounding context rather than the token format.

    Args:
        token: The token to analyze
        context_text: Full context (before + current + after lines joined)
        file_path: The file path for context
        line_content: The line containing the token
        embedded_in_url: Whether token was extracted from a URL/DSN

    Returns:
        Dict with 5 hex disambiguation features
    """
    # Only compute these features for hex-like tokens (40 or 64 chars)
    hex_chars = set("0123456789abcdefABCDEF")
    is_hex_candidate = (
        len(token) in (32, 40, 64) and
        all(c in hex_chars for c in token)
    )

    if not is_hex_candidate:
        # Return defaults for non-hex tokens
        return {
            "hex_context_git_keywords": False,
            "hex_context_crypto_keywords": False,
            "hex_adjacent_assignment_pattern": 0,
            "hex_in_url_or_dsn": False,
            "hex_file_suggests_secret": False,
        }

    context_lower = context_text.lower() if context_text else ""
    line_lower = line_content.lower() if line_content else ""

    # Feature 1: Git-related keywords in context (suggests NOT a secret)
    git_keywords = ["commit", "git", "merge", "branch", "checkout", "sha", "rev", "HEAD"]
    hex_context_git = any(kw in context_lower for kw in git_keywords)

    # Feature 2: Crypto/secret keywords in context (suggests IS a secret)
    crypto_keywords = ["encrypt", "decrypt", "sign", "key", "secret", "auth", "password", "credential", "token"]
    hex_context_crypto = any(kw in context_lower for kw in crypto_keywords)

    # Feature 3: Assignment pattern (helps identify source)
    # 0=unknown, 1=env_var, 2=config_assignment, 3=func_arg
    assignment_pattern = 0
    if "os.environ" in line_lower or "process.env" in line_lower or "env[" in line_lower:
        assignment_pattern = 1  # Environment variable
    elif ": " in line_content or "= " in line_content or "=" in line_content:
        assignment_pattern = 2  # Config assignment
    elif "(" in line_content and ")" in line_content:
        assignment_pattern = 3  # Function argument

    # Feature 4: Token embedded in URL/DSN
    hex_in_url = embedded_in_url or any(
        pattern in line_lower
        for pattern in ["http://", "https://", "://", "postgres://", "mysql://", "mongodb://"]
    )

    # Feature 5: File path suggests secrets
    hex_file_secret = False
    if file_path:
        path_lower = file_path.lower()
        secret_paths = [
            "secrets/", "secret/", ".env", "credentials/", "credential/",
            "keys/", "key/", "private/", "auth/", "vault/", "_secrets",
        ]
        hex_file_secret = any(sp in path_lower for sp in secret_paths)

    return {
        "hex_context_git_keywords": hex_context_git,
        "hex_context_crypto_keywords": hex_context_crypto,
        "hex_adjacent_assignment_pattern": assignment_pattern,
        "hex_in_url_or_dsn": hex_in_url,
        "hex_file_suggests_secret": hex_file_secret,
    }


# ---------------------------------------------------------------------------
# Env-loading pattern awareness (v0.2.0 — features 64-65)
# ---------------------------------------------------------------------------

_ENV_LOADER_PATTERNS = [
    re.compile(r"load_dotenv"),
    re.compile(r"from\s+dotenv\s+import"),
    re.compile(r"dotenv\.config"),
    re.compile(r"os\.getenv"),
    re.compile(r"os\.environ"),
    re.compile(r"process\.env\."),
    re.compile(r"ENV\["),
    re.compile(r"ENV\.fetch"),
    re.compile(r"viper\.\w+Get"),
    re.compile(r"System\.getenv"),
    re.compile(r"Environment\.GetEnvironmentVariable"),
    re.compile(r"from_envvar"),
    re.compile(r"env\(\s*['\"]"),
    re.compile(r"getenv\("),
    re.compile(r"std::env::var"),
]

_FALLBACK_LOCATORS = [
    re.compile(r"os\.getenv\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]"),
    re.compile(r"os\.environ\.get\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]"),
    re.compile(r"process\.env\.\w+\s*\|\|\s*['\"]"),
    re.compile(r"ENV\.fetch\(\s*['\"][^'\"]+['\"]\s*\)\s*\{\s*['\"]"),
    re.compile(r"env\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]"),
    re.compile(r"getenv\([^)]+\)\s*\?:\s*['\"]"),
    re.compile(r"\$\{[^:}]+:-"),
]


def _detect_env_loader_in_context(context_text: str, line: str) -> bool:
    """True when surrounding context contains any env-loading pattern."""
    text = f"{context_text}\n{line}"
    return any(pat.search(text) for pat in _ENV_LOADER_PATTERNS)


def _detect_env_fallback_value(line: str, token_start_index: int) -> bool:
    """True IFF the token at `token_start_index` sits inside the fallback/default
    argument of an env-loader call.

    Uses authoritative start index from upstream extraction (TokenMatch.start),
    NOT line.find() which grabs the wrong occurrence if the token appears twice.
    """
    for pattern in _FALLBACK_LOCATORS:
        match = pattern.search(line)
        if match:
            fallback_start = match.end()
            if token_start_index >= fallback_start - 1:
                return True
    return False


def extract_features(
    finding: "Finding",
    context: CodeContext,
    regex_match_type: int = 0,
) -> FeatureVector:
    """
    Extract all 65 features from a finding and its context.

    Args:
        finding: The Finding object with token and metadata
        context: CodeContext with surrounding code
        regex_match_type: Encoded type of regex match (0 = entropy-only)

    Returns:
        FeatureVector with all 65 features
    """
    token = finding.token or ""

    # Encode regex match type from finding type
    if regex_match_type == 0 and finding.type != "ENTROPY_CANDIDATE":
        # Encode common types
        type_map = {
            "AWS_ACCESS_KEY_ID": 1,
            "AWS_SECRET_ACCESS_KEY": 1,
            "GITHUB_TOKEN": 2,
            "STRIPE_SECRET_KEY": 3,
            "OPENAI_API_KEY": 4,
            "SLACK_TOKEN": 5,
            "GCP_API_KEY": 6,
            "JWT": 7,
            "AZURE_KEY_VAULT_URL": 8,
        }
        regex_match_type = type_map.get(finding.type, 9)

    # Extract variable name first (needed by context features)
    var_name = extract_var_name(context.line_content, token)

    # Extract features from each category
    token_features = _extract_token_features(
        token,
        regex_match_type,
        line=context.line_content,
        context_text=context.full_context,
    )

    var_features = _extract_var_features(var_name, context.line_content, token)

    context_features = _extract_context_features(context, token, var_name)

    # Extract Stage B precision improvement features
    stage_b_features = _extract_stage_b_precision_features(token, context.file_path)

    # Extract Stage B generalization improvement features (hex disambiguation)
    hex_features = _extract_hex_disambiguation_features(
        token=token,
        context_text=context.full_context,
        file_path=context.file_path,
        line_content=context.line_content,
        embedded_in_url=token_features.get("embedded_token_flag", False),
    )

    # Env-loading awareness features
    env_in_ctx = _detect_env_loader_in_context(context.full_context, context.line_content)
    token_pos = context.line_content.find(token)
    env_fallback = _detect_env_fallback_value(context.line_content, token_pos) if token_pos >= 0 else False

    # Combine all features
    return FeatureVector(
        **token_features,
        **var_features,
        **context_features,
        **stage_b_features,
        **hex_features,
        env_loader_in_context=env_in_ctx,
        is_env_fallback_value=env_fallback,
    )


def extract_features_from_record(record: dict) -> FeatureVector:
    """
    Extract features from a training data record.

    Args:
        record: Dict with token, line_content, context_before, context_after, etc.

    Returns:
        FeatureVector with all 65 features
    """
    from Harpocrates.core.result import EvidenceType, Finding

    token = record.get("token", "")
    line_content = record.get("line_content", "")
    file_path = record.get("file_path", "")
    context_before = record.get("context_before", [])
    context_after = record.get("context_after", [])

    # Detect if file is a test file based on path
    in_test_file = False
    if file_path:
        path_lower = file_path.lower()
        in_test_file = any(
            x in path_lower
            for x in ["test", "spec", "__tests__", "_test.", ".test.", "tests/"]
        )

    # Detect if line is a comment
    stripped = line_content.strip()
    in_comment = (
        stripped.startswith("#")
        or stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
        or stripped.startswith("--")
        or stripped.startswith("'")  # VB comments
        or stripped.startswith("REM")  # Batch comments
    )

    # Estimate line position based on context
    # Use position within visible context window as a proxy
    # This gives a value from 0 (at start of context) to 1 (at end of context)
    total_context_lines = len(context_before) + 1 + len(context_after)
    if total_context_lines > 1:
        # Position within context window (0=top, 1=bottom)
        estimated_line_number = len(context_before) + 1
        estimated_total_lines = total_context_lines
    else:
        # No context, assume middle of file
        estimated_line_number = 50
        estimated_total_lines = 100

    # Create a minimal Finding object
    finding = Finding(
        type=record.get("secret_type", "ENTROPY_CANDIDATE"),
        snippet=line_content,
        evidence=EvidenceType.ENTROPY,
        token=token,
    )

    # Create CodeContext from record with properly set fields
    context = CodeContext(
        line_content=line_content,
        lines_before=context_before,
        lines_after=context_after,
        file_path=file_path,
        in_test_file=in_test_file,
        in_comment=in_comment,
        line_number=estimated_line_number,
        total_lines=estimated_total_lines,
    )

    return extract_features(finding, context)


__all__ = [
    "FeatureVector",
    "extract_features",
    "extract_features_from_record",
    "KNOWN_PREFIXES",
    "SECRET_VAR_PATTERNS",
    "SAFE_VAR_PATTERNS",
]

