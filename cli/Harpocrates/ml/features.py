"""
Feature engineering for ML-based secrets verification.

Extracts 64 features from tokens and their context to enable
context-aware classification of potential secrets.

Features are organized into seven categories:
- Token features (20): Properties of the token itself including advanced
  entropy analysis, vendor prefix detection, and discriminative features
  (is_uuid_v4, jwt_structure_valid, entropy_charset_mismatch, has_hash_prefix)
- Variable name features (9): Properties of the variable/key name including
  N-gram scoring for secret and safe patterns
- Context features (16): Properties of surrounding code including semantic
  analysis and secret density
- Stage B precision features (6): Targeted FP reduction features
- Stage B generalization features (4): Hex disambiguation features
- Env-loading awareness features (2): Env-loader and fallback detection
- Phase 7.0 value-shape features (7): Value-shape FP suppression features
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from Harpocrates.detectors.entropy_detector import shannon_entropy
from Harpocrates.ml.context import (
    CodeContext,
    feature_var_name,
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


# AI-generated placeholder detection: values an LLM writes in place of a real key.
# Word-boundary anchors are intentionally omitted for underscore-joined patterns like
# dummy_token, DUMMY_SECRET, FAKE_API_KEY which are the primary LLM leftovers.
_AI_PLACEHOLDER_RE = re.compile(
    r"YOUR_|_HERE|dummy|CHANGEME|PLACEHOLDER|<[A-Z_]+>|TODO|REPLACE[_\-]|FAKE|TEST[_\-]",
    re.IGNORECASE,
)

# Connection-string and credential-context keywords
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


FEATURE_NAMES: Tuple[str, ...] = (
    # Token features (20 — dropped normalized_entropy, cryptographic_score, is_known_hash_length)
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
    "vendor_prefix_boost",
    "token_span_offset",
    "token_in_multiline_block",
    "embedded_token_flag",
    "token_quote_type",
    "is_uuid_v4",
    "jwt_structure_valid",
    "entropy_charset_mismatch",
    "has_hash_prefix",
    # Variable name features (9 — dropped var_contains_secret)
    "var_name_extracted",
    "var_contains_safe",
    "var_name_length",
    "var_is_uppercase",
    "var_is_camelcase",
    "assignment_type",
    "in_string_literal",
    "var_ngram_secret_score",
    "var_ngram_safe_score",
    # Context features (16 — dropped line_position_ratio, cross_line_entropy)
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
    "surrounding_secret_density",
    "surrounding_token_count",
    "key_value_distance",
    "json_path_hint",
    "adjacency_ngram_score",
    # Stage B precision improvement features (6 — dropped contains_example_keyword)
    "is_hex_len_40",
    "is_hex_len_64",
    "is_test_token",
    "file_is_git_related",
    "file_is_build",
    "file_is_example",
    # Stage B generalization improvement features (4 — dropped hex_context_git_keywords)
    "hex_context_crypto_keywords",
    "hex_adjacent_assignment_pattern",
    "hex_in_url_or_dsn",
    "hex_file_suggests_secret",
    # Env-loading awareness features (2)
    "env_loader_in_context",
    "is_env_fallback_value",
    # Value-shape features (7)
    "value_starts_with_slash",
    "value_contains_path_separator",
    "value_ends_with_known_ext",
    "value_is_lowercase_word",
    "value_is_dotted_quad_or_host",
    "value_is_template_syntax",
    "is_hex_with_no_alpha_mix",
)


@dataclass
class FeatureVector:
    """
    Description: 64 extracted features for ML classification. The model should learn from context, not token format.
    Parameters:
        token_length: Token length.
        token_entropy: Base-2 Shannon entropy of the candidate.
        char_class_count: Number of character classes present
        digit_ratio: Fraction of token characters that are digits.
        uppercase_ratio: Fraction of token characters that are uppercase.
        special_char_ratio: Fraction of token characters that are non-alphanumeric.
        is_base64_like: Is base64 like.
        has_padding: Ends with = or ==
        regex_match_type: 0 = entropy, 1+ = regex pattern
        token_structure_score: 0=random, 1=structured (words, separators)
        has_version_pattern: Contains v1.2.3, 1.0.0 patterns
        vendor_prefix_boost: Boost for known vendor prefixes (AKIA, ghp_, etc.)
        token_span_offset: Position within line (0=start, 1=end)
        token_in_multiline_block: Part of PEM/SSH block
        embedded_token_flag: Extracted from URL/DSN
        token_quote_type: 0=none, 1=single, 2=double, 3=backtick
        is_uuid_v4: Matches UUID v4 format (strong non-secret signal)
        jwt_structure_valid: JWT has valid JSON header
        entropy_charset_mismatch: High entropy but low charset diversity
        has_hash_prefix: Starts with sha256:, md5:, etc.
        var_name_extracted: Var name extracted.
        var_contains_safe: Var contains safe.
        var_name_length: Var name length.
        var_is_uppercase: CONSTANT_STYLE
        var_is_camelcase: Var is camelcase.
        assignment_type: 0=unknown, 1==, 2=:, 3=:=, 4=func_arg
        in_string_literal: In string literal.
        var_ngram_secret_score: Weighted sum of secret N-gram matches
        var_ngram_safe_score: Weighted sum of safe N-gram matches
        line_is_comment: Line is comment.
        context_mentions_test: Context mentions test.
        context_mentions_git: Context mentions git.
        context_mentions_hash: Context mentions hash.
        context_has_import: Context has import.
        context_has_function_def: Context has function def.
        file_is_test: File is test.
        file_is_config: File is config.
        file_extension_risk: 0=low, 1=medium, 2=high
        surrounding_entropy_avg: Mean entropy of up to ten nearby token candidates.
        semantic_context_score: -1=safe, 0=neutral, 1=risky
        surrounding_secret_density: Ratio of nearby secret-like tokens
        surrounding_token_count: Count of high-entropy tokens on same line
        key_value_distance: Distance between var name and token (-1 if no var)
        json_path_hint: Depth in JSON/YAML structure (0=root)
        adjacency_ngram_score: Sum of N-gram scores in nearby var names
        is_hex_len_40: Token is exactly 40 hex chars (git SHA/SHA1)
        is_hex_len_64: Token is exactly 64 hex chars (SHA256)
        is_test_token: Has _test_, _staging_, test_ prefix
        file_is_git_related: Path contains .git/, hash, commit
        file_is_build: Path contains build/, dist/, node_modules/
        file_is_example: Path contains example/, docs/, demo/
        hex_context_crypto_keywords: Context contains encrypt, sign, key, secret
        hex_adjacent_assignment_pattern: 0=unknown, 1=env_var, 2=config, 3=func_arg
        hex_in_url_or_dsn: Token is embedded in URL/DSN
        hex_file_suggests_secret: Path suggests secrets (secrets/, .env, credentials/)
        env_loader_in_context: Context contains env-loading pattern
        is_env_fallback_value: Token is the fallback/default arg in env-loader call
        value_starts_with_slash: "/" or "~" or "./" prefix → file path
        value_contains_path_separator: Slash or backslash in value (non-URL) → path
        value_ends_with_known_ext: .pem/.key/.crt/.jks/.p12/.json/.yaml/.txt
        value_is_lowercase_word: re.fullmatch([a-z]+) → enum/literal
        value_is_dotted_quad_or_host: 0.0.0.0, 127.0.0.1, localhost, ::1
        value_is_template_syntax: ${...}, {{...}}, __X__, <<...>>, or AI placeholder (YOUR_KEY,
        CHANGEME)
        is_hex_with_no_alpha_mix: all hex but no uppercase letter → hash
    Output: Instances provide 64 feature fields in canonical model input order; features removed for shortcut
        learning or collinearity are not fields.
    """

    # Token features (20) - dropped normalized_entropy, cryptographic_score, is_known_hash_length
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
    vendor_prefix_boost: float = 0.0  # Boost for known vendor prefixes (AKIA, ghp_, etc.)
    # Position and structural features
    token_span_offset: float = 0.0  # Position within line (0=start, 1=end)
    token_in_multiline_block: bool = False  # Part of PEM/SSH block
    embedded_token_flag: bool = False  # Extracted from URL/DSN
    token_quote_type: int = 0  # 0=none, 1=single, 2=double, 3=backtick
    # Discriminative features for precision boost
    is_uuid_v4: bool = False  # Matches UUID v4 format (strong non-secret signal)
    jwt_structure_valid: bool = False  # JWT has valid JSON header
    entropy_charset_mismatch: float = 0.0  # High entropy but low charset diversity
    has_hash_prefix: bool = False  # Starts with sha256:, md5:, etc.

    # Variable name features (9) - dropped var_contains_secret
    var_name_extracted: bool = False
    var_contains_safe: bool = False
    var_name_length: int = 0
    var_is_uppercase: bool = False  # CONSTANT_STYLE
    var_is_camelcase: bool = False
    assignment_type: int = 0  # 0=unknown, 1==, 2=:, 3=:=, 4=func_arg
    in_string_literal: bool = True
    # N-gram weighted scores
    var_ngram_secret_score: float = 0.0  # Weighted sum of secret N-gram matches
    var_ngram_safe_score: float = 0.0  # Weighted sum of safe N-gram matches

    # Context features (16) - dropped line_position_ratio, cross_line_entropy
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
    surrounding_secret_density: float = 0.0  # Ratio of nearby secret-like tokens
    # Advanced context features
    surrounding_token_count: int = 0  # Count of high-entropy tokens on same line
    key_value_distance: int = -1  # Distance between var name and token (-1 if no var)
    json_path_hint: int = 0  # Depth in JSON/YAML structure (0=root)
    adjacency_ngram_score: float = 0.0  # Sum of N-gram scores in nearby var names

    # STAGE B PRECISION IMPROVEMENT: Features targeting observed FP patterns
    is_hex_len_40: bool = False  # Token is exactly 40 hex chars (git SHA/SHA1)
    is_hex_len_64: bool = False  # Token is exactly 64 hex chars (SHA256)
    is_test_token: bool = False  # Has _test_, _staging_, test_ prefix
    file_is_git_related: bool = False  # Path contains .git/, hash, commit
    file_is_build: bool = False  # Path contains build/, dist/, node_modules/
    file_is_example: bool = False  # Path contains example/, docs/, demo/

    # STAGE B GENERALIZATION IMPROVEMENT: Hex disambiguation features
    # These help distinguish hex secrets from git SHAs/checksums
    hex_context_crypto_keywords: bool = False  # Context contains encrypt, sign, key, secret
    hex_adjacent_assignment_pattern: int = 0  # 0=unknown, 1=env_var, 2=config, 3=func_arg
    hex_in_url_or_dsn: bool = False  # Token is embedded in URL/DSN
    hex_file_suggests_secret: bool = False  # Path suggests secrets (secrets/, .env, credentials/)

    # Env-loading awareness features (2)
    env_loader_in_context: bool = False  # Context contains env-loading pattern
    is_env_fallback_value: bool = False  # Token is the fallback/default arg in env-loader call

    # Value-shape features, which suppress false positives
    value_starts_with_slash: bool = False        # "/" or "~" or "./" prefix → file path
    value_contains_path_separator: bool = False  # "/" or "\" in value (non-URL) → path
    value_ends_with_known_ext: bool = False      # .pem/.key/.crt/.jks/.p12/.json/.yaml/.txt
    value_is_lowercase_word: bool = False        # re.fullmatch([a-z]+) → enum/literal
    value_is_dotted_quad_or_host: bool = False   # 0.0.0.0, 127.0.0.1, localhost, ::1
    value_is_template_syntax: bool = False       # ${...}, {{...}}, __X__, <<...>>, or AI placeholder (YOUR_KEY, CHANGEME)
    is_hex_with_no_alpha_mix: bool = False       # all hex but no uppercase letter → hash

    def to_array(self) -> List[float]:
        """
        Description: Convert to numpy-compatible array of 64 floats.
        Parameters: None
        Output: List of 64 numeric feature values in canonical order; invalid field values may raise
        TypeError or ValueError during float conversion.
        """
        return [
            # Token features (20)
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
            self.vendor_prefix_boost,
            # Token position/structure features
            self.token_span_offset,
            float(self.token_in_multiline_block),
            float(self.embedded_token_flag),
            float(self.token_quote_type),
            # Discriminative features for precision boost
            float(self.is_uuid_v4),
            float(self.jwt_structure_valid),
            self.entropy_charset_mismatch,
            float(self.has_hash_prefix),
            # Variable name features (9)
            float(self.var_name_extracted),
            float(self.var_contains_safe),
            float(self.var_name_length),
            float(self.var_is_uppercase),
            float(self.var_is_camelcase),
            float(self.assignment_type),
            float(self.in_string_literal),
            self.var_ngram_secret_score,
            self.var_ngram_safe_score,
            # Context features (16)
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
            self.surrounding_secret_density,
            # Advanced context features
            float(self.surrounding_token_count),
            float(self.key_value_distance),
            float(self.json_path_hint),
            self.adjacency_ngram_score,
            # Stage B precision improvement features (6)
            float(self.is_hex_len_40),
            float(self.is_hex_len_64),
            float(self.is_test_token),
            float(self.file_is_git_related),
            float(self.file_is_build),
            float(self.file_is_example),
            # Stage B generalization improvement features (4)
            float(self.hex_context_crypto_keywords),
            float(self.hex_adjacent_assignment_pattern),
            float(self.hex_in_url_or_dsn),
            float(self.hex_file_suggests_secret),
            # Env-loading awareness features (2)
            float(self.env_loader_in_context),
            float(self.is_env_fallback_value),
            # Value-shape features (7)
            float(self.value_starts_with_slash),
            float(self.value_contains_path_separator),
            float(self.value_ends_with_known_ext),
            float(self.value_is_lowercase_word),
            float(self.value_is_dotted_quad_or_host),
            float(self.value_is_template_syntax),
            float(self.is_hex_with_no_alpha_mix),
        ]

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        Description: Get ordered list of 64 feature names.
        Parameters: None
        Output: New list of the 64 canonical feature names.
        """
        return list(FEATURE_NAMES)

def _get_char_class_count(token: str) -> int:
    """
    Description: Count distinct character classes in token.
    Parameters:
        token: Candidate text to analyze.
    Output: Number of lowercase, uppercase, digit, and non-alphanumeric classes present, from zero
        to four.
    """
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
    """
    Description: Get encoded prefix type for known secret formats.
    Parameters:
        token: Candidate text to analyze.
    Output: Category code of the first known prefix, or zero if none matches.
    """
    for prefix, prefix_type in KNOWN_PREFIXES.items():
        if token.startswith(prefix):
            return prefix_type
    return 0


def _is_base64_like(token: str) -> bool:
    """
    Description: Check if token looks like base64 encoded data.
    Parameters:
        token: Candidate text to analyze.
    Output: True for at least four characters with at least 95% standard or URL-safe base64
        characters; otherwise False.
    """
    if len(token) < 4:
        return False
    # Check if mostly base64 characters
    base64_chars = sum(1 for c in token if c in BASE64_CHARS)
    url_safe_chars = sum(1 for c in token if c in BASE64_URL_CHARS)
    return max(base64_chars, url_safe_chars) / len(token) >= 0.95


def _is_hex_like(token: str) -> bool:
    """
    Description: Check if token is pure hexadecimal.
    Parameters:
        token: Candidate text to analyze.
    Output: True for at least eight characters consisting entirely of hexadecimal digits; otherwise
        False.
    """
    if len(token) < 8:
        return False
    return all(c in HEX_CHARS for c in token)


def _is_camelcase(name: str) -> bool:
    """
    Description: Check if variable name uses camelCase.
    Parameters:
        name: Variable name to inspect.
    Output: True for a lowercase initial character and a later uppercase character; otherwise False.
    """
    if not name or name.isupper() or name.islower():
        return False
    # Has mix of upper and lower with lower start
    return name[0].islower() and any(c.isupper() for c in name[1:])


def _calculate_token_structure_score(token: str) -> float:
    """
    Description: Calculate how structured (non-random) a token appears. Low scores (near 0.0) suggest
        a random, likely secret value, and high scores (near 1.0) suggest a likely non-secret.
    Parameters:
        token: Candidate text to analyze.
    Output: Structure score capped at 1.0, or 0.0 for fewer than four characters.
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
    """
    Description: Check if token contains version-like patterns.
    Parameters:
        token: Candidate text to analyze.
    Output: True when a configured version pattern matches; otherwise False.
    """
    if not token:
        return False
    # Match patterns like v1.2.3, 1.0.0, v2.0, etc.
    version_patterns = [
        re.compile(r'v\d+\.\d+(\.\d+)?', re.I),  # v1.2.3
        re.compile(r'\d+\.\d+\.\d+'),  # 1.2.3
        re.compile(r'version[_-]?\d+', re.I),  # version1, version_1
    ]
    return any(p.search(token) for p in version_patterns)


def _get_vendor_prefix_boost(token: str) -> float:
    """
    Description: Return a boost score for tokens with known vendor secret prefixes. A known prefix
        indicates a high probability of a real secret.
    Parameters:
        token: Candidate text to analyze.
    Output: Prefix confidence of 1.0, 0.8, or 0.6, or 0.0 for no match.
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
    Description: Check if token matches UUID v4 format. UUIDs are identifiers, not credentials, so a
        match is a strong non-secret signal.
    Parameters:
        token: Candidate text to analyze.
    Output: True for a 36-character UUID v4 with a valid variant marker; otherwise False. The format
        is xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx, where x is a hex digit and y is one of 8, 9, a, or b.
    """
    if not token or len(token) != 36:
        return False
    return bool(UUID_V4_PATTERN.match(token))


def _is_hex_with_no_alpha_mix(token: str) -> bool:
    """
    Description: True if token is all lowercase hex with no uppercase — typical of hashes, not
    credentials.
    Parameters:
        token: Candidate text to analyze.
    Output: True for at least 20 lowercase hexadecimal characters; otherwise False.
    """
    return bool(re.fullmatch(r"[a-f0-9]+", token) and len(token) >= 20)


_KNOWN_EXT_RE = re.compile(
    r"\.(?:pem|key|crt|cer|jks|p12|pfx|der|json|yaml|yml|txt|env|conf|config)$",
    re.IGNORECASE,
)
_DOTTED_QUAD_RE = re.compile(
    r"^(?:10\.\d+\.\d+\.\d+|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|"
    r"192\.168\.\d+\.\d+|"
    r"127\.0\.0\.1|0\.0\.0\.0|localhost|::1)$",
    re.IGNORECASE,
)
_TEMPLATE_SYNTAX_RE = re.compile(
    r"\$\{[^}]+\}|\{\{[^}]+\}\}|__[A-Z][A-Z0-9_]+__|<<[^>]+>>",
)


def _value_shape_features(token: str) -> dict:
    """
    Description: Compute Phase 7.0 value-shape features for a token.
    Parameters:
        token: Candidate text to analyze.
    Output: Dictionary of seven boolean path, word, host, placeholder, and hexadecimal features.
    """
    starts_with_slash = token.startswith(("/", "~", "./", "../"))
    # Path separator: exclude any URL scheme (s3://, gs://, ftp://, postgres://, etc.)
    has_path_sep = ("/" in token or "\\" in token) and "://" not in token
    ends_with_ext = bool(_KNOWN_EXT_RE.search(token))
    is_lowercase_word = bool(re.fullmatch(r"[a-z]+", token))
    is_host = bool(_DOTTED_QUAD_RE.match(token))
    # Combines template syntax (${}, {{}}, __X__, <<>>) and AI placeholder patterns
    # (YOUR_KEY, CHANGEME, dummy_*) — both indicate non-secret values.
    is_placeholder = bool(_TEMPLATE_SYNTAX_RE.search(token) or _AI_PLACEHOLDER_RE.search(token))
    is_hex_no_alpha_mix = _is_hex_with_no_alpha_mix(token)
    return {
        "value_starts_with_slash": starts_with_slash,
        "value_contains_path_separator": has_path_sep,
        "value_ends_with_known_ext": ends_with_ext,
        "value_is_lowercase_word": is_lowercase_word,
        "value_is_dotted_quad_or_host": is_host,
        "value_is_template_syntax": is_placeholder,
        "is_hex_with_no_alpha_mix": is_hex_no_alpha_mix,
    }


def _is_jwt_structure_valid(token: str) -> bool:
    """
    Description: Check if a JWT-like token has valid structure, which suggests a real secret. Invalid
        or test JWTs (alg "none", fake payloads, expired timestamps) are likely not secrets.
    Parameters:
        token: Candidate text to analyze.
    Output: True if decoding and the header, subject, and expiry heuristics pass; False on any caught
        exception. A missing alg field is accepted, and only an expiry before 2020 is rejected.
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
    Description: Calculate mismatch between entropy and charset diversity. Hashes and UUIDs (high
        entropy, low diversity) and special-character words (low entropy, high diversity) suggest a
        non-secret.
    Parameters:
        token: Candidate text to analyze.
    Output: Mismatch score from 0.0 to 1.0, or 0.0 for short inputs or no detected mismatch. Scores of
        0.3 to 0.6 are moderate and suspicious; 0.6 to 1.0 are likely not secrets.
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
        return float(min(1.0, mismatch))

    # If we have low entropy with high charset, it's structured text (not concerning)
    if entropy < 3.0 and char_classes >= 3:
        return 0.2  # Slightly indicative but not definitive

    return 0.0


def _has_hash_prefix(token: str) -> bool:
    """
    Description: Check if token starts with a hash algorithm prefix. Such prefixes strongly indicate a
        hash, not a secret.
    Parameters:
        token: Candidate text to analyze.
    Output: True for a supported lowercase or uppercase hash algorithm prefix, such as sha256:, sha1:,
        md5:, or sha512:; otherwise False.
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
    Description: Check if token is exactly `length` hex characters.
    Parameters:
        token: The token to check.
        length: Expected length (40 for SHA1/git, 64 for SHA256).
    Output: True for a nonempty token of the requested length made only of hexadecimal digits;
        otherwise False.
    """
    if not token or len(token) != length:
        return False
    return all(c in HEX_CHARS for c in token)


def _is_test_token(token: str) -> bool:
    """
    Description: Check if token has test or staging prefix patterns. This targets false positives such
        as sk_test_, pk_test_, and rk_test_ tokens, which are sandbox credentials rather than secrets.
    Parameters:
        token: Candidate text to analyze.
    Output: True if any configured test or staging substring occurs; otherwise False.
    """
    if not token:
        return False

    test_patterns = [
        '_test_', '_staging_', 'test_', 'TEST_', '_demo_',
        'sk_test_', 'pk_test_', 'rk_test_',
    ]
    return any(p in token for p in test_patterns)


def _is_git_related_path(file_path: Optional[str]) -> bool:
    """
    Description: Check if file path is git-related. This targets false positives: 40-character hex
        tokens in .git/ directories or in git-named files that hold commit SHAs.
    Parameters:
        file_path: Optional source file path.
    Output: True for a configured Git-related path substring; False for absent paths or no match.
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
    Description: Check if file path is in a build or dist directory. This targets false positives: hex
        tokens in build artifacts, node_modules, or vendor directories that are checksums or asset
        hashes.
    Parameters:
        file_path: Optional source file path.
    Output: True for a configured build or artifact directory substring; otherwise False.
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
    Description: Check if file path is in an example or docs directory. This targets false positives:
        tokens in documentation, examples, or demo directories, which are often placeholder credentials.
    Parameters:
        file_path: Optional source file path.
    Output: True for README or CONTRIBUTING, or an example or documentation path substring; otherwise
        False.
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
    Description: Calculate weighted N-gram score for a variable name. Weights are summed rather than
        maxed, so several signals are needed for a high score and one signal cannot dominate.
    Parameters:
        var_name: The variable or key name to analyze.
        ngram_dict: Dictionary mapping N-grams to weights.
    Output: Sum of matching weights multiplied by 0.5, capped at 1.0; empty names produce zero.
    """
    if not var_name:
        return 0.0

    var_lower = var_name.lower()
    total_score = 0.0

    for ngram, weight in ngram_dict.items():
        if ngram in var_lower:
            total_score += weight * 0.5  # Scale factor to prevent score explosion

    return min(total_score, 1.0)  # Cap at 1.0


def _line_has_marker_then_word(text: str, marker: str, word: str) -> bool:
    """
    Description: Linear-time equivalent of re.search(re.escape(marker) + '.*' + word, text, re.I).
    Parameters:
        text: Text to search; lines are split on newline only, as '.' does not match it.
        marker: Literal marker that must precede the word on the same line.
        word: Literal word searched case-insensitively with the regex engine after the marker.
    Output: True when some line holds the marker followed later by the word.
    """
    for line in text.split('\n'):
        start = line.find(marker)
        if start >= 0 and re.search(word, line[start + len(marker):], re.I):
            return True
    return False


def _calculate_semantic_context_score(context_text: str) -> float:
    """
    Description: Calculate a weighted semantic context score. Negative values suggest a safe, non-secret
        context, and positive values suggest a secret.
    Parameters:
        context_text: Surrounding source text.
    Output: Weighted context score clamped to [-1, 1], with zero for empty context.
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

    # Comment patterns (#.*example, //.*test) are quadratic as regexes; matched per line instead.
    if _line_has_marker_then_word(context_text, '#', 'example'):
        score += -0.2  # Comments mentioning example
    if _line_has_marker_then_word(context_text, '//', 'test'):
        score += -0.2  # Comments mentioning test

    for pattern, weight in risky_patterns:
        if re.search(pattern, context_text, re.I):
            score += weight

    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, score))


def _calculate_surrounding_secret_density(context_text: str) -> float:
    """
    Description: Calculate the density of secret-like tokens in the surrounding context.
    Parameters:
        context_text: Surrounding source text.
    Output: Secret-like hit count divided by the first 20 candidates, or zero without candidates.
        Entropy and prefix hits can count the same token twice, so the result can exceed 1.0.
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
    Description: Calculate the position of the token within the line.
    Parameters:
        line: Source line containing the candidate.
        token: Candidate text to analyze.
    Output: Clamped token-center position ratio from 0.0 (start of line) to 1.0 (end); defaults to 0.5
        for missing text or token.
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
    Description: Detect if token is part of a multi-line secret block, such as a PEM or SSH key.
    Parameters:
        context_text: Surrounding source text.
        token: Candidate text to analyze.
    Output: True when the context and token are nonempty and a PEM or SSH marker is present; otherwise
        False. A marker alone does not prove the token is inside the block.
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
    Description: Detect if token was extracted from a URL, DSN, or connection string.
    Parameters:
        line: Source line containing the candidate.
        token: Candidate text to analyze.
    Output: True for a URL or DSN pattern, or a token query-parameter match, on a nonempty line;
        otherwise False.
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
    Description: Detect the type of quotes wrapping the token.
    Parameters:
        line: Source line containing the candidate.
        token: Candidate text to analyze.
    Output: Quote code: 0 for none, 1 for single quotes ('), 2 for double quotes ("), or 3 for
        backticks (`).
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
    Description: Count the number of high-entropy tokens on the same line.
    Parameters:
        line: Source line containing the candidate.
    Output: Count of candidates at least eight characters long with Shannon entropy greater than 3.5.
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
    Description: Calculate the distance, in characters, between a variable name and the token.
    Parameters:
        line: Source line containing the candidate.
        token: Candidate text to analyze.
        var_name: Optional variable or key name.
    Output: Absolute distance from the end of the variable name to the start of the token; -1 when
        an input is missing or the name or the token is not in the line.
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
    Description: Estimate the depth of the token in a JSON or YAML structure.
    Parameters:
        line: Source line containing the candidate.
    Output: Maximum of the two-space indentation depth and the opening brace or bracket count; zero for
        empty text. Zero means root level, and N means nested N levels deep.
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
    Description: Calculate N-gram score for variable names near the token.
    Parameters:
        context_text: Surrounding source text.
        token: Candidate text to analyze.
    Output: Sum of secret N-gram scores for neighboring variables, taken from context lines that do not
        contain the token; zero if the token is absent.
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


def _detect_assignment_type(line: str, token: str) -> int:
    """
    Description: Detect the type of assignment operator used.
    Parameters:
        line: Source line containing the candidate.
        token: Candidate text to analyze.
    Output: Code for the operator before the token: 3 for :=, 4 for =>, 2 for colon, 1 for equals, or
        0 for unknown.
    """
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
    Description: Check if the token itself is wrapped in quotes, as a string literal. This is stricter
        than checking whether quotes appear anywhere in the line.
    Parameters:
        line: The line of code containing the token.
        token: The token to check.
    Output: True when matching quotes surround the token, including within a longer string; otherwise
        False.
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
    """
    Description: Extract token properties and structure as model features. Prefix and hex-shape
        features are left out on purpose to prevent shortcut learning, while UUID, JWT,
        entropy-mismatch, and hash-prefix features are kept for precision.
    Parameters:
        token: Candidate text to analyze.
        regex_match_type: 0 for entropy, or 1 or more for a regex pattern.
        line: Source line containing the candidate.
        context_text: Surrounding source text.
    Output: Dictionary of 20 token features.
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
        "vendor_prefix_boost": _get_vendor_prefix_boost(token),
        # Token position/structure features
        "token_span_offset": _calculate_token_span_offset(line, token),
        "token_in_multiline_block": _detect_multiline_block(context_text, token),
        "embedded_token_flag": _detect_embedded_token(line, token),
        "token_quote_type": _detect_quote_type(line, token),
        # Discriminative features for precision boost
        "is_uuid_v4": _is_uuid_v4(token),
        "jwt_structure_valid": _is_jwt_structure_valid(token),
        "entropy_charset_mismatch": _calculate_entropy_charset_mismatch(token),
        "has_hash_prefix": _has_hash_prefix(token),
    }


def _extract_var_features(var_name: Optional[str], line: str, token: str) -> dict:
    """
    Description: Extract variable name features for the model.
    Parameters:
        var_name: Optional variable or key name.
        line: Source line containing the candidate.
        token: Candidate text to analyze.
    Output: Dictionary of nine variable and assignment features, with defaults for a missing variable
        name.
    """
    if not var_name:
        return {
            "var_name_extracted": False,
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
    """
    Description: Extract surrounding-code and file metadata as model context features.
    Parameters:
        context: Surrounding code and file metadata.
        token: Candidate text to analyze.
        var_name: Optional variable or key name.
    Output: Dictionary of 16 context features.
    """
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
        re.compile(
            r"^[ \t]*(?:(?:public|private|protected)[ \t]+)?(?:static[ \t]+)?\w+[ \t]+\w+[ \t]*\(",
            re.M,
        ),  # Java
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
        "semantic_context_score": _calculate_semantic_context_score(full_text),
        "surrounding_secret_density": _calculate_surrounding_secret_density(full_text),
        # Advanced context features
        "surrounding_token_count": _count_surrounding_tokens(line),
        "key_value_distance": _calculate_key_value_distance(line, token, var_name),
        "json_path_hint": _calculate_json_path_hint(line),
        "adjacency_ngram_score": _calculate_adjacency_ngram_score(full_text, token),
    }


def _extract_stage_b_precision_features(token: str, file_path: Optional[str]) -> dict:
    """
    Description: Extract the Stage B precision features that target observed false-positive patterns:
        40- and 64-character hex tokens (git SHAs, SHA256 hashes), test-mode tokens (sk_test_*,
        pk_test_*), and git, build, and example file paths.
    Parameters:
        token: The token to analyze.
        file_path: The file path for context.
    Output: Dictionary of six Stage B precision features.
    """
    return {
        "is_hex_len_40": _is_hex_exact_length(token, 40),
        "is_hex_len_64": _is_hex_exact_length(token, 64),
        "is_test_token": _is_test_token(token),
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
    Description: Extract Stage B generalization features that separate hex secrets from git SHAs and
        checksums. The features come from the surrounding context rather than the token format.
    Parameters:
        token: The token to analyze.
        context_text: Full context, with before, current, and after lines joined.
        file_path: The file path for context.
        line_content: The line containing the token.
        embedded_in_url: Whether the token was extracted from a URL or DSN.
    Output: Dictionary of four disambiguation features for 32-, 40-, or 64-character hexadecimal tokens,
        or defaults otherwise.
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
            "hex_context_crypto_keywords": False,
            "hex_adjacent_assignment_pattern": 0,
            "hex_in_url_or_dsn": False,
            "hex_file_suggests_secret": False,
        }

    context_lower = context_text.lower() if context_text else ""
    line_lower = line_content.lower() if line_content else ""

    # Feature 1: Crypto/secret keywords in context (suggests IS a secret)
    crypto_keywords = ["encrypt", "decrypt", "sign", "key", "secret", "auth", "password", "credential", "token"]
    hex_context_crypto = any(kw in context_lower for kw in crypto_keywords)

    # Feature 2: Assignment pattern (helps identify source)
    # 0=unknown, 1=env_var, 2=config_assignment, 3=func_arg
    assignment_pattern = 0
    if "os.environ" in line_lower or "process.env" in line_lower or "env[" in line_lower:
        assignment_pattern = 1  # Environment variable
    elif ": " in line_content or "= " in line_content or "=" in line_content:
        assignment_pattern = 2  # Config assignment
    elif "(" in line_content and ")" in line_content:
        assignment_pattern = 3  # Function argument

    # Feature 3: Token embedded in URL/DSN
    hex_in_url = embedded_in_url or any(
        pattern in line_lower
        for pattern in ["http://", "https://", "://", "postgres://", "mysql://", "mongodb://"]
    )

    # Feature 4: File path suggests secrets
    hex_file_secret = False
    if file_path:
        path_lower = file_path.lower()
        secret_paths = [
            "secrets/", "secret/", ".env", "credentials/", "credential/",
            "keys/", "key/", "private/", "auth/", "vault/", "_secrets",
        ]
        hex_file_secret = any(sp in path_lower for sp in secret_paths)

    return {
        "hex_context_crypto_keywords": hex_context_crypto,
        "hex_adjacent_assignment_pattern": assignment_pattern,
        "hex_in_url_or_dsn": hex_in_url,
        "hex_file_suggests_secret": hex_file_secret,
    }


# ---------------------------------------------------------------------------
# Env-loading pattern awareness (features 64-65)
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
    """
    Description: Check if the surrounding context contains any environment-loading pattern.
    Parameters:
        context_text: Surrounding source text.
        line: Source line containing the candidate.
    Output: True when the context or the current line matches an environment-loader pattern; otherwise
        False.
    """
    text = f"{context_text}\n{line}"
    return any(pat.search(text) for pat in _ENV_LOADER_PATTERNS)


def _detect_env_fallback_value(line: str, token_start_index: int) -> bool:
    """
    Description: Flag default values in environment-loader calls by comparing the token's start index
        with matched fallback starts. The start index comes from upstream extraction (TokenMatch.start),
        not line.find(), which grabs the wrong occurrence when a token appears twice.
    Parameters:
        line: Source line containing the candidate.
        token_start_index: Authoritative token start offset from upstream extraction.
    Output: True when the index is at or beyond a matched fallback start minus one; otherwise False.
        There is no closing-boundary check, so the token is not verified to end inside the argument.
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
    Description: Extract all 64 features from a finding and its context.
    Parameters:
        finding: The Finding object with token and metadata.
        context: CodeContext with surrounding code.
        regex_match_type: Encoded type of regex match (0 = entropy-only).
    Output: FeatureVector containing all 64 features inferred from the finding and context.
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
    var_name = feature_var_name(context.line_content, token)

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

    # Value-shape features
    value_shape = _value_shape_features(token)

    # Combine all features
    return FeatureVector(
        **token_features,
        **var_features,
        **context_features,
        **stage_b_features,
        **hex_features,
        env_loader_in_context=env_in_ctx,
        is_env_fallback_value=env_fallback,
        **value_shape,
    )


def extract_features_from_record(record: dict) -> FeatureVector:
    """
    Description: Extract features from a training data record.
    Parameters:
        record: Dict with token, line_content, context_before, context_after, etc.
    Output: FeatureVector containing all 64 features, using record metadata and inferred context
        defaults.
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

    token_start = record.get("token_start")
    token_end = record.get("token_end")

    # Create a minimal Finding object
    finding = Finding(
        type=record.get("secret_type", "ENTROPY_CANDIDATE"),
        snippet=line_content,
        evidence=EvidenceType.ENTROPY,
        token=token,
        token_start=token_start,
        token_end=token_end,
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
    if token_start is not None and token_end is not None:
        from Harpocrates.ml.tokens import TokenMatch
        context.token_match = TokenMatch(
            token=token, start=token_start, end=token_end, kind="entropy"
        )

    return extract_features(finding, context)


__all__ = [
    "FEATURE_NAMES",
    "FeatureVector",
    "extract_features",
    "extract_features_from_record",
    "KNOWN_PREFIXES",
    "SECRET_VAR_PATTERNS",
    "SAFE_VAR_PATTERNS",
]
