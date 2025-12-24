"""
Feature engineering for ML-based secrets verification.

Extracts 37 features from tokens and their context to enable
context-aware classification of potential secrets.

Features are organized into three categories:
- Token features (14): Properties of the token itself including advanced
  entropy analysis and vendor prefix detection
- Variable name features (10): Properties of the variable/key name including
  N-gram scoring for secret and safe patterns
- Context features (13): Properties of surrounding code including semantic
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

# Maximum possible entropy per character class
MAX_ENTROPY_ALPHANUMERIC = 5.954  # log2(62) for a-zA-Z0-9
MAX_ENTROPY_BASE64 = 6.0  # log2(64)
MAX_ENTROPY_HEX = 4.0  # log2(16)

# Variable name N-gram weights for secret detection
SECRET_NGRAMS: Dict[str, float] = {
    # Highest confidence secret indicators
    "api_key": 1.0, "apikey": 1.0, "api-key": 1.0,
    "secret": 0.9, "password": 0.9, "passwd": 0.9, "pwd": 0.85,
    "private_key": 1.0, "privatekey": 1.0,
    "access_key": 0.95, "accesskey": 0.95,
    "auth_token": 0.95, "authtoken": 0.95,
    "bearer": 0.9, "credential": 0.9,
    # Medium confidence
    "token": 0.7, "key": 0.5,  # Generic, needs context
    "client_secret": 0.95, "clientsecret": 0.95,
    "signing_key": 0.9, "signingkey": 0.9,
    "encryption_key": 0.9, "encryptionkey": 0.9,
}

# Variable name N-gram weights for safe (non-secret) detection
SAFE_NGRAMS: Dict[str, float] = {
    # Highest confidence safe indicators
    "commit": 0.9, "commit_sha": 1.0, "commitsha": 1.0,
    "sha": 0.8, "sha1": 0.9, "sha256": 0.95, "sha512": 0.95,
    "hash": 0.85, "checksum": 0.95, "digest": 0.9,
    "uuid": 0.95, "guid": 0.95, "identifier": 0.7,
    "version": 0.8, "revision": 0.85, "rev": 0.7,
    # Context-dependent safe indicators
    "example": 0.9, "sample": 0.85, "placeholder": 0.95,
    "test": 0.7, "mock": 0.85, "fake": 0.9, "dummy": 0.9,
    "fingerprint": 0.9, "session_id": 0.7, "trace_id": 0.8,
}


@dataclass
class FeatureVector:
    """
    37 extracted features for ML classification.

    Organized into three categories:
    - Token features (14): Properties of the token itself
    - Variable name features (10): Properties of the variable/key name
    - Context features (13): Properties of surrounding code

    NOTE: The following features were REMOVED to prevent shortcut learning:
    - has_known_prefix: Directly encodes token type based on prefix
    - prefix_type: Maps prefixes to secret categories
    - is_hex_like: Strongly correlated with non-secrets (git SHAs)

    The model should learn from CONTEXT, not token format.
    """

    # Token features (14) - includes advanced entropy analysis
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

    # Context features (13)
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

    def to_array(self) -> List[float]:
        """Convert to numpy-compatible array of 37 floats."""
        return [
            # Token features (14)
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
            # Context features (13)
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
        ]

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get ordered list of 37 feature names."""
        return [
            # Token features (14)
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
            # Context features (13)
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
    has_lower = any(c.islower() for c in token)
    has_upper = any(c.isupper() for c in token)
    has_digit = any(c.isdigit() for c in token)
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


def _calculate_ngram_score(var_name: str, ngram_dict: Dict[str, float]) -> float:
    """
    Calculate weighted N-gram score for a variable name.

    Args:
        var_name: The variable/key name to analyze
        ngram_dict: Dictionary mapping N-grams to weights

    Returns:
        0.0-1.0: Weighted match score (max of all matches)
    """
    if not var_name:
        return 0.0

    var_lower = var_name.lower()
    max_score = 0.0

    for ngram, weight in ngram_dict.items():
        if ngram in var_lower:
            max_score = max(max_score, weight)

    return max_score


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


def _extract_token_features(token: str, regex_match_type: int = 0) -> dict:
    """Extract token-level features (14 features).

    NOTE: The following features are intentionally NOT included to prevent
    shortcut learning:
    - has_known_prefix: Would allow model to trivially match AKIA*, ghp_*, sk-*
    - prefix_type: Would directly encode secret type from prefix
    - is_hex_like: Would strongly correlate with non-secrets (git SHAs)

    Instead, we use advanced entropy analysis and vendor prefix boost which
    provide the model with useful signal without enabling trivial shortcuts.
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
            "in_string_literal": "'" in line or '"' in line,
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
        "in_string_literal": "'" in line or '"' in line,
        # N-gram weighted scores for secret/safe variable name patterns
        "var_ngram_secret_score": _calculate_ngram_score(var_name, SECRET_NGRAMS),
        "var_ngram_safe_score": _calculate_ngram_score(var_name, SAFE_NGRAMS),
    }


def _extract_context_features(context: CodeContext) -> dict:
    """Extract context-level features."""
    full_text = context.full_context

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
    }


def extract_features(
    finding: "Finding",
    context: CodeContext,
    regex_match_type: int = 0,
) -> FeatureVector:
    """
    Extract all 37 features from a finding and its context.

    Args:
        finding: The Finding object with token and metadata
        context: CodeContext with surrounding code
        regex_match_type: Encoded type of regex match (0 = entropy-only)

    Returns:
        FeatureVector with all 37 features
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

    # Extract features from each category
    token_features = _extract_token_features(token, regex_match_type)

    var_name = extract_var_name(context.line_content, token)
    var_features = _extract_var_features(var_name, context.line_content, token)

    context_features = _extract_context_features(context)

    # Combine all features
    return FeatureVector(
        **token_features,
        **var_features,
        **context_features,
    )


def extract_features_from_record(record: dict) -> FeatureVector:
    """
    Extract features from a training data record.

    Args:
        record: Dict with token, line_content, context_before, context_after, etc.

    Returns:
        FeatureVector with all 37 features
    """
    from Harpocrates.core.result import EvidenceType, Finding

    token = record.get("token", "")
    line_content = record.get("line_content", "")

    # Create a minimal Finding object
    finding = Finding(
        type=record.get("secret_type", "ENTROPY_CANDIDATE"),
        snippet=line_content,
        evidence=EvidenceType.ENTROPY,
        token=token,
    )

    # Create CodeContext from record
    context = CodeContext(
        line_content=line_content,
        lines_before=record.get("context_before", []),
        lines_after=record.get("context_after", []),
        file_path=record.get("file_path"),
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
