"""
Core detection engine for Harpocrates.

Three-phase pipeline per line:
  1. CRITICAL regex  → Finding(CRITICAL, confidence=0.99) — skips ML
  2. HIGH regex      → Finding(HIGH,     confidence=0.95) — skips ML
  3. Entropy         → Finding(ENTROPY_CANDIDATE)         → ML verification

Regex hits bypass XGBoost entirely; only entropy candidates are forwarded
to the ML verifier. This keeps the fast path deterministic and CPU-free.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from Harpocrates.core.classification import extract_var_name, infer_category
from Harpocrates.core.result import EvidenceType, Finding, Severity
from Harpocrates.detectors.entropy_detector import looks_like_secret, shannon_entropy
from Harpocrates.detectors.regex_patterns import CRITICAL_SIGNATURES, HIGH_SIGNATURES
from Harpocrates.ml.context import HIGH_RISK_EXTENSIONS, LOW_NOISE_EXTENSIONS
from Harpocrates.utils.file_utils import iter_text_lines

if TYPE_CHECKING:
    from Harpocrates.core.classification import CategoryInference
    from Harpocrates.ml.context import CodeContext
    from Harpocrates.ml.verifier import Verifier


class MLVerificationError(RuntimeError):
    """Raised when an explicitly requested ML verification cannot complete."""


_ML_VERIFICATION_BATCH_SIZE = 1024


# Tier 2: tightened token alphabet — drops '.' so dotted identifiers and URLs
# no longer form single long tokens. Retains '-' for UUID-style and hyphenated
# API keys (Azure SAS, some OAuth tokens). Minimum length matches the
# looks_like_secret() floor, eliminating a redundant per-token short check.
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")

# Tier 2: strip URL and data: URI runs before tokenizing so CDN/IDP/doc URLs
# and inline sourcemap data URIs don't generate entropy candidates.
_URL_RE = re.compile(r"(?:https?://|data:)\S+")

# Tier 1: pure base64 lines of PEM / X.509 certificate bodies.
# RFC 7468 specifies exactly 64 chars per non-terminal body line. Using {64}
# (not a range) avoids false-matching 60–63 or 65–76 char real-credential
# lines in non-PEM contexts (raw AES-256 keys, JWT segments, etc.).
_PEM_BODY_RE = re.compile(r"^[A-Za-z0-9+/]{64}$")

# Comment detection — all common styles: Python/shell/Ruby/YAML (#),
# JS/TS/Go/Java/C/Rust (// and /* */), HTML/XML (<!-- -->), SQL/Lua (--).
# Single `*` catches continuation lines inside /* */ blocks.
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "<!--", "--")
_COMMENT_STRIP_RE = re.compile(r"^(?:#+|//+|/\*+|\*+|<!--|--)\s*")
# Arch override 2026-05-06: compiled regex replaces any() generator for the
# prose-comment fast-path. ~10x faster on heavily-commented files.
_PROSE_FILTER_RE = re.compile(r"[=:'\"]")

# Phase 2b: sensitive-variable assignment bypass — forwards low-entropy literals
# assigned to clearly credential-named variables directly to ML, skipping the
# entropy gate. Only fires when regex phases found nothing on the line.
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![a-zA-Z])(?:pass(?:word|wd|w)?|pwd|usr(?:name)?|user|host|conn(?:ection|str)?|secret|token|key|auth|cred)[a-z0-9_]{0,40}\s*[:=]\s*['\"]([^'\"]{3,100})['\"]"
)

# Phase 2c: unquoted KEY=VALUE for .env-style files. Restricted to
# HIGH_RISK_EXTENSIONS (detector.py gates by file extension). Covers cases
# _SENSITIVE_ASSIGNMENT_RE misses because .env files write values without quotes.
# Value stops at whitespace or quote to avoid double-matching quoted .env values.
# Intentionally excludes `host` and `user` — unquoted HOST=localhost and
# USER=postgres are standard config, not credentials; quoted variants are still
# caught by _SENSITIVE_ASSIGNMENT_RE.
_ENV_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![a-zA-Z])(?:pass(?:word|wd|w)?|pwd|conn(?:ection|str)?|secret|token|key|auth|cred|url|endpoint|callback)[a-z0-9_]{0,40}\s*=\s*([^\s'\"]{3,200})"
)

# Phase 2d: credential shapes the tokenizer cannot see, forwarded to ML as
# candidates (the verifier decides). Tokens are >=20 chars of [A-Za-z0-9+/=_-]
# and URLs are stripped, so these never became entropy candidates:
# - URI with userinfo password (postgres://user:pw@host, redis://:pw@host)
# Scheme is bounded and must not continue a longer [a-z0-9+.-] run: otherwise a long dotted/hyphenated
# run with no "://" is rescanned from every boundary (quadratic on minified lines).
_CONN_URI_RE = re.compile(r"(?<![a-z0-9+.\-])[a-z][a-z0-9+.\-]{0,19}://[^\s:/@'\"]*:[^\s@'\"/]{3,}@[^\s'\"]+")
# - key=value password fields in ADO/ODBC/JDBC strings (value not quote-delimited)
_KV_PASSWORD_RE = re.compile(r"(?i)(?:^|[;&?\s'\"])(?:password|pwd|passwd)\s*=\s*([^;&'\"\s]{3,})")
# - unquoted `name: value` credentials (YAML, compose, properties); value must be password-shaped
_COLON_CREDENTIAL_RE = re.compile(
    r"(?i)(?<![a-z])(?:pass(?:word|wd)?|pwd|secret|token|api_?key|auth)[a-z0-9_]{0,40}['\"]?\s*:\s+([^\s'\"#]{6,200})\s*$"
)
# - credential query parameters in URLs (run before URL stripping)
_URL_TOKEN_RE = re.compile(
    r"(?i)[?&](?:token|access_token|api_key|apikey|key|sig|signature|auth|secret|client_secret)=([^&\s'\"#]{8,})"
)
# - long lowercase hex (auth tokens, HMAC keys) that sits below the entropy gate
_HEX_SECRET_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{32,64}(?![0-9a-f])")
# - password-shaped quoted literals regardless of variable name: letters + digits
#   + (uppercase or a password symbol), no spaces/paths
_QUOTED_LITERAL_RE = re.compile(r"['\"]([^'\"\s/]{8,64})['\"]")
_PASSWORD_SYMBOLS = set("!@#$%^&*+=")
# Round 2 (bench/model_improvement.ipynb section 14), from uncovered secrets in LLM-written code:
# - header values: `Bearer <token>` (any length >= 6) and `X-Deploy-Key: <token>`
_BEARER_RE = re.compile(r"(?i)\bbearer\s+([^\s'\"\\<>`]{6,200})")
_HEADER_SECRET_RE = re.compile(r"(?i)\b[\w-]{0,40}(?:key|token|sig|signature|secret)[\w-]{0,20}:\s*([^\s'\"\\`]{12,200})")
# - secrets inside URLs: key-only userinfo (Sentry DSN), random path segments, any query value
_URL_IN_LINE_RE = re.compile(r"(?i)(?<![a-z0-9+.\-])[a-z][a-z0-9+.\-]{0,19}://[^\s'\"<>`]{1,2000}")
_URL_TRAILING = ").,;:!?]}"
_URL_PIECE_RE = re.compile(r"[^/?&#]+")
_USERINFO_KEY_RE = re.compile(r"[A-Za-z0-9_+\-]{16,200}(?=@)")
_WORDS_JOINED_RE = re.compile(r"[A-Za-z]+(?:[-_.~][A-Za-z0-9]+)*")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
# - symbol-heavy random values after = or : (generic random secrets the tokenizer splits apart)
_ASSIGNED_VALUE_RE = re.compile(r"[=:]\s*[\"']?([^\s'\"`]{12,120})")
_SOUP_SYMBOLS = set("!@#$%^&*()[]{}<>;,?|~")
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{5,}")


def _random_url_piece(value: str) -> bool:
    return (len(value) >= 16 and _password_shaped(value) and not _WORDS_JOINED_RE.fullmatch(value)
            and not _UUID_RE.fullmatch(value))


def _symbol_soup(value: str) -> bool:
    """Random punctuation-heavy secret: >= 3 distinct unusual symbols, letters and digits, and no
    word-like letter run (code and prose have long letter runs; random strings don't)."""
    return (len(_SOUP_SYMBOLS & set(value)) >= 3 and sum(c.isalpha() for c in value) >= 3
            and any(c.isdigit() for c in value) and not _LETTER_RUN_RE.search(value)
            and "://" not in value)


def _url_candidates(scan_target: str) -> List[Tuple[str, int, int]]:
    out = []
    for url in _URL_IN_LINE_RE.finditer(scan_target):
        rest = url.start() + url.group().index("://") + 3
        end = url.start() + len(url.group().rstrip(_URL_TRAILING))  # "(see https://x/y)." ends before ")."
        m = _USERINFO_KEY_RE.match(scan_target, rest, end)
        if m and m.end() < end:
            out.append((m.group(), m.start(), m.end()))
        query = scan_target.find("?", rest, end)
        query = end if query < 0 else query
        for piece in _URL_PIECE_RE.finditer(scan_target, rest, end):
            if piece.start() == rest:  # the host
                continue
            value, start = piece.group(), piece.start()
            if start > query and "=" in value:  # name=value only in the query; a path keeps its base64 padding
                cut = value.index("=") + 1
                value, start = value[cut:], start + cut
            if _random_url_piece(value):
                out.append((value, start, start + len(value)))
    return out


def _password_shaped(value: str) -> bool:
    letters = sum(c.isalpha() for c in value)
    digits = sum(c.isdigit() for c in value)
    if not letters or not digits:
        return False
    # Mixed case or a password symbol, or a weak lowercase+digits password like "hunter4242".
    return (any(c.isupper() for c in value) or bool(_PASSWORD_SYMBOLS & set(value))
            or (letters >= 3 and digits >= 2))


def _structured_candidates(scan_target: str) -> List[Tuple[str, int, int]]:
    """(value, start, end) for Phase 2d shapes, in scan_target coordinates."""
    out = [(m.group(), m.start(), m.end()) for m in _CONN_URI_RE.finditer(scan_target)] if "://" in scan_target else []
    for pattern in (_KV_PASSWORD_RE, _URL_TOKEN_RE, _HEX_SECRET_RE):
        for m in pattern.finditer(scan_target):
            group = 1 if pattern.groups else 0
            out.append((m.group(group), m.start(group), m.end(group)))
    for m in (*_QUOTED_LITERAL_RE.finditer(scan_target), *_COLON_CREDENTIAL_RE.finditer(scan_target),
              *_BEARER_RE.finditer(scan_target), *_HEADER_SECRET_RE.finditer(scan_target)):
        if _password_shaped(m.group(1)):
            out.append((m.group(1), m.start(1), m.end(1)))
    for m in _ASSIGNED_VALUE_RE.finditer(scan_target):
        if _symbol_soup(m.group(1)):
            out.append((m.group(1), m.start(1), m.end(1)))
    if "://" in scan_target:
        out += _url_candidates(scan_target)
    return out


# Characters kept on each side of a token when building ML context (NFR-01 latency bound).
_CONTEXT_WINDOW = 256


@functools.lru_cache(maxsize=512)
def _file_ext(file: str) -> str:
    """File extension used for risk routing, computed once per file instead of once per line."""
    ext = Path(file).suffix.lower()
    if not ext or ext not in HIGH_RISK_EXTENSIONS:
        basename = Path(file).name.lower()
        if basename == ".env" or basename.startswith(".env."):
            return ".env"
    return ext


def _calculate_entropy_confidence(entropy_val: Optional[float]) -> float:
    """Map entropy 4.0–5.5 linearly to confidence 0.6–0.8."""
    if entropy_val is None:
        return 0.6
    if entropy_val >= 5.5:
        return 0.8
    if entropy_val >= 4.0:
        return 0.6 + (entropy_val - 4.0) * (0.2 / 1.5)
    return 0.6


def _severity_from_entropy(inference: "CategoryInference") -> Severity:
    """Severity for entropy-stage findings (before ML verification, if any).

    Delegates to _severity_from_classification without ml_confidence so strong
    heuristic signals (e.g., PASSWORD 0.90, JWT 0.85) reach HIGH without
    requiring ML confirmation.  ML verification can still upgrade MEDIUM→HIGH
    via ml_confidence when the verifier runs.
    """
    return _severity_from_classification(inference)


def _severity_from_classification(
    inference: "CategoryInference",
    ml_confidence: Optional[float] = None,
) -> Severity:
    """
    Map (category, classification confidence, optional ML confidence) → Severity.

    Banding:
      INFO   — no/weak signal (cat_conf < 0.5, or GENERIC_SECRET with cat_conf < 0.7)
      MEDIUM — moderate signal (specific category cat_conf in [0.70, 0.85), or
               GENERIC_SECRET with cat_conf >= 0.70)
      HIGH   — strong signal (specific category cat_conf >= 0.85, or ml_confidence
               >= 0.85 with any non-generic category)

    CRITICAL is intentionally NOT reachable here. Regex tier owns CRITICAL
    (deterministic format + 0.99 confidence). Heuristic+ML cannot make that
    claim; HIGH is the ceiling for entropy/ML paths.
    """
    from Harpocrates.core.classification import ViolationCategory

    cat = inference.category
    cat_conf = inference.confidence

    if cat_conf < 0.5:
        return Severity.INFO

    if cat != ViolationCategory.GENERIC_SECRET and cat_conf >= 0.85:
        return Severity.HIGH

    if (
        ml_confidence is not None
        and ml_confidence >= 0.85
        and cat != ViolationCategory.GENERIC_SECRET
    ):
        return Severity.HIGH

    if cat != ViolationCategory.GENERIC_SECRET and cat_conf >= 0.70:
        return Severity.MEDIUM

    if cat == ViolationCategory.GENERIC_SECRET and cat_conf >= 0.70:
        return Severity.MEDIUM

    return Severity.INFO


def _scan_line(line: str, lineno: int, file: Optional[str]) -> List[Finding]:
    """
    Scan one line through all three detection phases.

    Returns findings ordered: CRITICAL regex → HIGH regex → entropy.
    Entropy phase is skipped if any regex hit is found on the line.
    """
    findings: List[Finding] = []
    stripped = line.strip()

    # Determine whether this file belongs to a credential-bearing extension set
    # (.env, .pem, .key, etc.) where URL bodies are credentials, not noise.
    # Also handles dotted-variant names like .env.local and .env.production.
    file_ext = _file_ext(file) if file is not None else ""

    if not stripped:
        return findings

    # Detect comment lines and strip the prefix so the payload can be scanned.
    # PEM headers start with five hyphens and must not be treated as SQL comments.
    is_comment = not stripped.startswith("-----BEGIN ") and stripped.startswith(_COMMENT_PREFIXES)
    scan_target = _COMMENT_STRIP_RE.sub("", stripped) if is_comment else stripped

    in_comment = True if is_comment else None

    # ------------------------------------------------------------------
    # Phase 1a: CRITICAL regex — deterministic, no ML needed.
    # ------------------------------------------------------------------
    for sig_name, pattern in CRITICAL_SIGNATURES.items():
        for match in pattern.finditer(scan_target):
            token = match.group()
            _inf = infer_category(signature_name=sig_name, var_name=None, token=token)
            findings.append(
                Finding(
                    type=sig_name,
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=shannon_entropy(token) if token else 0.0,
                    evidence=EvidenceType.REGEX,
                    severity=Severity.CRITICAL,
                    confidence=0.99,
                    token=token,
                    token_start=match.start(),
                    token_end=match.end(),
                    in_comment=in_comment,
                    category=_inf.category.value,
                    category_reason=_inf.reason,
                )
            )

    # ------------------------------------------------------------------
    # Phase 1b: HIGH regex — also deterministic, also bypasses ML.
    # ------------------------------------------------------------------
    for sig_name, pattern in HIGH_SIGNATURES.items():
        for match in pattern.finditer(scan_target):
            token = match.group()
            _inf = infer_category(signature_name=sig_name, var_name=None, token=token)
            findings.append(
                Finding(
                    type=sig_name,
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=shannon_entropy(token) if token else 0.0,
                    evidence=EvidenceType.REGEX,
                    severity=Severity.HIGH,
                    confidence=0.95,
                    token=token,
                    token_start=match.start(),
                    token_end=match.end(),
                    in_comment=in_comment,
                    category=_inf.category.value,
                    category_reason=_inf.reason,
                )
            )

    # ------------------------------------------------------------------
    # Phase 2: Entropy fallback — only when regex found nothing.
    # ------------------------------------------------------------------
    if not findings:
        # Prose-comment guard: comments with no `=`, `:`, or quote characters
        # cannot contain assignment-style secrets — skip entropy/ML to preserve
        # the 2ms budget on heavily-commented files (legal headers, JSDoc, etc.).
        if is_comment and not _PROSE_FILTER_RE.search(scan_target):
            return findings

        # Tier 1: skip PEM/X.509 certificate body lines (pure base64, 60-76
        # chars). The BEGIN header was already caught by the regex tier above.
        if _PEM_BODY_RE.match(scan_target):
            return findings

        # Tier 1b: skip entropy for markup/style/vector files — webpack hashes,
        # CSS class fingerprints, and SVG path data saturate the entropy threshold
        # but are never credentials. Regex tier already ran above and catches any
        # real structured API key in an HTML <script> block.
        if file_ext in LOW_NOISE_EXTENSIONS:
            return findings

        # Tier 2: strip URL substrings before tokenizing so CDN/IDP/doc URLs
        # don't generate entropy candidates from their path components.
        # Exception: skip stripping for credential-bearing file types (.env,
        # .pem, .key, etc.) where URLs ARE the credential (e.g. DATABASE_URL).
        if file_ext in HIGH_RISK_EXTENSIONS:
            scan_text = scan_target
        else:
            scan_text = _URL_RE.sub(" ", scan_target)

        # TODO(v0.3): switch to finditer to capture offsets → TokenMatch for entropy candidates.
        # URL stripping does not preserve length so offsets would be scan_text-relative;
        # re-derive against stripped once the ONNX is retrained on correct features.
        for token in _TOKEN_RE.findall(scan_text):
            if looks_like_secret(token):
                ent = shannon_entropy(token)
                _vn = extract_var_name(scan_target, token)
                _inf = infer_category(signature_name=None, var_name=_vn, token=token, line=scan_target)
                findings.append(
                    Finding(
                        type="ENTROPY_CANDIDATE",
                        file=file,
                        line=lineno,
                        snippet=stripped[:200],
                        entropy=ent,
                        evidence=EvidenceType.ENTROPY,
                        severity=_severity_from_entropy(_inf),
                        confidence=_calculate_entropy_confidence(ent),
                        token=token,
                        in_comment=in_comment,
                        category=_inf.category.value,
                        category_reason=_inf.reason,
                    )
                )

        found_tokens = {f.token for f in findings}
        # NOTE(v0.3): token_start/token_end here are relative to scan_text
        # (post-URL-strip, and for comment lines also post-prefix-strip). They
        # are NOT relative to the original line. On comment lines the offset
        # drift is: actual_start = token_start + len(comment_prefix_stripped).
        # This is acceptable for v0.2.x — TokenMatch consumers in features.py
        # still call line.find(token) and are not yet consuming these offsets.
        # Fix alongside the v0.3 finditer migration.
        for match in _SENSITIVE_ASSIGNMENT_RE.finditer(scan_text):
            value = match.group(1)
            if value not in found_tokens:
                ent = shannon_entropy(value)
                # var name: everything left of the value in the match, stripped
                _vn = scan_text[: match.start(1)].split("=")[0].split(":")[0].strip()
                _vn = _vn or None
                _inf = infer_category(signature_name=None, var_name=_vn, token=value, line=scan_target)
                findings.append(
                    Finding(
                        type="ML_CANDIDATE",
                        file=file,
                        line=lineno,
                        snippet=stripped[:200],
                        entropy=ent,
                        evidence=EvidenceType.ML,
                        severity=_severity_from_entropy(_inf),
                        confidence=0.5,
                        token=value,
                        token_start=match.start(1),
                        token_end=match.end(1),
                        in_comment=in_comment,
                        category=_inf.category.value,
                        category_reason=_inf.reason,
                    )
                )
                found_tokens.add(value)

        # Phase 2d: structured credential shapes (see _structured_candidates).
        for value, start, end in _structured_candidates(scan_target):
            if value in found_tokens:  # exact only: a password inside a whole-string candidate still gets its own
                continue
            _vn = extract_var_name(scan_target, value)
            _inf = infer_category(signature_name=None, var_name=_vn, token=value, line=scan_target)
            findings.append(
                Finding(
                    type="ML_CANDIDATE",
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=shannon_entropy(value),
                    evidence=EvidenceType.ML,
                    severity=_severity_from_entropy(_inf),
                    confidence=0.5,
                    token=value,
                    token_start=start,
                    token_end=end,
                    in_comment=in_comment,
                    category=_inf.category.value,
                    category_reason=_inf.reason,
                )
            )
            found_tokens.add(value)

        # Phase 2c: unquoted KEY=VALUE for .env-style files. Runs on scan_text
        # which equals scan_target for HIGH_RISK files (URL-strip is skipped
        # above), so offsets are consistent with found_tokens populated by the
        # entropy and SA passes. Deduplicates against found_tokens to avoid
        # double-emitting a value already caught above.
        if file_ext in HIGH_RISK_EXTENSIONS:
            for match in _ENV_ASSIGNMENT_RE.finditer(scan_text):
                value = match.group(1)
                if value in found_tokens:
                    continue
                _vn = scan_text[: match.start(1)].split("=")[0].split(":")[0].strip()
                _vn = _vn or None
                _inf = infer_category(signature_name=None, var_name=_vn, token=value, line=scan_target)
                findings.append(
                    Finding(
                        type="ENV_ASSIGNMENT",
                        file=file,
                        line=lineno,
                        snippet=stripped[:200],
                        entropy=shannon_entropy(value),
                        evidence=EvidenceType.REGEX,
                        severity=_severity_from_classification(_inf),
                        confidence=_inf.confidence,
                        token=value,
                        token_start=match.start(1),
                        token_end=match.end(1),
                        in_comment=in_comment,
                        category=_inf.category.value,
                        category_reason=_inf.reason,
                    )
                )
                found_tokens.add(value)

    return findings


def _apply_block_comment_flag(line_findings: List[Finding], in_block: bool) -> List[Finding]:
    """Set in_comment=True on findings from inside an open /* */ block.

    _scan_line only sets in_comment when the line STARTS with a comment prefix.
    Interior lines of /* */ blocks don't start with a prefix, so this patch
    propagates the open-block state tracked by the collector.
    """
    if not in_block or not line_findings:
        return line_findings
    import dataclasses

    return [
        dataclasses.replace(f, in_comment=True) if f.in_comment is None else f
        for f in line_findings
    ]


def _collect_text_findings(text: str) -> List[Finding]:
    """Raw scan of text — returns all findings including evidence=ML."""
    findings: List[Finding] = []
    in_block_comment = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        # Track /* */ block state: a line can open and close on the same line
        # (/* ... */), in which case we must NOT set in_block for the next line.
        opens = stripped.count("/*")
        closes = stripped.count("*/")
        was_in_block = in_block_comment and not stripped.startswith("/*")
        line_findings = _scan_line(line, lineno, file=None)
        findings.extend(_apply_block_comment_flag(line_findings, was_in_block))
        in_block_comment = (in_block_comment and closes == 0) or (opens > closes)
    return findings


def _collect_file_findings(path_obj: Path, max_bytes: Optional[int]) -> List[Finding]:
    """Raw scan of a file — returns all findings including evidence=ML."""
    file_name = str(path_obj)
    findings: List[Finding] = []
    in_block_comment = False
    for lineno, line in iter_text_lines(path_obj, max_bytes=max_bytes):
        stripped = line.strip()
        opens = stripped.count("/*")
        closes = stripped.count("*/")
        was_in_block = in_block_comment and not stripped.startswith("/*")
        line_findings = _scan_line(line, lineno, file=file_name)
        findings.extend(_apply_block_comment_flag(line_findings, was_in_block))
        in_block_comment = (in_block_comment and closes == 0) or (opens > closes)
    return findings


def detect_text(
    text: str,
    threshold: float = 4.0,
) -> List[Finding]:
    """
    Detect secrets in a text blob (regex + entropy, no ML).

    Args:
        text: Text content to scan
        threshold: Entropy threshold (reserved for future use)

    Returns:
        List of findings

    Example:
        >>> findings = detect_text("aws_key = AKIAIOSFODNN7EXAMPLE1234")
        >>> findings[0].type
        'AWS_ACCESS_KEY_ID'
    """
    _ = threshold
    return [f for f in _collect_text_findings(text) if f.evidence != EvidenceType.ML]


def detect_file(
    path: str | Path,
    threshold: float = 4.0,
    max_bytes: Optional[int] = None,
) -> List[Finding]:
    """
    Detect secrets in a file on disk (regex + entropy, no ML).

    Skips binary files, handles encoding errors gracefully, and
    returns an empty list for nonexistent files.

    Args:
        path: Path to the file to scan
        threshold: Entropy threshold (reserved)
        max_bytes: Maximum bytes to read

    Returns:
        List of findings
    """
    _ = threshold
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    return [f for f in _collect_file_findings(path_obj, max_bytes) if f.evidence != EvidenceType.ML]


# ---------------------------------------------------------------------------
# ML verification — applied only to entropy candidates, never regex hits.
# ---------------------------------------------------------------------------


_REASON_CONF_RE = re.compile(r"confidence=(\d+\.\d+)")


def _inference_from_finding(finding: Finding) -> "CategoryInference":
    """Reconstruct a CategoryInference from an existing finding's category fields.

    Used in _apply_ml_verification to recompute severity with ml_confidence.
    Recovers the original cat_conf from the structured reason string so that
    strong-signal findings (PASSWORD 0.90, JWT 0.85) are promoted to HIGH by
    cat_conf alone without requiring ml_confidence >= 0.85.
    """
    from Harpocrates.core.classification import CategoryInference, ViolationCategory

    if finding.category:
        try:
            cat = ViolationCategory(finding.category)
        except ValueError:
            cat = ViolationCategory.GENERIC_SECRET
        # Try to recover the original cat_conf from the reason string.
        # Format written by _reason(): "... confidence=0.90"
        cat_conf = 0.30 if cat == ViolationCategory.GENERIC_SECRET else 0.75
        if finding.category_reason:
            m = _REASON_CONF_RE.search(finding.category_reason)
            if m:
                cat_conf = float(m.group(1))
    else:
        cat = ViolationCategory.GENERIC_SECRET
        cat_conf = 0.30

    reason = finding.category_reason or "layer=fallback matched=reconstructed confidence=0.00"
    return CategoryInference(category=cat, reason=reason, confidence=cat_conf)


def _apply_ml_verification(
    findings: List[Finding],
    full_content: str,
    verifier: "Verifier",
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """Filter entropy candidates through the ML verifier."""
    lines = full_content.splitlines()
    verified: List[Finding] = []
    for start in range(0, len(findings), _ML_VERIFICATION_BATCH_SIZE):
        candidates = findings[start : start + _ML_VERIFICATION_BATCH_SIZE]
        findings_with_context = [
            (finding, _prepare_ml_context_from_lines(finding, lines))
            for finding in candidates
        ]
        verified.extend(
            _apply_ml_verification_with_contexts(
                findings_with_context,
                verifier=verifier,
                ml_threshold=ml_threshold,
            )
        )
    return verified


def _prepare_ml_context_from_lines(finding: Finding, lines: Sequence[str]) -> "CodeContext":
    """Build verifier context from a file split once for a candidate batch."""
    from Harpocrates.ml.context import extract_context_from_lines
    from Harpocrates.ml.tokens import TokenMatch

    context = extract_context_from_lines(
        lines=lines,
        line_number=finding.line or 1,
        file_path=finding.file,
    )
    start, end = finding.token_start, finding.token_end
    # NFR-01: features scan the whole line and its neighbours; on minified files that is
    # ~100k chars per candidate. Keep a window around the token instead (training and
    # scanning both come through here, so features stay consistent).
    line = context.line_content
    if len(line) > 2 * _CONTEXT_WINDOW:
        at = line.find(finding.token) if finding.token else -1
        if start is None or line[start:end] != finding.token:
            start = at if at >= 0 else None
            end = start + len(finding.token) if start is not None else None
        center = start if start is not None else 0
        lo = max(0, center - _CONTEXT_WINDOW)
        hi = min(len(line), (end or center) + _CONTEXT_WINDOW)
        line = line[lo:hi]
        start, end = (start - lo, end - lo) if start is not None else (None, None)
    context = dataclasses.replace(
        context,
        line_content=line,
        lines_before=[ln[: 2 * _CONTEXT_WINDOW] for ln in context.lines_before],
        lines_after=[ln[: 2 * _CONTEXT_WINDOW] for ln in context.lines_after],
    )
    if finding.token is not None and start is not None and end is not None:
        context = dataclasses.replace(context, token_match=TokenMatch(
            token=finding.token,
            start=start,
            end=end,
            kind=("regex" if finding.evidence == EvidenceType.REGEX else "sensitive_assignment"),
        ))
    return context


def _apply_ml_verification_with_contexts(
    findings_with_context: List[Tuple[Finding, "CodeContext"]],
    verifier: "Verifier",
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """Verify pre-built contexts in bounded batches and preserve alignment."""
    if not findings_with_context:
        return []

    verified: List[Finding] = []
    for start in range(0, len(findings_with_context), _ML_VERIFICATION_BATCH_SIZE):
        batch = findings_with_context[start : start + _ML_VERIFICATION_BATCH_SIZE]
        try:
            results = verifier.verify_batch(batch)
        except Exception as exc:
            raise MLVerificationError(f"ML verification failed: {exc}") from exc

        if len(results) != len(batch):
            raise MLVerificationError(
                "ML verification failed: verifier returned "
                f"{len(results)} results for {len(batch)} candidates"
            )

        for (finding, _context), result in zip(batch, results):
            if result.is_secret and result.combined_confidence >= ml_threshold:
                inference = _inference_from_finding(finding)
                new_severity = _severity_from_classification(
                    inference, ml_confidence=result.combined_confidence
                )
                verified.append(
                    Finding(
                        type=finding.type,
                        file=finding.file,
                        line=finding.line,
                        snippet=finding.snippet,
                        entropy=finding.entropy,
                        evidence=EvidenceType.HYBRID,
                        severity=new_severity,
                        # FR-CORE-03: report calibrated P(secret) when the model ships a map.
                        confidence=(
                            calibrated
                            if (calibrated := getattr(result, "calibrated", None)) is not None
                            else result.combined_confidence
                        ),
                        token=finding.token,
                        token_start=finding.token_start,
                        token_end=finding.token_end,
                        in_comment=finding.in_comment,
                        category=finding.category,
                        category_reason=finding.category_reason,
                    )
                )
    return verified


def detect_text_with_ml(
    text: str,
    verifier: "Verifier",
    threshold: float = 4.0,
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """
    Detect secrets in text with ML verification for entropy candidates.

    Regex hits (CRITICAL/HIGH) are returned immediately — no ML call.
    Entropy candidates are forwarded to XGBoost for false-positive filtering.

    Args:
        text: Text content to scan
        verifier: ML verifier instance
        threshold: Entropy threshold (reserved)
        ml_threshold: Minimum ML confidence to keep an entropy finding

    Returns:
        List of findings
    """
    _ = threshold
    findings = _collect_text_findings(text)
    if not findings:
        return findings

    regex_findings = [f for f in findings if f.evidence == EvidenceType.REGEX]
    entropy_findings = [f for f in findings if f.evidence != EvidenceType.REGEX]

    if not entropy_findings:
        return regex_findings

    verified_entropy = _apply_ml_verification(
        findings=entropy_findings,
        full_content=text,
        verifier=verifier,
        ml_threshold=ml_threshold,
    )

    return regex_findings + verified_entropy


def detect_file_with_ml(
    path: str | Path,
    verifier: "Verifier",
    threshold: float = 4.0,
    max_bytes: Optional[int] = None,
    ml_threshold: float = 0.5,
) -> List[Finding]:
    """
    Detect secrets in a file with ML verification for entropy candidates.

    Regex hits (CRITICAL/HIGH) are returned immediately — no ML call.
    Entropy candidates are forwarded to XGBoost for false-positive filtering.

    Args:
        path: Path to the file
        verifier: ML verifier instance
        threshold: Entropy threshold (reserved)
        max_bytes: Maximum bytes to read
        ml_threshold: Minimum ML confidence to keep an entropy finding

    Returns:
        List of findings
    """
    _ = threshold
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    findings = _collect_file_findings(path_obj, max_bytes)
    if not findings:
        return findings

    regex_findings = [f for f in findings if f.evidence == EvidenceType.REGEX]
    entropy_findings = [f for f in findings if f.evidence != EvidenceType.REGEX]

    if not entropy_findings:
        return regex_findings

    try:
        if max_bytes:
            with open(path_obj, "r", encoding="utf-8", errors="ignore") as f:
                full_content = f.read(max_bytes)
        else:
            full_content = path_obj.read_text(encoding="utf-8", errors="ignore")
    except (OSError, IOError):
        return [f for f in findings if f.evidence != EvidenceType.ML]

    verified_entropy = _apply_ml_verification(
        findings=entropy_findings,
        full_content=full_content,
        verifier=verifier,
        ml_threshold=ml_threshold,
    )

    return regex_findings + verified_entropy


__all__ = [
    "detect_text",
    "detect_file",
    "detect_text_with_ml",
    "detect_file_with_ml",
    "MLVerificationError",
    "Finding",
]
