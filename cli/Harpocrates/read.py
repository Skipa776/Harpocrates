"""Read tool (FR-READ-01, FR-READ-02, SEC-02): file reads and greps with secrets replaced by
typed placeholders ``<<HARPO:type:hash4>>`` before anything reaches the agent.

Invariant: only redacted text leaves this module. The whole file is scanned before a line
range is cut (a key block can start above the range), and grep matches run over redacted
text, so a pattern can never be used to probe a secret's value.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from Harpocrates.core.classification import secret_type
from Harpocrates.core.detector import _apply_ml_verification, _collect_text_findings
from Harpocrates.core.result import EvidenceType, Finding

logger = logging.getLogger(__name__)

MAX_BYTES = 10 * 1024 * 1024  # larger files are refused, not truncated: a cut can split a secret
MAX_GREP_MATCHES = 200
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "target"}
# The regex tier only matches a key's header line; the block itself is what must not leak.
# An unterminated block is redacted to the end of the file.
_PEM_BLOCK = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", re.S)
# Entropy tokens on unquoted `NAME=value` lines include the name; redact only the value.
_NAME_PREFIX = re.compile(r"^[A-Za-z_][\w.\-]*\s*[=:]\s*(?=\S)")
_PLACEHOLDER = re.compile(r"<<HARPO:\w+:[0-9a-f]{4}>>")


@lru_cache(maxsize=1)
def _gate_verifier():
    """
    Description: Lazily create and cache the recall-first gate verifier for FR-CORE-06.
    Parameters: None
    Output: Shared OnnxVerifier instance; imports and initializes the model on first use and
        propagates initialization errors.
    """
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    return OnnxVerifier(layer="gate")  # recall-first threshold (FR-CORE-06)


SNIFF_BYTES = 8192


def text_encoding(head: bytes) -> Optional[str]:
    """
    Description: Identify UTF-16 or UTF-32 text from a byte order mark or NULs on one byte
        parity, as in ASCII-heavy UTF-16, so secrets are not bypassed by hooks.py or hidden
        from safe_read as binary data.
    Parameters:
        head: Initial bytes of a file to inspect for an encoding signature or NUL pattern.
    Output: UTF-16 or UTF-32 codec name when recognized; None for UTF-8 and real binary.
        Both parities mostly NUL (executables, zero fill) stay binary. Trade-off: binary with
        the same NUL pattern is decoded and scanned as text, which can only cost a false
        positive. Such encoded text contains NULs but reads as plain text after decoding.
    """
    if head[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return "utf-32"
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    even, odd = head[0::2], head[1::2]
    if not even or not odd:
        return None
    low, high = even.count(0) / len(even), odd.count(0) / len(odd)
    if high >= 0.5 and low <= 0.1:
        return "utf-16-le"
    if low >= 0.5 and high <= 0.1:
        return "utf-16-be"
    return None


def _read_text(path: Path) -> str:
    """
    Description: Read a regular text file within the size limit so special or binary files
        are refused before redaction.
    Parameters:
        path: File path to validate and read.
    Output: Decoded text with invalid sequences replaced; reads the file and raises ValueError
        for nonregular files, files over MAX_BYTES, or binary data in the sniffed prefix.
        Filesystem errors propagate as OSError.
    """
    if not path.is_file():  # also false for FIFOs, sockets and devices, which could block
        raise ValueError(f"not a regular file: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"file over {MAX_BYTES} bytes, not read: {path}")
    data = path.read_bytes()
    encoding = text_encoding(data[:SNIFF_BYTES])
    if encoding is None and b"\x00" in data[:SNIFF_BYTES]:
        raise ValueError(f"binary file, not read: {path}")
    return data.decode(encoding or "utf-8", errors="replace")


class Redactor:
    """
    Description: Represent one agent session's secret redactor, whose HMAC key lives only
        in memory and dies with the process.
    Parameters: None
    Output: Instances provide session-keyed placeholders, redacted file reads, and searches;
        construction generates a random in-memory key.
    """

    def __init__(self) -> None:
        """
        Description: Generate the in-memory HMAC key used for session-scoped placeholders (SEC-02).
        Parameters: None
        Output: None; stores 32 random bytes on the instance.
        """
        self._key = secrets.token_bytes(32)  # SEC-02

    def placeholder(self, value: str, stype: str) -> str:
        """
        Description: Build a typed, session-keyed placeholder to refer to a value without
            returning the value itself.
        Parameters:
            value: Value to hash with the session's HMAC key.
            stype: Secret type label to include in the placeholder.
        Output: A <<HARPO:type:hash4>> string containing the first four hexadecimal HMAC characters.
        """
        digest = hmac.new(self._key, value.encode(), hashlib.sha256).hexdigest()[:4]
        return f"<<HARPO:{stype}:{digest}>>"

    def _findings(self, text: str) -> List[Finding]:
        """
        Description: Collect regex and other candidates and verify non-regex candidates with
            the gate model, failing closed by retaining every candidate if verification fails.
        Parameters:
            text: Full text to inspect for secrets.
        Output: Regex findings plus verified candidates, or all findings on verification failure;
            may initialize the cached model and logs only the exception type when it fails.
            Errors from initial finding collection propagate.
        """
        findings = _collect_text_findings(text)
        regex = [f for f in findings if f.evidence == EvidenceType.REGEX]
        candidates = [f for f in findings if f.evidence != EvidenceType.REGEX]
        if not candidates:
            return regex
        try:
            return regex + _apply_ml_verification(candidates, text, _gate_verifier(), ml_threshold=0.0)
        except Exception as exc:  # fail closed: without the model, every candidate is redacted
            logger.warning("ML verifier unavailable, redacting all candidates: %s", type(exc).__name__)
            return findings

    def secret_values(self, text: str, findings: Optional[List[Finding]] = None) -> dict:
        """
        Description: List every secret value in a text with its coarse type, the one place that turns findings into
            values to replace (the read tool and the egress gate both use it).
        Parameters:
            text: Text whose secret values should be found.
            findings: Findings already collected for this text, or None to collect them with the recall-first rules of
                _findings. A caller that must fail on a verifier error passes its own.
        Output: {value: secret type}, longest values first. Whole private-key blocks (an unterminated block runs to the
            end of the text) are one value; an entropy token on a NAME=value line is cut down to the value (unless that
            would leave under 8 characters). Detection errors propagate.
        """
        spans = {m.group(0): "private_key" for m in _PEM_BLOCK.finditer(text)}
        for f in self._findings(text) if findings is None else findings:
            value = f.token or ""
            stripped = _NAME_PREFIX.sub("", value, count=1)
            if len(stripped) >= 8:  # never shrink a token like "abcd==" to "=", which would rewrite every "="
                value = stripped
            if value.strip():
                spans.setdefault(value, secret_type(f.category))
        return {value: spans[value] for value in sorted(spans, key=len, reverse=True)}  # a block swallows its own header

    def redact(self, text: str) -> str:
        """
        Description: Replace every detected secret everywhere it occurs while keeping the line
            count, including entire private-key blocks and unterminated blocks through end of text.
        Parameters:
            text: Text whose detected secret values should be replaced.
        Output: Redacted text with session-keyed placeholders; detection may initialize the model
            or log a verification failure, and initial detection errors propagate.
        """
        for value, stype in self.secret_values(text).items():
            ph = self.placeholder(value, stype)
            text = text.replace(value, "\n".join([ph] * (value.count("\n") + 1)))
        return text

    def safe_read(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """
        Description: Read and redact the whole file before selecting lines so secrets spanning
            the requested range remain protected.
        Parameters:
            path: Text file path to read.
            start_line: Optional 1-based inclusive first line, clamped to at least one.
            end_line: Optional 1-based inclusive last line, bounded by the file's line count.
        Output: Numbered lines like `cat -n`, with secrets redacted and a placeholder warning when
            needed; reads the file. Raises ValueError for nonregular, oversized, or binary files;
            filesystem and detection errors propagate, and verification failures may be logged.
        """
        lines = self.redact(_read_text(Path(path))).split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        start = max(start_line or 1, 1)
        end = min(end_line or len(lines), len(lines))
        body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(start, end + 1))
        return body + _footer(body)

    def safe_grep(self, pattern: str, path: str = ".", max_matches: int = MAX_GREP_MATCHES) -> str:
        """
        Description: Search a file or directory for regex matches on redacted lines to avoid
            reporting secret values, using a raw-text prefilter to skip files that cannot match.
        Parameters:
            pattern: Python regular expression to search for.
            path: File or directory to search recursively, defaulting to the current directory.
            max_matches: Match count at which to stop collecting results.
        Output: `path:line:text` for each matching redacted line, with a limit notice when reached
            and a placeholder warning when needed; reads files and skips unreadable, binary, or
            oversized files. Raises ValueError for invalid regexes; detection errors propagate and
            verification failures may be logged. Searching for "<<HARPO" itself finds nothing
            through the raw prefilter unless that pattern is already present in the source.
        """
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        root = Path(path)
        out: List[str] = []
        for file in _walk(root):
            try:
                text = _read_text(file)
            except (ValueError, OSError):
                continue
            # raw prefilter skips files that can't match; the match that is reported still
            # runs on redacted text. Cost: a search for "<<HARPO" itself finds nothing.
            if not rx.search(text):
                continue
            rel = file.relative_to(root) if file != root else file
            for n, line in enumerate(self.redact(text).split("\n"), start=1):
                if rx.search(line):
                    out.append(f"{rel}:{n}:{line}")
                    if len(out) >= max_matches:
                        body = "\n".join(out)
                        return body + f"\n[harpo: stopped at {max_matches} matches]" + _footer(body)
        body = "\n".join(out)
        return body + _footer(body)


def _walk(root: Path):
    """
    Description: Enumerate files in stable name order while pruning configured directories
        such as version-control, dependency, environment, and cache directories.
    Parameters:
        root: Directory to traverse, or a nondirectory path to yield directly.
    Output: Iterator of Path objects; traverses the filesystem without following directory
        symlinks and uses os.walk's default handling of traversal errors.
    """
    if not root.is_dir():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _footer(body: str) -> str:
    """
    Description: Tell the agent placeholders are not real values so it does not write them
        into code and instead references the variable or config key.
    Parameters:
        body: Redacted output to inspect for distinct placeholders.
    Output: Warning footer with the distinct placeholder count, or an empty string if none
        occur; no side effects.
    """
    found = sorted(set(_PLACEHOLDER.findall(body)))
    if not found:
        return ""
    return (f"\n[harpo: {len(found)} secret(s) redacted as <<HARPO:type:hash>>. "
            "They are not the real values; reference the variable or config key instead of copying them.]")
