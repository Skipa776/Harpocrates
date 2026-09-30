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
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    return OnnxVerifier(layer="gate")  # recall-first threshold (FR-CORE-06)


def _read_text(path: Path) -> str:
    if not path.is_file():  # also false for FIFOs, sockets and devices, which could block
        raise ValueError(f"not a regular file: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"file over {MAX_BYTES} bytes, not read: {path}")
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ValueError(f"binary file, not read: {path}")
    return data.decode("utf-8", errors="replace")


class Redactor:
    """One per agent session: the HMAC key lives only in memory and dies with the process."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)  # SEC-02

    def placeholder(self, value: str, stype: str) -> str:
        digest = hmac.new(self._key, value.encode(), hashlib.sha256).hexdigest()[:4]
        return f"<<HARPO:{stype}:{digest}>>"

    def _findings(self, text: str) -> List[Finding]:
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

    def redact(self, text: str) -> str:
        """Replace every detected secret, everywhere it occurs, keeping the line count."""
        spans = {m.group(0): "private_key" for m in _PEM_BLOCK.finditer(text)}
        for f in self._findings(text):
            value = f.token or ""
            stripped = _NAME_PREFIX.sub("", value, count=1)
            if len(stripped) >= 8:  # never shrink a token like "abcd==" to "=", which would rewrite every "="
                value = stripped
            if value.strip():
                spans.setdefault(value, secret_type(f.category))
        for value in sorted(spans, key=len, reverse=True):  # longest first: a block swallows its own header
            ph = self.placeholder(value, spans[value])
            text = text.replace(value, "\n".join([ph] * (value.count("\n") + 1)))
        return text

    def safe_read(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Numbered lines (like `cat -n`) of a file, secrets redacted; optional 1-based inclusive range."""
        lines = self.redact(_read_text(Path(path))).split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        start = max(start_line or 1, 1)
        end = min(end_line or len(lines), len(lines))
        body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(start, end + 1))
        return body + _footer(body)

    def safe_grep(self, pattern: str, path: str = ".", max_matches: int = MAX_GREP_MATCHES) -> str:
        """`path:line:text` for each redacted line matching the regex, under a file or directory."""
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
            # ponytail: raw prefilter skips files that can't match; the match that is reported still
            # runs on redacted text. Cost: a search for "<<HARPO" itself finds nothing.
            if not rx.search(text):
                continue
            rel = file.relative_to(root) if file != root else file
            for n, line in enumerate(self.redact(text).split("\n"), start=1):
                if rx.search(line):
                    out.append(f"{rel}:{n}:{line}")
                    if len(out) >= max_matches:
                        body = "\n".join(out)
                        return body + f"\n[harpocrates: stopped at {max_matches} matches]" + _footer(body)
        body = "\n".join(out)
        return body + _footer(body)


def _walk(root: Path):
    if not root.is_dir():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _footer(body: str) -> str:
    """Tell the agent placeholders are not values, so it doesn't write them into code."""
    found = sorted(set(_PLACEHOLDER.findall(body)))
    if not found:
        return ""
    return (f"\n[harpocrates: {len(found)} secret(s) redacted as <<HARPO:type:hash>>. "
            "They are not the real values; reference the variable or config key instead of copying them.]")
