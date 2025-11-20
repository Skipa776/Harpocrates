from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from Harpocrates.scanner.entropy import shannon_entropy, looks_like_secret
from Harpocrates.scanner.regex_signatures import SIGNATURES
from Harpocrates.utils.file_loader import iter_text_lines

Finding = Dict[str, object]

_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-.]{8,}")

def _make_finding(
    type_: str,
    file: Optional[str],
    line: Optional[int],
    snippet: str,
    entropy: Optional[float],
    evidence: str,
) -> Finding:
    return {
        "type": type_,
        "file": file,
        "line": line,
        "snippet": snippet,
        "entropy": entropy,
        "evidence": evidence,
    }
    
def _scan_line(line: str, lineno: int, file: Optional[str]) -> List[Finding]:
    """Scan a single line for regex signatures and entropy-based canditates"""
    findings: List[Finding] = []
    stripped = line.strip()
    
    for sig_name, pattern in SIGNATURES.items():
        for match in pattern.finditer(stripped):
            token = match.group()
            entropy_val = shannon_entropy(token) if token else 0.0
            findings.append(
                _make_finding(
                    type_=sig_name,
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=entropy_val,
                    evidence="regex",
                ),
            )

    for token in _TOKEN_RE.findall(stripped):
        if looks_like_secret(token):
            findings.append(
                _make_finding(
                    type_="ENTROPY_CANDIDATE",
                    file=file,
                    line=lineno,
                    snippet=stripped[:200],
                    entropy=shannon_entropy(token),
                    evidence="entropy",
                )
            )
    return findings

def detect_text(
    text: str,
    threshold: float = 4.0,
) -> List[Finding]:
    """
    Dectect secrets in a text blob and return a list of uniform findings.
    
    Returns findings of the form:
    {
      "type": str,
      "file": str | None,
      "line": int | None,
      "snippet": str,
      "entropy": float | None,
      "evidence": "regex" | "entropy"
    }
    """
    _ = threshold
    findings: List[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        findings.extend(_scan_line(line, lineno, file = None))
    return findings

def detect_file(
    path: str | Path,
    threshold: float = 4.0,
    max_bytes: Optional[int] = None,
) -> List[Finding]:
    """
    Detects secrets in a file on disk using safe text iteration (binary-aware).
    
    Args:
        path (str | Path): path to scan
        threshold (float, optional): reserved for future tuning. Defaults to 4.0.
        max_bytes (Optional[int], optional): maximum number of bytes to read from the file. Defaults to None.

    Returns:
        List[Finding]: list of findings detected in the file
    """
    _ = threshold
    path_obj = Path(path)
    file_name = str(path_obj)
    
    findings: List[Finding] = []
    for lineno, line in iter_text_lines(path_obj, max_bytes=max_bytes):
        findings.extend(_scan_line(line, lineno, file=file_name))
    return findings