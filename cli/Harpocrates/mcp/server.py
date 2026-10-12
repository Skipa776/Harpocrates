"""
MCP server exposing Harpocrates scan APIs to other agents.

Stdio transport: ALL logging MUST go to stderr to avoid corrupting
JSON-RPC frames. We defensively bind logging to stderr at startup
(first call wins — must happen before any library import that might log).

ML-verified variants (detect_text_with_ml, detect_file_with_ml) are not
exposed in v1 — they require a Verifier instance with cold-start cost
(~50ms ONNX load). MCP tools should be cheap and stateless. Add
scan_text_ml / scan_file_ml in a future version using a module-level
lazy-loaded verifier.
"""
from __future__ import annotations

import logging
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from Harpocrates.core.classification import secret_type
from Harpocrates.core.detector import detect_file, detect_text
from Harpocrates.core.result import Finding
from Harpocrates.read import Redactor

# Defensive: bind logging to stderr BEFORE any import that might emit logs,
# so log lines never corrupt the JSON-RPC stdio frame stream.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = FastMCP("harpocrates")

# Hard ceiling on bytes read per scan_file call. Prevents resource exhaustion
# from large files or slow streams when the caller omits max_bytes.
_MAX_SCAN_BYTES: int = 10 * 1024 * 1024  # 10 MB


def _check_path(path: Path) -> None:
    """Raise ValueError for special files that could block or exhaust resources.

    Named pipes (FIFOs), sockets, and device nodes can block indefinitely on
    open() or yield infinite data. Regular files, symlinks, and non-existent
    paths are allowed through (non-existent returns [] from detect_file).
    """
    try:
        st = path.stat()
    except OSError:
        return  # Non-existent or inaccessible; detect_file handles this safely.
    mode = st.st_mode
    if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        raise ValueError(f"Refusing to scan special file: {path}")


# One redactor per server process: the MCP session. Its HMAC key keeps placeholders stable
# within the session and differs across sessions (FR-READ-02, SEC-02).
_redactor = Redactor()


def _safe_finding(f: Finding) -> Dict[str, Any]:
    """Agent-safe view of a finding (SEC-02).

    Carries type, severity, confidence, location and a session-keyed placeholder, never the
    token, the source snippet, entropy, the free-text category_reason (it can quote the match),
    or any hash of the value other than the 4-hex HMAC placeholder.
    Trade-off: agents lose snippet context; they can call safe_read on the file instead.
    """
    return {
        "type": f.type,
        "severity": f.severity.value,
        "evidence": f.evidence.value,
        "confidence": f.confidence,
        "category": f.category,
        "in_comment": f.in_comment,
        "file": f.file,
        "line": f.line,
        "column": f.token_start + 1 if f.token_start is not None else None,
        "placeholder": _redactor.placeholder(f.token or "", secret_type(f.category)),
    }


def _with_explanations(findings: List[Finding], text: Optional[str]) -> List[Dict[str, Any]]:
    # Lazy import: xgboost never loaded on the default scan path.
    from Harpocrates.ml.context import extract_context_from_finding
    from Harpocrates.ml.explain import explain_finding

    result = []
    for f in findings:
        ctx = (extract_context_from_finding(f, full_content=text) if text is not None
               else extract_context_from_finding(f))
        explanation = explain_finding(f, ctx)
        d = _safe_finding(f)
        if explanation:
            ed = explanation.to_dict()
            ed.pop("finding_id", None)  # derived from the token; not needed by agents
            d["explanation"] = ed
        else:
            d["explanation"] = None
        result.append(d)
    return result


@mcp.tool()
def scan_text(text: str, include_contributions: bool = False) -> List[Dict[str, Any]]:
    """
    Scan a text blob for secrets using regex + entropy detection.

    Secret values are never returned. Each finding has: type, severity, evidence,
    confidence, category (use it to choose remediation: rotate password vs revoke token vs
    regenerate keypair), in_comment, file, line, column and placeholder
    (<<HARPO:type:hash4>>, stable within this session).

    Args:
        text: The text content to scan.
        include_contributions: If True, each finding gains an `explanation` key
                               with TreeSHAP per-feature contributions. Requires
                               xgboost: pip install 'harpocrates[train]'.
    """
    findings: List[Finding] = detect_text(text)
    if not include_contributions:
        return [_safe_finding(f) for f in findings]
    return _with_explanations(findings, text)


@mcp.tool()
def scan_file(
    path: str,
    max_bytes: Optional[int] = None,
    include_contributions: bool = False,
) -> List[Dict[str, Any]]:
    """
    Scan a file on disk for secrets using regex + entropy detection.

    Reads at most 10 MB regardless of `max_bytes`. Refuses special files (FIFOs, sockets,
    devices). Runs with the filesystem scope of the user who launched the server; symlinks
    are followed.

    Secret values are never returned; findings have the same fields as scan_text.

    Args:
        path: Absolute path to the file to scan.
        max_bytes: Cap on bytes read. Capped at 10 MB regardless of this value.
        include_contributions: As in scan_text.

    Returns:
        List of finding dicts. Empty list if the path does not exist or is binary.
    """
    resolved = Path(path)
    _check_path(resolved)
    cap = min(max_bytes, _MAX_SCAN_BYTES) if max_bytes is not None else _MAX_SCAN_BYTES
    findings: List[Finding] = detect_file(resolved, max_bytes=cap)
    if not include_contributions:
        return [_safe_finding(f) for f in findings]
    return _with_explanations(findings, None)


@mcp.tool()
def safe_read(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Read a text file with every detected secret replaced by <<HARPO:type:hash4>>.

    Use this instead of the built-in file read. Returns numbered lines like `cat -n`;
    start_line and end_line (1-based, inclusive) select a range. The same secret always
    gets the same placeholder in this session. Placeholders are not the real values:
    never write them into files; reference the variable or config key instead.
    Refuses binary files, special files, and files over 10 MB.
    """
    return _redactor.safe_read(path, start_line, end_line)


@mcp.tool()
def safe_grep(pattern: str, path: str = ".", max_matches: int = 200) -> str:
    """
    Search files for a Python regex and return `file:line:text` matches with secrets redacted.

    path is a file or directory (searched recursively, skipping .git, node_modules and
    virtualenvs). Matching runs on redacted text, so searching for a secret's value
    finds nothing.
    """
    return _redactor.safe_grep(pattern, path, max_matches)


def main() -> None:
    """Entry point for the `harpocrates-mcp` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
