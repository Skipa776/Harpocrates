"""
MCP server exposing Harpo scan APIs to other agents.

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
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from Harpocrates.audit import FindingsLog
from Harpocrates.core.classification import secret_type
from Harpocrates.core.detector import detect_file, detect_text
from Harpocrates.core.result import Finding
from Harpocrates.read import Redactor

# Defensive: bind logging to stderr BEFORE any import that might emit logs,
# so log lines never corrupt the JSON-RPC stdio frame stream.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = FastMCP("harpo")

# Hard ceiling on bytes read per scan_file call. Prevents resource exhaustion
# from large files or slow streams when the caller omits max_bytes.
_MAX_SCAN_BYTES: int = 10 * 1024 * 1024  # 10 MB


def _check_path(path: Path) -> None:
    """
    Description: Reject special files that could block indefinitely on open() or yield
        infinite data and exhaust resources.
    Parameters:
        path: Filesystem path to inspect, following symlinks.
    Output: None; stats the path and raises ValueError for named pipes (FIFOs), sockets, and
        device nodes. Regular files, symlinks, and nonexistent paths pass through; nonexistent
        paths return [] from detect_file. Stat errors, including inaccessible paths, are left
        for detect_file to handle safely.
    """
    try:
        st = path.stat()
    except OSError:
        return  # Non-existent or inaccessible; detect_file handles this safely.
    mode = st.st_mode
    if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        raise ValueError(f"Refusing to scan special file: {path}")


# One redactor per server process: the MCP session. Its HMAC key keeps placeholders stable
# within the session and differs across sessions (FR-READ-02, SEC-02). What safe_read and
# safe_grep redact goes to ~/.harpo/findings.jsonl, once per session and file (FR-LOG-01).
_redactor = Redactor(FindingsLog("read", uuid.uuid4().hex))


def _safe_finding(f: Finding) -> Dict[str, Any]:
    """
    Description: Create an agent-safe view of a finding (SEC-02) so agents can inspect
        detection metadata without receiving secret values.
    Parameters:
        f: Finding to convert to agent-safe fields.
    Output: Dictionary carrying type, severity, evidence, confidence, category, comment status,
        location, and a session-keyed placeholder; never the token, source snippet, entropy,
        free-text category_reason (it can quote the match), or any hash of the value other
        than the 4-hex HMAC placeholder. Trade-off: agents lose snippet context; they can call
        safe_read on the file instead.
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
    """
    Description: Add context-based explanations to agent-safe findings while omitting the
        token-derived finding_id from explanation metadata.
    Parameters:
        findings: Findings to convert and explain.
        text: Full source text for context extraction, or None to use each finding's context.
    Output: List of safe finding dictionaries with explanation dictionaries or None; lazily
        imports explanation support and propagates import, context, or explanation errors.
    """
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
    Description: Scan a text blob for secrets using regex + entropy detection so agents can
        choose remediation without receiving secret values.
    Parameters:
        text: Text content to scan.
        include_contributions: Whether to add an explanation key with TreeSHAP per-feature
            contributions; requires xgboost: pip install 'harpocrates[explain]'.
    Output: List of findings with type, severity, evidence, confidence, category (use it to
        choose remediation: rotate password vs revoke token vs regenerate keypair), in_comment,
        file, line, column, and placeholder (<<HARPO:type:hash4>>, stable within this session).
        Secret values are never returned. Optional explanations load explanation support;
        detection, import, and explanation errors propagate.
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
    Description: Scan a file on disk for secrets using regex + entropy detection, reading
        at most 10 MB regardless of max_bytes to bound resource use.
    Parameters:
        path: Absolute path to the file to scan.
        max_bytes: Optional cap on bytes read, capped at 10 MB regardless of this value.
        include_contributions: As in scan_text, add TreeSHAP per-feature explanations;
            requires xgboost: pip install 'harpocrates[explain]'.
    Output: List of finding dictionaries with the same fields as scan_text; secret values are
        never returned. Returns an empty list if the path does not exist or is binary. Reads
        with the filesystem scope of the user who launched the server and follows symlinks.
        Raises ValueError for special files (FIFOs, sockets, devices); detection, import, and
        explanation errors propagate. Optional explanations load explanation support.
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
    Description: Read a text file with every detected secret replaced by <<HARPO:type:hash4>>.
        Use this instead of the built-in file read.
    Parameters:
        path: Text file path to read.
        start_line: Optional 1-based inclusive first line of the range to return.
        end_line: Optional 1-based inclusive last line of the range to return.
    Output: Numbered lines like `cat -n`; the same secret always gets the same placeholder in
        this session. Placeholders are not the real values: never write them into files;
        reference the variable or config key instead. Reads and redacts the whole file before
        selecting the range. Raises ValueError for binary files, special files, and files over
        10 MB; filesystem and detection errors propagate. May load the verifier and log its
        failure, failing closed by redacting all candidates if verification fails.
    """
    return _redactor.safe_read(path, start_line, end_line)


@mcp.tool()
def safe_grep(pattern: str, path: str = ".", max_matches: int = 200) -> str:
    """
    Description: Search files for a Python regex and return `file:line:text` matches with
        secrets redacted; matching runs on redacted text, so searching for a secret's value
        finds nothing.
    Parameters:
        pattern: Python regular expression to search for.
        path: File or directory to search recursively, skipping .git, node_modules and
            virtualenvs; defaults to the current directory.
        max_matches: Match count at which to stop collecting results, defaulting to 200.
    Output: Redacted matches with a limit notice when reached and a placeholder warning when
        needed; reads files and skips unreadable, binary, or oversized files. Raises ValueError
        for invalid regexes; detection errors propagate. May load the verifier and log its
        failure, failing closed by redacting all candidates if verification fails.
    """
    return _redactor.safe_grep(pattern, path, max_matches)


def main() -> None:
    """
    Description: Run the stdio MCP server as the entry point for the `harpo-mcp`
        console script so agents can call its tools.
    Parameters: None
    Output: None; runs the server over standard input and output and propagates server errors.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
