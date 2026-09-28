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

from Harpocrates.core.detector import detect_file, detect_text
from Harpocrates.core.result import Finding

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


@mcp.tool()
def scan_text(
    text: str,
    include_token: bool = False,
    include_contributions: bool = False,
) -> List[Dict[str, Any]]:
    """
    Scan a text blob for secrets using regex + entropy detection.

    Note: `include_token=False` redacts the `token` field only. The `snippet`
    field contains the raw source line and may still include the matched secret.
    Do not log or store findings in contexts where the snippet must be redacted.

    Each finding includes:
      - `category`: violation classification (e.g., "password", "api_token",
        "connection_string", "jwt", "private_key", "generic_secret"). Use this
        field to decide remediation (rotate password vs revoke token vs
        regenerate keypair).
      - `category_reason`: human-readable explanation of how the category
        was inferred. Surface this in agent-facing UIs.

    Args:
        text: The text content to scan.
        include_token: If True, include the raw matched token in each finding
                       (default False — token field is redacted).
        include_contributions: If True, each finding gains an `explanation` key
                               with TreeSHAP per-feature contributions. Loads
                               xgboost lazily — only pay the cost when needed.
                               Requires pip install harpocrates[ml].

    Returns:
        A list of finding dicts with type, severity, evidence, file, line,
        snippet, entropy, confidence, category, category_reason.
        Token is included only if include_token=True.
        Explanation is included only if include_contributions=True.
    """
    findings: List[Finding] = detect_text(text)
    if not include_contributions:
        return [f.to_json_dict(include_token=include_token) for f in findings]

    # Lazy import — xgboost never loaded on the default scan path.
    from Harpocrates.ml.context import extract_context_from_finding
    from Harpocrates.ml.explain import explain_finding

    result = []
    for f in findings:
        ctx = extract_context_from_finding(f, full_content=text)
        explanation = explain_finding(f, ctx)
        d = f.to_json_dict(include_token=include_token)
        d["explanation"] = explanation.to_dict() if explanation else None
        result.append(d)
    return result


@mcp.tool()
def scan_file(
    path: str,
    include_token: bool = False,
    max_bytes: Optional[int] = None,
    include_contributions: bool = False,
) -> List[Dict[str, Any]]:
    """
    Scan a file on disk for secrets using regex + entropy detection.

    Reads at most 10 MB regardless of `max_bytes` to prevent resource
    exhaustion. Refuses to open special files (FIFOs, sockets, devices).

    The harpocrates-mcp process has the same filesystem read scope as the
    user who launched it — symlinks are followed and any readable file may
    be scanned. Callers are responsible for passing safe paths.

    Note: `include_token=False` redacts the `token` field only. The `snippet`
    field contains the raw source line and may still include the matched secret.

    Each finding includes `category` and `category_reason` fields — see
    scan_text docstring for details.

    Args:
        path: Absolute path to the file to scan.
        include_token: If True, include the raw matched token (default False).
        max_bytes: Cap on bytes read. Capped at 10 MB regardless of this value.
        include_contributions: If True, each finding gains an `explanation` key
                               with TreeSHAP per-feature contributions. Loads
                               xgboost lazily. Requires pip install harpocrates[ml].

    Returns:
        List of finding dicts. Empty list if the path does not exist or is binary.
        Explanation is included only if include_contributions=True.
    """
    resolved = Path(path)
    _check_path(resolved)
    cap = min(max_bytes, _MAX_SCAN_BYTES) if max_bytes is not None else _MAX_SCAN_BYTES
    findings: List[Finding] = detect_file(resolved, max_bytes=cap)
    if not include_contributions:
        return [f.to_json_dict(include_token=include_token) for f in findings]

    # Lazy import — xgboost never loaded on the default scan path.
    from Harpocrates.ml.context import extract_context_from_finding
    from Harpocrates.ml.explain import explain_finding

    result = []
    for f in findings:
        ctx = extract_context_from_finding(f)
        explanation = explain_finding(f, ctx)
        d = f.to_json_dict(include_token=include_token)
        d["explanation"] = explanation.to_dict() if explanation else None
        result.append(d)
    return result


def main() -> None:
    """Entry point for the `harpocrates-mcp` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
