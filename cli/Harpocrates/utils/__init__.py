"""
Harpocrates utilities package.

Provides file handling and path filtering utilities.
"""
from __future__ import annotations

from Harpocrates.utils.file_utils import iter_text_lines
from Harpocrates.utils.path_filters import (
    DEFAULT_SKIP_DIRS,
    DEFAULT_SKIP_EXTS,
    load_ignore_patterns,
    should_scan_path,
)
from Harpocrates.utils.redaction import (
    redact_finding,
    redact_findings,
    redact_token,
)

__all__ = [
    "iter_text_lines",
    "DEFAULT_SKIP_DIRS",
    "DEFAULT_SKIP_EXTS",
    "load_ignore_patterns",
    "should_scan_path",
    "redact_finding",
    "redact_findings",
    "redact_token",
]
