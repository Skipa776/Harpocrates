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

__all__ = [
    "iter_text_lines",
    "DEFAULT_SKIP_DIRS",
    "DEFAULT_SKIP_EXTS",
    "load_ignore_patterns",
    "should_scan_path",
]
