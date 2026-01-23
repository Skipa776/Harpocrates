from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List, Optional

DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build"
}

DEFAULT_SKIP_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".rar",
    ".7z",
    ".class",
    ".jar",
    ".war",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".pyc"
}

DEFAULT_IGNORE_FILE = ".harpocratesignore"
GITIGNORE_FILE = ".gitignore"

def _read_ignore_file(path: Path) -> list[str]:
    out: List[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    except OSError:
        return []
    return out

def _pattern_matches(rel_str: str, is_dir: bool, patterns: str) -> bool:
    """
    Minimal gitignore-like matching against a relative posix path string.

    Semantics:
      - Leading '/' anchors to repo root.
      - Trailing '/' indicates a directory pattern; matches the directory and anything under it.
      - Otherwise, pattern is treated as unanchored and may match any path segment.
      - We use fnmatch with two tries to emulate unanchored behavior:
          1) direct match on rel_str
          2) '**/' + pattern (so 'dist/' can match nested 'pkg/dist/')
    """
    anchored = patterns.startswith("/")
    pat_core = patterns.lstrip("/").rstrip("/")
    target = rel_str if not is_dir else rel_str + "/"

    if anchored:
        return fnmatch.fnmatchcase(target, pat_core)
    else:
        if fnmatch.fnmatchcase(target, pat_core):
            return True
        implied = f"**/{pat_core}"
        return fnmatch.fnmatchcase(target, implied)

def load_ignore_patterns(
    root: Path,
    respect_gitignore: bool,
    extra_ignore_file: Optional[Path] = None
) -> list[str]:
    """
    Load ordered ignore patterns from .gitignore and/or a Harpocrates ignore file.

    - root: repository root (all matches evaluated against paths relative to this root)
    - respect_gitignore: if True and .gitignore exists at root, include its patterns
    - extra_ignore_file: if provided and exists, include; otherwise try '.harpocratesignore' at root

    Returns a flat, ordered list of patterns. Negations ('!pattern') are preserved; later
    entries take precedence during evaluation.
    """
    patterns: List[str] = []

    if respect_gitignore:
        gi = root / GITIGNORE_FILE
        if gi.exists():
            patterns.extend(_read_ignore_file(gi))
    if extra_ignore_file is not None and extra_ignore_file.exists():
        patterns.extend(_read_ignore_file(extra_ignore_file))
    else:
        hp = root / DEFAULT_IGNORE_FILE
        if hp.exists():
            patterns.extend(_read_ignore_file(hp))
    return patterns

def should_scan_path(
    path: Path,
    *,
    root: Path,
    ignores:list[str],
    default_skip_exts:bool = True,
    follow_symlinks:bool = False
) -> bool:
    """
    Decide whether 'path' should be scanned.

    Rules (in order):
      1) Normalize to a path relative to 'root'. If not under root: skip.
      2) Skip symlinks unless follow_symlinks=True (dir symlink loops are common).
      3) Skip if any ancestor directory is in DEFAULT_SKIP_DIRS.
      4) If file and default_skip_exts=True, skip if extension in DEFAULT_SKIP_EXTS.
      5) Apply ordered ignore patterns (gitignore-ish semantics with negation).
      6) Default: include.

    Notes:
      - Patterns are evaluated against posix-style relative paths (e.g., 'src/app/file.py').
      - A leading '/' in a pattern anchors it to root (e.g., '/secrets.txt').
      - A trailing '/' denotes a directory pattern (e.g., 'logs/').
      - Negations '!pattern' re-include matches; later lines win.
    """

    root_resolved = root.resolve()
    try:
        rel = path.resolve().relative_to(root_resolved)
    except ValueError:
        return False

    if rel==Path('.'):
        return True
    try:
        if path.is_symlink() and not follow_symlinks:
            return False
    except OSError:
        return False

    if any(part in DEFAULT_SKIP_DIRS for part in rel.parts if part):
        return False

    try:
        is_dir = path.is_dir()
    except OSError:
        return False

    if not is_dir and default_skip_exts:
        if path.suffix.lower() in DEFAULT_SKIP_EXTS:
            return False

    rel_str = rel.as_posix()
    decision: Optional[bool] = None

    for pat in ignores:
        if not pat or pat.startswith('#'):
            continue

        negated = pat.startswith('!')
        raw = pat[1:] if negated else pat
        if _pattern_matches(rel_str, is_dir, raw):
            decision = True if negated else False
    if decision is not None:
        return decision

    return True
