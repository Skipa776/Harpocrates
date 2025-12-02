from __future__ import annotations
import fnmatch

from typing import Iterable, List, Optional
from pathlib import Path

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
    """
    Read ignore patterns from a text file, returning non-empty, non-comment lines.
    
    Parameters:
        path (Path): Path to the ignore file to read.
    
    Returns:
        list[str]: List of ignore pattern strings (each line trimmed). Returns an empty list if the file cannot be read.
    """
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
    Determine whether a single ignore pattern matches a POSIX-style path relative to the repository root.
    
    The pattern follows minimal gitignore-like semantics:
    - A leading '/' anchors the pattern to the repository root.
    - A trailing '/' denotes a directory pattern and matches the directory and any descendant paths.
    - Patterns without a leading '/' may match any path segment (they are treated as unanchored).
    
    Parameters:
        rel_str (str): Path relative to the repository root, using POSIX separators (e.g., 'src/app/file.py').
        is_dir (bool): True when the target path is a directory.
        patterns (str): The single ignore pattern to test, possibly anchored ('/pat'), a directory pattern ('pat/'), or a negated form is handled elsewhere.
    
    Returns:
        bool: `True` if the pattern matches the target path (taking directory semantics into account), `False` otherwise.
    """
    anchored = patterns.startswith("/")
    dir_pat = patterns.endswith("/")
    pat_core = patterns.rstrip("/")
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
    Load ordered ignore patterns from repository ignore files.
    
    Parameters:
        root (Path): Repository root; patterns are evaluated relative to this path.
        respect_gitignore (bool): If True and a `.gitignore` exists at `root`, include its patterns.
        extra_ignore_file (Optional[Path]): Optional alternate ignore file; if provided and exists its
            patterns are appended. If not provided or not found, `.harpocratesignore` at `root` is used
            when present.
    
    Returns:
        list[str]: Flat, ordered list of ignore pattern strings. Negated patterns (those starting with
        '!') are preserved and later entries override earlier ones.
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
    Decide whether a filesystem path should be included for scanning relative to a repository root.
    
    Evaluation applies repository-relative normalization, optional symlink following, default directory and file-extension exclusions, and ordered ignore patterns with gitignore-like semantics (anchoring with leading '/', directory patterns with trailing '/', and negation with leading '!').
    
    Parameters:
        path (Path): Path to test.
        root (Path): Repository root used to compute a POSIX-style relative path for pattern matching.
        ignores (list[str]): Ordered ignore patterns; later entries override earlier ones.
        default_skip_exts (bool): If True, skip files with extensions listed in DEFAULT_SKIP_EXTS.
        follow_symlinks (bool): If False, symlinks are skipped.
    
    Returns:
        bool: `True` if the path should be scanned, `False` otherwise.
    """
    
    root_resolved = root.resolve()
    try:
        rel = path.resolve().relative_to(root_resolved)
    except Exception:
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