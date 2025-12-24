from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

from Harpocrates.core.detector import detect_file, detect_file_with_ml
from Harpocrates.core.result import Finding, ScanResult

if TYPE_CHECKING:
    from Harpocrates.ml.verifier import Verifier

DEFAULT_IGNORE_PATTERNS = {
    # Version control
    ".git", ".svn", ".hg", ".bzr",
    # Dependencies
    "node_modules", "venv", ".venv", "env", "__pycache__",
    # Build artifacts
    "dist", "build", "*.egg-info", "target",
    # IDE
    ".idea", ".vscode", "*.swp", "*.swo",
    # Compiled
    "*.pyc", "*.pyo", "*.so", "*.dll", "*.dylib",
    # Archives
    "*.zip", "*.tar", "*.gz", "*.bz2",
    # Media
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.mp3",
    # Docs (binary formats)
    "*.pdf", "*.doc", "*.docx",
}

def _should_scan_file(path: Path, ignore_patterns: Set[str]) -> bool:
    """
    Determine if a file should be scanned.

    Args:
        path: File path to check
        ignore_patterns: Set of patterns to ignore

    Returns:
        True if file should be scanned
    """
    # Skip symlinks to prevent infinite loops
    if path.is_symlink():
        return False

    # Check parent directories against ignore patterns
    for parent in path.parents:
        if parent.name in ignore_patterns:
            return False

    # Check filename against ignore patterns
    if path.name in ignore_patterns:
        return False

    # Check extension patterns (e.g., "*.pyc")
    for pattern in ignore_patterns:
        if pattern.startswith("*") and path.name.endswith(pattern[1:]):
            return False

    return path.is_file()

def scan_directory(
    directory: str | Path,
    recursive: bool = True,
    max_file_size: int = 10 * 1024 * 1024,
    ignore_patterns: Optional[Set[str]] = None,
    verifier: Optional["Verifier"] = None,
    ml_threshold: float = 0.5,
) -> ScanResult:
    """
    Scan a directory for secrets.

    Args:
        directory: Path to directory to scan
        recursive: If True, scan subdirectories
        max_file_size: Maximum file size to scan (in bytes)
        ignore_patterns: Additional patterns to ignore (merged with defaults)
        verifier: Optional ML verifier for false positive filtering
        ml_threshold: ML confidence threshold when verifier is enabled

    Returns:
        ScanResult containing all findings

    Example:
        >>> result = scan_directory("./my_project")
        >>> print(f"Found {len(result.findings)} secrets")
        >>> for finding in result.findings:
        ...     print(f"{finding.file}:{finding.line} - {finding.type}")
    """
    start_time = time.time()
    dir_path = Path(directory)

    if not dir_path.exists():
        return ScanResult(
            findings = [],
            errors = [f'Not a directory: {directory}']
        )

    if not dir_path.is_dir():
        return ScanResult(
            findings=[],
            errors=[f'Not a directory: {directory}']
        )

    ignore = DEFAULT_IGNORE_PATTERNS.copy()
    if ignore_patterns:
        ignore.update(ignore_patterns)

    all_findings: List[Finding] = []
    scanned_files = 0
    total_lines = 0
    errors: List[str] = []

    # Get all files and sort for deterministic ordering
    if recursive:
        file_iter = sorted(dir_path.rglob("*"))
    else:
        file_iter = sorted(dir_path.glob("*"))

    for file_path in file_iter:
        if not _should_scan_file(file_path, ignore):
            continue

        try:
            if file_path.stat().st_size > max_file_size:
                errors.append(f"Skipped large file: {file_path}")
                continue
        except (OSError, PermissionError) as e:
            errors.append(f"Cannot access {file_path}: {e}")
            continue

        try:
            # Use ML verification if verifier is provided
            if verifier is not None:
                findings = detect_file_with_ml(
                    file_path,
                    verifier=verifier,
                    max_bytes=max_file_size,
                    ml_threshold=ml_threshold,
                )
            else:
                findings = detect_file(file_path, max_bytes=max_file_size)

            all_findings.extend(findings)
            scanned_files += 1
            if findings:
                max_line = max(f.line for f in findings if f.line)
                total_lines += max_line
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines += sum(1 for _ in f)
                except Exception:
                    pass
        except Exception as e:
            errors.append(f"Error scanning {file_path}: {e}")

    duration = (time.time() - start_time) * 1000  # ms

    return ScanResult(
        findings=all_findings,
        scanned_files=scanned_files,
        total_lines=total_lines,
        duration_ms=duration,
        errors=errors
    )

def scan_file(
    filepath: str | Path,
    max_file_size: int = 10 * 1024 * 1024,
    verifier: Optional["Verifier"] = None,
    ml_threshold: float = 0.5,
) -> ScanResult:
    """
    Scan a single file for secrets.

    Args:
        filepath: Path to file
        max_file_size: Maximum file size to scan
        verifier: Optional ML verifier for false positive filtering
        ml_threshold: ML confidence threshold when verifier is enabled

    Returns:
        ScanResult containing findings from this file

    Example:
        >>> result = scan_file("config.env")
        >>> if result.found_secrets:
        ...     print(f"⚠️  Found {len(result.findings)} secrets!")
    """
    start_time = time.time()
    file_path = Path(filepath)

    if not file_path.exists():
        return ScanResult(
            findings=[],
            errors=[f"File not found: {filepath}"]
        )

    try:
        # Use ML verification if verifier is provided
        if verifier is not None:
            findings = detect_file_with_ml(
                file_path,
                verifier=verifier,
                max_bytes=max_file_size,
                ml_threshold=ml_threshold,
            )
        else:
            findings = detect_file(file_path, max_bytes=max_file_size)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = 0

        duration = (time.time() - start_time) * 1000

        return ScanResult(
            findings=findings,
            scanned_files=1,
            total_lines=total_lines,
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        return ScanResult(
            findings=[],
            scanned_files=0,
            total_lines=0,
            duration_ms=duration,
            errors=[f"Error scanning file: {e}"]
        )


__all__ = ["scan_directory", "scan_file", "ScanResult"]
