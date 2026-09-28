from __future__ import annotations

import fnmatch
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Set

from Harpocrates.core.detector import detect_file, detect_file_with_ml
from Harpocrates.core.result import Finding, ScanResult
from Harpocrates.utils.file_utils import iter_text_lines

if TYPE_CHECKING:
    from Harpocrates.core.rust_backend import RustScannerBackend
    from Harpocrates.ml.verifier import Verifier

ScanEngine = Literal["auto", "python", "rust"]

DEFAULT_IGNORE_PATTERNS = {
    # Version control
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    # Dependencies
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    # Tool and agent caches
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    # Build artifacts
    "dist",
    "build",
    "*.egg-info",
    "target",
    # IDE
    ".idea",
    ".vscode",
    "*.swp",
    "*.swo",
    # Compiled
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.dylib",
    # Archives
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bz2",
    # Media
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.mp4",
    "*.mp3",
    # Docs (binary formats)
    "*.pdf",
    "*.doc",
    "*.docx",
    # Tier 1: minified / transpiled bundles — high token density, zero secrets.
    "*.min.js",
    "*.min.css",
    "*.min.map",
    "*.bundle.js",
    "*.bundle.css",
    # Tier 1: package lockfiles — deterministic hashes, not secrets.
    "package-lock.json",
    "yarn.lock",
    "Pipfile.lock",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    # Tier 1: vendored third-party code.
    "vendor",
    # Tier 3: source map files — generated build artifacts, never contain secrets.
    "*.css.map",
    "*.js.map",
    # Tier 3: SAML/SP metadata XML — contain X.509 cert bodies, not credentials.
    "*-metadata*.xml",
    "*sp.xml",
    "*idp.xml",
}


def _should_scan_file(path: Path, ignore_patterns: Set[str]) -> bool:
    """
    Determine if a file should be scanned.

    Patterns without wildcards are matched exactly against any path component
    (file name or directory name). Glob patterns are matched with fnmatch.
    """
    if path.is_symlink():
        return False

    glob_patterns = {
        pattern
        for pattern in ignore_patterns
        if any(character in pattern for character in "*?[")
    }
    exact_patterns = ignore_patterns - glob_patterns

    # Exact match against any component (file name or ancestor directory name)
    path_names = {path.name} | {p.name for p in path.parents}
    if path_names & exact_patterns:
        return False

    # Glob match against file and ancestor directory components, mirroring
    # the native walker's early-pruning contract.
    for pattern in glob_patterns:
        if any(fnmatch.fnmatch(name, pattern) for name in path_names):
            return False

    return path.is_file()


def scan_directory(
    directory: str | Path,
    recursive: bool = True,
    max_file_size: int = 10 * 1024 * 1024,
    ignore_patterns: Optional[Set[str]] = None,
    verifier: Optional["Verifier"] = None,
    ml_threshold: float = 0.5,
    engine: ScanEngine = "auto",
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
        engine: ``auto`` prefers a built Rust scanner, ``rust`` requires it,
            and ``python`` uses the original detector

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
        return ScanResult(findings=[], errors=[f"Not a directory: {directory}"])

    if not dir_path.is_dir():
        return ScanResult(findings=[], errors=[f"Not a directory: {directory}"])

    ignore = DEFAULT_IGNORE_PATTERNS.copy()
    if ignore_patterns:
        ignore.update(ignore_patterns)

    try:
        native_backend = _resolve_native_backend(engine)
    except (RuntimeError, ValueError) as exc:
        return ScanResult(
            findings=[],
            duration_ms=(time.time() - start_time) * 1000,
            errors=[str(exc)],
        )
    all_findings: List[Finding] = []
    scanned_files = 0
    total_lines = 0
    errors: List[str] = []
    eligible_files: list[Path] = []

    if native_backend is not None:
        try:
            if verifier is None:
                batch = native_backend.scan_directory(
                    dir_path,
                    recursive=recursive,
                    max_file_size=max_file_size,
                    ignore_patterns=ignore,
                )
            else:
                batch = native_backend.scan_directory_with_ml(
                    dir_path,
                    verifier=verifier,
                    recursive=recursive,
                    max_file_size=max_file_size,
                    ignore_patterns=ignore,
                    ml_threshold=ml_threshold,
                )
        except Exception as exc:
            if engine == "rust":
                errors.append(f"Rust directory scan failed: {exc}")
                return ScanResult(
                    findings=[],
                    scanned_files=0,
                    total_lines=0,
                    duration_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                )
            errors.append(f"Rust directory scan failed; used Python fallback: {exc}")
            native_backend = None
        else:
            all_findings.extend(batch.findings)
            scanned_files += batch.scanned_files
            total_lines += batch.total_lines
            errors.extend(batch.errors)
            return ScanResult(
                findings=all_findings,
                scanned_files=scanned_files,
                total_lines=total_lines,
                duration_ms=(time.time() - start_time) * 1000,
                errors=errors,
            )

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

        eligible_files.append(file_path)

    for file_path in eligible_files:
        try:
            # Native directory scans return above; this loop is the Python
            # engine or auto-mode fallback after a native failure.
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
            total_lines += sum(
                1 for _ in iter_text_lines(file_path, max_bytes=max_file_size)
            )
        except Exception as e:
            errors.append(f"Error scanning {file_path}: {e}")

    duration = (time.time() - start_time) * 1000  # ms

    return ScanResult(
        findings=all_findings,
        scanned_files=scanned_files,
        total_lines=total_lines,
        duration_ms=duration,
        errors=errors,
    )


def scan_file(
    filepath: str | Path,
    max_file_size: int = 10 * 1024 * 1024,
    verifier: Optional["Verifier"] = None,
    ml_threshold: float = 0.5,
    engine: ScanEngine = "auto",
) -> ScanResult:
    """
    Scan a single file for secrets.

    Args:
        filepath: Path to file
        max_file_size: Maximum file size to scan
        verifier: Optional ML verifier for false positive filtering
        ml_threshold: ML confidence threshold when verifier is enabled
        engine: ``auto`` prefers a built Rust scanner, ``rust`` requires it,
            and ``python`` uses the original detector

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
        return ScanResult(findings=[], errors=[f"File not found: {filepath}"])

    try:
        native_backend = _resolve_native_backend(engine)
    except Exception as exc:
        return ScanResult(
            findings=[],
            scanned_files=0,
            total_lines=0,
            duration_ms=(time.time() - start_time) * 1000,
            errors=[f"Error scanning file: {exc}"],
        )

    errors: List[str] = []
    if native_backend is not None:
        try:
            if verifier is not None:
                findings = native_backend.scan_file_with_ml(
                    file_path,
                    verifier=verifier,
                    max_bytes=max_file_size,
                    ml_threshold=ml_threshold,
                )
            else:
                findings = native_backend.scan_file(file_path, max_bytes=max_file_size)
        except Exception as exc:
            if engine == "rust":
                return ScanResult(
                    findings=[],
                    scanned_files=0,
                    total_lines=0,
                    duration_ms=(time.time() - start_time) * 1000,
                    errors=[f"Rust file scan failed: {exc}"],
                )
            errors.append(f"Rust file scan failed; used Python fallback: {exc}")
            native_backend = None

    if native_backend is None:
        try:
            if verifier is not None:
                findings = detect_file_with_ml(
                    file_path,
                    verifier=verifier,
                    max_bytes=max_file_size,
                    ml_threshold=ml_threshold,
                )
            else:
                findings = detect_file(file_path, max_bytes=max_file_size)
        except Exception as exc:
            return ScanResult(
                findings=[],
                scanned_files=0,
                total_lines=0,
                duration_ms=(time.time() - start_time) * 1000,
                errors=[*errors, f"Error scanning file: {exc}"],
            )

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = 0

    duration = (time.time() - start_time) * 1000

    return ScanResult(
        findings=findings,
        scanned_files=1,
        total_lines=total_lines,
        duration_ms=duration,
        errors=errors,
    )


def _resolve_native_backend(engine: ScanEngine) -> Optional["RustScannerBackend"]:
    """Return a native backend for auto/rust, or None for Python."""
    if engine not in {"auto", "python", "rust"}:
        raise ValueError(f"Unknown scan engine: {engine}")
    if engine == "python":
        return None

    from Harpocrates.core.rust_backend import RustScannerBackend

    return RustScannerBackend.discover(required=engine == "rust")


__all__ = ["scan_directory", "scan_file", "ScanEngine", "ScanResult"]
