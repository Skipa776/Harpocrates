"""
Command-line interface for Harpocrates secrets detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, cast

import typer
from rich.console import Console
from rich.table import Table

from Harpocrates.core.result import ScanResult, Severity
from Harpocrates.core.scanner import ScanEngine, scan_directory, scan_file, scan_git_history

app = typer.Typer(
    name="harpocrates",
    help="Harpocrates - Secrets detection for code repositories",
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)

# Severity ranking (higher value = more severe). Used by --fail-on to
# determine which findings should cause a non-zero exit code.
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

# String tokens accepted by --fail-on. "none" disables the gate entirely
# (findings are reported but never cause a non-zero exit code).
_FAIL_ON_NONE = "none"
_VALID_FAIL_ON_LEVELS = (
    _FAIL_ON_NONE,
    Severity.INFO.value,
    Severity.LOW.value,
    Severity.MEDIUM.value,
    Severity.HIGH.value,
    Severity.CRITICAL.value,
)
_VALID_ENGINES = ("auto", "python", "rust")


def _resolve_fail_on(raw: str) -> Optional[Severity]:
    """Parse --fail-on into a minimum Severity, or None to disable the gate."""
    normalized = raw.strip().lower()
    if normalized == _FAIL_ON_NONE:
        return None
    try:
        return Severity(normalized)
    except ValueError as exc:
        valid = ", ".join(_VALID_FAIL_ON_LEVELS)
        raise typer.BadParameter(
            f"Invalid --fail-on value {raw!r}. Must be one of: {valid}"
        ) from exc


def _fail_on_callback(value: str) -> str:
    """Validate --fail-on at argument-parse time so bad input fails fast."""
    _resolve_fail_on(value)
    return value


def _engine_callback(value: str) -> str:
    """Validate the selected scan engine at argument-parse time."""
    normalized = value.strip().lower()
    if normalized not in _VALID_ENGINES:
        valid = ", ".join(_VALID_ENGINES)
        raise typer.BadParameter(f"Invalid --engine value {value!r}. Must be one of: {valid}")
    return normalized


def _should_fail(findings, min_severity: Optional[Severity]) -> bool:
    """Return True if any finding meets or exceeds the minimum severity."""
    if min_severity is None:
        return False
    threshold = _SEVERITY_RANK[min_severity]
    return any(_SEVERITY_RANK[f.severity] >= threshold for f in findings)


@app.command()
def scan(
    paths: Optional[List[Path]] = typer.Argument(default=None, help="Files or directories to scan"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r",
        help="Scan directories recursively"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    max_file_size: int = typer.Option(10, "--max-size", help="Max file size in MB"),
    engine: str = typer.Option(
        "auto",
        "--engine",
        help="Scanning engine: auto, python, or rust. Auto prefers a built Rust binary.",
        callback=_engine_callback,
    ),
    ignore: Optional[str] = typer.Option(
        None, "--ignore",
        help="Comma-separated patterns to ignore"
    ),
    ml_threshold: float = typer.Option(
        0.19, "--ml-threshold",
        help=(
            "Extra floor on combined confidence (0.0-1.0, default: 0.19). "
            "The model's own commit threshold in model_config.json decides "
            "first; at the default this floor never removes a finding."
        ),
    ),
    show_secrets: bool = typer.Option(
        False, "--show-secrets",
        help="Display full secret tokens instead of redacted versions"
    ),
    fail_on: str = typer.Option(
        Severity.MEDIUM.value, "--fail-on",
        help=(
            "Minimum severity that causes a non-zero exit code. One of: "
            "critical, high, medium, low, info, none. Default: medium."
        ),
        callback=_fail_on_callback,
    ),
    history: bool = typer.Option(
        False, "--history", help="Scan lines added in every git commit instead of files on disk"
    ),
    explain: bool = typer.Option(
        False, "--explain",
        help=(
            "Emit JSON with per-feature ML contribution scores (TreeSHAP). "
            "Loads xgboost lazily — cost paid only when set."
        ),
    ),
) -> None:
    """
    Scan a file or directory for secrets.

    By default, detected secret tokens are redacted in both table and JSON
    output. Pass --show-secrets to display full tokens (use with caution).

    Examples:

        # Scan a single file
        harpocrates scan config.env

        # Scan a directory recursively
        harpocrates scan ./my_project

        # Output as JSON
        harpocrates scan ./my_project --json

        # Ignore specific patterns
        harpocrates scan ./my_project --ignore "*.test.js,test_*"

        # Fewer, higher-confidence findings
        harpocrates scan ./my_project --ml-threshold 0.7

        # Require the native regex/entropy scanner; ML still runs in Python
        harpocrates scan ./my_project --engine rust

        # Display full token values (NOT recommended outside local debugging)
        harpocrates scan config.env --show-secrets

        # Scan every commit in git history (finds secrets deleted from disk)
        harpocrates scan . --history

        # Only fail CI on high or critical findings
        harpocrates scan ./my_project --fail-on high

        # Report findings but never return a non-zero exit code
        harpocrates scan ./my_project --fail-on none

        # XAI: emit JSON with per-feature SHAP contributions
        harpocrates scan ./my_project --explain | jq '.findings[0].explanation.top_positive'
    """
    # Zero-file early return — pre-commit passes no files when nothing is staged.
    if not paths:
        raise typer.Exit(code=0)

    if not (0.0 <= ml_threshold <= 1.0):
        error_console.print(
            f"[red]✗[/red] --ml-threshold must be between 0.0 and 1.0, got: {ml_threshold}"
        )
        raise typer.Exit(code=2)

    if engine == "rust":
        from Harpocrates.core.rust_backend import RustScannerBackend, RustScannerError

        try:
            RustScannerBackend.discover(required=True)
        except RustScannerError as exc:
            error_console.print(f"[red]✗[/red] {exc}")
            raise typer.Exit(code=2) from exc

    # Already validated by _fail_on_callback at parse time; safe to call.
    fail_on_severity = _resolve_fail_on(fail_on)

    ignore_patterns = set(ignore.split(",")) if ignore else set()

    max_bytes = max_file_size * 1024 * 1024

    # The shipped model always runs. Fail closed: without it, regex and entropy alone would pass
    # secrets the model is there to catch.
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    try:
        verifier = OnnxVerifier(layer="commit", lazy_load=False)
    except Exception as e:
        error_console.print(f"[red]✗[/red] ML model could not be loaded: {e}")
        raise typer.Exit(code=2) from e

    # Scan all paths, aggregating findings across files and directories.
    all_findings = []
    total_files = 0
    total_lines = 0
    total_duration = 0.0
    all_errors: list = []

    for path in paths:
        if not path.exists():
            error_console.print(f"[yellow]⚠[/yellow] Path not found, skipping: {path}")
            continue

        if history:
            try:
                r = scan_git_history(path)
            except ValueError as e:
                error_console.print(f"[red]✗[/red] {e}")
                raise typer.Exit(code=2) from e
        elif path.is_dir():
            r = scan_directory(
                path,
                recursive=recursive,
                max_file_size=max_bytes,
                ignore_patterns=ignore_patterns,
                verifier=verifier,
                ml_threshold=ml_threshold,
                engine=cast(ScanEngine, engine),
            )
        else:
            try:
                r = scan_file(
                    path,
                    max_file_size=max_bytes,
                    verifier=verifier,
                    ml_threshold=ml_threshold,
                    engine=cast(ScanEngine, engine),
                )
            except UnicodeDecodeError:
                error_console.print(
                    f"[yellow]⚠[/yellow] Skipping binary/unreadable file: {path}"
                )
                continue
            except (OSError, PermissionError) as e:
                error_console.print(
                    f"[yellow]⚠[/yellow] Skipping unreadable file {path}: {e}"
                )
                continue

        all_findings.extend(r.findings)
        total_files += r.scanned_files
        total_lines += r.total_lines
        total_duration += r.duration_ms
        if r.errors:
            all_errors.extend(r.errors)

    result = ScanResult(
        findings=all_findings,
        scanned_files=total_files,
        total_lines=total_lines,
        duration_ms=total_duration,
        errors=all_errors,
    )

    # Handle errors
    if result.errors:
        for error in result.errors:
            error_console.print(f"[yellow]⚠[/yellow]  {error}")
        fatal_errors = [
            error
            for error in result.errors
            if "; used Python fallback:" not in error
        ]
        if fatal_errors:
            raise typer.Exit(code=2)

    exit_on_findings = (
        1 if _should_fail(result.findings, fail_on_severity) else 0
    )

    if explain:
        # Lazy import — xgboost is never loaded on the default scan path.
        try:
            from Harpocrates.ml.context import extract_context_from_finding
            from Harpocrates.ml.explain import explain_finding
        except ImportError:
            error_console.print("[red]✗[/red] --explain requires xgboost: pip install xgboost")
            raise typer.Exit(code=2)

        payload = []
        for f in result.findings:
            ctx = extract_context_from_finding(f)
            explanation = explain_finding(f, ctx)
            payload.append({
                "finding": f.to_json_dict(include_token=show_secrets),
                "explanation": explanation.to_dict() if explanation else None,
            })
        print(json.dumps({"findings": payload}, indent=2))
        raise typer.Exit(code=exit_on_findings)

    if json_output:
        print(json.dumps(result.to_dict(include_token=show_secrets), indent=2))
        raise typer.Exit(code=exit_on_findings)

    if not result.found_secrets:
        console.print("[green]✓[/green] No secrets detected")
        console.print(
            f"Scanned {result.scanned_files} files ({result.total_lines} lines) "
            f"in {result.duration_ms:.0f}ms"
        )
        raise typer.Exit(code=0)

    table = Table(
        title=f"Found {len(result.findings)} potential secrets",
        show_header=True
    )
    table.add_column("Severity", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Location")
    table.add_column("Secret")

    for finding in sorted(
        result.findings,
        key=lambda f: (f.severity.value, f.file or "", f.line or 0)
    ):
        # Color severity
        severity_colors = {
            Severity.CRITICAL: "red bold",
            Severity.HIGH: "red",
            Severity.MEDIUM: "yellow",
            Severity.LOW: "blue",
            Severity.INFO: "white",
        }
        severity_style = severity_colors.get(finding.severity, "white")
        severity_text = f"[{severity_style}]{finding.severity.value.upper()}[/{severity_style}]"

        # Format location
        location = f"{finding.file}:{finding.line}" if finding.file else "text input"

        # Redacted by default; full token only when --show-secrets is set
        if show_secrets:
            secret_cell = finding.token or finding.snippet or "—"
        else:
            secret_cell = finding.redacted_token or "—"

        table.add_row(
            severity_text,
            finding.type,
            location,
            secret_cell,
        )

    console.print(table)
    console.print(
        f"\nScanned {result.scanned_files} files ({result.total_lines} lines) "
        f"in {result.duration_ms:.0f}ms"
    )
    console.print(
        f"[yellow]⚠[/yellow]  Found {result.critical_count} critical, "
        f"{result.high_count} high severity secrets"
    )

    raise typer.Exit(code=exit_on_findings)


@app.command()
def version() -> None:
    """Display version information."""
    from importlib.metadata import version as _version
    console.print(f"Harpocrates version {_version('harpocrates')}")


@app.command()
def setup(
    harness: Optional[str] = typer.Argument(
        None, help="claude-code or codex: print its read-tool setup. Omit to list detected harnesses."
    ),
) -> None:
    """Show each detected agent harness's support tier, or print one harness's setup."""
    from Harpocrates.harness import detect, setup_instructions, tier_report

    if harness is None:
        typer.echo(tier_report(detect()))
        return
    try:
        typer.echo(setup_instructions(harness))  # plain echo: rich would eat [toml] tables as markup
    except ValueError as exc:
        typer.echo(f"✗ {exc}", err=True)  # plain: the message echoes user input
        raise typer.Exit(code=2) from exc


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
