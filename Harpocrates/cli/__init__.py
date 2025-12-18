"""
Command-line interface for Harpocrates secrets detection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from Harpocrates.core.result import Severity
from Harpocrates.core.scanner import scan_directory, scan_file

app = typer.Typer(
    name="harpocrates",
    help="🔒 Harpocrates - ML-powered secrets detection for code repositories",
    add_completion=False,
)
console = Console()


@app.command()
def scan(
    path: Path = typer.Argument(..., help="File or directory to scan"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", 
                                   help="Scan directories recursively"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    max_file_size: int = typer.Option(10, "--max-size", help="Max file size in MB"),
    ignore: Optional[str] = typer.Option(None, "--ignore", 
                                         help="Comma-separated patterns to ignore"),
) -> None:
    """
    Scan a file or directory for secrets.

    Examples:

        # Scan a single file
        harpocrates scan config.env

        # Scan a directory recursively
        harpocrates scan ./my_project

        # Output as JSON
        harpocrates scan ./my_project --json

        # Ignore specific patterns
        harpocrates scan ./my_project --ignore "*.test.js,test_*"
    """
    if not path.exists():
        console.print(f"[red]✗[/red] Path not found: {path}")
        raise typer.Exit(code=1)

    ignore_patterns = set(ignore.split(",")) if ignore else set()

    max_bytes = max_file_size * 1024 * 1024

    # Perform scan
    if path.is_file():
        result = scan_file(path, max_file_size=max_bytes)
    else:
        result = scan_directory(
            path, 
            recursive=recursive,
            max_file_size=max_bytes,
            ignore_patterns=ignore_patterns
        )

    # Handle errors
    if result.errors:
        for error in result.errors:
            console.print(f"[yellow]⚠[/yellow]  {error}", file=sys.stderr)

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
        raise typer.Exit(code=2 if result.found_secrets else 0)

    if not result.found_secrets:
        console.print("[green]✓[/green] No secrets detected")
        console.print(f"Scanned {result.scanned_files} files ({result.total_lines} lines) "
                     f"in {result.duration_ms:.0f}ms")
        raise typer.Exit(code=0)

    table = Table(title=f"🔍 Found {len(result.findings)} potential secrets", 
                  show_header=True)
    table.add_column("Severity", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Location")
    table.add_column("Evidence")

    for finding in sorted(result.findings, 
                         key=lambda f: (f.severity.value, f.file or "", f.line or 0)):
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

        table.add_row(
            severity_text,
            finding.type,
            location,
            finding.evidence.value,
        )

    console.print(table)
    console.print(f"\nScanned {result.scanned_files} files ({result.total_lines} lines) "
                 f"in {result.duration_ms:.0f}ms")
    console.print(f"[yellow]⚠[/yellow]  Found {result.critical_count} critical, "
                 f"{result.high_count} high severity secrets")

    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Display version information."""
    from Harpocrates import __version__
    console.print(f"Harpocrates version {__version__}")


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()