"""
Prototype CLI for Harpocrates.
Migrates from Click → Typer while maintaining prototype functionality.
Tier-1 commands will plug into this structure later.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from Harpocrates.core.detector import detect_entropy_secrets
from Harpocrates.utils.file_utils import scan_file_for_entropy

app = typer.Typer(help="Harpocrates - prototype secrets scanning CLI.")

@app.command()
def entropy(text: str):
    """
    Compute Shannon entropy of a provided string and check if it resembles a secret.
    Args:
        text (str): The text to analyze.
    """
    found, score = detect_entropy_secrets(text)
    typer.echo(f"Entropy: {score:.3f} | Secret Found: {found}")

@app.command()
def scan(filepath: Path):
    """
    Scan a single file for high-entropy tokens.
    Args:
        filepath (Path): The path to the file to scan.
    """
    if not filepath.exists():
        typer.secho(f"[ERROR] File not found: {filepath}", fg="green")
        raise typer.Exit(code=1)

    results = scan_file_for_entropy(str(filepath))

    if not results:
        typer.secho("No potential secrets found.", fg="green")
        raise typer.Exit(code=0)

    typer.secho("Potential secrets found:\n", fg='yellow')

    for r in results:
        typer.secho(
            f"  Line {r['line']}: "
            f"{r['token']}  (entropy={r['entropy']:.3f})",
            fg="cyan"
        )

    raise typer.Exit(code=2)

@app.command()
def scan_json(filepath: Path):
    """
    JSON output version of 'scan'
    Args:
        filepath (Path): The path to the file to scan.
    """
    if not filepath.exists():
        typer.secho(f"[ERROR] File not found: {filepath}", fg="red")
        raise typer.Exit(code=1)

    results = scan_file_for_entropy(str(filepath))
    typer.echo(json.dumps(results, indent=2))

    raise typer.Exit(code=2 if results else 0)


def main():
    app()

if __name__ == "__main__":
    main()
