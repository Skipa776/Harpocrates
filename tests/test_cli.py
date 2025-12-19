"""Tests for the CLI module."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from Harpocrates.cli import app

runner = CliRunner()


def test_cli_scan_file_with_secrets(tmp_path: Path) -> None:
    """Test CLI scanning a file with secrets."""
    # Create file with secret
    file_path = tmp_path / "secrets.env"
    file_path.write_text("GITHUB_TOKEN=ghp_" + "a" * 36 + "\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(file_path)])

    # Exit code 1 = findings detected
    assert result.exit_code == 1
    assert "GITHUB_TOKEN" in result.stdout or "Found" in result.stdout


def test_cli_scan_file_without_secrets(tmp_path: Path) -> None:
    """Test CLI scanning a file without secrets."""
    file_path = tmp_path / "config.txt"
    file_path.write_text("APP_NAME=Test\nVERSION=1.0.0\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(file_path)])

    # Exit code 0 = no findings
    assert result.exit_code == 0
    assert "No secrets detected" in result.stdout


def test_cli_scan_nonexistent_path() -> None:
    """Test CLI with nonexistent path."""
    result = runner.invoke(app, ["scan", "/nonexistent/path/file.txt"])

    # Exit code 2 = error
    assert result.exit_code == 2
    assert "not found" in result.stdout.lower() or "not found" in (result.stderr or "").lower()


def test_cli_scan_json_output(tmp_path: Path) -> None:
    """Test CLI JSON output format."""
    # Create file with secret
    file_path = tmp_path / "secrets.env"
    file_path.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(file_path), "--json"])

    # Exit code 1 = findings
    assert result.exit_code == 1

    # Output should be valid JSON
    data = json.loads(result.stdout)
    assert "findings" in data
    assert "scanned_files" in data
    assert len(data["findings"]) >= 1


def test_cli_scan_json_output_no_secrets(tmp_path: Path) -> None:
    """Test CLI JSON output with no secrets."""
    file_path = tmp_path / "config.txt"
    file_path.write_text("APP_NAME=Test\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(file_path), "--json"])

    # Exit code 0 = no findings
    assert result.exit_code == 0

    data = json.loads(result.stdout)
    assert data["findings"] == []


def test_cli_scan_json_excludes_token(tmp_path: Path) -> None:
    """Test that JSON output doesn't include raw tokens."""
    file_path = tmp_path / "secrets.env"
    file_path.write_text("GITHUB_TOKEN=ghp_" + "a" * 36 + "\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(file_path), "--json"])

    data = json.loads(result.stdout)

    # Token field should not be present in any finding
    for finding in data["findings"]:
        assert "token" not in finding


def test_cli_scan_directory(tmp_path: Path) -> None:
    """Test CLI scanning a directory."""
    # Create file with secret
    file_path = tmp_path / "secrets.env"
    file_path.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 1  # Findings detected


def test_cli_scan_with_ignore(tmp_path: Path) -> None:
    """Test CLI with ignore pattern."""
    # Create file with secret
    file_path = tmp_path / "secrets.env"
    file_path.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(tmp_path), "--ignore", "*.env"])

    assert result.exit_code == 0  # No findings (file was ignored)


def test_cli_version() -> None:
    """Test version command."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "version" in result.stdout.lower()
