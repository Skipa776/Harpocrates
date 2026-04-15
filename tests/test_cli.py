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


# ---------------------------------------------------------------------------
# --fail-on severity gate (PRD-01 Task 2)
# ---------------------------------------------------------------------------


def _write_critical_secret(tmp_path: Path, filename: str = "secrets.env") -> Path:
    """Write a file containing an AWS key (detected as CRITICAL severity)."""
    file_path = tmp_path / filename
    file_path.write_text("AWS_KEY=AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")
    return file_path


def test_cli_scan_fail_on_default_medium_exits_one(tmp_path: Path) -> None:
    """Default --fail-on medium: critical finding triggers exit 1."""
    file_path = _write_critical_secret(tmp_path)

    result = runner.invoke(app, ["scan", str(file_path)])

    assert result.exit_code == 1


def test_cli_scan_fail_on_critical_exits_one_for_critical(tmp_path: Path) -> None:
    """--fail-on critical: critical finding still triggers exit 1."""
    file_path = _write_critical_secret(tmp_path)

    result = runner.invoke(app, ["scan", str(file_path), "--fail-on", "critical"])

    assert result.exit_code == 1


def test_cli_scan_fail_on_none_exits_zero_with_findings(tmp_path: Path) -> None:
    """--fail-on none: findings are reported but exit code stays 0."""
    file_path = _write_critical_secret(tmp_path)

    result = runner.invoke(app, ["scan", str(file_path), "--fail-on", "none"])

    assert result.exit_code == 0


def test_cli_scan_fail_on_none_json_exits_zero(tmp_path: Path) -> None:
    """--fail-on none also applies to JSON output."""
    file_path = _write_critical_secret(tmp_path)

    result = runner.invoke(
        app, ["scan", str(file_path), "--json", "--fail-on", "none"]
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data["findings"]) >= 1


def test_cli_scan_fail_on_invalid_value_exits_two(tmp_path: Path) -> None:
    """Invalid --fail-on value returns exit 2 (argument error)."""
    file_path = _write_critical_secret(tmp_path)

    result = runner.invoke(app, ["scan", str(file_path), "--fail-on", "bogus"])

    assert result.exit_code == 2


def test_cli_scan_fail_on_case_insensitive(tmp_path: Path) -> None:
    """--fail-on accepts uppercase severity names (case-insensitive)."""
    file_path = _write_critical_secret(tmp_path)

    result = runner.invoke(app, ["scan", str(file_path), "--fail-on", "HIGH"])

    assert result.exit_code == 1


def test_cli_scan_fail_on_no_findings_exits_zero(tmp_path: Path) -> None:
    """No findings: exit code is 0 regardless of --fail-on value."""
    file_path = tmp_path / "clean.txt"
    file_path.write_text("APP_NAME=Test\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(file_path), "--fail-on", "info"])

    assert result.exit_code == 0


def test_cli_scan_fail_on_help_documents_flag() -> None:
    """--help output mentions --fail-on and lists accepted values."""
    result = runner.invoke(app, ["scan", "--help"])

    assert result.exit_code == 0
    assert "--fail-on" in result.stdout
