"""Tests for the CLI module."""
from __future__ import annotations

import json
import re
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
    """Non-existent paths produce a warning and are skipped (exit 0, no crash)."""
    result = runner.invoke(app, ["scan", "/nonexistent/path/file.txt"])

    # Multi-file mode: warn-and-continue, exit 0 when nothing was found
    assert result.exit_code == 0
    assert "not found" in result.output.lower()


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


# --- --show-secrets flag ----------------------------------------------------


def _write_github_token(tmp_path: Path) -> Path:
    file_path = tmp_path / "secrets.env"
    token = "ghp_" + "a" * 36
    file_path.write_text(f"GITHUB_TOKEN={token}\n", encoding="utf-8")
    return file_path


_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def test_cli_scan_show_secrets_flag_in_help() -> None:
    """--show-secrets should be documented in scan --help."""
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "--show-secrets" in _ANSI_RE.sub("", result.stdout)


def test_cli_scan_table_redacts_by_default(tmp_path: Path) -> None:
    """Default table output must not include the full token."""
    file_path = _write_github_token(tmp_path)
    full_token = "ghp_" + "a" * 36

    result = runner.invoke(
        app, ["scan", str(file_path)], env={"COLUMNS": "200"}
    )

    assert result.exit_code == 1
    assert full_token not in result.stdout
    # Redacted format keeps first 4 and last 4 chars with '...' between
    assert "ghp_" in result.stdout
    assert "..." in result.stdout


def test_cli_scan_table_shows_full_token_with_flag(tmp_path: Path) -> None:
    """--show-secrets must surface the full token in the table."""
    file_path = _write_github_token(tmp_path)
    full_token = "ghp_" + "a" * 36

    result = runner.invoke(
        app,
        ["scan", str(file_path), "--show-secrets"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 1
    assert full_token in result.stdout


def test_cli_scan_json_omits_token_by_default(tmp_path: Path) -> None:
    """JSON output must not include a raw token field by default.

    Note: the snippet field may still contain the raw token — snippet-level
    redaction is handled by the reusable redaction helper (Task 4).
    """
    file_path = _write_github_token(tmp_path)

    result = runner.invoke(app, ["scan", str(file_path), "--json"])

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["findings"], "expected at least one finding"
    for finding in data["findings"]:
        assert "token" not in finding


def test_cli_scan_json_includes_token_with_flag(tmp_path: Path) -> None:
    """--show-secrets + --json must include the full token field."""
    file_path = _write_github_token(tmp_path)
    full_token = "ghp_" + "a" * 36

    result = runner.invoke(
        app, ["scan", str(file_path), "--json", "--show-secrets"]
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["findings"], "expected at least one finding"
    tokens = [f.get("token") for f in data["findings"]]
    assert full_token in tokens


def test_cli_scan_show_secrets_no_findings(tmp_path: Path) -> None:
    """--show-secrets is a no-op when no findings exist."""
    file_path = tmp_path / "config.txt"
    file_path.write_text("APP_NAME=Test\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(file_path), "--show-secrets"])

    assert result.exit_code == 0
    assert "No secrets detected" in result.stdout


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
    assert "--fail-on" in _ANSI_RE.sub("", result.stdout)


def test_cli_scan_fail_on_below_threshold_exits_zero(
    tmp_path: Path, monkeypatch
) -> None:
    """Findings below the --fail-on threshold do NOT trigger exit 1.

    Guards against the comparison collapsing to '==' instead of '>=' — a LOW
    finding with --fail-on high must still produce exit 0.
    """
    from Harpocrates.core.result import EvidenceType, Finding, ScanResult, Severity

    file_path = tmp_path / "low_severity.env"
    file_path.write_text("APP_NAME=Test\n", encoding="utf-8")

    low_finding = Finding(
        type="ENTROPY_CANDIDATE",
        snippet="APP_NAME=Test",
        evidence=EvidenceType.ENTROPY,
        severity=Severity.LOW,
        file=str(file_path),
        line=1,
    )
    fake_result = ScanResult(
        findings=[low_finding], scanned_files=1, total_lines=1, duration_ms=0.1
    )

    monkeypatch.setattr(
        "Harpocrates.cli.scan_file", lambda *args, **kwargs: fake_result
    )

    result = runner.invoke(app, ["scan", str(file_path), "--fail-on", "high"])

    assert result.exit_code == 0, (
        f"LOW finding with --fail-on high should exit 0, got {result.exit_code}"
    )


def test_cli_scan_ml_threshold_default_is_0_19() -> None:
    """PRD-01 Task 9: --ml-threshold default must align with tuned model (0.19)."""
    import inspect

    from Harpocrates.cli import scan

    sig = inspect.signature(scan)
    default = sig.parameters["ml_threshold"].default
    # Typer wraps defaults in OptionInfo objects; unwrap when present.
    resolved = getattr(default, "default", default)
    assert resolved == 0.19


def test_cli_scan_ml_threshold_default_documented_in_help() -> None:
    """--help should surface the 0.19 default so operators see it."""
    result = runner.invoke(app, ["scan", "--help"])

    assert result.exit_code == 0
    assert "0.19" in result.stdout


def test_scan_multiple_files(tmp_path: Path) -> None:
    """harpocrates scan a.txt b.txt c.env processes all three files."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.env"
    a.write_text("APP_NAME=myapp\n", encoding="utf-8")
    b.write_text("VERSION=1.0.0\n", encoding="utf-8")
    c.write_text("DB_HOST=localhost\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(a), str(b), str(c)])

    # No secrets in any file — clean exit
    assert result.exit_code == 0
    assert "No secrets detected" in result.output


def test_scan_zero_files() -> None:
    """harpocrates scan with no arguments exits 0 (pre-commit no-staged-files case)."""
    result = runner.invoke(app, ["scan"])

    assert result.exit_code == 0
    # No error message, no usage error
    assert "error" not in result.output.lower()


def test_scan_skips_binary_file(tmp_path: Path) -> None:
    """Binary files are silently skipped; other files in the same call still process."""
    binary = tmp_path / "image.bin"
    binary.write_bytes(b"\x80\x81\x82\xff\xfe\xfd")

    clean = tmp_path / "clean.txt"
    clean.write_text("APP_NAME=myapp\n", encoding="utf-8")

    result = runner.invoke(app, ["scan", str(binary), str(clean)])

    # No crash; scanner silently skips binary via _looks_binary heuristic
    assert result.exit_code == 0
    assert "No secrets detected" in result.output
