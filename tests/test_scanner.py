"""Tests for the scanner module."""
from __future__ import annotations

from pathlib import Path

from Harpocrates.core.scanner import scan_directory, scan_file


def test_scan_file_with_secrets(tmp_path: Path) -> None:
    """Test scanning a file with secrets."""
    content = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
    file_path = tmp_path / "secrets.env"
    file_path.write_text(content, encoding="utf-8")

    result = scan_file(file_path)

    assert result.found_secrets
    assert result.scanned_files == 1
    assert len(result.findings) >= 1
    assert any(f.type == "AWS_ACCESS_KEY_ID" for f in result.findings)


def test_scan_file_without_secrets(tmp_path: Path) -> None:
    """Test scanning a file without secrets."""
    content = "APP_NAME=MyApp\nVERSION=1.0.0\n"
    file_path = tmp_path / "config.env"
    file_path.write_text(content, encoding="utf-8")

    result = scan_file(file_path)

    assert not result.found_secrets
    assert result.scanned_files == 1
    assert len(result.findings) == 0


def test_scan_file_nonexistent(tmp_path: Path) -> None:
    """Test scanning a nonexistent file."""
    file_path = tmp_path / "does_not_exist.txt"

    result = scan_file(file_path)

    assert not result.found_secrets
    assert result.scanned_files == 0
    assert result.errors
    assert "not found" in result.errors[0].lower()


def test_scan_directory_with_secrets(tmp_path: Path) -> None:
    """Test scanning a directory with secrets."""
    # Create a file with secrets
    secrets_file = tmp_path / "secrets.env"
    secrets_file.write_text("GITHUB_TOKEN=ghp_" + "a" * 36 + "\n", encoding="utf-8")

    # Create a safe file
    safe_file = tmp_path / "config.txt"
    safe_file.write_text("APP_NAME=Test\n", encoding="utf-8")

    result = scan_directory(tmp_path)

    assert result.found_secrets
    assert result.scanned_files >= 1
    assert any(f.type == "GITHUB_TOKEN" for f in result.findings)


def test_scan_directory_recursive(tmp_path: Path) -> None:
    """Test recursive directory scanning."""
    # Create nested directory structure
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Create secret in subdirectory
    secrets_file = subdir / "secrets.env"
    secrets_file.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")

    # Scan recursively
    result = scan_directory(tmp_path, recursive=True)
    assert result.found_secrets

    # Scan non-recursively
    result_no_recurse = scan_directory(tmp_path, recursive=False)
    assert not result_no_recurse.found_secrets


def test_scan_directory_ignore_patterns(tmp_path: Path) -> None:
    """Test that ignore patterns work."""
    # Create file in node_modules (should be ignored by default)
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    secrets_file = node_modules / "secrets.env"
    secrets_file.write_text("GITHUB_TOKEN=ghp_" + "a" * 36 + "\n", encoding="utf-8")

    result = scan_directory(tmp_path)

    # File in node_modules should be ignored
    assert not result.found_secrets


def test_scan_directory_custom_ignore(tmp_path: Path) -> None:
    """Test custom ignore patterns."""
    # Create a .env file
    env_file = tmp_path / "secrets.env"
    env_file.write_text("GITHUB_TOKEN=ghp_" + "a" * 36 + "\n", encoding="utf-8")

    # Scan with custom ignore pattern
    result = scan_directory(tmp_path, ignore_patterns={"*.env"})

    # The .env file should be ignored
    assert not result.found_secrets


def test_scan_directory_nonexistent(tmp_path: Path) -> None:
    """Test scanning a nonexistent directory."""
    nonexistent = tmp_path / "nonexistent"

    result = scan_directory(nonexistent)

    assert not result.found_secrets
    assert result.errors


def test_scan_directory_deterministic_order(tmp_path: Path) -> None:
    """Test that findings are in deterministic order."""
    # Create multiple files with secrets
    for i in range(5):
        file_path = tmp_path / f"file_{i}.txt"
        file_path.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")

    # Scan multiple times and verify order is consistent
    result1 = scan_directory(tmp_path)
    result2 = scan_directory(tmp_path)

    files1 = [f.file for f in result1.findings]
    files2 = [f.file for f in result2.findings]

    assert files1 == files2


def test_scan_directory_symlink_skipped(tmp_path: Path) -> None:
    """Test that symlinks are skipped."""
    # Create a real file with secrets
    real_file = tmp_path / "real_secrets.txt"
    real_file.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")

    # Create a symlink to it
    symlink = tmp_path / "link_to_secrets.txt"
    try:
        symlink.symlink_to(real_file)
    except OSError:
        # Skip test on systems that don't support symlinks
        return

    result = scan_directory(tmp_path)

    # Should only find the secret once (in real file, not symlink)
    assert len(result.findings) == 1


def test_scan_result_summary(tmp_path: Path) -> None:
    """Test ScanResult summary properties."""
    # Create files with different severity secrets
    aws_file = tmp_path / "aws.txt"
    aws_file.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")

    result = scan_directory(tmp_path)

    assert result.found_secrets
    assert result.critical_count >= 0
    assert result.high_count >= 0
    # Duration should be set
    assert result.duration_ms >= 0
