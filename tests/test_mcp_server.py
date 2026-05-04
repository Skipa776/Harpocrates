"""Tests for the MCP server module.

Uses pytest.importorskip("mcp") so the test file is silently skipped in
environments where the [mcp] extra is not installed (e.g. CI without it).
"""
from __future__ import annotations

from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp", reason="mcp package not installed; install harpocrates[mcp]")


def test_scan_text_tool_returns_finding_dicts() -> None:
    """scan_text returns a list of finding dicts; token is absent by default."""
    from Harpocrates.mcp.server import scan_text

    known_secret = "ghp_" + "a" * 36
    text = f'api_key = "{known_secret}"\n'

    results = scan_text(text)

    assert isinstance(results, list)
    assert len(results) >= 1

    for item in results:
        assert isinstance(item, dict)
        assert "type" in item
        assert "severity" in item
        assert "evidence" in item
        assert "token" not in item, "token must be redacted by default"
        assert "token_start" not in item, "token_start must not appear in MCP output"
        assert "token_end" not in item, "token_end must not appear in MCP output"


def test_scan_text_tool_include_token() -> None:
    """scan_text with include_token=True exposes the matched token."""
    from Harpocrates.mcp.server import scan_text

    known_secret = "ghp_" + "b" * 36
    text = f"secret = {known_secret}\n"

    results = scan_text(text, include_token=True)

    assert results
    token_values = [r.get("token") for r in results if r.get("token")]
    assert any(known_secret in (t or "") for t in token_values), (
        f"Expected {known_secret!r} to appear in at least one finding token"
    )


def test_scan_file_tool_detects_secret(tmp_path: Path) -> None:
    """scan_file returns findings for a file containing a known secret."""
    from Harpocrates.mcp.server import scan_file

    known_secret = "ghp_" + "c" * 36
    secret_file: Path = tmp_path / "config.txt"
    secret_file.write_text(f"token={known_secret}\n", encoding="utf-8")

    results = scan_file(str(secret_file))

    assert isinstance(results, list)
    assert len(results) >= 1
    assert all("token" not in r for r in results), "token must be redacted by default"


def test_scan_file_tool_with_include_token(tmp_path: Path) -> None:
    """scan_file with include_token=True includes the raw token."""
    from Harpocrates.mcp.server import scan_file

    known_secret = "ghp_" + "d" * 36
    secret_file: Path = tmp_path / "secrets.env"
    secret_file.write_text(f"GITHUB_TOKEN={known_secret}\n", encoding="utf-8")

    results = scan_file(str(secret_file), include_token=True)

    assert results
    token_values = [r.get("token") for r in results if r.get("token")]
    assert any(known_secret in (t or "") for t in token_values)


def test_scan_file_tool_nonexistent_returns_empty(tmp_path: Path) -> None:
    """scan_file returns [] for a nonexistent path."""
    from Harpocrates.mcp.server import scan_file

    results = scan_file(str(tmp_path / "does_not_exist.py"))
    assert results == []


def test_scan_text_tool_empty_input() -> None:
    """scan_text returns [] for empty input."""
    from Harpocrates.mcp.server import scan_text

    assert scan_text("") == []


def test_scan_file_refuses_fifo(tmp_path: Path) -> None:
    """scan_file raises ValueError for named pipes to prevent indefinite blocking."""
    import os

    from Harpocrates.mcp.server import scan_file

    fifo_path = tmp_path / "evil.pipe"
    os.mkfifo(fifo_path)

    with pytest.raises(ValueError, match="Refusing to scan special file"):
        scan_file(str(fifo_path))


def test_scan_file_respects_max_bytes_ceiling(tmp_path: Path) -> None:
    """scan_file caps reads at _MAX_SCAN_BYTES regardless of caller-supplied max_bytes."""
    from Harpocrates.mcp.server import _MAX_SCAN_BYTES, scan_file

    # Create a file just over the ceiling
    big_file = tmp_path / "big.txt"
    big_file.write_bytes(b"x" * (_MAX_SCAN_BYTES + 1024))

    # Passing a larger max_bytes must still be capped internally
    results = scan_file(str(big_file), max_bytes=_MAX_SCAN_BYTES * 10)
    # Test passes if no exception is raised and result is a list
    assert isinstance(results, list)
