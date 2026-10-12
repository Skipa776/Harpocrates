"""SEC-02 / FR-READ-02: MCP scan tools never hand a secret value to the agent.

Canaries are generated at runtime; nothing here is a real credential.
"""
from __future__ import annotations

import json
import secrets
import string
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp package not installed; install harpocrates[mcp]")

_ALNUM = string.ascii_letters + string.digits
_RUN = 12


def _canaries() -> list[str]:
    return [
        "ghp_" + "".join(secrets.choice(_ALNUM) for _ in range(36)),
        "AKIA" + "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(16)),
    ]


def _assert_clean(response: object, canary: str) -> None:
    blob = json.dumps(response)
    assert canary not in blob
    for i in range(len(canary) - _RUN + 1):
        assert canary[i : i + _RUN] not in blob, "12-char run of the secret reached the response"


def _assert_fields(findings: list[dict]) -> None:
    allowed = {"type", "severity", "evidence", "confidence", "category", "in_comment",
               "file", "line", "column", "placeholder", "explanation"}
    for f in findings:
        assert set(f) <= allowed, set(f) - allowed
        assert f["placeholder"].startswith("<<HARPO:") and f["placeholder"].endswith(">>")


@pytest.mark.parametrize("idx", [0, 1])
def test_sec_02_scan_text_no_raw_value(idx: int) -> None:
    from Harpocrates.mcp.server import scan_text

    canary = _canaries()[idx]
    findings = scan_text(f'api_key = "{canary}"\n')
    assert findings
    _assert_clean(findings, canary)
    _assert_fields(findings)
    assert all(f["line"] == 1 for f in findings)


@pytest.mark.parametrize("idx", [0, 1])
def test_sec_02_scan_file_no_raw_value(tmp_path: Path, idx: int) -> None:
    from Harpocrates.mcp.server import scan_file

    canary = _canaries()[idx]
    p = tmp_path / "settings.env"
    p.write_text(f"TOKEN={canary}\n", encoding="utf-8")
    findings = scan_file(str(p))
    assert findings
    _assert_clean(findings, canary)
    _assert_fields(findings)
    assert all(f["file"] for f in findings)


def test_sec_02_include_token_flag_is_gone() -> None:
    """The opt-in that exposed raw tokens no longer exists."""
    import inspect

    from Harpocrates.mcp import server

    for fn in (server.scan_text, server.scan_file):
        assert "include_token" not in inspect.signature(fn).parameters


def test_sec_02_contributions_no_raw_value(tmp_path: Path) -> None:
    from Harpocrates.mcp.server import scan_file, scan_text

    canary = _canaries()[0]
    try:
        text_res = scan_text(f"x = {canary}\n", include_contributions=True)
    except ImportError:
        pytest.skip("xgboost not installed")
    _assert_clean(text_res, canary)
    p = tmp_path / "a.txt"
    p.write_text(f"x = {canary}\n", encoding="utf-8")
    _assert_clean(scan_file(str(p), include_contributions=True), canary)


def test_sec_02_safe_read_and_grep_no_raw_value(tmp_path: Path) -> None:
    from Harpocrates.mcp.server import safe_grep, safe_read

    canary = _canaries()[0]
    p = tmp_path / "c.env"
    p.write_text(f"GITHUB_TOKEN={canary}\n", encoding="utf-8")
    _assert_clean(safe_read(str(p)), canary)
    _assert_clean(safe_grep("TOKEN", str(tmp_path)), canary)
