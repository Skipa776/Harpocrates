"""Tests for the core detector module."""
from __future__ import annotations

from pathlib import Path

from Harpocrates.core.detector import detect_file, detect_text
from Harpocrates.core.result import EvidenceType, Finding


def test_detect_text_finds_github_token() -> None:
    """Test that GitHub tokens are detected via regex."""
    github_token = "ghp_" + "a" * 36
    text = f"Here is a GitHub token: {github_token}\n"

    findings = detect_text(text)

    assert findings
    assert all(isinstance(f, Finding) for f in findings)
    assert any(f.type == "GITHUB_TOKEN" for f in findings)
    assert any(f.evidence == EvidenceType.REGEX for f in findings)
    # Confidence should be set for regex matches
    assert all(f.confidence is not None for f in findings)
    assert all(f.confidence >= 0.9 for f in findings)


def test_detect_text_finds_aws_key() -> None:
    """Test that AWS access key IDs are detected."""
    # Use a fake but valid-format AWS key
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    text = f"AWS_ACCESS_KEY_ID={aws_key}\n"

    findings = detect_text(text)

    assert findings
    assert any(f.type == "AWS_ACCESS_KEY_ID" for f in findings)
    assert any(f.evidence == EvidenceType.REGEX for f in findings)


def test_detect_text_no_false_positives() -> None:
    """Test that normal text doesn't trigger false positives."""
    text = """
    This is a normal configuration file.
    username = john_doe
    password = please_change_me
    api_url = https://api.example.com
    """

    findings = detect_text(text)

    # Should not detect anything in this normal text
    assert not findings


def test_detect_text_empty_input() -> None:
    """Test handling of empty input."""
    findings = detect_text("")
    assert findings == []


def test_detect_text_comments_skipped() -> None:
    """Test that commented lines are skipped."""
    text = "# ghp_" + "a" * 36 + "\n"  # Commented out token

    findings = detect_text(text)

    # Should not detect commented tokens
    assert not findings


def test_detect_file_smoke(tmp_path: Path) -> None:
    """Smoke test for file detection."""
    github_token = "ghp_" + "b" * 36
    content = f"token={github_token}\nno secret here\n"
    file_path = tmp_path / "test_secrets.txt"
    file_path.write_text(content, encoding="utf-8")

    findings = detect_file(file_path)

    assert findings
    assert all(isinstance(f, Finding) for f in findings)
    assert any(f.type == "GITHUB_TOKEN" for f in findings)
    # All findings should report the correct file path
    assert all(str(file_path) in (f.file or "") for f in findings)


def test_detect_file_nonexistent(tmp_path: Path) -> None:
    """Test handling of nonexistent files."""
    file_path = tmp_path / "does_not_exist.txt"

    findings = detect_file(file_path)

    # Should return empty list for nonexistent files
    assert findings == []


def test_finding_redacted_token() -> None:
    """Test that redacted_token property works correctly."""
    finding = Finding(
        type="AWS_ACCESS_KEY_ID",
        snippet="AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        evidence=EvidenceType.REGEX,
        token="AKIAIOSFODNN7EXAMPLE",
    )

    # Redacted token should show first 4 and last 4 chars
    assert finding.redacted_token == "AKIA...MPLE"


def test_finding_to_json_dict_excludes_token() -> None:
    """Test that to_json_dict() excludes the token field."""
    finding = Finding(
        type="GITHUB_TOKEN",
        snippet="token=ghp_xxx",
        evidence=EvidenceType.REGEX,
        token="ghp_abcdefghijklmnopqrstuvwxyz123456789",
    )

    json_dict = finding.to_json_dict()

    assert "token" not in json_dict
    assert json_dict["type"] == "GITHUB_TOKEN"


def test_finding_confidence_score() -> None:
    """Test that findings have appropriate confidence scores."""
    # Regex match should have high confidence
    text = "AKIAIOSFODNN7EXAMPLE"
    findings = detect_text(text)

    if findings:
        # Regex matches should have 0.95 confidence
        regex_findings = [f for f in findings if f.evidence == EvidenceType.REGEX]
        for f in regex_findings:
            assert f.confidence == 0.95
