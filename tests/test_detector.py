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
    assert any(f.type == "GITHUB_PAT" for f in findings)
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


def test_detect_text_detects_secret_in_python_comment() -> None:
    """Commented-out secrets are still leaks — they must be detected with in_comment=True."""
    text = "# ghp_" + "a" * 36 + "\n"

    findings = detect_text(text)

    assert findings, "commented-out GitHub PAT must be detected"
    assert any(f.type == "GITHUB_PAT" for f in findings)
    assert all(f.in_comment is True for f in findings)


def test_detect_text_detects_secret_in_js_comment() -> None:
    """// and /* */ style comments are scanned for secrets."""
    token = "ghp_" + "b" * 36
    assert detect_text(f"// old token: {token}\n"), "// comment must be scanned"
    assert detect_text(f"/* {token} */\n"), "/* */ comment must be scanned"


def test_detect_text_detects_aws_key_in_html_comment() -> None:
    """HTML <!-- --> comments are scanned."""
    text = "<!-- AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE -->\n"
    findings = detect_text(text)
    assert findings
    assert all(f.in_comment is True for f in findings)


def test_detect_text_detects_aws_key_in_sql_comment() -> None:
    """SQL -- comments are scanned."""
    text = "-- AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
    findings = detect_text(text)
    assert findings
    assert all(f.in_comment is True for f in findings)


def test_detect_text_prose_comment_skipped() -> None:
    """Prose comments with no = : or quotes produce no findings (budget guard)."""
    text = "# This is a description of the authentication module\n"
    assert detect_text(text) == []


def test_detect_text_non_comment_in_comment_is_none() -> None:
    """Findings from non-comment lines must have in_comment=None, not False."""
    token = "ghp_" + "c" * 36
    findings = detect_text(f"token = {token}\n")
    assert findings
    assert all(f.in_comment is None for f in findings)


def test_detect_text_entropy_candidate_in_comment_has_in_comment_flag() -> None:
    """High-entropy token in a comment must produce ENTROPY_CANDIDATE with in_comment=True.

    The line contains '=' so it passes the prose-comment guard. The value is
    a 30-char mixed-case alphanumeric string (entropy ~4.5 bits) that triggers
    the entropy phase. This confirms in_comment propagates to the entropy path,
    not only the regex path.
    """
    # Mixed-case alphanumeric, 30 chars — entropy well above 4.0 bits.
    text = "# secret_key = aB3dEfGhIjKlMnOpQrStUvWxYz012\n"

    findings = detect_text(text)

    entropy_findings = [f for f in findings if f.evidence == EvidenceType.ENTROPY]
    assert entropy_findings, "high-entropy token in comment must produce ENTROPY_CANDIDATE"
    assert all(f.in_comment is True for f in entropy_findings)


def test_detect_file_smoke(tmp_path: Path) -> None:
    """Smoke test for file detection."""
    github_token = "ghp_" + "b" * 36
    content = f"token={github_token}\nno secret here\n"
    file_path = tmp_path / "test_secrets.txt"
    file_path.write_text(content, encoding="utf-8")

    findings = detect_file(file_path)

    assert findings
    assert all(isinstance(f, Finding) for f in findings)
    assert any(f.type == "GITHUB_PAT" for f in findings)
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
        # CRITICAL regex matches have 0.99; HIGH have 0.95
        regex_findings = [f for f in findings if f.evidence == EvidenceType.REGEX]
        for f in regex_findings:
            assert f.confidence >= 0.95


def test_detect_text_does_not_expose_ml_candidates() -> None:
    """detect_text (non-ML entrypoint) must never return evidence=ML findings."""
    # This line matches _SENSITIVE_ASSIGNMENT_RE and would generate an ML_CANDIDATE
    # inside _scan_line — but detect_text must filter them before returning.
    text = 'password = "hunter2_long_enough_to_match_threshold"\n'

    findings = detect_text(text)

    ml_findings = [f for f in findings if f.evidence == EvidenceType.ML]
    assert ml_findings == [], (
        f"detect_text returned {len(ml_findings)} unverified ML_CANDIDATE(s) — "
        "non-ML entrypoints must strip evidence=ML findings before returning"
    )


def test_detect_text_with_ml_drops_ml_candidates_on_verifier_failure() -> None:
    """On verifier exception, detect_text_with_ml must drop ML_CANDIDATEs."""
    from unittest.mock import MagicMock

    from Harpocrates.core.detector import detect_text_with_ml

    failing_verifier = MagicMock()
    failing_verifier.verify.side_effect = RuntimeError("simulated verifier failure")

    text = 'secret = "aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890_long"\n'

    findings = detect_text_with_ml(text, verifier=failing_verifier)

    ml_findings = [f for f in findings if f.evidence == EvidenceType.ML]
    assert ml_findings == [], (
        "On verifier failure, ML_CANDIDATEs must be dropped — "
        f"got {len(ml_findings)} unverified finding(s)"
    )


def test_detect_text_never_exposes_ml_candidates_for_any_input() -> None:
    """detect_text must not return evidence=ML findings for any input.

    This is a second ML-filter invariant check using a URL-containing line
    where the sensitive-assignment pattern might fire after URL stripping.
    Distinct from test_detect_text_does_not_expose_ml_candidates which uses
    a plain assignment — together they confirm the filter holds across input shapes.
    """
    text = "endpoint = https://user:secret_password_value_long@host.example.com/db\n"

    findings = detect_text(text)

    ml_findings = [f for f in findings if f.evidence == EvidenceType.ML]
    assert ml_findings == [], (
        "detect_text must not expose ML_CANDIDATEs regardless of input shape"
    )
