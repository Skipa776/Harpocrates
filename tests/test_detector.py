from __future__ import annotations

from pathlib import Path

from Harpocrates.scanner.detector import detect_file, detect_text


def test_detect_text_finds_github_token_and_entropy() -> None:
    github_token = "ghp_" + "a" * 36
    entropy_token = "a9fK3LmP8QzX1cV7bW2Y"  # mixed-case/digits, high-ish entropy

    text = (
        f"Here is a GitHub token: {github_token}\n"
        "Some non-secret line.\n"
        f"another_secret={entropy_token}\n"
    )

    findings = detect_text(text)
    assert findings
    assert any("GITHUB" in str(f["type"]) for f in findings)
    assert any(f["evidence"] == "entropy" for f in findings)


def test_detect_file_smoke(tmp_path: Path) -> None:
    github_token = "ghp_" + "b" * 36
    content = f"token={github_token}\nno secret here\n"
    file_path = tmp_path / "test_secrets.txt"
    file_path.write_text(content, encoding="utf-8")

    findings = detect_file(file_path)

    assert findings
    assert any("GITHUB" in str(f["type"]) for f in findings)
    assert all(str(file_path) == f["file"] for f in findings)
