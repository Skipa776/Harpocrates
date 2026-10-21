"""TEST-04: the pull request template lists each new or changed test, and the PR checklist check fails
while one of them is unticked."""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "pull_request_template.md"
WORKFLOW = ROOT / ".github" / "workflows" / "pr-checklist.yml"


def _run_block(lines: list[str]) -> list[str]:
    """Return the lines under the run: | key (the bash script), without the key itself."""
    start = next(i for i, line in enumerate(lines) if line.strip() == "run: |")
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return block


def _run_check(body: str) -> subprocess.CompletedProcess:
    """Run the workflow's bash script with the given PR body passed as BODY."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    script = textwrap.dedent("\n".join(_run_block(lines)))
    return subprocess.run(["bash", "-c", script], env={**os.environ, "BODY": body},
                          capture_output=True, text=True, timeout=60)


def test_test_04_the_pr_template_has_a_tests_checklist():
    """The template has a Tests section with at least one unticked test line."""
    lines = TEMPLATE.read_text(encoding="utf-8").splitlines()
    assert "## Tests" in [line.rstrip() for line in lines]
    assert any(line.startswith("- [ ] `test_") for line in lines)


def test_test_04_the_check_reads_the_pr_body_only_through_env():
    """The PR body enters the check only through the BODY env var, never through ${{ }} in a script."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    body_lines = [line for line in lines if "github.event.pull_request.body" in line]
    assert len(body_lines) == 1
    assert "BODY:" in body_lines[0]
    for line in _run_block(lines):
        assert "${{" not in line


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_test_04_the_check_counts_unticked_tests():
    """The check fails on an unticked test or a missing Tests section, and passes when all are ticked."""
    unticked = "## Summary\n\nx\n\n## Tests\n- [ ] `test_fr_gate_03_x` (FR-GATE-03): proves x\n\n## Checks\n"
    all_ticked = ("## Tests\r\n- [x] `test_a` (FR-A-01): a\r\n- [X] `test_b` (FR-A-02): b\r\n"
                  "\r\n## Checks\r\n")
    no_tests_section = "## Summary\n\nNo tests listed.\n"

    result = _run_check(unticked)
    assert result.returncode != 0
    assert "1 test(s) not yet ticked in the Tests section" in result.stderr

    result = _run_check(all_ticked)
    assert result.returncode == 0, result.stderr
    assert "2 test(s) ticked" in result.stdout

    result = _run_check(no_tests_section)
    assert result.returncode != 0
    assert "List the new and changed tests in the Tests section" in result.stderr
