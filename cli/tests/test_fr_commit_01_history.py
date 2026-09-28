"""FR-COMMIT-01: the CLI scans full git history, so a secret deleted from disk is still found."""

import subprocess

import pytest
from typer.testing import CliRunner

from Harpocrates.cli import app
from Harpocrates.core.scanner import scan_git_history

# Concatenated so this file never contains the canary literal.
FAKE_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo_with_deleted_secret(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "config.py").write_text(f'aws_key = "{FAKE_AWS_KEY}"\n')
    _git(tmp_path, "add", "config.py")
    _git(tmp_path, "commit", "-qm", "add config")
    first_sha = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "rm", "-q", "config.py")
    _git(tmp_path, "commit", "-qm", "remove config")
    return tmp_path, first_sha


def test_fr_commit_01_history_finds_deleted_secret(repo_with_deleted_secret):
    repo, first_sha = repo_with_deleted_secret
    findings = scan_git_history(repo).findings
    aws = [f for f in findings if f.type.startswith("AWS")]
    assert aws, findings
    assert aws[0].file == f"{first_sha[:12]}:config.py"
    assert aws[0].line == 1


def test_fr_commit_01_cli_history_flag(repo_with_deleted_secret):
    repo, _ = repo_with_deleted_secret
    runner = CliRunner()
    assert runner.invoke(app, ["scan", str(repo), "--history"]).exit_code == 1
    assert runner.invoke(app, ["scan", str(repo)]).exit_code == 0


def test_fr_commit_01_non_git_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        scan_git_history(tmp_path)


def test_fr_commit_01_secret_added_in_merge_resolution(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "checkout", "-qb", "side")
    (tmp_path / "a.py").write_text("x = 2\n")
    _git(tmp_path, "commit", "-qam", "side")
    _git(tmp_path, "checkout", "-q", "main")
    (tmp_path / "a.py").write_text("x = 3\n")
    _git(tmp_path, "commit", "-qam", "main")
    subprocess.run(["git", "-C", str(tmp_path), "merge", "-q", "side"], capture_output=True)
    (tmp_path / "a.py").write_text(f'aws_key = "{FAKE_AWS_KEY}"\n')
    _git(tmp_path, "commit", "-qam", "resolve")
    merge_sha = _git(tmp_path, "rev-parse", "HEAD")
    files = {f.file for f in scan_git_history(tmp_path).findings if f.type.startswith("AWS")}
    assert f"{merge_sha[:12]}:a.py" in files


def test_fr_commit_01_added_line_starting_with_plus_plus(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "weird.py").write_text(f'x = 1\n++ b/injected\naws_key = "{FAKE_AWS_KEY}"\n')
    _git(tmp_path, "add", "weird.py")
    _git(tmp_path, "commit", "-qm", "add")
    sha = _git(tmp_path, "rev-parse", "HEAD")
    result = scan_git_history(tmp_path)
    aws = [f for f in result.findings if f.type.startswith("AWS")]
    assert [(f.file, f.line) for f in aws] == [(f"{sha[:12]}:weird.py", 3)]
    assert result.total_lines == 3
