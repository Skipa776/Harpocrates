"""FR-NAME-05: .pre-commit-hooks.yaml has `id: harpo` and keeps `id: harpocrates` as an alias.

Both ids run the same command (`harpo scan`): `harpocrates` is a deprecated console script that still works, but the
alias does not need it. The yaml contract is always checked. The live run, `pre-commit try-repo` over a temp git repo
with a staged canary, needs the `pre-commit` tool and a network for the hook's environment, so it is an integration
test and is skipped when `pre-commit` is not installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".pre-commit-hooks.yaml"
CANARY = "AWS_KEY=AKIAIOSFODNN7EXAMPLE\n"  # AWS's documented example key, not a credential


def _hooks() -> dict[str, dict]:
    loaded = yaml.safe_load(HOOKS.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    return {hook["id"]: hook for hook in loaded}


def test_fr_name_05_both_hook_ids() -> None:
    hooks = _hooks()
    assert list(hooks) == ["harpo", "harpocrates"]  # the new id first; the old one stays as an alias


def test_fr_name_05_both_hook_ids_run_the_same_command() -> None:
    hooks = _hooks()
    assert hooks["harpo"]["name"] == "Harpo secrets scan"
    for hook in hooks.values():
        assert hook["entry"] == "harpo scan"
        assert hook["language"] == "python"
        assert hook["types_or"] == ["text"]
        assert hook["pass_filenames"] is True
    # Everything but the id and the display name is identical, so the alias cannot drift from the new id.
    strip = {k: v for k, v in hooks["harpo"].items() if k not in ("id", "name")}
    assert strip == {k: v for k, v in hooks["harpocrates"].items() if k not in ("id", "name")}
    assert "harpocrates" not in hooks["harpocrates"]["name"].lower()  # the alias is not shown under the old name


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("pre-commit") is None, reason="the pre-commit tool is not installed")
@pytest.mark.parametrize("hook_id", ["harpo", "harpocrates"])
def test_fr_name_05_both_hook_ids_run_in_a_fixture_repo(hook_id: str, tmp_path: Path) -> None:
    """Each id blocks a staged canary: `pre-commit try-repo` over this repo exits non-zero and prints the finding."""
    repo = tmp_path / "fixture"
    repo.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "PRE_COMMIT_HOME": str(tmp_path / "pc")}
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "config.env").write_text(CANARY, encoding="utf-8")
    subprocess.run([*git, "add", "config.env"], cwd=repo, check=True, env=env)
    done = subprocess.run(["pre-commit", "try-repo", str(ROOT), hook_id, "--files", "config.env"], cwd=repo,
                          capture_output=True, text=True, timeout=900, check=False, env=env)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "AKIA" in done.stdout + done.stderr or "aws" in (done.stdout + done.stderr).lower()
