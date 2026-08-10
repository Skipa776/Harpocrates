"""Regression checks for CI dependency and action compatibility."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mcp_extras_remain_on_the_v1_api() -> None:
    """The server imports FastMCP from the v1-only module path."""
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert project.count('"mcp>=1.27,<2"') == 2


def test_workflows_use_node24_actions() -> None:
    """CI must not invoke action majors that bundle deprecated Node 20."""
    workflows = [
        PROJECT_ROOT / ".github/workflows/ci.yml",
        PROJECT_ROOT / ".github/workflows/publish.yml",
    ]

    for workflow in workflows:
        content = workflow.read_text(encoding="utf-8")
        assert "actions/checkout@v4" not in content
        assert "actions/setup-python@v5" not in content
        assert "actions/checkout@v6" in content

    assert "actions/setup-python@v6" in workflows[0].read_text(encoding="utf-8")
    assert "actions/setup-python@v6" in workflows[1].read_text(encoding="utf-8")
