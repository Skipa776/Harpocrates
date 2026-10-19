"""FR-NAME-01: the user-facing name is Harpo. A text search finds no user-facing "Harpocrates" outside the changelog
and the migration notes.

What counts as user-facing, and is searched here: the README, the docs, the plugin README, the `--help` of every
command, every string a person or an agent can be shown by the package (messages, refusals, notices, generated
config and notes, MCP tool descriptions), and the MCP server name.

What is allowed to keep the old spelling, each with its reason (the allowlists below are deliberately short):
  * ALLOWED_TOKENS: pieces of a line that are not the product name, such as the import package `Harpocrates`.
  * ALLOWED_FILES / ALLOWED_SECTIONS: files and sections that are history or migration by nature.
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from Harpocrates import harness, hooks, install
from Harpocrates.cli import app
from Harpocrates.mcp import server

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "cli" / "Harpocrates"
OLD = re.compile("harpocrates", re.IGNORECASE)

# (regex of a piece that is not the product name, reason)
ALLOWED_TOKENS: tuple[tuple[str, str], ...] = (
    (r"github\.com/Skipa776/[Hh]arpocrates|Skipa776/[Hh]arpocrates",
     "the repository and marketplace id; the repo is renamed last (FR-NAME-07)"),
    (r"\.harpocrates\b", "the pre-rename state folder, named where it is migrated or still protected (FR-NAME-04)"),
    (r"migrated-from-harpocrates", "the marker file the migration leaves in the new state folder"),
    (r"(?<![\w-])Harpocrates(?=[./(])|(?<=-m )Harpocrates|(?<=from )Harpocrates|(?<=import )Harpocrates",
     "the import package and module paths (`python -m Harpocrates.hooks`), not user-facing"),
    (r"mcp__harpocrates__\w*", "tool names of an install made before the rename, named in refusals until it is re-run"),
    (r"HARPOCRATES_\w*", "the pre-rename environment variable names, still read (FR-NAME-04)"),
    (r"harpocrates(?=\[)|(?<=pip install )harpocrates|(?<=pip install ')harpocrates|(?<=pip install \")harpocrates",
     "the PyPI distribution name until it is renamed (FR-NAME-03)"),
    (r"harpocrates-prompt", "the Claude Code plugin id; the plugin is renamed with the repo (FR-NAME-07)"),
    (r"harpocrates-rust-scanner", "the file name of the native scanner binary, not shown as a product name"),
    (r"harpocrates:(?:start|end)|[<>]{3} harpocrates [<>]{3}",
     "the block markers a pre-rename install wrote, recognised so install can move them and uninstall remove them"),
)
# Exact string literals that are identifiers, not prose.
ALLOWED_LITERALS = {
    "harpocrates": "the deprecated command name and the PyPI distribution name, as an identifier",
    "harpocrates-mcp": "the deprecated MCP command name, as an identifier",
    "Harpocrates": "the import package name, as an identifier",
}
# Files that are history or migration by nature, as (path from the repo root, reason).
ALLOWED_FILES: dict[str, str] = {
    "cli/Harpocrates/deprecated.py": "it is the deprecation notice for the old command names (FR-NAME-02)",
    "cli/Harpocrates/allowlist.py": "it recognises the old command names and both state folders in agent shell "
                                    "commands, which it must refuse (FR-NAME-02, FR-NAME-04)",
    "docs/cleanup-report.md": "a record of the cleanup that came before the rename",
    "docs/r0-specs.md": "the approved rename spec, which describes the rename itself",
    "docs/DO_NOT_TRACK.md": "the rename brief, history",
    "docs/design.md": "the spec; its decision rows and history keep the old name",
}
# Sections that are the changelog or the migration notes, as (file, heading).
ALLOWED_SECTIONS: tuple[tuple[str, str], ...] = (
    ("README.md", "## Past versions"),
    ("README.md", "## Renamed from Harpocrates"),
    ("docs/setup.md", "## Migrating from ~/.harpocrates"),
)
SKIP_PACKAGE_DIRS = {"training"}  # the training code is not shipped surface


def _strip(text: str) -> str:
    for pattern, _reason in ALLOWED_TOKENS:
        text = re.sub(pattern, "", text)
    return text


def _found(text: str) -> list[str]:
    """The lines of text that still say Harpocrates once the allowed pieces are taken out."""
    return [line.strip()[:160] for line in text.splitlines() if OLD.search(_strip(line))]


def _without_sections(rel: str, text: str) -> str:
    for file, heading in ALLOWED_SECTIONS:
        if file != rel or heading not in text:
            continue
        level = len(heading) - len(heading.lstrip("#"))
        head, _, rest = text.partition(heading)
        lines = rest.split("\n")[1:]
        for i, line in enumerate(lines):
            if re.match(rf"#{{1,{level}}} ", line):
                head += "\n".join(lines[i:])
                break
        text = head
    return text


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "plugins" / "harpocrates-prompt" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    return [f for f in files if f.relative_to(ROOT).as_posix() not in ALLOWED_FILES]


def test_fr_name_01_no_user_facing_harpocrates_in_the_readme_and_docs() -> None:
    bad = {}
    for path in _markdown_files():
        rel = path.relative_to(ROOT).as_posix()
        found = _found(_without_sections(rel, path.read_text(encoding="utf-8")))
        if found:
            bad[rel] = found
    assert not bad, json.dumps(bad, indent=1)


def test_fr_name_01_the_migration_note_is_in_the_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Renamed from Harpocrates" in readme
    note = readme.partition("## Renamed from Harpocrates")[2].partition("\n## ")[0]
    for needle in ("harpo-scan", "harpo install", "~/.harpo", "harpo-mcp", "id: harpo", "MCP server"):
        assert needle in note, needle


def test_fr_name_01_allowlists_stay_short_and_explained() -> None:
    assert len(ALLOWED_TOKENS) <= 12 and len(ALLOWED_FILES) <= 8 and len(ALLOWED_SECTIONS) <= 4
    for _pattern, reason in ALLOWED_TOKENS:
        assert len(reason) > 20
    assert all(len(reason) > 20 for reason in (*ALLOWED_FILES.values(), *ALLOWED_LITERALS.values()))
    for rel in ALLOWED_FILES:
        assert (ROOT / rel).exists(), rel  # a stale entry is removed, not kept


def _commands() -> list[list[str]]:
    import typer.main

    group = typer.main.get_command(app)
    return [[], *([name] for name in sorted(group.commands))]  # type: ignore[attr-defined]


def test_fr_name_01_no_user_facing_harpocrates_in_any_help() -> None:
    runner = CliRunner()
    bad = {}
    for argv in _commands():
        result = runner.invoke(app, [*argv, "--help"], env={"COLUMNS": "200", "TERM": "dumb"})
        assert result.exit_code == 0, (argv, result.output)
        found = _found(result.output)
        if found:
            bad[" ".join(argv) or "(root)"] = found
    assert not bad, json.dumps(bad, indent=1)
    assert "harpo" in runner.invoke(app, ["--help"], env={"COLUMNS": "200"}).output.lower()


def test_fr_name_01_version_commands_say_harpo() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--version"]).output.startswith("harpo ")
    assert runner.invoke(app, ["version"]).output.startswith("Harpo version ")


def _literals(path: Path) -> list[str]:
    """Every string constant of a module that is not a docstring (an f-string's fixed parts included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)}
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_fr_name_01_no_user_facing_harpocrates_in_package_strings() -> None:
    bad = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if path.relative_to(PACKAGE).parts[0] in SKIP_PACKAGE_DIRS or rel in ALLOWED_FILES:
            continue
        found = [text.strip()[:160] for text in _literals(path)
                 if text not in ALLOWED_LITERALS and OLD.search(_strip(text))]
        if found:
            bad[rel] = found
    assert not bad, json.dumps(bad, indent=1)


def test_fr_name_01_no_user_facing_harpocrates_in_generated_text() -> None:
    shown = {
        "setup claude-code": harness.setup_instructions("claude-code"),
        "setup codex": harness.setup_instructions("codex"),
        "tier report": harness.tier_report({"Claude Code": True, "Codex CLI": True, "Cursor": True}),
        "claude settings": json.dumps(harness.claude_code_settings()),
        "codex config": harness.codex_config(),
        "notes": harness.CLAUDE_NOTE + harness.CODEX_NOTE,
        "install notes": install.HOOKS_TRUST_NOTE + install.NO_RECORD_NOTE,
        "refusals": "\n".join([hooks.SCAN_REFUSAL, hooks.SCAN_UNCHECKABLE, hooks.STATE_REFUSAL]),
    }
    shown = {name: text.replace(sys.executable, "<python>") for name, text in shown.items()}  # the venv path may say it
    bad = {name: _found(text) for name, text in shown.items() if _found(text)}
    assert not bad, json.dumps(bad, indent=1)


def test_fr_name_01_the_mcp_server_is_named_harpo_and_its_tools_do_not_say_harpocrates() -> None:
    assert server.mcp.name == "harpo"
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == {"scan_text", "scan_file", "safe_read", "safe_grep"}
    bad = {t.name: _found(f"{t.name}\n{t.description or ''}") for t in tools if _found(f"{t.name}\n{t.description or ''}")}
    assert not bad, json.dumps(bad, indent=1)


@pytest.mark.parametrize("surface", ["README.md", "plugins/harpocrates-prompt/README.md"])
def test_fr_name_01_the_old_commands_are_not_taught(surface: str) -> None:
    """The docs teach `harpo`; the old command names appear only in the migration note."""
    text = _without_sections(surface, (ROOT / surface).read_text(encoding="utf-8"))
    assert not re.search(r"(?<![\w/.-])harpocrates(?:-mcp)?\s+(scan|install|uninstall|setup|allow|--version)\b", text)
