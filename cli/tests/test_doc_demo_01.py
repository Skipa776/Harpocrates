"""DOC-DEMO-01 / DOC-DEMO-02: the README demos are VHS tapes rendered to GIF in CI, using canary secrets only."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAPES = sorted((ROOT / "docs" / "tapes").glob("*.tape"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
WORKFLOW = ROOT / ".github" / "workflows" / "docs-demos.yml"
GIF_LINK = re.compile(r"docs/assets/(demo-[\w-]+\.gif)")

# Strings a tape must never contain: real home paths, state-writing commands, and flags that print full secrets.
FORBIDDEN = ("~", "/Users/", "$HOME/.", ".ssh", ".aws", "harpo install", "harpo uninstall", "harpo allow",
             "harpo setup", "--show-secrets", "--no-redact", "--unredacted")


def test_doc_demo_01_tapes_use_only_canaries() -> None:
    """Every tape builds its own canary repo in a throwaway HOME and touches nothing real."""
    assert {tape.name for tape in TAPES} == {"scan.tape", "safe-read.tape"}
    for tape in TAPES:
        text = tape.read_text(encoding="utf-8")
        assert "mktemp" in text, f"{tape.name}: no throwaway directory"
        assert re.search(r"export [^\n]*\bHOME=\$\(mktemp", text), f"{tape.name}: HOME is not redirected"
        assert "bench/canary_repo.py" in text, f"{tape.name}: not built from the generated canary repo"
        assert "Hide" in text and "Show" in text, f"{tape.name}: setup is not hidden"
        for bad in FORBIDDEN:
            assert bad not in text, f"{tape.name}: contains {bad!r}"
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            assert not re.search(r"\bharpo\s+(install|uninstall|allow|setup)\b", code), f"{tape.name}: {line}"


def test_doc_demo_01_scanner_scene_numbers_match_the_measurement() -> None:
    """The README quotes the measured counts, and the measurement never stores a canary value."""
    path = ROOT / "bench" / "results" / "demos" / "scan-vs-gitleaks.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    totals = result["totals"]
    assert totals["canaries"] == 15
    assert f"{totals['harpo_found']} of {totals['canaries']}" in README
    assert f"{totals['gitleaks_found']} of {totals['canaries']}" in README
    caption = (ROOT / "docs" / "tapes" / "scan.tape").read_text(encoding="utf-8")
    assert f"harpo found {totals['harpo_found']} of 15 canaries, gitleaks {totals['gitleaks_found']} of 15" in caption
    manifest_free = path.read_text(encoding="utf-8")
    assert "value" not in manifest_free
    assert not re.search(r"(AKIA[A-Z0-9]{16}|sk_live_|ghp_[A-Za-z0-9]{20})", manifest_free)


def test_doc_demo_01_ci_job_renders_every_tape() -> None:
    """Each tape is rendered by the docs-demos workflow, and each README demo GIF comes from a tape."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "charmbracelet/vhs-action@" in workflow
    assert "workflow_dispatch" in workflow and "docs/tapes/**" in workflow
    assert "git commit" not in workflow and "git push" not in workflow
    assert "upload-artifact" in workflow
    for tape in TAPES:
        assert f"docs/tapes/{tape.name}" in workflow, f"{tape.name} is not rendered in CI"
    gifs = set(GIF_LINK.findall(README))
    assert gifs == {"demo-scan.gif", "demo-safe-read.gif"}
    for gif in gifs:
        tape = ROOT / "docs" / "tapes" / (gif.removeprefix("demo-").removesuffix(".gif") + ".tape")
        assert tape.exists(), f"{gif} has no tape"
        assert f"docs/assets/{gif}" in tape.read_text(encoding="utf-8"), f"{tape.name} does not write {gif}"
        assert f"docs/assets/{gif}" in workflow, f"CI does not check {gif}"
        assert (ROOT / "docs" / "assets" / gif).stat().st_size <= 5 * 1024 * 1024
