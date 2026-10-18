#!/usr/bin/env python3
"""FR-CLEAN-04 line-of-code counts for tracked files (git ls-files), grouped by package / top-level directory.

    python bench/results/r0-before/tools/loc.py > bench/results/r0-before/loc.md

No cloc needed. Per file: total physical lines, blank lines, comment-only lines (by language family), code
lines = total - blank - comment. Binary and non-text files count as files only (0 lines).
Comment rules (line-based, no multi-line string/doc handling beyond block comments for C-like and HTML):
  .py/.sh/.toml/.yml/.yaml/.tsv/.cfg/.ini/.gitignore etc: lines starting with '#'
  .rs/.js/.ts/.json5: lines starting with '//', plus /* ... */ blocks;  .md/.html/.ipynb/.json/.txt/.onnx: no comment rules
Deterministic output (sorted).
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HASH = {".py", ".sh", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".tsv", ".gitignore", ".gitattributes", ".gitleaksignore"}
SLASH = {".rs", ".js", ".ts", ".tsx", ".mjs", ".c", ".h", ".go", ".java"}


def group_of(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "cli":
        if len(parts) >= 3 and parts[1] == "Harpocrates":
            # Harpocrates/<subpackage>/... or Harpocrates/<module>.py
            return "cli/Harpocrates/" + (parts[2] if len(parts) > 3 else "(top-level modules)")
        if len(parts) >= 2 and parts[1] == "tests":
            return "cli/tests"
        return "cli/(other)"
    if parts[0] in {"core", "bench", "scripts", "docs", "plugins", "comparison"}:
        return parts[0]
    return "(repo root + .github + .claude-plugin)"


def count(path: Path) -> tuple[int, int, int, int]:
    try:
        text = path.read_bytes().decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return 0, 0, 0, 0
    if "\x00" in text:
        return 0, 0, 0, 0
    suffix = path.suffix if path.suffix else path.name
    total = blank = comment = 0
    in_block = False
    for line in text.splitlines():
        total += 1
        s = line.strip()
        if not s:
            blank += 1
        elif suffix in HASH and s.startswith("#"):
            comment += 1
        elif suffix in SLASH:
            if in_block:
                comment += 1
                if "*/" in s:
                    in_block = False
            elif s.startswith("//"):
                comment += 1
            elif s.startswith("/*"):
                comment += 1
                in_block = "*/" not in s
    return total, blank, comment, total - blank - comment


def main() -> None:
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\0")
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])  # files, total, blank, comment, code
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rel in sorted(f for f in files if f):
        p = ROOT / rel
        if not p.is_file():
            continue
        t, b, c, code = count(p)
        g = agg[group_of(rel)]
        g[0] += 1
        g[1] += t
        g[2] += b
        g[3] += c
        g[4] += code
        lang = p.suffix or p.name
        by_lang[lang][0] += 1
        by_lang[lang][1] += code
    print("# LOC (tracked files, `git ls-files`, tools/loc.py)\n")
    print("| group | files | total lines | blank | comment-only | code lines |")
    print("|---|---:|---:|---:|---:|---:|")
    tot = [0] * 5
    for name in sorted(agg):
        row = agg[name]
        tot = [a + b for a, b in zip(tot, row)]
        print(f"| {name} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    print(f"| **TOTAL** | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {tot[4]} |")
    print("\n## Code lines by file type\n")
    print("| type | files | code lines |")
    print("|---|---:|---:|")
    for lang, (n, code) in sorted(by_lang.items(), key=lambda kv: (-kv[1][1], kv[0])):
        print(f"| {lang} | {n} | {code} |")
    print("\nNote: `comparison/` has no tracked files on this commit (0 rows above means absent from the table).")


if __name__ == "__main__":
    main()
