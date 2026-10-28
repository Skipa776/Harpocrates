#!/usr/bin/env python3
"""Write the sorted pytest node-id list (`pytest -q --co`), with secret-like [param] parts replaced by sha256[:12].

    .venv/bin/python bench/results/r0-before/tools/tests_ids.py OUT_FILE

Several parametrized test ids embed canary/secret-like strings; the package scanner decides which lines are
flagged, those (and any param longer than 60 chars) are hashed so the file is safe to store and diff.
The pass/skip/fail summary (pytest run itself) is NOT produced here; run `python -m pytest -q` for that.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "cli"))
from Harpocrates.core.scanner import scan_file  # noqa: E402


def h(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()[:12]


def main(out: Path) -> None:
    raw = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "--co", "-p", "no:cacheprovider"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    ids = sorted({line.strip() for line in raw.splitlines() if "::" in line})
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "ids.txt"
        probe.write_text("\n".join(ids) + "\n")
        flagged = {f.line for f in scan_file(probe, ml_threshold=0.19).findings}
    clean, hashed = [], 0
    for n, i in enumerate(ids, 1):
        m = re.match(r"^(.*?::[^\[]*)\[(.*)\]$", i)
        if m and (n in flagged or len(m.group(2)) > 60):
            i, hashed = f"{m.group(1)}[{h(m.group(2))}]", hashed + 1
        elif n in flagged:
            i, hashed = h(i) + " (whole-id hashed)", hashed + 1
        clean.append(i)
    summary = Path(__file__).with_name("tests_summary.txt").read_text().rstrip("\n").splitlines()
    out.write_text("\n".join([*summary,
                              f"# node ids (pytest -q --co, python-sorted); {hashed} ids had their [param] replaced by "
                              "sha256[:12] because they contain secret-like values", *clean]) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
