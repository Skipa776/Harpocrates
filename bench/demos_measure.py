#!/usr/bin/env python3
"""Measure which canaries Harpo and gitleaks each find in the canary repo (DOC-DEMO-02 scene 1).

    python bench/demos_measure.py [--seed N] [--out bench/results/demos/scan-vs-gitleaks.json]

Generates the canary repo in a throwaway directory, runs `gitleaks dir` and
`harpo scan --json --show-secrets` on it (the match reads each finding's snippet, which is
redacted by default), and writes one record per canary: its file, its kind, and which tool
found it. Canary values are used for matching only and never written to the output. A tool "found" a canary when it reported a finding in the
canary's file whose text contains the start of the value (or, for a multi-line value, whose line
range includes the canary's first line).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "cli"), str(ROOT)]

from bench.canary_repo import build  # noqa: E402

# Canaries a comparison must not lean on. The generator draws the AWS key id from A-Z0-9,
# but gitleaks' rule only accepts the base32 alphabet (A-Z2-7), which real ids use; a
# base32-valid key is found by gitleaks, this canary is not. db_password is the password
# inside db_url, so one finding covers both.
CAVEATS = {
    "aws_access_key_id": "canary is not base32-valid (contains 0/1/8/9), so gitleaks's aws rule cannot match it; not evidence",
    "db_password": "embedded in db_url; the same finding covers both",
}

DEFAULT_OUT = ROOT / "bench" / "results" / "demos" / "scan-vs-gitleaks.json"


def _hit(value: str, file: Path, text: str, start: int, end: int, finding_file: str,
         first_line: int) -> bool:
    """True when a finding (file, line range, text) accounts for the canary."""
    if Path(finding_file).resolve() != file.resolve():
        return False
    if "\n" in value:
        return start <= first_line <= end
    # A snippet may be cut short (a 153-character JWT is), so match on the value's start.
    return value[:40] in text


def _first_line(path: Path, value: str) -> int:
    head = value.splitlines()[0]
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if head in line:
            return number
    return 0


def _run_gitleaks(repo: Path, report: Path) -> list[dict]:
    subprocess.run(["gitleaks", "dir", str(repo), "-f", "json", "-r", str(report), "--no-banner"],
                   capture_output=True, check=False)
    return json.loads(report.read_text())


def _run_harpo(repo: Path) -> list[dict]:
    proc = subprocess.run([sys.executable, "-m", "Harpocrates.cli", "scan", str(repo), "--json", "--show-secrets"],
                          capture_output=True, text=True, check=False)
    return json.loads(proc.stdout)["findings"]


def _version(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip().splitlines()[-1]


def measure(seed: int) -> dict:
    """Build the canary repo, run both scanners and return the per-canary result."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        manifest = build(repo, seed)
        gl = _run_gitleaks(repo, Path(tmp) / "gl.json")
        hp = _run_harpo(repo)
        rows = []
        for entry in manifest:
            path, value = repo / entry["file"], entry["value"]
            line = _first_line(path, value)
            gl_hit = any(_hit(value, path, f"{f.get('Secret', '')}\n{f.get('Match', '')}",
                              f["StartLine"], f["EndLine"], f["File"], line) for f in gl)
            hp_hit = any(_hit(value, path, f.get("snippet", ""), f["line"], f.get("end_line", f["line"]),
                              f["file"], line) for f in hp)
            rows.append({"file": entry["file"], "kind": entry["kind"], "harpo": hp_hit,
                         "gitleaks": gl_hit, **({"caveat": CAVEATS[entry["kind"]]}
                                                if entry["kind"] in CAVEATS else {})})
        return {
            "seed": seed,
            "tools": {"harpo": _version([sys.executable, "-m", "Harpocrates.cli", "--version"]),
                      "gitleaks": _version(["gitleaks", "version"])},
            "canaries": rows,
            "totals": {
                "canaries": len(rows),
                "harpo_found": sum(r["harpo"] for r in rows),
                "gitleaks_found": sum(r["gitleaks"] for r in rows),
                "only_harpo": [r["kind"] for r in rows if r["harpo"] and not r["gitleaks"]],
                "only_harpo_without_caveat": [r["kind"] for r in rows
                                              if r["harpo"] and not r["gitleaks"] and "caveat" not in r],
                "only_gitleaks": [r["kind"] for r in rows if r["gitleaks"] and not r["harpo"]],
                "neither": [r["kind"] for r in rows if not r["harpo"] and not r["gitleaks"]],
                "harpo_findings_total": len(hp),
                "gitleaks_findings_total": len(gl),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = measure(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
