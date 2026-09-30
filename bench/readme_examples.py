#!/usr/bin/env python3
"""Reproduce the README's "only Harpocrates catches it" examples.

    python bench/readme_examples.py --model-dir data/models/cand_v16_gate

Runs TruffleHog, gitleaks, detect-secrets, CredSweeper and Harpocrates over
docs/samples/readme_examples.jsonl (fake values) and prints who flags each one.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "cli"), str(ROOT)]

import bench.compare_scanners as cmp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, help="Harpocrates model (default: shipped)")
    args = parser.parse_args()
    records = [json.loads(line) for line in (ROOT / "docs/samples/readme_examples.jsonl").read_text().splitlines()]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        placed = cmp.materialize(records, root)
        found = {"trufflehog": cmp.run_trufflehog(root, records, placed),
                 "gitleaks": cmp.run_gitleaks(root, records, placed),
                 "detect-secrets": cmp.run_detect_secrets(root, records, placed),
                 "credsweeper": cmp.run_credsweeper(root, records, placed, "medium")[0],
                 "harpocrates": cmp.run_harpocrates(root, records, placed, args.model_dir, 0.19)[0]}
    print(f"{'example':32}" + "".join(f"{n:>15}" for n in found))
    for i, r in enumerate(records):
        print(f"{r['example']:32}" + "".join(f"{'yes' if found[n][i] else '-':>15}" for n in found))


if __name__ == "__main__":
    main()
