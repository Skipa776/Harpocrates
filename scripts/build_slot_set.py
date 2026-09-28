#!/usr/bin/env python3
"""Turn LLM-written code files with typed slots into labeled records.

    python scripts/build_slot_set.py data/llm_slots/ data/eval/benchmark_llm_v1.jsonl --seed 1

The LLM writes code only, marking where values go:
    {{SECRET:<kind>}}     kind from build_eval_set.POSITIVES (e.g. stripe_key, password)
    {{NONSECRET:<kind>}}  kind from build_eval_set.NEGATIVES (e.g. git_sha, uuid)
Every slot is filled with a deterministic fake value, so labels are correct by
construction: the LLM never writes or labels a secret. Used as a held-out
generator benchmark; never train on it.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_set import NEGATIVES, POSITIVES

SLOT = re.compile(r"\{\{(SECRET|NONSECRET):([a-z0-9_]+)\}\}")
ANY_MARKER = re.compile(r"\{\{[^}]*\}\}")
GENERATOR_VERSION = "1"


def fill_file(path: Path, root: Path, seed: int) -> list[dict]:
    text = path.read_text(errors="replace")
    for marker in ANY_MARKER.findall(text):
        if not SLOT.fullmatch(marker):
            raise ValueError(f"{path}: malformed marker {marker}")
    random.seed(f"{seed}:{path.relative_to(root).as_posix()}")  # provider generators use the global RNG

    slots = []  # (line index, value, label, kind) in file order

    def fill(match: re.Match) -> str:
        category, kind = match.groups()
        table = POSITIVES if category == "SECRET" else NEGATIVES
        if kind not in table:
            raise ValueError(f"{path}: unknown {category} kind {kind!r}")
        value = table[kind]()
        if "\n" in value:  # line indices come from the unfilled text; multi-line values would shift them
            raise ValueError(f"{path}: {kind} produced a multi-line value; not supported")
        slots.append((text.count("\n", 0, match.start()), value, int(category == "SECRET"), kind))
        return value

    lines = SLOT.sub(fill, text).splitlines()
    rel = str(path.relative_to(root))
    return [
        {
            "token": value, "label": label, "secret_type": kind, "line_content": lines[idx],
            "context_before": lines[max(0, idx - 3):idx], "context_after": lines[idx + 1:idx + 4],
            "file_path": rel, "file_type": path.suffix, "source": "llm_slot",
            "generator": "build_slot_set", "generator_version": GENERATOR_VERSION, "seed": seed,
        }
        for idx, value, label, kind in slots
    ]


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    records, errors = [], []
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        try:
            records.extend(fill_file(path, src, seed))
        except ValueError as e:
            errors.append(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps({"out": str(out), "records": len(records),
                      "positives": sum(r["label"] for r in records),
                      "rejected_files": len(errors), "first_errors": errors[:5]}, indent=2))


if __name__ == "__main__":
    main()
