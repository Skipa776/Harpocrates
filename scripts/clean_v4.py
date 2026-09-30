#!/usr/bin/env python3
"""Clean the 40k synthetic training set (see EDA_REPORT.md).

    python scripts/clean_v4.py data/synthetic_v4.jsonl data/synthetic_v4_clean.jsonl

Rules, in order:
1. Drop exact duplicates of (token, line_content).
2. Relabel positives that are clearly not secrets (identifiers, words, URLs, paths,
   numbers) as negatives; they become hard negatives. Marked relabeled/original_label.
3. Drop every record whose token still appears with both labels (we can't tell which is right).
4. Tag LLM rows as llm_synthetic (DATA-08).

Round 2 (2026-09-30, bench/model_improvement.ipynb section 12): also relabel word paths
(America/New_York, actions/checkout) and function names in call position, and drop hex/base64
positives in hash-named variables, which may be content hashes or HMAC keys.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda_datasets import _looks_non_secret

# Word path: every /-separated segment is words or numbers joined by _ - . (no random mixed-case runs).
_WORD_PIECE = re.compile(r"(?:[a-z]+|[A-Z]+|[A-Z][a-z]+)\d*|\d+")


def _word_path(token: str) -> bool:
    segments = token.split("/")
    return len(segments) > 1 and all(
        seg and all(_WORD_PIECE.fullmatch(piece) for piece in re.split(r"[_.-]", seg)) for seg in segments)


HASH_NAME = re.compile(r"(?i)hash|checksum|digest|integrity|\bsri\b|sha\d*\b")
HEX_OR_B64 = re.compile(r"^(?:[0-9a-fA-F]{16,}|[A-Za-z0-9+/_-]{16,}={0,2})$")


def _context_non_secret(record: dict) -> str | None:
    token, line = record["token"], record["line_content"]
    if _word_path(token):
        return "word path"
    if re.fullmatch(r"[A-Za-z_][\w:]*", token) and re.search(rf"(?<![\w'\"]){re.escape(token)}\s*(?:\(|->)", line):
        return "function call"
    return None


def _ambiguous(record: dict) -> bool:
    """Hex/base64 positive assigned to a hash-named identifier (only the name it's assigned to counts)."""
    before = record["line_content"].split(record["token"])[0]
    name = re.search(r"([\w.$-]+)[\"'\]]*\s*(?::=|=>|->|[:=(,])\s*[\"'`]?\s*$", before)
    return bool(name and HEX_OR_B64.match(record["token"]) and HASH_NAME.search(name.group(1)))


def clean(records: list[dict]) -> tuple[list[dict], dict]:
    seen, unique = set(), []
    for r in records:
        key = (r["token"], r["line_content"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    relabeled_records, relabeled = [], 0
    for r in unique:
        reason = (_looks_non_secret(r["token"]) or _context_non_secret(r)) if r["label"] == 1 else None
        source = "llm_synthetic" if r.get("source") == "llm" else r.get("source")
        extra = {"relabeled": True, "original_label": 1, "relabel_reason": reason, "label": 0} if reason else {}
        relabeled += bool(reason)
        relabeled_records.append({**r, "source": source, "generator": "synthetic_v4", **extra})

    # Conflicts are checked after relabeling, so an identifier labeled both ways becomes a clean negative.
    labels = defaultdict(set)
    for r in relabeled_records:
        labels[r["token"]].add(r["label"])
    consistent = [r for r in relabeled_records if len(labels[r["token"]]) == 1]
    out = [r for r in consistent if not (r["label"] == 1 and _ambiguous(r))]

    stats = {"input": len(records), "duplicates": len(records) - len(unique),
             "conflicting_dropped": len(unique) - len(consistent),
             "ambiguous_dropped": len(consistent) - len(out), "relabeled": relabeled,
             "output": len(out)}
    return out, stats


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    records = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    out, stats = clean(records)
    dst.write_text("".join(json.dumps(r) + "\n" for r in out))
    labels = [r["label"] for r in out]
    print(json.dumps({**stats, "positives": sum(labels), "negatives": len(labels) - sum(labels)}, indent=2))


if __name__ == "__main__":
    main()
