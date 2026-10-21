#!/usr/bin/env python3
"""False-positive check for FR-GATE-11 (docs/gate-base64-text-spec.md): replay every recorded request through the
encoded-text rule alone and count the requests it would block.

    python -m bench.gate_encoded_fp [--recordings bench/results/mock] [--out bench/results/gate/encoded-fp-<date>.json]

Each unique request body on a scan route is walked with the gate's own walkers; every class R and S string goes through
encoded.decoded_texts, and each decoding through the gate's detector (Finder, gate threshold). The report holds counts
and, for each request the rule would block, the recording folder, request index, route, path prefix and finding types:
never a value or decoded text. Rule 5: these recordings measure the rule; they are never used to tune it.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))

from Harpocrates.gate import encoded, walk_anthropic, walk_responses  # noqa: E402
from Harpocrates.gate.blocks import RefusalError  # noqa: E402
from Harpocrates.gate.finder import Finder  # noqa: E402
from Harpocrates.gate.jsonspan import BodyParseError, parse_strict  # noqa: E402
from Harpocrates.gate.walk_common import Leaf  # noqa: E402

SCAN_ROUTES = ("anthropic.messages", "anthropic.count_tokens", "openai.responses")


def first_hit(leaves: list[Leaf], finder: Finder) -> Optional[dict]:
    """
    Description: Find the first string whose base64 decodings hold a finding.
    Parameters:
        leaves: the request's class R and S strings.
        finder: the gate's finder.
    Output: {"path": first four path parts, "types": finding types, "decoded_chars": n}, or None.
    """
    for leaf in leaves:
        for text in encoded.strings_of(leaf.text, leaf.as_json):
            for decoded in encoded.decoded_texts(text):
                spans = finder.spans(decoded, "B")
                if spans:
                    return {"path": [str(p) for p in leaf.path[:4]], "types": sorted({s[2] for s in spans}),
                            "decoded_chars": len(decoded)}
    return None


def check(recordings: Path, finder: Optional[Finder] = None) -> dict:
    """
    Description: Replay every recorded request through the encoded-text rule.
    Parameters:
        recordings: the folder of mock recordings (requests.jsonl files, at any depth).
        finder: the finder to use, or None for the gate's default.
    Output: {"counts": {...}, "would_block": [...]} with counts of requests, duplicates, unwalkable bodies, runs and
        text decodings, and one entry per request the rule would block.
    """
    finder = finder or Finder()
    counts: collections.Counter = collections.Counter()
    blocks, seen = [], set()
    for path in sorted(recordings.rglob("requests.jsonl")):
        folder = str(path.parent.relative_to(recordings))
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            route = rec.get("route") or ""
            if route not in SCAN_ROUTES:
                continue
            body = (rec.get("body") or "").encode("utf-8")
            digest = hashlib.sha256(body).digest()
            if digest in seen:
                counts["duplicates"] += 1
                continue
            seen.add(digest)
            try:
                _, tree = parse_strict(body)
                walk = walk_anthropic.walk if route.startswith("anthropic.") else walk_responses.walk
                leaves = [op for op in walk(tree) if isinstance(op, Leaf)]
            except (RefusalError, BodyParseError):
                counts["not_walkable"] += 1
                continue
            counts["requests"] += 1
            for leaf in leaves:
                for text in encoded.strings_of(leaf.text, leaf.as_json):
                    counts["runs"] += sum(1 for _ in encoded.runs(text))
                    counts["text_decodings"] += len(encoded.decoded_texts(text))
            hit = first_hit(leaves, finder)
            if hit is not None:
                blocks.append({"recording": folder, "index": rec.get("index"), "route": route, **hit})
    return {"counts": dict(counts), "would_block": blocks}


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Run the check and write the counts-only report.
    Parameters:
        argv: arguments, or None for sys.argv.
    Output: Exit status 0. Prints the counts and each would-block entry.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, default=ROOT / "bench" / "results" / "mock")
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "gate" / f"encoded-fp-{date.today()}.json")
    args = parser.parse_args(argv)
    report = {"requirement": "FR-GATE-11", "min_run": encoded.MIN_RUN, **check(args.recordings)}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["counts"])
    for entry in report["would_block"]:
        print(" ", entry)
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
