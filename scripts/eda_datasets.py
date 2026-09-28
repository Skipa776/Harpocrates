#!/usr/bin/env python3
"""Compare the structure and label quality of secret-detection datasets.

    python scripts/eda_datasets.py data/synthetic_v4.jsonl data/eval/train_v1.jsonl > eda.md

Prints markdown. Only aggregate statistics and redacted examples; no raw tokens.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))

from Harpocrates.detectors.entropy_detector import shannon_entropy
from Harpocrates.detectors.regex_patterns import SIGNATURES

SECRET_WORDS = re.compile(r"key|secret|token|passw|pwd|auth|credential|api", re.I)
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z_]*$")          # letters/underscores only
# URL scheme, leading slash, or a path ending in a file extension / starting at a domain.
URL_OR_PATH = re.compile(r"^(https?:|//|/|\./)|^[\w.-]+(/[\w.-]+)*\.[a-z]{1,5}$|^[a-z0-9-]+\.[a-z]{2,}/")
NUMBERISH = re.compile(r"^\+?[\d\s().-]+$")
EXT_SYNTAX = {  # line syntax that should not appear in a file of this type
    ".py": re.compile(r"(\bconst |:= |;\s*$|\bvar \w+ =|\bString \w+ =)"),
    ".go": re.compile(r"(\bdef |\bconst \w+ = \"|;\s*$)"),
    ".js": re.compile(r"(\bdef |:= )"), ".ts": re.compile(r"(\bdef |:= )"),
    ".java": re.compile(r"(\bdef |:= |\bconst )"),
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def _skeleton(record: dict) -> str:
    line = record["line_content"].replace(record["token"], "<T>")
    return re.sub(r"\d+", "0", line.strip())


def _looks_non_secret(token: str) -> str | None:
    if URL_OR_PATH.match(token):
        return "url/path"
    if NUMBERISH.match(token):
        return "number/phone"
    if IDENTIFIER.match(token):
        return "identifier/word"
    if len(token) < 8:
        return "too short (<8)"
    return None


def _provider_format(token: str) -> bool:
    return any(pattern.search(token) for pattern in SIGNATURES.values())


def profile(path: Path) -> dict:
    rs = _load(path)
    pos = [r for r in rs if r["label"] == 1]
    neg = [r for r in rs if r["label"] == 0]
    skeletons = Counter(_skeleton(r) for r in rs)
    token_labels: dict[str, set] = {}
    for r in rs:
        token_labels.setdefault(r["token"], set()).add(r["label"])
    kw_pos = sum(bool(SECRET_WORDS.search(r["line_content"].replace(r["token"], ""))) for r in pos)
    kw_neg = sum(bool(SECRET_WORDS.search(r["line_content"].replace(r["token"], ""))) for r in neg)
    mismatched = sum(
        bool(EXT_SYNTAX[Path(r.get("file_path") or "").suffix].search(r["line_content"]))
        for r in rs if Path(r.get("file_path") or "").suffix in EXT_SYNTAX
    )
    typed = sum(Path(r.get("file_path") or "").suffix in EXT_SYNTAX for r in rs)

    def tok_stats(group: list[dict]) -> dict:
        lengths = [len(r["token"]) for r in group] or [0]
        return {
            "median_len": statistics.median(lengths),
            "mean_entropy": round(statistics.mean(shannon_entropy(r["token"]) for r in group), 2) if group else 0,
            "provider_format": _pct(sum(_provider_format(r["token"]) for r in group), len(group)),
        }

    return {
        "name": path.name, "records": len(rs), "positives": len(pos), "negatives": len(neg),
        "sources": Counter((r.get("source"), r["label"]) for r in rs),
        "secret_types": len({r.get("secret_type") for r in rs}),
        "extensions": Counter(Path(r.get("file_path") or "none").suffix or "none" for r in rs),
        "token_in_line": _pct(sum(r["token"] in r["line_content"] for r in rs), len(rs)),
        "with_context": _pct(sum(bool(r.get("context_before")) for r in rs), len(rs)),
        "unique_tokens": _pct(len(token_labels), len(rs)),
        "conflicting_tokens": sum(len(labels) > 1 for labels in token_labels.values()),
        "exact_dupes": len(rs) - len({(r["token"], r["line_content"]) for r in rs}),
        "unique_skeletons": _pct(len(skeletons), len(rs)),
        "top10_skeleton_share": _pct(sum(c for _, c in skeletons.most_common(10)), len(rs)),
        "pos_non_secret": Counter(filter(None, (_looks_non_secret(r["token"]) for r in pos))),
        "neg_provider_format": sum(_provider_format(r["token"]) for r in neg),
        "kw_rate_pos": _pct(kw_pos, len(pos)), "kw_rate_neg": _pct(kw_neg, len(neg)),
        "p_secret_given_kw": _pct(kw_pos, kw_pos + kw_neg),
        "lang_mismatch": _pct(mismatched, typed),
        "pos_tokens": tok_stats(pos), "neg_tokens": tok_stats(neg),
    }


def main() -> None:
    profiles = [profile(Path(p)) for p in sys.argv[1:]]
    rows = [
        ("Records", "records"), ("Positives", "positives"), ("Negatives", "negatives"),
        ("Distinct secret_type values", "secret_types"), ("Token appears in line", "token_in_line"),
        ("Has context_before", "with_context"), ("Unique tokens", "unique_tokens"),
        ("Tokens with conflicting labels", "conflicting_tokens"), ("Exact duplicate records", "exact_dupes"),
        ("Unique line skeletons", "unique_skeletons"), ("Top-10 skeletons' share", "top10_skeleton_share"),
        ("Negatives in a provider format", "neg_provider_format"),
        ("Secret word on line: positives", "kw_rate_pos"), ("Secret word on line: negatives", "kw_rate_neg"),
        ("P(secret | secret word on line)", "p_secret_given_kw"),
        ("Line syntax mismatches file type", "lang_mismatch"),
    ]
    print("| Metric | " + " | ".join(p["name"] for p in profiles) + " |")
    print("| --- |" + " --- |" * len(profiles))
    for label, key in rows:
        print(f"| {label} | " + " | ".join(str(p[key]) for p in profiles) + " |")
    for side in ("pos_tokens", "neg_tokens"):
        for stat in ("median_len", "mean_entropy", "provider_format"):
            print(f"| {side.split('_')[0]} token {stat} | " + " | ".join(str(p[side][stat]) for p in profiles) + " |")
    for p in profiles:
        print(f"\n**{p['name']}**")
        print(f"- Sources (source, label): {dict(p['sources'])}")
        print(f"- Top extensions: {p['extensions'].most_common(8)}")
        print(f"- Positives that look like non-secrets: {dict(p['pos_non_secret'])} "
              f"(total {sum(p['pos_non_secret'].values())} of {p['positives']})")


if __name__ == "__main__":
    main()
