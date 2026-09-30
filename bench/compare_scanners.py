#!/usr/bin/env python3
"""Head-to-head: Harpocrates vs TruffleHog vs gitleaks on the same labeled records.

    python bench/compare_scanners.py data/eval/benchmark_llm_v3.jsonl --model-dir data/models/cand_B_all_llm

Each record becomes one file (context + line, real extension) so language-aware
rules apply. A scanner "flags" a record only with a finding on the record's own
line whose value overlaps the labeled token, the same rule Harpocrates is held
to in bench/eval_detector.py. TruffleHog runs with --no-verification (all values
are fake). Prints JSON with overall and per-type results plus, per scanner, the
positives only it caught. Never prints token values.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))

from bench.eval_detector import _onnx_verifier, evaluate  # noqa: E402


def overlaps(found: str, token: str) -> bool:
    return bool(found) and len(found) >= 4 and (found in token or token in found)


def materialize(records: list[dict], root: Path) -> list[tuple[str, int]]:
    """Write one file per record; return (relative file name, 1-based target line) per record."""
    placed = []
    for i, r in enumerate(records):
        suffix = r.get("file_type") or Path(r.get("file_path") or "").suffix or ".txt"
        name = f"r{i:06d}{suffix if suffix.startswith('.') else '.txt'}"
        before = r.get("context_before", [])
        text = "\n".join([*before, r["line_content"], *r.get("context_after", [])]) + "\n"
        (root / name).write_text(text)
        placed.append((name, len(before) + 1))
    return placed


def _hits(findings: dict[tuple[str, int], list[str]], records, placed) -> list[bool]:
    return [any(overlaps(v, r["token"]) for v in findings.get(p, [])) for r, p in zip(records, placed)]


def run_trufflehog(root: Path, records, placed) -> list[bool]:
    out = subprocess.run(["trufflehog", "filesystem", str(root), "--json", "--no-verification", "--no-update"],
                         capture_output=True, text=True, check=True).stdout
    found: dict[tuple[str, int], list[str]] = defaultdict(list)
    for line in out.splitlines():
        if not line.startswith("{"):
            continue  # log lines, not findings
        f = json.loads(line)
        meta = f["SourceMetadata"]["Data"]["Filesystem"]
        found[(Path(meta["file"]).name, meta.get("line", 0))] += [f.get("Raw", ""), f.get("RawV2", "")]
    return _hits(found, records, placed)


def run_gitleaks(root: Path, records, placed) -> list[bool]:
    with tempfile.NamedTemporaryFile(suffix=".json") as report:
        subprocess.run(["gitleaks", "dir", str(root), "--report-format", "json", "--report-path", report.name,
                        "--no-banner", "--exit-code", "0", "--log-level", "error"], check=True)
        results = json.loads(Path(report.name).read_text() or "[]")
    found: dict[tuple[str, int], list[str]] = defaultdict(list)
    for f in results:
        # Secret only: Match includes surrounding text and would make overlap lenient to gitleaks.
        found[(Path(f["File"]).name, f["StartLine"])].append(f.get("Secret", ""))
    return _hits(found, records, placed)


def run_harpocrates(root: Path, records, placed, model_dir, ml_threshold) -> tuple[list[bool], list[bool]]:
    """Scan each file as `harpocrates scan --ml` does. Also returns candidate coverage: whether any
    raw candidate (before ML) overlaps the labeled token, the recall ceiling for any threshold."""
    from Harpocrates.core.detector import _collect_file_findings, detect_file_with_ml

    verifier = _onnx_verifier(model_dir)
    flagged, covered = [], []
    for r, (name, line) in zip(records, placed):
        path = root / name
        on_line = lambda fs: [f for f in fs if f.line == line and overlaps(f.token or "", r["token"])]  # noqa: E731
        flagged.append(bool(on_line(detect_file_with_ml(path, verifier, ml_threshold=ml_threshold))))
        covered.append(bool(on_line(_collect_file_findings(path, None))))
    return flagged, covered


def summarize(records: list[dict], flags: list[bool]) -> dict:
    lookup = {id(r): f for r, f in zip(records, flags)}
    by_type = defaultdict(list)
    for r in records:
        by_type[r.get("secret_type", "unknown")].append(r)
    return {**evaluate(records, lambda r: lookup[id(r)]),
            "by_type": {k: evaluate(v, lambda r: lookup[id(r)])["recall"] for k, v in sorted(by_type.items())
                        if any(x["label"] == 1 for x in v)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path, help="candidate Harpocrates model (default: shipped)")
    parser.add_argument("--ml-threshold", type=float, default=0.19)
    args = parser.parse_args()
    records = [json.loads(line) for p in args.paths for line in p.read_text().splitlines() if line.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        placed = materialize(records, root)
        flags = {"trufflehog": run_trufflehog(root, records, placed),
                 "gitleaks": run_gitleaks(root, records, placed)}
        flags["harpocrates"], covered = run_harpocrates(root, records, placed, args.model_dir, args.ml_threshold)

    positives = [c for r, c in zip(records, covered) if r["label"] == 1]
    report = {"data": [str(p) for p in args.paths], "model_dir": str(args.model_dir) if args.model_dir else "shipped",
              "harpocrates_candidate_coverage": round(sum(positives) / max(len(positives), 1), 4),
              "scanners": {name: summarize(records, f) for name, f in flags.items()}, "only": {}}
    for name, f in flags.items():
        others = [g for n, g in flags.items() if n != name]
        only = [r.get("secret_type", "unknown") for i, r in enumerate(records)
                if r["label"] == 1 and f[i] and not any(g[i] for g in others)]
        report["only"][name] = {"count": len(only), "by_type": dict(sorted(
            {t: only.count(t) for t in set(only)}.items(), key=lambda kv: -kv[1]))}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
