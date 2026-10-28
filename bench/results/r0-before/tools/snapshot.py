#!/usr/bin/env python3
"""FR-CLEAN-02 BEFORE/AFTER findings snapshots. Deterministic: sorted, no timestamps, token values sha256-hashed.

    .venv/bin/python bench/results/r0-before/tools/snapshot.py holdout  OUT_DIR
    .venv/bin/python bench/results/r0-before/tools/snapshot.py evalsets OUT_DIR
    .venv/bin/python bench/results/r0-before/tools/snapshot.py oss      OUT_DIR
    .venv/bin/python bench/results/r0-before/tools/snapshot.py canary   OUT_DIR

Reuses (imports, never edits) bench/eval_detector.py, bench/compare_scanners.py, bench/canary_repo.py and
scripts/build_eval_set.py. Layers: commit = OnnxVerifier(layer="commit") like `scan` on the CLI; gate =
OnnxVerifier(layer="gate") like the read tool. Never prints or stores raw token values.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

COMMIT_T, GATE_T = 0.5559, 0.1904  # model_config.json thresholds.commit / thresholds.gate


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def verifier(layer: str):
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    return OnnxVerifier(layer=layer, lazy_load=False)


def dump_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def finding_row(f, base: Path) -> dict:
    rel = Path(f.file).resolve().relative_to(base.resolve()).as_posix() if f.file else None
    return {"file": rel, "line": f.line, "type": f.type, "evidence": f.evidence.value, "severity": f.severity.value,
            "confidence": None if f.confidence is None else round(f.confidence, 5), "category": f.category,
            "category_reason": f.category_reason, "in_comment": f.in_comment,
            "token_sha256_16": sha(f.token or ""), "token_len": len(f.token or ""),
            "token_start": f.token_start, "token_end": f.token_end}


def sort_rows(rows):
    return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))


# ---------------------------------------------------------------- holdout (bench/eval_detector.py default set)
def holdout(out: Path) -> None:
    from bench.eval_detector import HOLDOUT, evaluate
    from Harpocrates.core.detector import detect_text_with_ml
    from Harpocrates.ml.features import extract_features_from_record

    records = [json.loads(line) for line in HOLDOUT.read_text().splitlines() if line.strip()]
    vc, vg = verifier("commit"), verifier("gate")
    # name -> (verifier, ml_threshold); ml_threshold None = ML stage alone (verifier._route)
    configs = {"ml_gate_0.1904": (vg, None), "ml_commit_0.5559": (vc, None),
               "pipeline_gate_0.1904": (vg, GATE_T), "pipeline_commit_0.5559": (vc, COMMIT_T),
               "pipeline_cli_default_0.19": (vc, 0.19)}
    rows, flags = [], {n: [] for n in configs}
    for i, r in enumerate(records):
        row = {"i": i, "secret_type": r.get("secret_type"), "label": r["label"], "token_sha256_16": sha(r["token"])}
        before = r.get("context_before", [])
        text = "\n".join([*before, r["line_content"], *r.get("context_after", [])])
        for name, (v, thr) in configs.items():
            if thr is None:
                flagged, prob, route = v._route(extract_features_from_record(r))
                row[name] = {"flagged": bool(flagged), "p": round(float(prob), 6), "route": route}
            else:
                hit = [f for f in detect_text_with_ml(text, v, ml_threshold=thr)
                       if f.line == len(before) + 1 and f.token and (r["token"] in f.token or f.token in r["token"])]
                flagged = bool(hit)
                row[name] = {"flagged": flagged, "type": hit[0].type if hit else None,
                             "confidence": round(hit[0].confidence, 5) if hit and hit[0].confidence is not None else None}
            flags[name].append(flagged)
        rows.append(row)
    dump_jsonl(out / "holdout_findings.jsonl", rows)
    summary = {}
    for n in configs:
        look = {id(r): f for r, f in zip(records, flags[n])}
        summary[n] = evaluate(records, lambda r: look[id(r)])
    dump_json(out / "holdout_summary.json", {"data": "cli/tests/fixtures/positive_holdout_v4.jsonl", "results": summary})


# ---------------------------------------------------------------- labeled eval sets, full-file scan (compare_scanners unit)
def evalsets(out: Path) -> None:
    from bench.compare_scanners import materialize, overlaps
    from bench.eval_detector import evaluate
    from Harpocrates.core.detector import _collect_file_findings, detect_file_with_ml

    layers = {"gate": verifier("gate"), "commit": verifier("commit")}
    # (layer, ml_threshold) combos tried against model_config.json metrics; all kept in the snapshot
    combos = {"gate_ml0.0": ("gate", 0.0), "gate_ml0.1904": ("gate", GATE_T), "gate_ml0.19": ("gate", 0.19),
              "commit_ml0.0": ("commit", 0.0), "commit_ml0.19": ("commit", 0.19), "commit_ml0.5559": ("commit", COMMIT_T)}
    summary, rows_all = {}, []
    for name in ("test_v5", "benchmark_llm_v5"):
        records = [json.loads(line) for line in (ROOT / f"data/eval/{name}.jsonl").read_text().splitlines() if line.strip()]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            placed = materialize(records, root)
            on_line = lambda fs, r, line: [f for f in fs if f.line == line and overlaps(f.token or "", r["token"])]  # noqa: E731
            covered = [bool(on_line(_collect_file_findings(root / n, None), r, ln)) for r, (n, ln) in zip(records, placed)]
            flags = {c: [] for c in combos}
            for r, (n, ln) in zip(records, placed):
                for c, (layer, thr) in combos.items():
                    flags[c].append(bool(on_line(detect_file_with_ml(root / n, layers[layer], ml_threshold=thr), r, ln)))
        for i, r in enumerate(records):
            rows_all.append({"set": name, "i": i, "secret_type": r.get("secret_type"), "label": r["label"],
                             "token_sha256_16": sha(r["token"]), "candidate_covered": covered[i],
                             "flags": {c: flags[c][i] for c in combos}})
        pos = [c for r, c in zip(records, covered) if r["label"] == 1]
        summary[name] = {"records": len(records), "candidate_coverage": round(sum(pos) / max(len(pos), 1), 4)}
        for c in combos:
            look = {id(r): f for r, f in zip(records, flags[c])}
            res = evaluate(records, lambda r: look[id(r)])
            summary[name][c] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
    cfg = json.loads((ROOT / "cli/Harpocrates/ml/models/model_config.json").read_text())["metrics"]
    dump_jsonl(out / "evalsets_records.jsonl", rows_all)
    dump_json(out / "evalsets_summary.json", {"model_config_metrics": cfg, "measured": summary})


# ---------------------------------------------------------------- OSS test-split repos, CLI scan semantics
def oss(out: Path) -> None:
    import build_eval_set as b

    from Harpocrates.core.scanner import scan_directory

    repos = sorted(r["repo"] for r in csv.DictReader(open(ROOT / "scripts/oss_repos.tsv"), delimiter="\t")
                   if b.split_of(r["repo"]) == "test")
    summary = {"split": "test", "repos": len(repos), "layers": {}}
    for layer, thr in (("commit", 0.19), ("gate", 0.0)):
        v = verifier(layer)
        rows, files, lines, errors = [], 0, 0, []
        for repo in repos:
            d = b.CORPUS / repo.replace("/", "__")
            if not d.is_dir():
                errors.append(f"{repo}: missing")
                continue
            res = scan_directory(d, verifier=v, ml_threshold=thr)
            files += res.scanned_files
            lines += res.total_lines
            errors += [f"{repo}: {e}" for e in res.errors]
            rows += [{"repo": repo, **finding_row(f, d)} for f in res.findings]
        rows = sort_rows(rows)
        dump_jsonl(out / f"oss_findings_{layer}.jsonl", rows)
        summary["layers"][layer] = {
            "ml_threshold": thr, "files": files, "lines": lines, "findings": len(rows),
            "findings_per_1k_files": round(1000 * len(rows) / max(files, 1), 2),
            "by_evidence": {e: sum(r["evidence"] == e for r in rows) for e in sorted({r["evidence"] for r in rows})},
            "errors": sorted(errors)}
    dump_json(out / "oss_summary.json", summary)


# ---------------------------------------------------------------- canary repo (bench/canary_repo.py --seed 0)
def canary(out: Path) -> None:
    import subprocess

    from bench.canary_repo import build
    from Harpocrates.core.scanner import scan_directory

    with tempfile.TemporaryDirectory(prefix="canary_") as tmp:
        repo = Path(tmp) / "repo"
        manifest = build(repo, seed=0)
        gen = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(repo.rglob("*")) if p.is_file()}
        mrows = [{"file": m["file"], "kind": m["kind"], "value_sha256_16": sha(m["value"])} for m in manifest]
        summary = {"seed": 0, "generated_file_sha256": gen, "manifest": mrows, "layers": {}}
        for layer, thr in (("commit", 0.19), ("gate", 0.0)):
            res = scan_directory(repo, verifier=verifier(layer), ml_threshold=thr)
            rows = sort_rows([finding_row(f, repo) for f in res.findings])
            dump_jsonl(out / f"canary_findings_{layer}.jsonl", rows)
            hit = sorted({m["kind"] for m in manifest for f in res.findings
                          if f.file and Path(f.file).resolve().relative_to(repo.resolve()).as_posix() == m["file"]
                          and f.token and (m["value"] in f.token or f.token in m["value"])})
            summary["layers"][layer] = {"findings": len(rows), "canary_kinds_found": hit,
                                        "canary_found": len(hit), "canary_total": len(manifest)}
        cli = subprocess.run([str(ROOT / ".venv/bin/harpocrates"), "scan", str(repo), "--json", "--fail-on", "none"],
                             capture_output=True, text=True)
        try:
            j = json.loads(cli.stdout)
            n = len(j.get("findings", [])) if isinstance(j, dict) else len(j)
        except Exception:
            n = f"unparsed(exit={cli.returncode})"
        summary["cli_scan_json_findings"] = n
        summary["cli_exit"] = cli.returncode
    dump_json(out / "canary_summary.json", summary)


if __name__ == "__main__":
    cmd, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    {"holdout": holdout, "evalsets": evalsets, "oss": oss, "canary": canary}[cmd](outdir)
