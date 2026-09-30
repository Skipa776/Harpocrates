#!/usr/bin/env python3
"""Train a candidate v1.1 detector and report whether it meets the target.

    python scripts/train_v11.py --train data/synthetic_v4_clean.jsonl data/eval/train_v1.jsonl

Target: recall >= 0.90 with precision 0.80-0.85 (0.85 ideal).
Validation mixes generators so the threshold isn't tuned to one generator's quirks:
val_v1 (OSS-insert generator) + a fixed 10% holdout of every other training file.
The threshold is the highest one reaching recall >= 0.90 on that mix. Test sets are
scored once with that threshold; nothing is tuned on them (DATA-09). The model is
saved under data/models/ (git-ignored); the shipped model is not touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))

from Harpocrates.core.detector import _collect_text_findings, _prepare_ml_context_from_lines
from Harpocrates.core.result import EvidenceType
from Harpocrates.ml.features import extract_features, extract_features_from_record

sys.path.insert(0, str(ROOT))
from bench.compare_scanners import overlaps  # noqa: E402

TARGET_RECALL = 0.90
# Shipped v0.4 hyperparameters (cli/Harpocrates/training/train_model.py) so only data changes.
PARAMS = dict(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.7,
              reg_alpha=0.5, reg_lambda=3.0, random_state=42, eval_metric="logloss")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def features(records: list[dict]) -> np.ndarray:
    return np.array([extract_features_from_record(r).to_array() for r in records], dtype=np.float32)


def pipeline_rows(records: list[dict]) -> list[tuple[list[float], int, str]]:
    """(features, label, candidate token) for every ML candidate the scanner extracts on each
    record's own line, featurized exactly as the verifier does at scan time. A candidate is
    positive only if the record is and the candidate overlaps the labeled token."""
    rows = []
    for r in records:
        before = r.get("context_before", [])
        lines = [*before, r["line_content"], *r.get("context_after", [])]
        target = len(before) + 1
        file = "record" + (r.get("file_type") or "")
        for f in _collect_text_findings("\n".join(lines)):
            if f.line != target or f.evidence == EvidenceType.REGEX or not f.token:
                continue
            f = dataclasses.replace(f, file=file)  # same file-type signal a file scan carries
            x = extract_features(f, _prepare_ml_context_from_lines(f, lines)).to_array()
            hit = r["label"] == 1 and overlaps(f.token, r["token"])  # same rule as bench/compare_scanners.py
            rows.append((x, int(hit), f.token))
    return rows


CACHE = ROOT / "data" / "models" / "feature_cache"
# Code that determines features: the whole package plus this script; editing any of it invalidates the cache.
_FEATURE_CODE = sorted((ROOT / "cli/Harpocrates").rglob("*.py")) + [Path(__file__)]


def xy(records: list[dict], pipeline: bool) -> tuple[np.ndarray, np.ndarray]:
    """Features/labels for records, cached on disk by content + feature code (pipeline mode is slow)."""
    h = hashlib.sha256(str(pipeline).encode())
    for path in _FEATURE_CODE:
        h.update(path.read_bytes())
    for r in records:
        h.update(json.dumps(r, sort_keys=True).encode())
    cached = CACHE / f"{h.hexdigest()[:24]}.npz"
    if cached.exists():
        data = np.load(cached)
        return data["X"], data["y"]
    if pipeline:
        rows = pipeline_rows(records)
        X = np.array([x for x, _y, _t in rows], dtype=np.float32)
        y = np.array([y for _x, y, _t in rows])
    else:
        X, y = features(records), np.array([r["label"] for r in records])
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_name(cached.stem + ".partial.npz")
    np.savez(tmp, X=X, y=y)
    tmp.replace(cached)  # atomic: an interrupted run never leaves a half-written cache file
    return X, y


def is_holdout(record: dict) -> bool:
    key = f'{record["token"]}|{record["line_content"]}'.encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % 10 == 0


def scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {"records": len(y), "auc": round(roc_auc_score(y, p), 4) if pos and neg else None,
            "recall": round(tp / pos, 4) if pos else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp and neg else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=ROOT / "data/eval/val_v3.jsonl")
    parser.add_argument("--test", nargs="*", type=Path,
                        default=[ROOT / "data/eval/test_v3.jsonl", ROOT / "data/trufflehog_golden.jsonl"])
    parser.add_argument("--out", type=Path, default=ROOT / "data/models/v11_candidate.json")
    parser.add_argument("--pipeline-features", action="store_true",
                        help="train/score on the scanner's own ML candidates instead of record features")
    parser.add_argument("--target-recall", type=float, nargs="+", default=[TARGET_RECALL],
                        help="val recall(s) a threshold must reach (candidate-level with --pipeline-features); "
                             "one model, one threshold per value, e.g. 0.97 (gate) 0.90 (commit)")
    args = parser.parse_args()

    train, val = [], load(args.val)
    for path in args.train:
        for r in load(path):
            # train_vN is repo-split (its val is val_vN); other files donate a fixed 10% holdout.
            (val if not path.name.startswith("train_v") and is_holdout(r) else train).append(r)
    # The same (token, line) can recur across repos; never score a row the model trained on.
    train_keys = {(r["token"], r["line_content"]) for r in train}
    unseen = lambda rs: [r for r in rs if (r["token"], r["line_content"]) not in train_keys]  # noqa: E731
    val_all = len(val)
    val = unseen(val)
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    log(f"features: train ({len(train)} records)")
    X_train, y_train = xy(train, args.pipeline_features)
    log(f"features: val ({len(val)} records)")
    X_val, y_val = xy(val, args.pipeline_features)
    log("training")

    model = xgb.XGBClassifier(**PARAMS).fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    chosen = {}
    for target in args.target_recall:
        reach = np.flatnonzero(recall[:-1] >= target)
        if reach.size == 0:
            sys.exit(f"no threshold reaches recall >= {target} on val; best is {recall[:-1].max():.3f}")
        chosen[str(target)] = float(thresholds[reach[-1]])  # highest threshold still meeting the target
    threshold = chosen[str(args.target_recall[0])]

    report = {"train_files": [str(p) for p in args.train], "train_records": len(train),
              "unit": "scanner candidate" if args.pipeline_features else "record", "train_rows": len(y_train),
              "note": ("candidate-level recall: secrets the scanner never extracts are not counted; "
                       "end-to-end recall comes from bench/compare_scanners.py") if args.pipeline_features else "",
              "train_positive_share": round(float(y_train.mean()), 3),
              "threshold": round(threshold, 4), "thresholds": {k: round(v, 4) for k, v in chosen.items()},
              "val_dropped_seen_in_train": val_all - len(val),
              "val": {k: scores(y_val, p_val, v) for k, v in chosen.items()}, "test": {}}
    for path in args.test:
        if path.exists():
            loaded = load(path)
            rs = unseen(loaded)
            log(f"features: test {path.name}")
            X, y = xy(rs, args.pipeline_features)
            p = model.predict_proba(X)[:, 1]
            report["test"][path.name] = {"dropped_seen_in_train": len(loaded) - len(rs),
                                         **{k: scores(y, p, v) for k, v in chosen.items()}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(args.out)
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
