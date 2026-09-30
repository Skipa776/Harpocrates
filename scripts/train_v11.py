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
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))

from Harpocrates.ml.features import extract_features_from_record

TARGET_RECALL = 0.90
# Shipped v0.4 hyperparameters (cli/Harpocrates/training/train_model.py) so only data changes.
PARAMS = dict(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.7,
              reg_alpha=0.5, reg_lambda=3.0, random_state=42, eval_metric="logloss")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def features(records: list[dict]) -> np.ndarray:
    return np.array([extract_features_from_record(r).to_array() for r in records], dtype=np.float32)


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
    y_train = np.array([r["label"] for r in train])
    y_val = np.array([r["label"] for r in val])

    model = xgb.XGBClassifier(**PARAMS).fit(features(train), y_train)
    p_val = model.predict_proba(features(val))[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    reach = np.flatnonzero(recall[:-1] >= TARGET_RECALL)
    if reach.size == 0:
        sys.exit(f"no threshold reaches recall >= {TARGET_RECALL} on val; best is {recall[:-1].max():.3f}")
    threshold = float(thresholds[reach[-1]])  # highest threshold still meeting the recall target

    report = {"train_files": [str(p) for p in args.train], "train_records": len(train),
              "train_positive_share": round(float(y_train.mean()), 3),
              "threshold": round(threshold, 4), "val_dropped_seen_in_train": val_all - len(val),
              "val": scores(y_val, p_val, threshold), "test": {}}
    for path in args.test:
        if path.exists():
            loaded = load(path)
            rs = unseen(loaded)
            y = np.array([r["label"] for r in rs])
            report["test"][path.name] = {**scores(y, model.predict_proba(features(rs))[:, 1], threshold),
                                         "dropped_seen_in_train": len(loaded) - len(rs)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(args.out)
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
