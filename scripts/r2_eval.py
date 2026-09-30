#!/usr/bin/env python3
"""Round-2 evaluation of one (detector, data, params) configuration on validation evidence only.

    python scripts/r2_eval.py LABEL --data r1|r2 [--depth 5 --trees 300]

Reports: record-level recall at 95% / 90% precision on val (all, and clean generators only), the
candidate ceiling, leave-one-LLM-family-out recall, and ML alarms per 1,000 real files in the
val-split OSS repos at the val-chosen thresholds. Writes data/models/r2/<LABEL>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_common as C  # noqa: E402
import train_v11 as T  # noqa: E402

ZERO = [C.NAMES.index(n) for n in json.loads((C.ROOT / "data/models/v16.report.json").read_text())["zeroed_features"]]
MONO = "(" + ",".join(str(T.MONOTONE.get(n, 0)) for n in C.NAMES) + ")"
DATA = {"r1": C.ROOT / "data/synthetic_v4_clean_r1.jsonl", "r2": C.ROOT / "data/synthetic_v4_clean.jsonl"}


def fitter(depth: int, trees: int):
    params = {**T.PARAMS, "max_depth": depth, "n_estimators": trees, "monotone_constraints": MONO, "scale_pos_weight": 2.0}
    params.pop("eval_metric")

    def fit(X, y):
        X = X.copy()
        X[:, ZERO] = 0
        model = xgb.XGBClassifier(**params).fit(X, y)

        def predict(Z):
            Z = Z.copy()
            Z[:, ZERO] = 0
            return model.predict_proba(Z)[:, 1]
        return predict
    return fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--data", choices=DATA, default="r2")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--real-negatives", type=int, default=0,
                    help="add ML candidates from N sampled train-split OSS files as negatives")
    args = ap.parse_args()

    train, val = C.split([DATA[args.data]] + C.TRAIN_FILES[1:])
    dtr, dva = C.rows(train), C.rows(val)
    fit = fitter(args.depth, args.trees)
    if args.real_negatives:
        # Unlabeled real code, nearly secret-free. Appended with rec=-1 so record metrics ignore them.
        neg = C.oss_rows("train", args.real_negatives)
        base_fit = fit

        def fit(X, y):  # noqa: F811 - also applied inside every LOLO fold
            return base_fit(np.vstack([X, neg["X"]]), np.r_[y, np.zeros(len(neg["fid"]), dtype=int)])
        report_extra = {"real_negative_rows": len(neg["fid"])}
    else:
        report_extra = {}
    predict = fit(dtr["X"], dtr["y"])
    s = C.record_scores(dva, predict(dva["X"]))
    lab = dva["label"]
    clean = np.array([(r.get("generator") or r.get("source")) != "synthetic_v4" for r in val])
    report = {"label": args.label, "data": args.data, "depth": args.depth, "trees": args.trees,
              "train_rows": len(dtr["y"]), **report_extra, "val": {}, "thresholds": {}}
    for name, k in (("all", np.ones_like(clean)), ("clean", clean)):
        L, S = lab[k], s[k]
        report["val"][name] = {"ceiling": float((S[L == 1] > 0).mean()), "auc": float(roc_auc_score(L, S)),
                               "R@P95": C.recall_at_precision(L, S, 0.95)[0], "R@P90": C.recall_at_precision(L, S, 0.90)[0]}
    for layer, target in (("commit", 0.95), ("gate", 0.90)):
        report["thresholds"][layer] = C.recall_at_precision(lab, s, target)[1]
    oss = C.oss_rows()
    p_oss = predict(oss["X"]) if len(oss["fid"]) else np.zeros(0)
    report["oss"] = {"files": int(oss["n_files"]), "candidates": len(oss["fid"]),
                     **{f"alarms_per_1k_{k}": C.oss_alarms(oss, p_oss, t) for k, t in report["thresholds"].items()}}
    # Equal real-code false-alarm budget: shipped v16 alarms per 1k val-split OSS files.
    budget = {"commit": 14.9, "gate": 56.8}  # shipped v16 at 0.8016 / 0.5505 (section 13)
    for layer, per_1k in budget.items():
        t = C.threshold_for_budget(oss, p_oss, per_1k)
        report["thresholds"][f"{layer}_budget"] = t
        for name, k in (("all", np.ones_like(clean)), ("clean", clean)):
            report["val"][name][f"R@budget_{layer}"] = float((s[k][lab[k] == 1] >= t).mean())
    report["lolo"] = C.lolo(train, dtr, fit, budgets={"oss": oss, "per_1k": budget})["mean"]
    out = C.ROOT / "data/models/r2" / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, default=float))


if __name__ == "__main__":
    main()
