#!/usr/bin/env python3
"""Release checks for v17 against shipped v16 (bench/model_improvement.ipynb section 18).

    python scripts/r2_audit.py dump OUT.npz          # candidates under the current detector
    python scripts/r2_audit.py report OLD.npz NEW.npz

`dump` runs once with the pre-round-2 detector (v16's) and once with the current one (v17's).
`report` then measures:
1. alarm rate per repo, val-split and test-split OSS repos (thresholds are frozen; test repos
   are reported, never used to choose anything);
2. the recall gain on validation with a paired bootstrap over repos / files;
3. mined real-code negatives that a model trained without them scores most secret-like;
4. a blinded sample of val-split alarms for human labeling.
Audit files go to data/audit/ (git-ignored: they contain real strings from the OSS corpus).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_common as C  # noqa: E402
import train_v11 as T  # noqa: E402

from bench.compare_scanners import overlaps  # noqa: E402

ROOT = C.ROOT
AUDIT = ROOT / "data/audit"
V16 = (ROOT / "cli/Harpocrates/ml/models/xgboost_model.json", {"commit": 0.8016, "gate": 0.5505})


def _v17() -> tuple[Path, dict]:
    t = json.loads((ROOT / "data/models/v17.report.json").read_text())["thresholds"]
    return ROOT / "data/models/v17.json", {"commit": t["commit"], "gate": t["gate"]}


def _common_val() -> list[dict]:
    """Validation records both models are scored on: the clean generators (the old-40k holdout
    differs between cleaning rounds)."""
    _, val = C.split()
    return [r for r in val if (r.get("generator") or r.get("source")) != "synthetic_v4"]


def dump(out: Path) -> None:
    val = _common_val()
    d = {f"val_{k}": v for k, v in C.rows(val).items()}
    for split in ("val", "test"):
        d.update({f"oss_{split}_{k}": v for k, v in C.oss_rows(split).items()})
    np.savez(out, **d)
    print(f"wrote {out}: {len(val)} val records, "
          f"{len(d['oss_val_fid'])} / {len(d['oss_test_fid'])} val / test OSS candidates", file=sys.stderr)


def _sub(d: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def _predict(model_path: Path, X: np.ndarray) -> np.ndarray:
    b = xgb.Booster()
    b.load_model(str(model_path))
    return b.predict(xgb.DMatrix(X)) if len(X) else np.zeros(0)


def _repo_of(path: str) -> str:
    return path.split("/", 1)[0]


def alarm_rates(old: dict, new: dict, split: str) -> dict:
    """ML alarms per 1,000 files, overall and per repo, for v16 (old detector) and v17 (new)."""
    out = {}
    for name, (model, thr), d in (("v16", V16, _sub(old, f"oss_{split}_")), ("v17", _v17(), _sub(new, f"oss_{split}_"))):
        p = _predict(model, d["X"])
        repos = np.array([_repo_of(x) for x in d["path"]])
        files_per_repo = {r: int((repos == r).sum()) for r in set(repos)}
        cand_repo = repos[d["fid"]]
        for layer, t in thr.items():
            alarms = p >= t
            per_repo = {r: 1000 * int((alarms & (cand_repo == r)).sum()) / n for r, n in files_per_repo.items()}
            out[f"{name}_{layer}"] = {"overall_per_1k": 1000 * int(alarms.sum()) / int(d["n_files"]),
                                      "per_repo": per_repo}
    return out


def recall_gain(old: dict, new: dict, val: list[dict], n_boot: int = 2000) -> dict:
    """Paired bootstrap over groups (repo for OSS inserts, file for LLM slots) of the recall gain."""
    caught = {}
    for name, (model, thr), d in (("v16", V16, _sub(old, "val_")), ("v17", _v17(), _sub(new, "val_"))):
        s = C.record_scores(d, _predict(model, d["X"]))
        caught.update({f"{name}_{layer}": s >= t for layer, t in thr.items()})
    label = np.array([r["label"] for r in val])
    group = np.array([r.get("repo") or r.get("file_path") or str(i) for i, r in enumerate(val)])
    pos = label == 1
    groups = np.unique(group[pos])
    idx = {g: np.flatnonzero(pos & (group == g)) for g in groups}
    rng = np.random.default_rng(0)
    out = {}
    for layer in ("commit", "gate"):
        a, b = caught[f"v16_{layer}"], caught[f"v17_{layer}"]
        diffs = []
        for _ in range(n_boot):
            sample = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
            diffs.append(b[sample].mean() - a[sample].mean())
        out[layer] = {"v16_recall": float(a[pos].mean()), "v17_recall": float(b[pos].mean()),
                      "gain": float(b[pos].mean() - a[pos].mean()),
                      "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                      "v17_only": int((b & ~a & pos).sum()), "v16_only": int((a & ~b & pos).sum()),
                      "groups": len(groups)}
    return out


def _context(path: str, line: int, token: str, width: int = 3) -> str | None:
    """Lines around the candidate with it marked, or None if the token can't be found near `line`
    (line counting can differ on files with form feeds or other Unicode line breaks)."""
    lines = (ROOT / "data/oss" / path).read_text(encoding="utf-8", errors="ignore").split("\n")
    near = [n for n in (line - 1, *range(line - 6, line + 5)) if 0 <= n < len(lines) and token in lines[n]]
    if not near:
        return None
    at = near[0]
    shown = []
    for n in range(max(at - width, 0), min(at + width + 1, len(lines))):
        text = lines[n]
        if n == at:  # window the long line around the candidate so the mark is always visible
            col = text.index(token)
            text = text[max(col - 80, 0):col] + f"⟦{token}⟧" + text[col + len(token):col + len(token) + 80]
        shown.append(f"{n + 1:>5}{'>' if n == at else ' '} {text[:400]}")
    return "\n".join(shown)


def mined_negative_audit(k: int = 100) -> dict:
    """Train v17's recipe *without* mined negatives, score the mined negatives, write the top k.
    High scores are the unlabeled real-code strings most likely to be hidden secrets."""
    rep = json.loads((ROOT / "data/models/v17.report.json").read_text())
    train, _ = C.split([Path(p) if Path(p).is_absolute() else ROOT / p for p in rep["train_files"]])
    d = C.rows(train)
    zero = [C.NAMES.index(n) for n in rep["zeroed_features"]]
    X = d["X"].copy()
    X[:, zero] = 0
    params = {**T.PARAMS, "max_depth": rep["max_depth"], "n_estimators": rep["n_estimators"],
              "scale_pos_weight": rep["scale_pos_weight"]}
    if rep["monotone"]:
        params["monotone_constraints"] = "(" + ",".join(str(T.MONOTONE.get(n, 0)) for n in C.NAMES) + ")"
    params.pop("eval_metric")
    model = xgb.XGBClassifier(**params).fit(X, d["y"])
    neg = C.oss_rows("train", rep.get("real_negative_files", 30000))
    Z = neg["X"].copy()
    Z[:, zero] = 0
    p = model.predict_proba(Z)[:, 1]
    AUDIT.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(AUDIT / "mined_negatives_top.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score_without_mined_negatives", "file", "line", "context", "label", "note"])
        for i in np.argsort(-p):
            path = str(neg["path"][neg["fid"][i]])
            ctx = _context(path, int(neg["line"][i]), str(neg["token"][i]))
            if ctx is None:
                continue
            w.writerow([f"m{written:03}", round(float(p[i]), 4), path, int(neg["line"][i]), ctx, "", ""])
            written += 1
            if written == k:
                break
    return {"mined_rows": len(p), "score_ge_0.9": int((p >= 0.9).sum()), "score_ge_0.5": int((p >= 0.5).sum()),
            "top_k_written": k}


def blind_sample(old: dict, new: dict, per_group: dict[str, int], seed: int = 0) -> dict:
    """Gate-threshold alarms on val-split OSS files, grouped v17-only / v16-only / both by location
    and token overlap, sampled and shuffled. Labels file has no model information; key is separate."""
    alarms = {}
    for name, (model, thr), d in (("v16", V16, _sub(old, "oss_val_")), ("v17", _v17(), _sub(new, "oss_val_"))):
        p = _predict(model, d["X"])
        alarms[name] = [(str(d["path"][d["fid"][i]]), int(d["line"][i]), str(d["token"][i]), float(p[i]),
                         bool(p[i] >= thr["commit"])) for i in np.flatnonzero(p >= thr["gate"])]

    def same(a, b):
        return a[0] == b[0] and a[1] == b[1] and overlaps(a[2], b[2])

    groups = {"both": [], "v17_only": [], "v16_only": []}
    unmatched = list(alarms["v16"])
    for a in alarms["v17"]:  # one-to-one: each v16 alarm pairs with at most one v17 alarm
        b = next((x for x in unmatched if same(a, x)), None)
        if b is not None:
            unmatched.remove(b)
        groups["both" if b else "v17_only"].append((a, b))
    groups["v16_only"] = [(None, b) for b in unmatched]
    rng = random.Random(seed)
    picked = []
    for g, n in per_group.items():
        pool = [x for x in groups[g] if _context(*(x[0] or x[1])[:3]) is not None]  # the mark must be visible
        picked += [(g, x) for x in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(picked)
    key = {}
    AUDIT.mkdir(parents=True, exist_ok=True)
    with open(AUDIT / "alarms_blind.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "line", "context", "label (secret / not_secret / unsure)", "note"])
        for n, (g, (a, b)) in enumerate(picked):
            path, line, token = (a or b)[:3]
            w.writerow([f"a{n:03}", path, line, _context(path, line, token), "", ""])  # never None: filtered above
            key[f"a{n:03}"] = {"group": g, "v17_score": a and round(a[3], 4), "v16_score": b and round(b[3], 4),
                               "v17_commit": bool(a and a[4]), "v16_commit": bool(b and b[4])}
    (AUDIT / "alarms_key.json").write_text(json.dumps(key, indent=1))
    return {"alarms": {k: len(v) for k, v in alarms.items()}, "groups": {k: len(v) for k, v in groups.items()},
            "sampled": dict(zip(per_group, [sum(1 for g, _ in picked if g == k) for k in per_group]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump").add_argument("out", type=Path)
    rep = sub.add_parser("report")
    rep.add_argument("old", type=Path)
    rep.add_argument("new", type=Path)
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.out)
        return
    old, new = dict(np.load(args.old)), dict(np.load(args.new))
    report = {"alarm_rates": {s: alarm_rates(old, new, s) for s in ("val", "test")},
              "recall_gain": recall_gain(old, new, _common_val()),
              "mined_negatives": mined_negative_audit(),
              "blind_sample": blind_sample(old, new, {"v17_only": 80, "v16_only": 40, "both": 80})}
    (ROOT / "data/models/r2/release_checks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "alarm_rates"}, indent=1))


if __name__ == "__main__":
    main()
