"""Shared loading for the round-2 recall experiments (bench/model_improvement.ipynb, sections 11+).

Candidate rows keep their record index, so recall can be measured per secret (a secret is caught
if any overlapping candidate scores over the threshold), not only per candidate.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "cli"), str(ROOT / "scripts"), str(ROOT)]

import train_v11 as T  # noqa: E402

from bench.compare_scanners import overlaps  # noqa: E402
from Harpocrates.core.detector import (  # noqa: E402
    _collect_text_findings,
    _prepare_ml_context_from_lines,
)
from Harpocrates.core.result import EvidenceType  # noqa: E402
from Harpocrates.ml.features import FeatureVector, extract_features  # noqa: E402

E = ROOT / "data" / "eval"
NAMES = FeatureVector.get_feature_names()
TRAIN_FILES = [ROOT / "data/synthetic_v4_clean.jsonl", E / "train_v5.jsonl", E / "llm_train_v5.jsonl"]
CACHE = ROOT / "data/models/feature_cache"


def split(train_files: list[Path] = TRAIN_FILES) -> tuple[list[dict], list[dict]]:
    """The exact train/val split train_v11 uses (val_v5 + fixed 10% holdout of non-repo files)."""
    train, val = [], T.load(E / "val_v5.jsonl")
    for p in train_files:
        for r in T.load(p):
            (val if not p.name.startswith("train_v") and T.is_holdout(r) else train).append(r)
    keys = {(r["token"], r["line_content"]) for r in train}
    return train, [r for r in val if (r["token"], r["line_content"]) not in keys]


def rows(records: list[dict]) -> dict[str, np.ndarray]:
    """X, y, rec (record index), regex (record caught by a provider regex on its line), cached."""
    h = hashlib.sha256(b"r2rows")
    for path in T._FEATURE_CODE + [Path(__file__)]:
        h.update(path.read_bytes())
    for r in records:
        h.update(json.dumps(r, sort_keys=True).encode())
    cached = CACHE / f"r2_{h.hexdigest()[:24]}.npz"
    if cached.exists():
        return dict(np.load(cached))
    X, y, rec, regex = [], [], [], np.zeros(len(records), dtype=bool)
    for i, r in enumerate(records):
        before = r.get("context_before", [])
        lines = [*before, r["line_content"], *r.get("context_after", [])]
        target = len(before) + 1
        file = "record" + (r.get("file_type") or "")
        for f in _collect_text_findings("\n".join(lines)):
            if f.line != target or not f.token:
                continue
            hit = r["label"] == 1 and overlaps(f.token, r["token"])
            if f.evidence == EvidenceType.REGEX:
                # A key header regex hit covers the key body on the same line ("-----BEGIN...\\n<body>").
                pem = "PRIVATE_KEY" in f.type and "PRIVATE KEY" in r["line_content"] and r["label"] == 1
                # A provider-regex hit on a non-secret's line is a false alarm at every threshold.
                regex[i] |= hit or pem or r["label"] == 0
                continue
            f = dataclasses.replace(f, file=file)
            X.append(extract_features(f, _prepare_ml_context_from_lines(f, lines)).to_array())
            y.append(int(hit))
            rec.append(i)
    out = {"X": np.array(X, dtype=np.float32), "y": np.array(y), "rec": np.array(rec), "regex": regex,
           "label": np.array([r["label"] for r in records])}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cached, **out)
    return out


def record_scores(d: dict, p: np.ndarray) -> np.ndarray:
    """Per record: max score over its overlapping candidates for positives, over all its candidates for
    negatives; regex-caught positives score 1; records with no candidate score 0 (the ceiling)."""
    s = np.zeros(len(d["label"]))
    pos_cand = d["y"] == 1
    neg_rec = d["label"][d["rec"]] == 0
    keep = pos_cand | neg_rec  # a positive record is caught only through a candidate overlapping its secret
    np.maximum.at(s, d["rec"][keep], p[keep])
    s[d["regex"]] = 1.0
    return s


def recall_at_precision(label: np.ndarray, s: np.ndarray, target: float) -> tuple[float, float]:
    """Highest record-level recall with precision >= target, and the threshold that gives it."""
    order = np.argsort(-s)
    tp = np.cumsum(label[order] == 1)
    prec = tp / np.arange(1, len(s) + 1)
    ranked = s[order]
    boundary = np.r_[ranked[1:] != ranked[:-1], True]  # a threshold admits a whole tie group, never part of it
    ok = np.flatnonzero((prec >= target) & boundary)
    if not ok.size:
        return 0.0, 1.0
    j = ok[-1]
    return tp[j] / label.sum(), float(ranked[j])


FAMILIES = {"claude": ("claude", "opus"), "gemini": ("gemini",), "gemma+qwen": ("gemma", "qwen"),
            "deepseek": ("deepseek",), "gpt": ("gpt",)}


def family(record: dict) -> str | None:
    model = record.get("generator_model") or ""
    return next((f for f, keys in FAMILIES.items() if any(k in model for k in keys)), None)


def lolo(train: list[dict], d: dict, fit, targets=(0.95, 0.90), budgets=None) -> dict:
    """Leave-one-LLM-family-out: for each family, fit on training rows from every other record
    and score that family's records, which the model never saw. fit(X, y) -> predict_proba-like fn.
    Mirrors the held-out benchmark (unseen LLMs) without touching it."""
    fam = np.array([family(r) for r in train])
    out = {}
    for f in FAMILIES:
        held = fam == f
        rows_held = held[d["rec"]]
        predict = fit(d["X"][~rows_held], d["y"][~rows_held])
        thresholds = {}
        if budgets:  # this fold's model, thresholded to raise the same alarms on real code
            p_oss = predict(budgets["oss"]["X"])
            thresholds = {k: threshold_for_budget(budgets["oss"], p_oss, b) for k, b in budgets["per_1k"].items()}
        sub = {"X": d["X"][rows_held], "y": d["y"][rows_held], "rec": np.searchsorted(np.flatnonzero(held), d["rec"][rows_held]),
               "regex": d["regex"][held], "label": d["label"][held]}
        s = record_scores(sub, predict(sub["X"]))
        L = sub["label"]
        out[f] = {"n_pos": int(L.sum()), "ceiling": float((s[L == 1] > 0).mean()),
                  **{f"R@P{int(t * 100)}": recall_at_precision(L, s, t)[0] for t in targets},
                  **{f"R@budget_{k}": float((s[L == 1] >= t).mean()) for k, t in thresholds.items()}}
    out["mean"] = {k: float(np.mean([out[f][k] for f in FAMILIES])) for k in out[next(iter(FAMILIES))]}
    return out


def oss_rows(split_name: str = "val", max_files: int | None = None) -> dict[str, np.ndarray]:
    """Every ML candidate in the val-split OSS repos (unlabeled real code, nearly secret-free), for a
    false-alarm rate on real code; split_name="train" gives real-code training negatives.
    Test-split repos are never used here (DATA-09)."""
    import csv

    import build_eval_set as B

    from Harpocrates.core.detector import _collect_file_findings

    with open(ROOT / "scripts/oss_repos.tsv") as f:
        repos = [r["repo"] for r in csv.DictReader(f, delimiter="\t") if B.split_of(r["repo"]) == split_name]
    files = sorted(p for repo in repos for p in B._files(ROOT / "data/oss" / repo.replace("/", "__")))
    if max_files is not None:
        files = [files[i] for i in np.random.default_rng(0).choice(len(files), min(max_files, len(files)), replace=False)]
    h = hashlib.sha256(b"r2oss" + split_name.encode())
    for path in T._FEATURE_CODE + [Path(__file__)]:
        h.update(path.read_bytes())
    h.update("\n".join(map(str, files)).encode())
    cached = CACHE / f"r2oss_{h.hexdigest()[:24]}.npz"
    if cached.exists():
        return dict(np.load(cached))
    X, fid, line, token, regex = [], [], [], [], np.zeros(len(files), dtype=int)
    for i, path in enumerate(files):
        findings = _collect_file_findings(path, None)
        if not findings:
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for f in findings:
            if f.evidence == EvidenceType.REGEX:
                regex[i] += 1
            elif f.token:
                X.append(extract_features(f, _prepare_ml_context_from_lines(f, lines)).to_array())
                fid.append(i)
                line.append(f.line)
                token.append(f.token[:256])  # fixed-width array: one huge minified token would pad every row
    # path/line/token locate each candidate for audits (local only: they are real strings from the corpus)
    out = {"X": np.array(X, dtype=np.float32).reshape(-1, len(NAMES)), "fid": np.array(fid, dtype=int),
           "line": np.array(line, dtype=int), "token": np.array(token, dtype=str),
           "path": np.array([str(p.relative_to(ROOT / "data/oss")) for p in files], dtype=str),
           "regex": regex, "n_files": np.array(len(files))}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cached, **out)
    return out


def oss_alarms(d: dict, p: np.ndarray, threshold: float) -> float:
    """ML alarms (candidates at or over threshold) per 1,000 real files."""
    return float((p >= threshold).sum() * 1000 / int(d["n_files"]))


def threshold_for_budget(d: dict, p: np.ndarray, per_1k: float) -> float:
    """Lowest threshold whose ML alarms on the val-split OSS files stay within per_1k per 1,000 files."""
    allowed = int(per_1k * int(d["n_files"]) / 1000)
    ranked = np.sort(p)[::-1]
    return float(ranked[allowed]) + 1e-9 if allowed < len(ranked) else 0.0
