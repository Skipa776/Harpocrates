#!/usr/bin/env python3
"""Build bench/model_improvement.ipynb (analysis notebook for the v1.1 detector).

    python bench/build_model_improvement_nb.py && \
      jupyter nbconvert --to notebook --execute --inplace bench/model_improvement.ipynb

Kept as a script so the notebook's code is reviewable in diffs. Never prints token values.
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("""# Model improvement: v1.1 detector

Goal: raise **full-scan recall** at the gate (target ≥ 95%, precision ≥ 80%) and cut scan latency.

Rules for this notebook:
- **Diagnose on validation only.** The held-out LLM benchmark (Kimi + MiniMax) and the OSS test split are never used to pick features, extractors, or thresholds (DATA-09). They are scored once, at the end, per final candidate.
- **No secret values are printed.** Examples are shown as character-class shapes (`A`=upper run, `a`=lower run, `9`=digits).
- Model under study: `data/models/v15.json` (data v5, trained on scanner candidates; gate threshold from `v15.report.json`)."""),
    code("""import json, sys, time, re, collections
from pathlib import Path
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "bench" else Path.cwd()
sys.path[:0] = [str(ROOT / "cli"), str(ROOT / "scripts"), str(ROOT)]
import train_v11 as T
from Harpocrates.core.detector import _collect_text_findings, _prepare_ml_context_from_lines
from Harpocrates.core.result import EvidenceType
from Harpocrates.ml.features import FeatureVector, extract_features
from bench.compare_scanners import overlaps

E = ROOT / "data" / "eval"
NAMES = FeatureVector.get_feature_names()
report = json.loads((ROOT / "data/models/v15.report.json").read_text())
GATE = report["thresholds"]["0.97"]
booster = xgb.Booster(); booster.load_model(str(ROOT / "data/models/v15.json"))

def shape(s):
    return re.sub(r"[a-z]+", "a", re.sub(r"[A-Z]+", "A", re.sub(r"\\d+", "9", s)))[:24]

# Rebuild the exact train/val split used by train_v11 (same files, same hash holdout).
train_files = [ROOT / "data/synthetic_v4_clean.jsonl", E / "train_v5.jsonl", E / "llm_train_v5.jsonl"]
train, val = [], T.load(E / "val_v5.jsonl")
for p in train_files:
    for r in T.load(p):
        (val if not p.name.startswith("train_v") and T.is_holdout(r) else train).append(r)
keys = {(r["token"], r["line_content"]) for r in train}
val = [r for r in val if (r["token"], r["line_content"]) not in keys]
print(f"train records {len(train)}, val records {len(val)} (val positives {sum(r['label'] for r in val)}), gate threshold {GATE:.4f}")"""),
    md("""## 1. Recall ceiling: which secrets never become a candidate?

A secret the scanner never extracts cannot be caught by any threshold. For each validation positive we check whether *any* candidate on its line overlaps it (regex hits included), and if not, classify why."""),
    code("""def candidates(r):
    before = r.get("context_before", [])
    lines = [*before, r["line_content"], *r.get("context_after", [])]
    target = len(before) + 1
    return lines, target, [f for f in _collect_text_findings("\\n".join(lines)) if f.line == target and f.token]

def miss_reason(tok, line_cands):
    if line_cands:
        return "candidate on line, no overlap"
    if re.search(r"\\s", tok): return "contains whitespace"
    if len(tok) < 8: return "shorter than 8"
    if len(tok) < 20 and not re.search(r"[:=@;&?]", tok): return "8-19 chars, no name/structure hook"
    if re.search(r"[;:@/?&]", tok): return "delimiters split the token"
    return "other"

rows = []
t0 = time.perf_counter()
for i, r in enumerate(val):
    if r["label"] != 1: continue
    _lines, _t, cands = candidates(r)
    hit = [f for f in cands if overlaps(f.token, r["token"])]
    rows.append(dict(i=i, type=r.get("secret_type", "unknown"), source=r.get("source"), covered=bool(hit),
                     regex=any(f.evidence == EvidenceType.REGEX for f in hit),
                     reason=None if hit else miss_reason(r["token"], cands), shape=shape(r["token"])))
print(f"{len(rows)} positives in {time.perf_counter()-t0:.0f}s; ceiling = {np.mean([x['covered'] for x in rows]):.3f}")
by_type = collections.defaultdict(lambda: [0, 0])
for x in rows: by_type[x["type"]][0] += x["covered"]; by_type[x["type"]][1] += 1
print(f"{'type':26}{'coverage':>9}{'n':>6}")
for t, (c, n) in sorted(by_type.items(), key=lambda kv: kv[1][0] / kv[1][1])[:14]:
    print(f"{t:26}{c/n:9.2f}{n:6}")"""),
    code("""miss = [x for x in rows if not x["covered"]]
print(f"uncovered positives: {len(miss)}")
print("by reason:", collections.Counter(x["reason"] for x in miss).most_common())
print()
for t in sorted({x['type'] for x in miss}, key=lambda t: -sum(x['type'] == t for x in miss))[:8]:
    xs = [x for x in miss if x["type"] == t]
    print(f"{t:24} n={len(xs):4}  reasons={dict(collections.Counter(x['reason'] for x in xs).most_common(3))}")
    print(f"{'':24} shapes={[s for s, _ in collections.Counter(x['shape'] for x in xs).most_common(4)]}")"""),
    md("""## 2. Model rejections: covered secrets the gate still drops

For covered positives whose best overlapping candidate scores below the gate threshold, which features push the score down? SHAP values come from XGBoost's exact TreeSHAP (`pred_contribs`)."""),
    code("""rej_X, acc_X, rej_types = [], [], []
for x in rows:
    if not x["covered"]: continue
    r = val[x["i"]]
    lines, target, cands = candidates(r)
    best = None
    for f in cands:
        if f.evidence == EvidenceType.REGEX or not overlaps(f.token, r["token"]):
            continue
        import dataclasses
        f = dataclasses.replace(f, file="record" + (r.get("file_type") or ""))
        v = extract_features(f, _prepare_ml_context_from_lines(f, lines)).to_array()
        p = float(booster.predict(xgb.DMatrix(np.array([v], dtype=np.float32)))[0])
        if best is None or p > best[0]: best = (p, v)
    if best is None: continue  # caught by regex, no ML decision
    (acc_X if best[0] >= GATE else rej_X).append(best[1])
    if best[0] < GATE: rej_types.append(x["type"])
print(f"ML-decided positives: accepted {len(acc_X)}, rejected {len(rej_X)}  -> rejection rate {len(rej_X)/max(len(rej_X)+len(acc_X),1):.3f}")
print("rejected by type:", collections.Counter(rej_types).most_common(8))"""),
    code("""def shap_mean(X):
    c = booster.predict(xgb.DMatrix(np.array(X, dtype=np.float32)), pred_contribs=True)
    return c[:, :-1].mean(axis=0)  # drop bias column
s_rej, s_acc = shap_mean(rej_X), shap_mean(acc_X)
order = np.argsort(s_rej - s_acc)[:12]  # features pushing rejected positives down the most, relative to accepted
print(f"{'feature':34}{'rejected':>10}{'accepted':>10}")
for j in order: print(f"{NAMES[j]:34}{s_rej[j]:10.3f}{s_acc[j]:10.3f}")
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([NAMES[j] for j in order][::-1], (s_rej - s_acc)[order][::-1], color="#b44")
ax.set_xlabel("mean SHAP (rejected - accepted positives), log-odds"); ax.set_title("What pushes real secrets below the gate")
plt.tight_layout(); plt.show()"""),
    md("""### Findings from the first pass (model v14, data v4), and what changed for v15

| Finding | Evidence | Change in v15 |
| --- | --- | --- |
| Ceiling 89.7% on val; 303 of 403 uncovered "secrets" were old-40k label noise | dotted names (`signer.verifySignature`), kebab placeholders (`your-github-client-id`) labeled 1 | `clean_v4` relabels dotted/kebab identifiers |
| Slack coverage 69% | generator emitted `xoxb-` + 3 all-digit parts; our regex capped the secret at 34 chars, so real bot tokens (`xoxb-<d>-<d>-<24 alnum>`) could never match | real-format generator; regex cap 48 |
| Passwords 64% | lowercase+digit passwords (`hunter42`) not "password-shaped" | accept ≥3 letters + ≥2 digits |
| Telegram tokens missed inside bot URLs | URL stripping; no Telegram rule | `TELEGRAM_BOT_TOKEN` HIGH regex + category |
| ML rejects only 3.4% of covered positives | SHAP: low entropy / low digit ratio (mostly the mislabeled records) | fixed by relabeling, not by the model |
| p99 94 ms, max 4.2 s per file | features scanned whole 100k-char lines; file extension recomputed per line | ±256-char context window; per-file extension cache |
"""),
    md("""## 3. Speed: where does scan time go?

Profile the full `detect_file_with_ml` path (the CLI's `scan --ml`) over files sampled from the OSS test-split repos, using the gate model."""),
    code("""import cProfile, pstats, random, csv
sys.path.insert(0, str(ROOT / "bench"))
from bench.eval_detector import _onnx_verifier
from Harpocrates.core.detector import detect_file_with_ml
import build_eval_set as B

verifier = _onnx_verifier(ROOT / "data/models/cand_v15_gate")
with open(ROOT / "scripts/oss_repos.tsv") as f:
    repos = [r["repo"] for r in csv.DictReader(f, delimiter="\\t") if B.split_of(r["repo"]) == "test"]
files = [p for repo in repos for p in B._files(ROOT / "data/oss" / repo.replace("/", "__"))]
random.seed(0); sample = random.sample(files, 400)
times = []
prof = cProfile.Profile(); prof.enable()
for p in sample:
    t = time.perf_counter(); detect_file_with_ml(p, verifier, ml_threshold=0.19)
    times.append((time.perf_counter() - t, p.stat().st_size, max((len(l) for l in p.read_text(errors="replace").splitlines()), default=0)))
prof.disable()
ts = np.array([t for t, _, _ in times]) * 1000
print(f"{len(ts)} files: p50={np.percentile(ts,50):.1f}ms  p95={np.percentile(ts,95):.1f}ms  p99={np.percentile(ts,99):.1f}ms  max={ts.max():.0f}ms  total={ts.sum()/1000:.1f}s")
slow = sorted(times, reverse=True)[:5]
print("slowest files (ms, bytes, longest line):", [(round(t*1000), b, l) for t, b, l in slow])
pstats.Stats(prof).sort_stats("cumulative").print_stats(14)"""),
    md("""## 4. Feature importance (validation, candidate level)

Mean |SHAP| per feature over validation candidates, and XGBoost gain. Features with high importance but no causal link to "is a secret" (names, file hints) are shortcut risks."""),
    code("""Xtr, ytr = T.xy(train, True)
Xva, yva = T.xy(val, True)
print(f"train candidates {len(ytr)} (pos {ytr.mean():.3f}), val candidates {len(yva)} (pos {yva.mean():.3f})")
contrib = booster.predict(xgb.DMatrix(Xva), pred_contribs=True)[:, :-1]
imp = np.abs(contrib).mean(axis=0)
top = np.argsort(imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([NAMES[j] for j in top][::-1], imp[top][::-1], color="#46a")
ax.set_xlabel("mean |SHAP| (log-odds)"); ax.set_title("Top features, validation candidates")
plt.tight_layout(); plt.show()"""),
    md("""## 5. Feature-group ablation

Retrain with one group zeroed (same params, cached features) and compare validation **AUC** and **recall at 5% FPR**. A group whose removal barely hurts, or helps, is carrying shortcuts."""),
    code("""from sklearn.metrics import roc_auc_score, roc_curve
GROUPS = {
    "token shape": [n for n in NAMES if n.startswith(("token_", "char_", "digit_", "uppercase_", "special_", "is_", "has_", "entropy_", "jwt_", "embedded_", "vendor_", "regex_"))],
    "value shape": [n for n in NAMES if n.startswith("value_")],
    "variable name": [n for n in NAMES if n.startswith(("var_", "assignment_", "key_value_", "adjacency_"))],
    "surrounding context": [n for n in NAMES if n.startswith(("context_", "surrounding_", "semantic_", "line_", "in_string", "json_", "env_", "hex_context", "hex_adjacent", "hex_in"))],
    "file type": [n for n in NAMES if n.startswith(("file_", "hex_file"))],
}
P = dict(T.PARAMS); P.pop("eval_metric", None)
def fit_eval(Xa, Xb, **extra):
    m = xgb.XGBClassifier(**P, **extra).fit(Xa, ytr)
    p = m.predict_proba(Xb)[:, 1]
    fpr, tpr, _ = roc_curve(yva, p)
    return roc_auc_score(yva, p), float(np.interp(0.05, fpr, tpr)), m
base_auc, base_r5, _ = fit_eval(Xtr, Xva)
print(f"{'dropped group':22}{'n feats':>8}{'AUC':>8}{'recall@5%FPR':>14}")
print(f"{'(none)':22}{0:8}{base_auc:8.4f}{base_r5:14.4f}")
for g, feats in GROUPS.items():
    idx = [NAMES.index(f) for f in feats]
    Za, Zb = Xtr.copy(), Xva.copy(); Za[:, idx] = 0; Zb[:, idx] = 0
    auc, r5, _ = fit_eval(Za, Zb)
    print(f"{g:22}{len(idx):8}{auc:8.4f}{r5:14.4f}")"""),
    md("""## 6. Recall-oriented training: class weighting and monotonic constraints

- `scale_pos_weight` makes missed secrets costlier than false alarms (the gate's priority).
- Monotonic constraints forbid nonsense directions: higher entropy, a vendor prefix, a secret-sounding name or a failed-placeholder check can never *lower* the score."""),
    code("""mono = {"token_entropy": 1, "vendor_prefix_boost": 1, "var_ngram_secret_score": 1, "jwt_structure_valid": 1,
        "value_is_template_syntax": -1, "is_uuid_v4": -1, "has_version_pattern": -1}
constraints = "(" + ",".join(str(mono.get(n, 0)) for n in NAMES) + ")"
print(f"{'variant':34}{'AUC':>8}{'recall@5%FPR':>14}")
print(f"{'baseline':34}{base_auc:8.4f}{base_r5:14.4f}")
for label, extra in [("scale_pos_weight=2", dict(scale_pos_weight=2.0)),
                     ("scale_pos_weight=4", dict(scale_pos_weight=4.0)),
                     ("monotonic", dict(monotone_constraints=constraints)),
                     ("monotonic + scale_pos_weight=2", dict(monotone_constraints=constraints, scale_pos_weight=2.0))]:
    auc, r5, _ = fit_eval(Xtr, Xva, **extra)
    print(f"{label:34}{auc:8.4f}{r5:14.4f}")"""),
    md("""## 7. Calibration (FR-CORE-03)

Is a score of 0.8 right 80% of the time? A calibrated score makes the gate threshold stable across retrains."""),
    code("""from sklearn.calibration import calibration_curve
p_val = booster.predict(xgb.DMatrix(Xva))
frac, mean_p = calibration_curve(yva, p_val, n_bins=10, strategy="quantile")
bins = np.quantile(p_val, np.linspace(0, 1, 11)); ids = np.clip(np.digitize(p_val, bins[1:-1]), 0, 9)
ece = sum(abs(p_val[ids == b].mean() - yva[ids == b].mean()) * (ids == b).mean() for b in range(10) if (ids == b).any())
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 1], [0, 1], "--", color="#999"); ax.plot(mean_p, frac, "o-", color="#46a")
ax.set_xlabel("predicted probability"); ax.set_ylabel("observed fraction secret"); ax.set_title(f"Reliability (ECE={ece:.3f})")
plt.tight_layout(); plt.show()"""),
    md("""## 8. Distribution shift: training vs held-out benchmark candidates

A classifier tries to tell training candidates from benchmark candidates using **features only (no benchmark labels)**. AUC near 0.5 means no shift; the most separating features are where the model may not generalize. This is the one place the benchmark is touched before final scoring, and only its unlabeled features."""),
    code("""bench = [r for r in T.load(E / "benchmark_llm_v5.jsonl") if (r["token"], r["line_content"]) not in keys]
Xb, _ = T.xy(bench, True)
rng = np.random.default_rng(0); sub = rng.choice(len(Xtr), size=min(len(Xtr), 4 * len(Xb)), replace=False)
Xs, ys = np.vstack([Xtr[sub], Xb]), np.r_[np.zeros(len(sub)), np.ones(len(Xb))]
perm = rng.permutation(len(ys)); cut = int(0.7 * len(ys))
adv = xgb.XGBClassifier(max_depth=4, n_estimators=150, learning_rate=0.1).fit(Xs[perm[:cut]], ys[perm[:cut]])
print(f"adversarial AUC (train vs benchmark candidates) = {roc_auc_score(ys[perm[cut:]], adv.predict_proba(Xs[perm[cut:]])[:, 1]):.3f}")
g = adv.get_booster().get_score(importance_type="gain")
print("most separating features:", [(NAMES[int(k[1:])], round(v, 1)) for k, v in sorted(g.items(), key=lambda kv: -kv[1])[:8]])"""),
    md("""## 9. Head-to-head (final scoring, full file scans)

Scored once per final candidate on the held-out benchmark and OSS test split. detect-secrets is matched per line (it reports only a hash of each secret), which is lenient in its favor. CredSweeper's AUC uses its ML probabilities at `--ml_threshold 0`."""),
    code("""rows = []
for s in ("benchmark_llm_v5", "test_v5"):
    d = json.loads((ROOT / f"data/models/cmp5_gate_{s}.json").read_text())
    for n, v in d["scanners"].items():
        fpr = v["fp"] / max(v["fp"] + v["tn"], 1)
        rows.append((s, n, v["recall"], v["precision"], fpr, v.get("auc_full_curve")))
    print(f"{s}: Harpocrates candidate coverage (recall ceiling) = {d['harpocrates_candidate_coverage']:.3f}")
print()
print(f"{'set':18}{'scanner':16}{'recall':>8}{'precision':>10}{'FPR':>7}{'AUC':>8}")
for s, n, r, p, f, a in rows:
    print(f"{s:18}{n:16}{r:8.3f}{p:10.3f}{f:7.3f}{(f'{a:.3f}' if a is not None else '  -'):>8}")"""),
    md("""## 10. Decision: v16 (variable-name features removed, monotonic, 2x positive weight)

Chosen from validation evidence (sections 5-6: removing names *raised* val AUC and recall@5%FPR; monotonic + weighting added a little). The held-out sets below only confirm it; they did not pick it. Data v5 with the JWT-safe relabel rule."""),
    code("""f = lambda x: "  n/a" if x is None else f"{x:.3f}"
print(f"{'set':18}{'model':12}{'recall':>8}{'precision':>10}{'ceiling':>9}")
for s in ("benchmark_llm_v5", "test_v5"):
    for tag, pre in (("v15 gate", "cmp5_gate"), ("v16 gate", "cmp6_gate"), ("v15 commit", "cmp5_commit"), ("v16 commit", "cmp6_commit")):
        d = json.loads((ROOT / f"data/models/{pre}_{s}.json").read_text()); v = d["scanners"]["harpocrates"]
        print(f"{s:18}{tag:12}{f(v['recall']):>8}{f(v['precision']):>10}{d['harpocrates_candidate_coverage']:9.3f}")
r16 = json.loads((ROOT / "data/models/v16.report.json").read_text())
print()
print("v16 thresholds:", r16["thresholds"], "| zeroed:", len(r16["zeroed_features"]), "features | TruffleHog-set candidate recall (gate):",
      r16["test"]["trufflehog_golden.jsonl"]["0.97"]["recall"])"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "harpocrates", "display_name": "Harpocrates (.venv)", "language": "python"}
    out = Path(__file__).with_name("model_improvement.ipynb")
    nbf.write(nb, out)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
