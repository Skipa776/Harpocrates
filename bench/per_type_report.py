"""FR-CORE-02 per-type precision and recall for the shipped detector, at both layer thresholds.

    python bench/per_type_report.py data/eval/test_v5.jsonl data/eval/benchmark_llm_v5.jsonl

Recall per true type: share of labeled secrets of that type with an overlapping finding on their line.
Precision per reported type: share of findings the detector labeled with that type that overlap a real
secret. Findings on a non-secret's line are false alarms; other findings on a secret's line are ignored,
as in bench/compare_scanners.py. Scored on held-out sets only; nothing here is tuned.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "cli"), str(ROOT)]

from bench.compare_scanners import materialize, overlaps  # noqa: E402
from Harpocrates.core.classification import SECRET_TYPES, secret_type  # noqa: E402

# Ground truth: dataset secret_type -> the FR-CORE-02 type a correct detector should report.
TRUE_TYPE = {
    "aws_access_key": "cloud_key", "aws_secret_key": "cloud_key", "gcp_api_key": "cloud_key",
    "connection_uri": "db_credential", "jdbc_url": "db_credential", "ado_connection_string": "db_credential",
    "azure_connection_string": "cloud_key",  # storage account key in a connection string
    "password": "password", "generic_random": "generic",
    **{t: "api_token" for t in ("sendgrid_key", "npm_token", "slack_token", "discord_token", "openai_key",
                                "pypi_token", "github_fine_grained", "github_token", "token_url", "vault_token",
                                "telegram_token", "twilio_sid", "jwt", "digitalocean_token", "stripe_key")},
}
# Commit: CLI defaults (precision-first threshold, --ml-threshold 0.19). Gate: the read tool's settings.
LAYERS = {"commit": (None, 0.19), "gate": ("gate", 0.0)}


def summarize(rows: list[tuple[str | None, list[tuple[str, bool]]]]) -> dict:
    """rows: (true type or None for a non-secret, [(reported type, overlaps the secret)]) per record."""
    secrets, caught, findings, correct = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    agree = hits = 0
    for true, found in rows:
        if true is not None:
            found = [(t, ok) for t, ok in found if ok]  # other findings on a secret's line: not scored
            secrets[true] += 1
            if found:
                caught[true] += 1
                hits += 1
                agree += any(t == true for t, _ in found)
        for t, ok in found:
            findings[t] += 1
            correct[t] += ok
    out: dict = {t: {"secrets": secrets[t], "recall": caught[t] / secrets[t] if secrets[t] else None,
                     "findings": findings[t], "precision": correct[t] / findings[t] if findings[t] else None}
                 for t in SECRET_TYPES if secrets[t] or findings[t]}
    out["_type_agreement"] = agree / hits if hits else None
    return out


def scan(records: list[dict], layer: str | None, ml_threshold: float) -> list:
    from Harpocrates.core.detector import detect_file_with_ml
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    verifier = OnnxVerifier(layer=layer)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for r, (name, line) in zip(records, materialize(records, root)):
            fs = [f for f in detect_file_with_ml(root / name, verifier, ml_threshold=ml_threshold)
                  if f.line == line and f.token]
            if r["label"] == 1 and r.get("secret_type") not in TRUE_TYPE:
                continue  # e.g. synthetic ENTROPY_CANDIDATE rows: no known true type
            true = TRUE_TYPE[r["secret_type"]] if r["label"] == 1 else None
            rows.append((true, [(secret_type(f.category), r["label"] == 1 and overlaps(f.token, r["token"]))
                                for f in fs]))
    return rows


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    report = {}
    for path in args.paths:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for layer, (name, ml_threshold) in LAYERS.items():
            out = summarize(scan(records, name, ml_threshold))
            report[f"{path.stem}/{layer}"] = out
            print(f"\n{path.stem}, {layer} layer (type agreement on caught secrets: "
                  f"{_pct(out['_type_agreement'])})\n\n| Type | Secrets | Recall | Findings | Precision |\n"
                  "| --- | --- | --- | --- | --- |")
            for t in SECRET_TYPES:
                s = out.get(t, {"secrets": 0, "recall": None, "findings": 0, "precision": None})
                print(f"| {t} | {s['secrets']} | {_pct(s['recall'])} | {s['findings']} | {_pct(s['precision'])} |")
    (ROOT / "data/models").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/models/per_type.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
