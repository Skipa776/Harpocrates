#!/usr/bin/env python3
"""Reproduce the shipped ML model's precision and recall on labeled JSONL.

    python bench/eval_detector.py                    # committed 300-secret holdout
    python bench/eval_detector.py data/labeled.jsonl # any file with "label" 0/1

A record counts as flagged when the model routes it to "review" or "blocked"
(probability >= threshold_low), matching golden_metrics in model_config.json.
Precision is null when the set has no negatives: an all-positive set can't
measure it (model_config.json records 1.0 for the same all-positive holdout).
Runs from a repo clone; bench/ is not in the installed package. Prints JSON;
never prints tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))

HOLDOUT = ROOT / "cli" / "tests" / "fixtures" / "positive_holdout_v4.jsonl"


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def evaluate(records: Iterable[dict], predict: Callable[[dict], bool]) -> dict:
    tp = fp = fn = tn = 0
    for record in records:
        flagged, positive = predict(record), record["label"] == 1
        tp += flagged and positive
        fn += not flagged and positive
        fp += flagged and not positive
        tn += not flagged and not positive
    return {
        "records": tp + fp + fn + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": _ratio(tp, tp + fn),
        "precision": _ratio(tp, tp + fp) if fp + tn else None,
    }


def run(paths: list[Path]) -> dict:
    from Harpocrates.ml.features import extract_features_from_record
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    verifier = OnnxVerifier()  # loading verifies the model's SHA-256 manifest
    verifier._ensure_loaded()
    records = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    metrics = evaluate(records, lambda r: verifier._route(extract_features_from_record(r))[0])
    return {"data": [str(p) for p in paths], **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[HOLDOUT])
    print(json.dumps(run(parser.parse_args().paths), indent=2))


if __name__ == "__main__":
    main()
