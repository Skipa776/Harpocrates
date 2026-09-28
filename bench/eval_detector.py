#!/usr/bin/env python3
"""Reproduce the shipped ML model's precision and recall on labeled JSONL.

    python bench/eval_detector.py                    # local 300-record holdout (gitignored)
    python bench/eval_detector.py data/labeled.jsonl # any file with "label" 0/1
    python bench/eval_detector.py --pipeline data/eval/test_v1.jsonl  # full scan, not just ML

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


SLICE_FIELDS = ("secret_type", "name_style", "insertion_style", "source")


def _pipeline_predictor(ml_threshold: float) -> Callable[[dict], bool]:
    from Harpocrates.core.detector import detect_text_with_ml
    from Harpocrates.ml.ensemble import get_verifier

    verifier = get_verifier("auto")

    def predict(record: dict) -> bool:
        before = record.get("context_before", [])
        text = "\n".join([*before, record["line_content"], *record.get("context_after", [])])
        target_line, token = len(before) + 1, record["token"]
        # Only findings on the record's own line count; the scanner already decided to flag them.
        return any(
            f.line == target_line and f.token and (token in f.token or f.token in token)
            for f in detect_text_with_ml(text, verifier, ml_threshold=ml_threshold)
        )

    return predict


def _ml_predictor() -> Callable[[dict], bool]:
    from Harpocrates.ml.features import extract_features_from_record
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    verifier = OnnxVerifier()  # loading verifies the model's SHA-256 manifest
    verifier._ensure_loaded()
    return lambda r: verifier._route(extract_features_from_record(r))[0]


def run(paths: list[Path], pipeline: bool = False, ml_threshold: float = 0.19) -> dict:
    predict = _pipeline_predictor(ml_threshold) if pipeline else _ml_predictor()
    records = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    flagged = {id(r): predict(r) for r in records}
    lookup = lambda r: flagged[id(r)]  # noqa: E731
    slices = {
        field: {value: evaluate([r for r in records if r.get(field) == value], lookup)
                for value in sorted({r[field] for r in records if field in r})}
        for field in SLICE_FIELDS if any(field in r for r in records)
    }
    return {"data": [str(p) for p in paths], "stage": "pipeline" if pipeline else "ml",
            **evaluate(records, lookup), "slices": slices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[HOLDOUT])
    parser.add_argument("--pipeline", action="store_true",
                        help="score the full scan (regex + entropy + ML) instead of the ML stage alone")
    parser.add_argument("--ml-threshold", type=float, default=0.19, help="pipeline ML threshold (CLI default)")
    args = parser.parse_args()
    print(json.dumps(run(args.paths, args.pipeline, args.ml_threshold), indent=2))


if __name__ == "__main__":
    main()
