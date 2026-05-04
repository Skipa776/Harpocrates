#!/usr/bin/env python3
"""Re-extract features for all records in one or more JSONL files.

Reads each record, calls extract_features_from_record (which now consumes
token_start/token_end and attaches TokenMatch to CodeContext), and writes
the updated record with a fresh features_65 array.  Idempotent — existing
features_65 values are replaced, not appended.

Usage:
    python scripts/regenerate_features.py \
        --input Harpocrates/training/data/train_transformed.jsonl \
        --input data/synthetic_v2_augmented.jsonl \
        --output Harpocrates/training/data/train_transformed_v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Harpocrates.ml.features import extract_features_from_record


def _process(input_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = ok = failed = 0
    with open(output_path, "w") as out:
        for input_path in input_paths:
            with open(input_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    record = json.loads(line)
                    try:
                        fv = extract_features_from_record(record)
                        record["features_65"] = fv.to_array()
                        ok += 1
                    except Exception as e:
                        record["features_65"] = None
                        failed += 1
                        if failed <= 5:
                            print(f"  [WARN] feature extraction failed: {e}",
                                  file=sys.stderr)
                    out.write(json.dumps(record) + "\n")
    print(f"Processed {total} records → {output_path}")
    print(f"  ok: {ok}  failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-extract features for JSONL training data")
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Input JSONL file(s); repeat for multiple inputs")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file")
    args = parser.parse_args()
    _process(args.input, args.output)


if __name__ == "__main__":
    main()
