#!/bin/sh
# Literal bench/eval_detector.py runs on its default holdout set (cli/tests/fixtures/positive_holdout_v4.jsonl).
# Usage: sh bench/results/r0-before/tools/eval_detector_cli.sh OUT_DIR   (run from the repo root)
set -e
out="$1"; mkdir -p "$out"
PY=.venv/bin/python
$PY bench/eval_detector.py                                   > "$out/eval_detector_ml.json"
$PY bench/eval_detector.py --pipeline                        > "$out/eval_detector_pipeline_default0.19.json"
$PY bench/eval_detector.py --pipeline --ml-threshold 0.1904  > "$out/eval_detector_pipeline_0.1904.json"
$PY bench/eval_detector.py --pipeline --ml-threshold 0.5559  > "$out/eval_detector_pipeline_0.5559.json"
