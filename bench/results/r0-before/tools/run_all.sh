#!/bin/sh
# Regenerate every deterministic FR-CLEAN-02 snapshot into OUT_DIR (run from the repo root, ~5 min).
# BEFORE:  sh bench/results/r0-before/tools/run_all.sh /some/dir_before
# AFTER:   sh bench/results/r0-before/tools/run_all.sh /some/dir_after ; then diff -r the two dirs
#          (or compare the sha256 table in README.md). speed.md and env.txt are not part of this set.
set -e
out="$1"; mkdir -p "$out"
T=bench/results/r0-before/tools
PY=.venv/bin/python
sh $T/eval_detector_cli.sh "$out"
$PY $T/snapshot.py holdout  "$out"
$PY $T/snapshot.py evalsets "$out"
$PY $T/snapshot.py oss      "$out"
$PY $T/snapshot.py canary   "$out"
$PY $T/tests_ids.py "$out/tests.txt"
$PY $T/loc.py > "$out/loc.md"
