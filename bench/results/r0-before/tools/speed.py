#!/usr/bin/env python3
"""Baseline speed numbers (median / p95) for the cleanup BEFORE/AFTER comparison.

    .venv/bin/python bench/results/r0-before/tools/speed.py [--runs 20] > speed_raw.md

1. Model inference (ONNX shipped model, commit layer): one candidate and a batch of 256, already loaded.
   `session.run` alone, and `_run_session` (what OnnxVerifier calls: np.array + run + probability extraction).
   Feature extraction is not included. A second table repeats it with 1000 runs for a stable median.
2. Hook wall time: `python -m Harpocrates.hooks` as a subprocess with a PreToolUse Read event on stdin
   (the way cli/tests/test_hooks.py drives it), for a clean file (no candidate, model not loaded), a file with a
   regex hit (deny), and a file whose only candidate goes to the model (model loads, p>=gate -> deny or allow).
   Whether the model loaded is verified in-process (onnxruntime in sys.modules) and printed.
Fixtures are generated under a temp dir; the high-entropy values are random strings generated here, not credentials.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv/bin/python")


def p95(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] if len(s) < 100 else s[int(0.95 * len(s))]


def row(name: str, xs: list[float], unit: str) -> str:
    return f"| {name} | {len(xs)} | {statistics.median(xs):.4f} {unit} | {p95(xs):.4f} {unit} | {min(xs):.4f} {unit} |"


def model_timing(runs: int) -> list[str]:
    import numpy as np

    from Harpocrates.ml.features import extract_features_from_record
    from Harpocrates.ml.onnx_verifier import OnnxVerifier, _run_session

    v = OnnxVerifier(layer="commit", lazy_load=False)
    rec = next(json.loads(line) for line in (ROOT / "cli/tests/fixtures/positive_holdout_v4.jsonl").read_text().splitlines() if line.strip())
    vec = extract_features_from_record(rec).to_array()
    one, batch = [vec], [vec] * 256
    arr1, arr256 = np.array(one, dtype=np.float32), np.array(batch, dtype=np.float32)
    name = v._session.get_inputs()[0].name
    cases = {
        "session.run, 1 candidate": lambda: v._session.run(None, {name: arr1}),
        "session.run, batch 256 (total)": lambda: v._session.run(None, {name: arr256}),
        "_run_session, 1 candidate": lambda: _run_session(v._session, one),
        "_run_session, batch 256 (total)": lambda: _run_session(v._session, batch),
    }
    out = []
    for n in (runs, 1000):
        out += [f"\n### Model inference, {n} runs each (warm, after 50 warm-up calls)\n",
                "| case | runs | median | p95 | min |", "|---|---:|---:|---:|---:|"]
        for label, fn in cases.items():
            for _ in range(50):
                fn()
            xs = []
            for _ in range(n):
                t = time.perf_counter_ns()
                fn()
                xs.append((time.perf_counter_ns() - t) / 1e6)
            out.append(row(label, xs, "ms"))
            if "256" in label:
                out.append(row(label.replace("(total)", "(per candidate)"), [x / 256 for x in xs], "ms"))
    return out


def hook_timing(runs: int) -> list[str]:
    rnd = random.Random(0)
    alnum = string.ascii_letters + string.digits
    with tempfile.TemporaryDirectory(prefix="speed_") as tmp:
        d = Path(tmp)
        (d / "clean.py").write_text("import os\n\n\ndef load(path):\n    with open(path) as f:\n        return f.read()\n")
        (d / "regex.py").write_text('GITHUB_TOKEN = "ghp_' + "".join(rnd.choice(alnum) for _ in range(36)) + '"\n')
        (d / "model.py").write_text('client_secret = "' + "".join(rnd.choice(alnum) for _ in range(40)) + '"\n')
        events = {n: json.dumps({"tool_name": "Read", "cwd": str(d), "tool_input": {"file_path": n}}) for n in ("clean.py", "regex.py", "model.py")}
        out = ["\n### Hook wall time (`python -m Harpocrates.hooks`, Read event on stdin), subprocess, "
               f"{runs} runs after 3 warm-ups\n", "| case | model loaded | decision | runs | median | p95 | min |", "|---|---|---|---:|---:|---:|---:|"]
        for n, ev in events.items():
            probe = subprocess.run([PY, "-c",
                                    "import sys,json;from Harpocrates import hooks;c=hooks.decide(sys.stdin.read());"
                                    "sys.stderr.write('LOADED' if 'onnxruntime' in sys.modules else 'NOT_LOADED')"],
                                   input=ev, capture_output=True, text=True)
            loaded = "yes" if "NOT_LOADED" not in probe.stderr and "LOADED" in probe.stderr else "no"
            decision = "deny" if '"deny"' in probe.stdout else "allow"
            for _ in range(3):
                subprocess.run([PY, "-m", "Harpocrates.hooks"], input=ev, capture_output=True, text=True)
            xs = []
            for _ in range(runs):
                t = time.perf_counter()
                subprocess.run([PY, "-m", "Harpocrates.hooks"], input=ev, capture_output=True, text=True)
                xs.append((time.perf_counter() - t) * 1000)
            out.append(row(n, xs, "ms").replace(f"| {n} |", f"| {n} | {loaded} | {decision} |", 1))
        base = []
        for _ in range(runs):
            t = time.perf_counter()
            subprocess.run([PY, "-c", "pass"], capture_output=True)
            base.append((time.perf_counter() - t) * 1000)
        out.append(row("(reference) python -c pass", base, "ms").replace("| (reference) python -c pass |", "| (reference) python -c pass | - | - |", 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    print("\n".join(model_timing(a.runs) + hook_timing(a.runs)))
