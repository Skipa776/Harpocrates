# Baseline speed (BEFORE), commit b2f9873, Apple M3 Pro / 12 cores / 18 GB, macOS 27.0.1, Python 3.12.12 (.venv)

Reproduce: `.venv/bin/python bench/results/r0-before/tools/speed.py --runs 20` (timings are machine and load dependent,
so this file is NOT in the byte-identical set; compare medians, not hashes).

Reference numbers from Oct 16 (caller-supplied): 0.013 ms single, 2.0 ms per 256, ~90 ms clean hook, 230-280 ms model-loading hook.
Measured here: single 0.007-0.010 ms (session.run / _run_session), batch 256 0.41 / 0.71 ms total, clean hook median 68 ms,
model-loading hook median 171 ms (p95 212 ms). All faster than the Oct 16 reference; the reference was probably taken on a
busier machine or a different definition of the batch case. A bare `python -c pass` is ~12 ms of the hook time.
"model.py" is the model-loading case: its only candidate is an entropy token, so the gate-layer ONNX model is loaded
(verified by onnxruntime being imported in-process). "regex.py" is a regex hit and does not load the model.


### Model inference, 20 runs each (warm, after 50 warm-up calls)

| case | runs | median | p95 | min |
|---|---:|---:|---:|---:|
| session.run, 1 candidate | 20 | 0.0073 ms | 0.0082 ms | 0.0068 ms |
| session.run, batch 256 (total) | 20 | 0.4112 ms | 0.4944 ms | 0.3100 ms |
| session.run, batch 256 (per candidate) | 20 | 0.0016 ms | 0.0019 ms | 0.0012 ms |
| _run_session, 1 candidate | 20 | 0.0097 ms | 0.0099 ms | 0.0093 ms |
| _run_session, batch 256 (total) | 20 | 0.7074 ms | 0.8393 ms | 0.5758 ms |
| _run_session, batch 256 (per candidate) | 20 | 0.0028 ms | 0.0033 ms | 0.0022 ms |

### Model inference, 1000 runs each (warm, after 50 warm-up calls)

| case | runs | median | p95 | min |
|---|---:|---:|---:|---:|
| session.run, 1 candidate | 1000 | 0.0090 ms | 0.0236 ms | 0.0068 ms |
| session.run, batch 256 (total) | 1000 | 0.3842 ms | 0.5074 ms | 0.3038 ms |
| session.run, batch 256 (per candidate) | 1000 | 0.0015 ms | 0.0020 ms | 0.0012 ms |
| _run_session, 1 candidate | 1000 | 0.0104 ms | 0.0220 ms | 0.0089 ms |
| _run_session, batch 256 (total) | 1000 | 0.6542 ms | 0.9277 ms | 0.5603 ms |
| _run_session, batch 256 (per candidate) | 1000 | 0.0026 ms | 0.0036 ms | 0.0022 ms |

### Hook wall time (`python -m Harpocrates.hooks`, Read event on stdin), subprocess, 20 runs after 3 warm-ups

| case | model loaded | decision | runs | median | p95 | min |
|---|---|---|---:|---:|---:|---:|
| clean.py | no | allow | 20 | 68.2024 ms | 107.2597 ms | 61.7838 ms |
| regex.py | no | deny | 20 | 69.8512 ms | 111.9278 ms | 61.9440 ms |
| model.py | yes | deny | 20 | 171.0086 ms | 212.2316 ms | 152.8384 ms |
| (reference) python -c pass | - | - | 20 | 11.9211 ms | 25.5679 ms | 11.6394 ms |
