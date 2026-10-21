# r0-before: BEFORE snapshot for the cleanup (FR-CLEAN-02, FR-CLEAN-04)

Taken at commit `b2f98737c134d4fd05f415de6f2d25801e6c7133` (branch `r0/cleanup`, tracked tree clean), Apple M3 Pro / 12 cores / 18 GB,
macOS 27.0.1, Python 3.12.12 from `.venv`, cargo/rustc 1.96.0. See `env.txt`. Nothing tracked was edited; everything here is untracked.
All commands run from the repo root `/Users/joshuado/harpocrates`. The environment is the existing `.venv` (editable install of this repo;
its pip metadata still says harpocrates 0.4.3 while pyproject says 1.3.0, a stale egg-info only; imports resolve to `cli/Harpocrates`).
`cargo build --release --manifest-path core/Cargo.toml` was already up to date.

## Reproduce everything at once

    sh bench/results/r0-before/tools/run_all.sh /path/to/out_dir      # ~3 min; writes the 16 files in the table below
    # AFTER the cleanup: same command into another dir, then `diff -r before after` or compare the sha256 table.

`run_all.sh` was run a third time (into a scratch dir) after the stored copies were made; every file was byte-identical.
Each snapshot was also produced twice during the work (run 1 == run 2 for every file). Result: **all snapshots are deterministic**.
No source change was needed. The Redactor's per-session random HMAC key (read.py) is the only random element and is not part of any
snapshot (the read tool's detection layer is covered by the gate-layer scans instead).

## Individual commands

| snapshot | command |
|---|---|
| tests | `.venv/bin/python -m pytest -q -p no:cacheprovider` (1157 passed, 1 skipped, 0 failed, 152-162 s); ids: `.venv/bin/python bench/results/r0-before/tools/tests_ids.py OUT/tests.txt` |
| holdout, literal eval_detector | `sh bench/results/r0-before/tools/eval_detector_cli.sh OUT` (runs `bench/eval_detector.py`, `--pipeline`, `--pipeline --ml-threshold 0.1904`, `--pipeline --ml-threshold 0.5559`) |
| holdout, per record | `.venv/bin/python bench/results/r0-before/tools/snapshot.py holdout OUT` |
| labeled eval sets, full-file scan | `.venv/bin/python bench/results/r0-before/tools/snapshot.py evalsets OUT` (data/eval/test_v5.jsonl and benchmark_llm_v5.jsonl, the unit of the model_config.json metrics, ~100 s) |
| OSS test split | `.venv/bin/python bench/results/r0-before/tools/snapshot.py oss OUT` (~90 s) |
| canary repo | `.venv/bin/python bench/results/r0-before/tools/snapshot.py canary OUT` (bench/canary_repo.py seed 0 in a temp dir, scanned with the library scanner; the CLI scan of that dir with `--json --fail-on none` is cross-checked: same 17 findings, exit 0) |
| LOC | `.venv/bin/python bench/results/r0-before/tools/loc.py > OUT/loc.md` |
| speed | `.venv/bin/python bench/results/r0-before/tools/speed.py --runs 20` (not deterministic, compare medians) |

Layers: `commit` = `OnnxVerifier(layer="commit")` (threshold 0.5559, what the CLI scan uses, scan ml_threshold 0.19);
`gate` = `OnnxVerifier(layer="gate")` (0.1904, what the read tool uses, ml_threshold 0.0). Finding rows hold file, line, type, evidence,
severity, confidence, category and `token_sha256_16` (sha256 prefix of the token); no raw token values are stored anywhere.

## What each file is

- `holdout_findings.jsonl`, `holdout_summary.json`, `eval_detector_*.json`: bench/eval_detector.py default set
  `cli/tests/fixtures/positive_holdout_v4.jsonl` (300 records, ALL positive, so precision is null).
  ML stage: recall 0.2967 (89/300) at commit 0.5559, 0.3633 (109/300) at gate 0.1904. Full pipeline: 0.33 (99/300) at ml-threshold
  0.19 or 0.5559 (commit verifier), 0.36 (108/300) with the gate verifier at 0.1904. These are low, but they are what this old (May 10)
  fixture gives with the current model; model_config.json records no holdout metric any more, so there is nothing to compare to.
- `evalsets_records.jsonl`, `evalsets_summary.json`: test_v5 (3000) and benchmark_llm_v5 (4352) as full-file scans (bench/compare_scanners.py unit),
  several (layer, ml_threshold) combos kept. Reproduces the model_config.json metrics exactly (at the CLI-default ml_threshold 0.19, also at 0.0):
  benchmark_llm_v5 gate R/P 0.9496/0.9041 (config 0.9496/0.9041), commit 0.9394/0.9436 (0.9394/0.9436);
  test_v5 gate 0.9933/0.9454 (0.9933/0.9454), commit 0.9873/0.9711 (0.9873/0.9711).
- `oss_findings_commit.jsonl`, `oss_findings_gate.jsonl`, `oss_summary.json`: `scan_directory` over the 42 test-split repos of scripts/oss_repos.tsv in data/oss
  (split rule build_eval_set.split_of; 11,354 files, 2.12 M lines). Unlabeled real code, so no recall/precision; a findings snapshot:
  commit 333 findings (29.3 per 1k files: 39 regex + 294 hybrid), gate 473 (41.7 per 1k files). No scan errors.
- `canary_*`: seed 0 canary repo, 15 planted canaries; 14/15 found at both layers (17 findings). The miss is `postgres_password`
  (human password in docker-compose.yml); existing behavior, recorded so the AFTER run can be compared.
- `loc.md`, `tests.txt`: as named. `speed.md`, `env.txt`: not byte-compared.

## sha256 of each snapshot file

| file | sha256 |
|---|---|
| `canary_findings_commit.jsonl` | `b91625cfd9a536c7d26bb288b68a1c120ac3b39faa558e53699d82eafc0ece62` |
| `canary_findings_gate.jsonl` | `b91625cfd9a536c7d26bb288b68a1c120ac3b39faa558e53699d82eafc0ece62` |
| `canary_summary.json` | `f62541d7cdc90cf49f9eac4349013a45c79b716f0027fff50958355d644dcb63` |
| `eval_detector_ml.json` | `56f7c62393c8eb5e75e111302ac1d7793a8089dc00b8050b08c873b33d91010d` |
| `eval_detector_pipeline_0.1904.json` | `776aad7f58e6ae84975ed26b7228ebe9e8aec4d2c77d20364eaa0bc4c830199e` |
| `eval_detector_pipeline_0.5559.json` | `776aad7f58e6ae84975ed26b7228ebe9e8aec4d2c77d20364eaa0bc4c830199e` |
| `eval_detector_pipeline_default0.19.json` | `776aad7f58e6ae84975ed26b7228ebe9e8aec4d2c77d20364eaa0bc4c830199e` |
| `evalsets_records.jsonl` | `97a344830bc3ee76dfe7875db8c91e243c01f002c51bd440f9937798fc5bc286` |
| `evalsets_summary.json` | `620db3a7071211a2d05df4db54f664a6a35571d689a4be4d3851c13c49359169` |
| `holdout_findings.jsonl` | `f65686e2855694ce1c2446af25f1c7b6133d22f4b18b716219e42408daa3f44f` |
| `holdout_summary.json` | `78bf7786e6855749c797525bbe2b1378462bdf5d313f6718f2ea21771f604f41` |
| `loc.md` | `5c393e090977327736d62700639b52c9564b7e6dce88135b853780a03de71e6b` |
| `oss_findings_commit.jsonl` | `53234e461bb4e4b645712c62cb231dfb5e3f5e7cd0127850f531de074c10e9e4` |
| `oss_findings_gate.jsonl` | `ae71ffa4d8991ca970c7f79b10cb645d7a9eef31fb35e6edc0a3d4f86d8ad42e` |
| `oss_summary.json` | `0ea55a5b18e4ede79e2db2aa0fbdf308235ce6b341c7ed07795f3f977b4e7b8e` |
| `tests.txt` | `b79b8feca42d4083ea79a58090843e4a6567b6cb63e7a1c6123b9268d2ae82b6` |

Tools (untracked, written for this snapshot; sha256 for provenance):

| file | sha256 |
|---|---|
| `tools/eval_detector_cli.sh` | `eb2efc98b9b374151520b365076ffd82077076bf9c4d52d0d25849848afe8160` |
| `tools/loc.py` | `26f0f5f869a7c79b627b1e7340ed51f96d58b7973c555af4ddc33653225a41ba` |
| `tools/run_all.sh` | `8df4b58b496620fe79ae0e109b3b9b3fea6e1989c1fcf52ceb4a69ea87577ea0` |
| `tools/snapshot.py` | `481849a82b23711a2107a3adea7e2627b6b47ceed1a85f52f9e16e07a17c6d59` |
| `tools/speed.py` | `fa8c441402af99953f3a80a365736580a5c2888bf72a2d6768f4614712ae66d4` |
| `tools/tests_ids.py` | `fd78e9c06c69d97b044182c2d4faf0daddf0e5c55e10fb320ce43c93b3f70144` |
| `tools/tests_summary.txt` | `03ae9ed963901038471aa1562a43fb0ec3f138509a263c429359d47ffb5de7ad` |

## Not run / caveats

- Not run: `scripts/r2_eval.py` (re-trains models and writes `data/models/r2/*.json`; not an inference snapshot) and `bench/compare_scanners.py`
  itself (needs trufflehog, gitleaks, credsweeper, detect-secrets; its Harpocrates part is reproduced by snapshot.py evalsets using its own helpers).
- `comparison/` has no tracked files at this commit, so it has no row in loc.md. `cli/Harpocrates/api` and `cli/Harpocrates/llm` exist on disk
  but hold no tracked files (also absent from loc.md).
- 1 test skipped: cli/tests/test_hooks.py:598, this file system rejects non-UTF-8 names.
- Some parametrized pytest ids contain canary-like strings; 154 had their `[param]` replaced by `sha256:<12 hex>` in tests.txt.
- The security hook blocks built-in reads of some result files (hex hashes look like generic secrets); read them with safe_read or process them with python.
