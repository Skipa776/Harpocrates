# R0 cleanup report (FR-CLEAN-01)

Oct 18, 2026 · branch `r0/cleanup` · status: **§1–§3 approved and executed. §4 still awaits decisions.**

Executed:
- **§1, items 1–14:** commits `23434e1`..`0edec5a`.
- **§2, items 15–22:** commits `ee24aa2`..`de0f5cf`.

Verification:
- **Findings:** `run_all.sh` output is byte-identical to `bench/results/r0-before/` after items 15, 19 and 21, and after the last commit.
- **Clean-checkout tests:** 1,029 passed, 2 skipped, 1 known environmental failure (`test_packaging.py::test_wheel_installs_with_ml_pipeline`, which failed before the cuts too). The 64 removed tests covered only untracked training code.
- **Mixed test file:** `test_ml_pipeline.py` kept its feature and verifier tests. Only its two training classes were removed.

Stale references left on purpose. Each still works, or will be fixed by a later requirement:
- `model_config.json` lines 14 and 45: shipped and hashed, so not edited.
- Comments citing the notebook in `detector.py:108` and `test_candidate_extractors.py:61`.
- `onnx_verifier.py:175`, which says "Re-convert with onnxmltools".
- `AGENTS.md`, `CLAUDE.md` and `docs/samples/samples/06_explain_demo.java` say `[train]`. They keep working through the deprecated alias, and FR-NAME-01 rewrites them anyway.
- `.gitleaksignore:16` names `generate_data.py`, which is now untracked.
- `bench/compare_scanners.py:154` lazily imports scikit-learn, which is no longer in any extra.

Method: `/ponytail-audit` over the whole repo. Then a read-only evidence pass per candidate: Python imports (AST, including lazy imports), references in docs, CI, `pyproject.toml`, `.gitignore` and pre-commit, and tests that exercise each item. Behavior before cleanup is snapshotted in `bench/results/r0-before/` (FR-CLEAN-02). Every cut is diffed against that snapshot.

Decision recorded from Josh (Oct 18): remove the training scripts, AI prompts and AI-planning docs from the repo. They were kept only for reproducibility.

**How a cut is done (Josh, Oct 18):** a file cut is `git rm --cached` plus a `.gitignore` entry. The file leaves the repo but stays on Josh's disk, so local reproducibility is kept.

Legend: **cut** means untrack as above. **move** means relocate with no behavior change. **edit** is the follow-up a cut needs. ⚠ marks items that need Josh's sign-off under FR-CLEAN-03 or carry a risk.

## 1. Safe cuts: no importers, no references, no tests

One commit each.

| # | Cut | Evidence | LOC |
| --- | --- | --- | --- |
| 1 | `scripts/train_experiments.py` | No importers or references. Reads `training/data/training_data_v2.pkl`, which no longer exists | 602 |
| 2 | `scripts/train_experiments_v2.py` | Same | 606 |
| 3 | `scripts/train_experiments_v3.py` | No references. Imports `training.generators`, so it keeps training alive but nothing depends on it | 344 |
| 4 | `scripts/train_v4.py` | No importers or references | 346 |
| 5 | `scripts/analyze_fp.py` | No importers or references. Stale `.pkl` path | 151 |
| 6 | `scripts/evaluate_overfit.py` | No importers or references | 498 |
| 7 | `scripts/regenerate_features.py` | No importers or references. Stale `training/data` paths | 75 |
| 8 | `scripts/generate_go_nogo_report.py` | No tracked references. ⚠ The untracked `M2_PLAN.md` could not be checked | 594 |
| 9 | `ml/context.py`: `has_safe_context`, `has_risky_context` and their `__all__` lines | Zero callers anywhere, including tests | ~12 |
| 10 | `pyproject.toml`: `examples*` and `comparison*` in `packages.find.exclude` | No such packages under `cli/` | 1 |
| 11 | `.gitignore`: `scripts/evaluate_stress_tests.py` and `scripts/train_simple.py` entries | Files do not exist | 2 |
| 12 | `docs/assets/divider-*.png` (6 files) and `docs/assets/ASSETS.md` | Dividers are referenced only by `ASSETS.md`, and README uses none. `ASSETS.md` is a brand catalog, and most images it links (banners B–E, lockups, OG cards, palette) are not in the repo | binary + 181 |
| 13 | `docs/ml_feature_vector_and_synthetic_data.md` | No inbound links. ⚠ Confirm `docs/data.md` supersedes it | 270 |
| 14 | `docs/samples/TESTING_GUIDE.txt` | No inbound references. ⚠ Confirm it is unused | 391 |

## 2. Removing the training pipeline (your decision)

Nothing in the runtime package imports `Harpocrates.training`. Training is not a runtime dependency. Four things block a plain delete:

1. **The canary generator lives in training.** `training/generators/secret_templates.py` (1,833 LOC) is imported by `bench/canary_repo.py`. That harness feeds four security suites (`test_hooks.py`, `test_hooks_allowlist.py`, `test_hooks_grants.py`, `test_read_tool.py`) and `bench/live_canary_check.py`. **Move**, don't cut: put it in `bench/canaries/`, the location `design.md` already plans for. ⚠ It touches the harness that proves FR-READ-01, so this needs your sign-off.
2. **The OSS test set is rebuilt by `fetch_oss_corpus.py`, `build_eval_set.py` and `oss_repos.tsv`.** FR-CLEAN-02 compares findings on that set before and after. The snapshot uses the data already on disk, so the comparison works without the scripts. Rebuilding the set from scratch would no longer be possible from `main`.
3. **M6 needs a retrain command.** FR-ML-02 asks for one command that retrains from a pinned snapshot. Today that path is spread over `train_v11.py`, `calibrate.py`, `convert_to_onnx.py` and `r2_common.py`, which import each other in a loop. Deleting them means M6 builds that command fresh. It would have to be rebuilt as one command anyway.
4. **Docs point at the scripts.** These are `docs/data.md` (DOC-02 provenance), `detector-report.md`, the `onnx_verifier.py` error messages ("Re-run scripts/convert_to_onnx.py"), and the `explain.py` error message ("python -m Harpocrates.training.train_model").

**Recommendation:** delete the pipeline from `main` and pin reproducibility to the `v1.3.0` tag, where every script still exists. `docs/data.md` then says "reproduce the v17 model at tag v1.3.0" instead of listing script paths. M6 adds back one clean retrain command. Side benefit: about 5,300 LOC of training code no longer ships in every wheel.

If you approve, the cuts go in this order, one per commit, each with the edits it needs:

| # | Cut or move | Follow-up edit | LOC |
| --- | --- | --- | --- |
| 15 | **move** `secret_templates.py` to `bench/canaries/` ⚠ | Update imports in `canary_repo.py`, `build_eval_set.py` and the tests. The canary snapshot must stay byte-identical | 1,833 |
| 16 | `bench/build_model_improvement_nb.py` + `bench/model_improvement.ipynb` | `docs/data.md` lines 21 and 31; the `bench/*.ipynb` ruff rule | 774 + nb |
| 17 | `scripts/r2_audit.py`, `scripts/r2_eval.py` | None after #16 | 353 |
| 18 | `scripts/generate_synthetic_data.py` | `docs/data.md` provenance line | 3,035 |
| 19 | `scripts/{train_v11,calibrate,convert_to_onnx,r2_common}.py` | Re-point the `onnx_verifier.py` messages and `pyproject.toml:41`; `detector-report.md`; `test_r2_metrics.py`, `test_train_pipeline_features.py` | 704 |
| 20 | `scripts/{build_eval_set,build_slot_set,generate_llm_slots,clean_v4,eda_datasets,fetch_oss_corpus}.py`, `oss_repos.tsv` | `docs/data.md`; `test_data_07_eval_set.py`, `test_slot_set.py`, `test_clean_v4.py`, `test_generate_llm_slots.py` | ~1,060 |
| 21 | `cli/Harpocrates/training/` (rest) | `explain.py` message; ruff ignores; `test_data_generator.py`, `test_ml_pipeline.py`, `test_ml_shortcuts.py`, `test_data_02/03_*.py` | ~3,560 |
| 22 | `[train]` extra in `pyproject.toml` | CI installs `.[dev,train,mcp]` (ci.yml lines 45, 78, 104). Keep `xgboost`, because `--explain` loads `xgboost_model.json` at runtime. Move it to an `explain` extra | – |

Tests that only cover deleted training code are deleted with it. FR-CLEAN-02 then means: every remaining test passes, and the findings snapshots are byte-identical.

## 3. AI prompts and planning docs

| # | Item | Evidence | Proposal |
| --- | --- | --- | --- |
| 23 | `design.md` "Developing with an AI assistant" section (kickoff prompts, target layout) | AI prompts | **done**: the section is cut and the target layout moved under Architecture. The removed working rules are all still in `AGENTS.md` |
| 24 | `AGENTS.md` and `CLAUDE.md` | Correction: `CLAUDE.md` is already a symlink to `AGENTS.md` (since Sep 27). An earlier evidence pass wrongly said they had drifted | **keep**, no change needed |
| 25 | Untracked `M0_PLAN.md`, `M2_PLAN.md`, `ML_PLAN.md`, `EDA_REPORT.md`, `allowed-models.txt` | Not in git. Deleting them is permanent | Josh: leave them alone. No change |

## 4. Not cuts: needs sign-off, consolidation or a fix

| # | Item | Why it is not a plain cut |
| --- | --- | --- |
| 26 | `utils/redaction.py`: `redact_finding`, `redact_findings` | Zero runtime callers, only tests. ⚠ Redaction surface: FR-CLEAN-03 sign-off |
| 27 | `core/rust_backend.py`: `scan_text_with_ml` | Only tests call it. Cut it together with its tests |
| 28 | `extract_var_name`, two copies (`core/classification.py`, `ml/context.py`) | Consolidate. It feeds model features, so FR-CLEAN-02 byte-identity is the guard |
| 29 | `ml/explain.py`: `_DEFAULT_HASHES_PATH` (unused) | ⚠ **Don't cut, fix.** `xgboost_model.json` loads with no integrity check, while `model.onnx` is hash-checked. Wire the check in as a separate security fix |
| 30 | `xgboost_model.json` missing from wheel `package-data` | Bug: `--explain` fails on pip installs, and the error says "reinstall", which does not help. Separate fix |
| 31 | Dual engine (Rust + Python) | Keep both. Wheels ship no Rust binary, so Python is the only engine for pip users |
| 32 | `plugins/harpocrates-prompt` | Keep, it is a shipped plugin. Its TypeScript tests don't run in CI: coverage gap |

Security modules (`hooks.py`, `allowlist.py`, `grants.py`, `harness.py`, `mcp/`, `read.py`): every public symbol has a live caller. **No cuts proposed.**

## 5. Lines of code before and after (FR-CLEAN-04)

Code lines (non-blank, non-comment) in tracked files. "Before" is `main` at `b2f9873`, counted by `bench/results/r0-before/tools/loc.py`. "After" is measured after the last approved cut.

| Package | Files | Before | After |
| --- | ---: | ---: | ---: |
| `cli/Harpocrates` top-level modules | 7 | 2,161 | |
| `cli/Harpocrates/cli` | 2 | 432 | |
| `cli/Harpocrates/core` | 6 | 1,990 | |
| `cli/Harpocrates/detectors` | 4 | 218 | |
| `cli/Harpocrates/mcp` | 2 | 144 | |
| `cli/Harpocrates/ml` | 12 | 2,477 | |
| `cli/Harpocrates/training` | 8 | 4,133 | |
| `cli/Harpocrates/utils` | 3 | 132 | |
| `cli/tests` | 53 | 8,086 | |
| `core` (Rust) | 9 | 1,593 | |
| `bench` | 9 | 3,177 | |
| `scripts` | 22 | 6,865 | |
| `docs` | 24 | 1,456 | |
| `plugins` | 11 | 1,704 | |
| root, `.github`, `.claude-plugin` | 14 | 601 | |
| **Total** | 186 | **35,169** | |

The line counts in §1–§3 are total lines from `wc -l`, so they run higher than the code-line counts here.

## 6. Baseline behavior (FR-CLEAN-02)

Captured Oct 18, 2026 on an Apple M3 Pro (12 cores, 18 GB), Python 3.12.12. Commands and sha256 hashes are in `bench/results/r0-before/README.md`. All 16 snapshot files were byte-identical across three runs.

- **Tests:** 1,157 passed, 1 skipped (non-UTF-8 filename on APFS), 0 failed.
- **Benchmark (`benchmark_llm_v5`):** gate recall 94.96% / precision 90.41%; commit 93.94% / 94.36%. Matches `model_config.json`.
- **OSS test set (`test_v5`):** gate 99.33% / 94.54%; commit 98.73% / 97.11%. Matches.
- **OSS real code (42 repos, 11,354 files):** 473 gate findings (41.7 per 1,000 files), 333 commit findings (29.3 per 1,000 files).
- **Canary repo (seed 0):** 14 of 15 canaries found by the scanner. The scanner misses `postgres_password`. The live result of 0 of 15 leaked still holds, because the read hook blocks the whole file.
- **Speed (median / p95, 20 runs):**
  - one candidate: 0.0073 / 0.0082 ms
  - batch of 256: 0.41 / 0.49 ms
  - clean-file hook call: 68 / 107 ms
  - model-loading hook call: 171 / 212 ms

  All are faster than the Oct 16 reference numbers.

Rule for every cut: rerun `bench/results/r0-before/tools/run_all.sh`. The findings diff must be empty and the remaining tests must pass.

net (if everything in §1–§3 is approved): about −15,000 lines before deleted tests (1,833 moved, not removed); −0 runtime deps, while `scikit-learn` and `onnxmltools` leave the dev install.
