# Data provenance (DOC-02)

Every dataset used to train, validate, or benchmark the Harpocrates detector. All secrets are fake, generated values; none are live credentials. Raw data files are git-ignored (`*.jsonl`) and not published; the scripts that build them are. The exception is `docs/samples/readme_examples.jsonl` (7 fake examples behind the README comparison).

## Shipped model (v1.2, model v17)

`cli/Harpocrates/ml/models/model.onnx` and `model_config.json` (version 1.2.0) hold a single-stage XGBoost model:
- 64 features, with the 10 variable-name features zeroed in training;
- depth 7, 600 trees, monotonic constraints.

`xgboost_model.json` is the same model (used by `--explain`). There is no native fallback: if the ONNX model cannot load, `harpocrates scan` exits 2. The previous shipped model (v1.1, model v16) is archived locally as `data/models/v16.json`.

**How it was trained:** `scripts/train_v11.py --pipeline-features` on **scanner candidates**. The scanner's own extraction runs over each record, and each candidate is labeled by overlap with the true secret. The training data is:
- 94,183 candidate rows from the three training files below;
- 62,883 real-code negative rows: every ML candidate in 30,000 files sampled from the 142 train-split OSS repos, excluding the 363 that a model trained without them scored ≥ 0.9, which were mostly credential-shaped test fixtures (`--reliable-below 0.9`).

**Thresholds (FR-CORE-06):**
- `commit` 0.5559: the CLI default, precision-first;
- `gate` 0.1904: read tool and egress gate, recall-first.

Each is set so the model raises v1.1's ML alarm rate on the val-split OSS repos (14.9 and 56.8 per 1,000 files). Measured results are in `model_config.json` and the README; the full log is in `bench/model_improvement.ipynb`.

## Training data

| File | Rows | Secrets / non-secrets | Origin | Built by | License |
| --- | --- | --- | --- | --- | --- |
| `data/synthetic_v4_clean.jsonl` | 34,652 | 11,186 / 23,466 | The original 40k set, cleaned: 4,835 duplicates removed, 7,659 non-secret "secrets" (identifiers, dotted/kebab names, URLs, paths) relabeled, 498 conflicting records dropped | `scripts/clean_v4.py` on `data/synthetic_v4.jsonl` (see below) | Project |
| `data/eval/train_v5.jsonl` | 30,000 | 15,000 / 15,000 | Fake secrets inserted into real files from 142 train-split repos; lookalike negatives; real scanner candidates from the same code as negatives | `scripts/fetch_oss_corpus.py`, `scripts/build_eval_set.py --split train` | Project (generators); inserted into code from the OSS corpus below |
| `data/eval/llm_train_v5.jsonl` | 23,937 | 11,655 / 12,282 | Code written by LLMs with typed `{{SECRET:kind}}` / `{{NONSECRET:kind}}` slots, filled with generated fake values | `scripts/generate_llm_slots.py`, `scripts/build_slot_set.py` | Project (values); LLM-written code, see below |

**Round 2 (model v17, shipped as v1.2; `bench/model_improvement.ipynb` round 2):**
- `synthetic_v4_clean.jsonl` is rebuilt with three more `clean_v4.py` rules:
  - word paths (`application/json`, `actions/checkout`) relabeled: 241
  - function names in call position relabeled: 96
  - hex/base64 positives in hash-named variables dropped as ambiguous: 218

  The result is 34,704 records (10,719 / 23,985). v16's input is kept as `synthetic_v4_clean_r1.jsonl`.
- v17 also trains on **real-code negatives** (see Shipped model above). They're unlabeled real code. The most secret-like are filtered out, but a few real secrets in fixtures could remain, which the DATA-05 review should check.
- Thresholds are set to match v16's ML alarm rate on the val-split OSS repos (14.9 / 56.8 per 1,000 files).

**LLM training generators** (records): GPT-6-luna 4,976; DeepSeek V4 Flash 5,019; GPT-5.6-luna 4,846; Claude Sonnet 5.5 1,464; DeepSeek V4 Pro 1,464; Gemini 3.8 Flash 1,259; Gemini 3.7 Flash 1,196; Onyx Gemma-4-31B 907; Qwen 3.8 Flash 854; Gemini 3.6 Flash 711; Gemini 3.5 Flash Lite 663; Opus 5.5 Light 578. Run through Codex, Claude Code, opencode and agy with all tools disabled (text only).

**Original 40k set** (`data/synthetic_v4.jsonl`): 20,000 script-generated and 20,000 LLM-generated records (LM Studio, default model `gemma-4-e2b`). About 31% of its "secrets" were not secrets, which is why it is cleaned before use.

## Evaluation data (never trained on, never tuned on; DATA-09)

| File | Rows | Secrets / non-secrets | Purpose |
| --- | --- | --- | --- |
| `data/eval/val_v5.jsonl` | 3,000 | 1,500 / 1,500 | Validation (16 val-split repos) plus a fixed 10% hash holdout of the non-repo training files. Thresholds and model choices are made here. |
| `data/eval/test_v5.jsonl` | 3,000 | 1,500 / 1,500 | Open-source test set (42 test-split repos). Scored once per final candidate. |
| `data/eval/benchmark_llm_v5.jsonl` | 4,352 | 2,064 / 2,288 | Held-out benchmark: code written by Kimi K2.6 (1,789) and MiniMax M2.5 (2,563), which generated no training data. Scored once per final candidate. |
| `data/trufflehog_golden.jsonl` | 606 | 521 / 85 | Local-only sanity check, never trained on and never published. Extracted from TruffleHog detector fixtures: **AGPL-3.0**. Open decision: whether any derived use is compatible with this repo's license. |

The earlier `positive_holdout_v4.jsonl` (300 records, about 37% mislabeled) is retired.

## Open-source corpus

200 repos, 20 per language across Python, JavaScript, TypeScript, Go, Java, Ruby, PHP, Rust, C# and shell. Licenses: Apache-2.0 80, BSD-3-Clause 64, MIT 56. Each is pinned to a commit in `scripts/oss_repos.tsv` and shallow-fetched into git-ignored `data/oss/`. No third-party code is committed. Repos are split train / val / test by a hash of their name (142 / 16 / 42), so no repo spans two splits (DATA-07). Real strings that gitleaks or Harpocrates' provider regexes flag are queued for review, never labeled as negatives.

## Rules (DATA-01..10) and known gaps

- **Provenance (DATA-06):** every generated record carries `source`, `generator`, `generator_version` (currently 5), `seed`, and `repo` or `generator_model`.
- **LLM share (DATA-08):** records whose *secret value* was written by an LLM (the old Gemma rows) are 8.6% of training positives, within the 10–20% cap. LLM-slot records are 31% of positives, but there the LLM wrote only the surrounding code; every value comes from the deterministic generators, so labels are correct by construction.
- **No external real-world test set.** SecretBench access was not available; the benchmark is synthetic (LLM-written code, generated values).
- **Label noise:** the review queues (`data/eval/*_review_queue.jsonl`) have not been manually reviewed (DATA-05's ≥500-sample review was not done).
- **Unpublished:** the benchmark and test files are not published yet.
