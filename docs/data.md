# Data provenance (DOC-02)

Every dataset used to train, validate, or benchmark the Harpocrates detector. All secrets are fake, generated values; none are live credentials. Raw data files are git-ignored (`*.jsonl`) and not published; the generator scripts are.

## Shipped model

`cli/Harpocrates/ml/models/model.onnx`, described by `model_config.json` (v2.1.0, single-stage XGBoost, 64 features).
Trained from `Harpocrates/training/data/training_data_v3.pkl` (see `scripts/train_v4.py`): 22,856 train / 2,658 validation / 2,655 calibration / 3,965 test samples. The "golden" metrics in `model_config.json` come from the TruffleHog set below.

## Sources

| File (`data/`) | Rows | Label split (1 / 0) | Origin (`source` field) | Generator | License |
| --- | --- | --- | --- | --- | --- |
| `synthetic_v2_augmented.jsonl` | 13,000 | 8,000 / 5,000 | `script_augment` | `cli/Harpocrates/training/generators/` | Project (repo LICENSE) |
| `synthetic_v3_script.jsonl` | 13,000 | 8,000 / 5,000 | `script_augment` | same | Project |
| `synthetic_v3_llm.jsonl` | 5,496 | 2,496 / 3,000 | `script` 5,000, `llm` 496 | `scripts/generate_synthetic_data.py` | Project; LLM rows generated with Gemma (see below) |
| `synthetic_v3_llm_retry.jsonl` | 15,000 | 7,500 / 7,500 | `script` 7,500, `llm` 7,500 | same | same |
| `synthetic_v4.jsonl` | 40,000 | 20,000 / 20,000 | `script` 20,000, `llm` 20,000 | same | same |
| `trufflehog_golden.jsonl` | 606 | 521 / 85 | `trufflehog` | Extracted from TruffleHog detector test fixtures | **AGPL-3.0 (TruffleHog). Open: confirm whether this use is compatible with the repo license before publishing any derived data.** |

**LLM rows:** generated via LM Studio's OpenAI-compatible API; the default model in `scripts/generate_synthetic_data.py` is `gemma-4-e2b`. That differs from the design doc's "Gemma 4 4B". Confirm which model produced each file.

## Known gaps (tracked for DATA-01..10, v1.1)

- No per-sample `generator_version`, `seed`, or `repo_id` (DATA-02, DATA-06).
- LLM rows are tagged `llm` rather than `llm_synthetic`, and make up 50% of `synthetic_v4` positives, above the 10–20% cap (DATA-08).
- Splits are random, not by repository (DATA-07).
- No external test set yet. SecretBench overlap check is pending until it is adopted (DATA-01).
- Which `data/*.jsonl` files went into `training_data_v3.pkl` isn't recorded. Reconstruct it from `cli/Harpocrates/training/` before v1.1.
