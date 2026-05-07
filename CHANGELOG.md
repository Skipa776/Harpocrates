# Changelog

## [0.3.2] — 2026-05-06

### Fixed

- **Severity calibration restored.** Entropy/ML findings with strong variable-name signal (e.g. `APIM_SECRET_KEY`, `DB_PASSWORD`) now surface as `MEDIUM` or `HIGH` instead of being buried at `INFO`. The `_entropy_severity()` stub that always returned `INFO` has been replaced with a two-tier system: pre-ML findings cap at `MEDIUM`; ML-verified findings can reach `HIGH` when the classifier confidence ≥ 0.85.
- **OpenAI legacy keys now detected.** Pre-2024 OpenAI keys (`sk-RQMJj8ELDjv7TRc-dS9sSw` style, shorter or containing hyphens in the body) are now caught by the new `OPENAI_API_KEY_LEGACY` pattern in the HIGH tier. The strict 48-char pure-alphanum `sk-...` and the `sk-proj-...` forms remain in CRITICAL.
- **Variable-name lexicon expanded.** Suffix-style identifiers (`APIM_CLIENT_KEY`, `APIM_SECRET_KEY`, `MY_TOKEN`, `API_CREDENTIALS`) now match the lexicon and receive appropriate severity. Bare `_KEY` suffix maps to `GENERIC_SECRET` (0.65 confidence → `INFO`) to avoid false positives on Django/SQLAlchemy column names.
- **Comment-scanning hot-path optimised.** Replaced `any(c in scan_target for c in "=:\"'")` generator with a compiled `_PROSE_FILTER_RE` (~10× faster on heavily-commented files).

### Security

- `CRITICAL` severity is now reserved exclusively for deterministic regex-tier matches. Entropy and ML findings are capped at `HIGH`.
- `GENERIC_SECRET` findings are never promoted to `HIGH` regardless of ML confidence, preventing over-alerting on enum constants and file-path assignments.

## [0.3.1] — 2026-05-03

- feat: LLM + script generators for commented secrets, runtime generation, dev/prod swaps
- fix: detect commented-out secrets across all comment styles (Finding.in_comment flag)
- feat: synthetic data augmentation for duplicate-token + getenv-fallback coverage
- feat: add MCP server exposing scan_text/scan_file; bump python to 3.10

## [0.3.0] — 2026-05-01

- feat: add ViolationCategory classification (heuristic) for ML/entropy findings
- feat: ship-dark TokenMatch infrastructure for v0.3 retrain
- fix: stop leaking unverified ML_CANDIDATEs as CRITICAL; align _SENSITIVE_ASSIGNMENT_RE with scan_text
- chore: refresh CLAUDE.md to v0.2.x scope; add LFS rule for *.onnx
