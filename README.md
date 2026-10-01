![Harpocrates](docs/assets/banner-a-classic-parchment.png)

<h1 align="center">Harpocrates</h1>

<p align="center">
  <strong>ML-powered secrets detection that catches what regex can't see</strong>
</p>

---

AI coding tools leak secrets at **2× the rate of human-written code** — 3.2% vs 1.6% of commits. 29 million secrets were exposed in 2025, up 34% year over year. One misconfigured environment variable cost a team **$87k in a single night**.

Regex scanners look for `AWS_ACCESS_KEY_ID` and `GITHUB_TOKEN`. They miss `client_secret`, `ENCRYPTION_KEY`, and `API_SECRET` — the names developers actually use. They miss secrets buried in comments. They miss env-var fallbacks. They miss the code your AI coding assistant just generated.

Harpocrates catches what slips through.

---

## Harpocrates vs other secret scanners

Five scanners, same files, full scans. Harpocrates results are exactly what `harpocrates scan --ml` runs (regex + entropy + ML). Two held-out test sets, never used for training or tuning:

- **Held-out benchmark:** 4,352 records. The code was written by LLMs (Kimi K2.6, MiniMax M2.5) that generated none of the training data. Half the records carry a secret, half a harmless look-alike.
- **Open-source code:** 3,000 records from 42 permissively licensed GitHub repos, with fake secrets inserted and real high-entropy strings from the same code as negatives.

Recall and precision columns are held-out benchmark first, then open-source code.

| Scanner | Recall | Precision | False-positive rate | Recall | Precision | AUC (benchmark / OSS) |
|---|---|---|---|---|---|---|
| TruffleHog 3.95 | 40.0% | 97.6% | 0.9% | 39.5% | 98.8% | n/a (yes/no only) |
| gitleaks 8.30 | 52.3% | 89.3% | 5.7% | 48.9% | 100.0% | n/a (yes/no only) |
| detect-secrets 1.5 ¹ | 55.3% | 63.8% | 28.3% | 74.0% | 78.7% | n/a (yes/no only) |
| CredSweeper 1.18 | 68.3% | 72.8% | 23.1% | 59.3% | 93.3% | 0.72 / 0.81 |
| **Harpocrates (gate)** | **95.0%** | **90.4%** | 9.1% | **99.3%** | **94.5%** | **0.986 / 1.000** ² |
| Harpocrates (commit) | 93.9% | 94.4% | 5.1% | 98.7% | 97.1% | same model |

¹ detect-secrets reports only a hash of each secret, so it is credited for any finding on the right line. That is lenient in its favor.
² Harpocrates AUC is measured over the candidates the scanner extracts. TruffleHog, gitleaks and detect-secrets output yes/no only, so they have one point on the ROC curve, not a curve. CredSweeper's AUC uses its ML probabilities (`--ml_threshold 0`).

**Two operating points, one model.** The *gate* threshold favors recall: it is for blocking secrets before they reach an AI model, where a missed secret is the costly mistake. The *commit* threshold favors precision for pre-commit hooks, where false alarms interrupt developers.

**What changed in v1.2 (model v17), compared with v1.1 (model v16):**

| | Benchmark recall | Benchmark precision | Open-source recall | Open-source precision |
|---|---|---|---|---|
| Commit | 90.2% → **93.9%** | 95.1% → 94.4% | 95.3% → **98.7%** | 97.3% → 97.1% |
| Gate | 93.3% → **95.0%** | 92.5% → 90.4% | 98.7% → **99.3%** | 93.5% → **94.5%** |

- **How the thresholds are set.** Both are chosen so the model raises the same number of alarms on real open-source code as v1.1 did.
- **Real code got quieter.** On 42 repositories no threshold was tuned on, v1.2 raises less than half as many gate alarms as v1.1 (62.7 against 136.7 per 1,000 files).
- **Where precision dropped.** Benchmark gate precision fell because the benchmark's non-secrets are LLM-written look-alikes, not real code.

The full analysis is in `bench/model_improvement.ipynb`.

**Secrets only one scanner caught** (held-out benchmark): Harpocrates **202**, detect-secrets 12, CredSweeper 10, TruffleHog 3, gitleaks 0.

### Recall by secret type (held-out benchmark)

Sorted by Harpocrates' lead over the best competitor. **Bold** = Harpocrates is best or tied.

| Secret type | TruffleHog | gitleaks | CredSweeper | Harpocrates |
|---|---|---|---|---|
| Twilio auth token | 1.4% | 5.7% | 7.1% | **94.3%** |
| Telegram bot token | 33.7% | 0.0% | 44.2% | **100.0%** |
| Password | 1.8% | 6.3% | 20.7% | **73.9%** |
| AWS secret key | 0.0% | 49.4% | 56.5% | **100.0%** |
| PyPI token | 0.0% | 54.4% | 57.0% | **100.0%** |
| OpenAI key | 1.1% | 51.6% | 57.0% | **100.0%** |
| Discord bot token | 0.0% | 34.6% | 53.1% | **95.1%** |
| Generic random secret | 0.0% | 0.0% | 42.5% | **78.3%** |
| DigitalOcean token | 0.0% | 37.6% | 59.1% | **94.6%** |
| Token in URL query | 26.8% | 63.9% | 62.9% | **95.9%** |
| Database/broker URI with password | 24.2% | 0.0% | 70.7% | **90.9%** |
| ADO.NET connection string | 78.7% | 3.2% | 58.5% | **91.5%** |
| AWS access key | 2.6% | 51.9% | 98.7% | **100.0%** |
| Stripe key | 98.9% | 93.7% | 97.9% | **100.0%** |
| GCP API key | 0.0% | 89.7% | 100.0% | **100.0%** |
| GitHub token | 100.0% | 100.0% | 65.0% | **100.0%** |
| npm token | 100.0% | 96.0% | 50.7% | **100.0%** |
| SendGrid key | 100.0% | 96.7% | 100.0% | **100.0%** |
| Slack bot token | 100.0% | 100.0% | 98.9% | **100.0%** |
| Azure storage connection string | 98.8% | 96.3% | 95.1% | 97.6% |
| JDBC URL with password | 100.0% | 3.2% | 66.7% | 96.8% |
| JWT | 0.0% | 93.4% | 98.7% | 94.7% |
| Vault token | 0.0% | 61.6% | 98.6% | 93.2% |
| GitHub fine-grained token | 98.9% | 100.0% | 100.0% | 94.3% |

**Where competitors miss the most:** credentials without a provider prefix to anchor a regex on. That covers Twilio auth tokens (bare 32-hex), passwords, generic random secrets, and passwords inside connection strings and URLs. It also covers tokens embedded in larger strings, such as a Telegram token inside a bot URL or a token in a URL query parameter.

### Real-world cases none of the others catch

Each case below was scanned by all five tools. Only Harpocrates flagged the secret. Values are fake and shortened here; the exact files and a script that reproduces this table are in `docs/samples/readme_examples.jsonl` and `bench/readme_examples.py`.

| Situation | Code | TruffleHog | gitleaks | detect-secrets | CredSweeper | Harpocrates |
|---|---|---|---|---|---|---|
| Database password in a YAML config | `pass: "Summer2024!…"` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Password passed as a keyword argument | `psycopg2.connect(…, password="hunter…")` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Postgres URL with a percent-encoded password | `postgresql://orders_app:Wint3r%23…@db.internal/orders` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Twilio auth token passed positionally | `twilio('AC4f2a…', '9f2c4e1a…')` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Random session secret in a Rails config | `config.session_value = "vQ3#kL9$…"` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Telegram bot token inside an API URL | `…api.telegram.org/bot7318849210:AAHx9…/getMe` | ❌ | ❌ | ❌ | ❌ | ✅ |

Why these slip past the others: none of them has a provider prefix like `AKIA`, `ghp_` or `sk_live_`. Several use neutral names (`pass`, `session_value`, a positional argument), and two are embedded in a URL. Harpocrates extracts candidates from connection strings, URL parameters and password-shaped literals, and an ML model trained on real code decides which are secrets.

### Where the others win

- **Precision at the extreme:** TruffleHog flags almost nothing that isn't a secret (97.6% precision). It also verifies credentials against provider APIs; Harpocrates never makes network calls.
- **Specific formats:**
  - CredSweeper catches more JWTs (98.7% vs 94.7%) and Vault tokens (98.6% vs 93.2%).
  - gitleaks and CredSweeper catch every GitHub fine-grained token (100% vs 94.3%).
  - TruffleHog catches every JDBC URL (100% vs 96.8%) and more Azure storage connection strings (98.8% vs 97.6%).
- **A known miss:** a 64-character hex HMAC signing key (`[]byte("7d4e9a1f…")` in Go) is flagged by detect-secrets and CredSweeper but not by Harpocrates. It is in the reproduce script.

**Recommended setup:** Harpocrates as the pre-commit hook and AI-agent gate (widest coverage), TruffleHog in CI (confirms which leaked credentials are live).

### Methodology and caveats

- Harness: `bench/compare_scanners.py` (needs the scanners on `PATH`). Tool versions: TruffleHog 3.95.2 (`--no-verification`), gitleaks 8.30.1, detect-secrets 1.5.0, CredSweeper 1.18.5 (default `medium` threshold).
- The benchmark and open-source test files are not published yet (the open-source set contains third-party code; it is rebuilt from a pinned repo list with `scripts/fetch_oss_corpus.py` and `scripts/build_eval_set.py`). The real-world cases above are published and reproducible today with `bench/readme_examples.py`.
- A scanner is credited only for a finding on the right line whose value overlaps the labeled secret (≥ 4 characters). detect-secrets is the exception noted above.
- Labels are correct by construction: LLMs wrote code with typed placeholders, and fake values of a known kind were filled in afterwards. The benchmark is still synthetic; real repositories will differ.
- Harpocrates' recall ceiling (the share of secrets its scanner turns into a candidate at all) is 95.5% on the benchmark and 99.5% on open-source code.
- This benchmark has now been scored for several model versions, so it is retired for choosing models. These numbers are a one-time report for v1.2. Future models are selected on leave-one-LLM-family-out validation and real-code alarm rates.
- These results are for the **v1.2 detector** (model v17), shipped in this repository.
  - `harpocrates scan --ml` uses the precision-first *commit* threshold (0.5559) by default.
  - The recall-first *gate* threshold (0.1904) is in `model_config.json`, and the read tool uses it.
  - The current PyPI release still ships the earlier v0.4 model.
- Per-type precision and recall, and how the reported confidence is calibrated, are in [docs/detector-report.md](docs/detector-report.md).

---

![Installation](docs/assets/divider-installation.png)

## Install

```bash
pip install harpocrates
harpocrates scan .
```

With ML verification (recommended; see [Harpocrates vs other secret scanners](#harpocrates-vs-other-secret-scanners) for measured recall and precision):

```bash
pip install "harpocrates[ml]"
harpocrates scan . --ml
```

With REST API server:

```bash
pip install "harpocrates[api]"
harpocrates serve
```

Everything at once:

```bash
pip install "harpocrates[all]"
```

---

![Usage](docs/assets/divider-usage.png)

## Usage

```bash
# Scan a directory
harpocrates scan ./my_project

# Scan a single file
harpocrates scan config.env

# Output as JSON (pipe-friendly)
harpocrates scan ./my_project --json

# Enable ML verification to suppress false positives
harpocrates scan ./my_project --ml

# Use the Rust regex/entropy engine and the existing Python ML verifier
harpocrates scan ./my_project --engine rust --ml

# Only fail CI on high or critical findings
harpocrates scan ./my_project --fail-on high

# Ignore specific patterns
harpocrates scan ./my_project --ignore "*.test.js,fixtures/*"
```

### Pre-commit hook

```yaml
repos:
  - repo: https://github.com/Skipa776/Harpocrates
    rev: v0.2.0
    hooks:
      - id: harpocrates
```

Add to `.pre-commit-config.yaml`, then run `pre-commit install`. Harpocrates scans every staged file before each commit.

---

![Configuration](docs/assets/divider-configuration.png)

## Configuration

### `harpocrates scan` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ml` | off | Enable ML verification to reduce false positives |
| `--ml-threshold FLOAT` | `0.19` | Extra floor on combined confidence, `0.0–1.0`. The model's commit threshold (`model_config.json`) decides first, so at the default this floor never removes a finding. Raise it for fewer, higher-confidence findings |
| `--engine ENGINE` | `auto` | `auto` prefers an available Rust binary; `rust` requires it; `python` uses the original engine |
| `--fail-on LEVEL` | `medium` | Severity that triggers exit code `1`: `critical` \| `high` \| `medium` \| `low` \| `info` \| `none` |
| `--json` | off | Output results as JSON instead of a table |
| `--show-secrets` | off | Print full token values instead of redacted previews |
| `--ignore TEXT` | — | Comma-separated glob patterns to skip (e.g. `"*.test.js,fixtures/*"`) |
| `--max-size INTEGER` | `10` | Maximum file size to scan, in MB |
| `--recursive / --no-recursive` | `--recursive` | Scan subdirectories recursively |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No findings at or above `--fail-on` severity |
| `1` | One or more findings detected at or above `--fail-on` severity |
| `2` | Error (bad argument, unreadable file, etc.) |

### Other commands

```bash
harpocrates version          # Print version
harpocrates setup            # Support tier of each detected agent harness, and what it leaves unprotected
harpocrates setup codex      # Read-tool setup for Codex (or: claude-code)
harpocrates serve            # Start the REST API server (requires harpocrates[api])
harpocrates --help           # Full command list
```

---

![How it scans](docs/assets/divider-how-it-scans.png)

## How it scans

Harpocrates runs a three-phase pipeline on every line of every file:

1. **Regex** — deterministic patterns for known credential formats (AWS, GitHub, Stripe, private keys, and more). No ML required. High-confidence, zero false positives on well-formed keys.

2. **Entropy analysis** — Shannon entropy flags high-randomness tokens that don't match any known pattern. Catches credentials stored under ambiguous variable names (`my_key`, `token`, `secret`) that regex scanners miss entirely.

3. **ML verification** (opt-in via `--ml`) — a single-stage XGBoost classifier extracts 64 features from the token, its variable name, and the surrounding code context. It learns to distinguish `api_secret = "AKIA..."` (secret) from `commit_sha = "a1b2c..."` (Git SHA) without relying on the variable name alone. Inference runs via ONNX Runtime when available, with native XGBoost as fallback.

### Native Rust scanner

The optional `core` crate owns file reading, binary-file rejection,
comment handling, regex matching, entropy calculation, and candidate generation.
Directory scans use one versioned batch subprocess, native traversal with early
ignore-directory pruning, bounded file reads, and deterministic bounded worker
threads. Rust returns a compact JSON candidate stream to Python. Python remains
responsible for violation classification, context extraction, and the existing
batched ML verifier; known regex hits still bypass ML.

Build it from a source checkout:

```bash
cargo build --release --manifest-path core/Cargo.toml
harpocrates scan ./my_project --engine rust
harpocrates scan ./my_project --engine rust --ml
```

`--engine auto` (the default) finds release/debug binaries in the source tree or
an installed `harpocrates-rust-scanner` on `PATH`. Set
`HARPOCRATES_RUST_SCANNER=/absolute/path/to/harpocrates-rust-scanner` to use a
custom build. Use `--engine python` to force the original implementation.
Set `HARPOCRATES_RUST_WORKERS` to control native directory-scan parallelism; the
default is the available CPU count capped at 8. Native directory subprocesses
have a 300-second watchdog; set `HARPOCRATES_RUST_TIMEOUT_SECONDS` to tune it.

### Scanner performance

The old "2ms" figure is not a valid repository-scan claim. On an Apple M3 Pro
(12 cores, 18 GB, macOS 26.5.1), the source-view benchmark on 2026-08-09 scanned
164 files / 46,982 lines and produced the same 580 token-level detections with
both engines:

| Stage | p50 | p95 |
|---|---:|---:|
| Rust directory scan, 8 workers | 32.473 ms | 34.934 ms |
| Python directory scan | 1908.268 ms | 1929.185 ms |
| ONNX verification, 64 candidates including feature extraction | 6.802 ms | 16.907 ms |

That is a 58.76x end-to-end speedup for the regex and entropy source scan on
this workload. It is not a universal per-file guarantee. Reproduce it with:

```bash
cargo build --release --manifest-path core/Cargo.toml
HARPOCRATES_RUST_WORKERS=8 python bench/benchmark_scanner.py . \
  --iterations 20 --warmups 3
```

The benchmark's default source view excludes generated data, model artifacts,
coverage output, and repository metadata. Pass `--full-tree` to disable those
benchmark-only exclusions; the scanner's normal ignore policy still applies.

**Ships pre-trained. No user training required.**

The ML model is bundled with the package. `pip install "harpocrates[ml]"` is all you need.

---

## MCP Server

Harpocrates ships an MCP (Model Context Protocol) server over stdio. It serves two jobs: scanning for an agent (`scan_text`, `scan_file`), and the **read tool** (`safe_read`, `safe_grep`). The read tool hands an agent file contents with every detected secret replaced by a typed placeholder such as `<<HARPO:api_token:3f9a>>`.

To make Claude Code or Codex read secret-bearing files only through the read tool, run `harpocrates setup claude-code` or `harpocrates setup codex` and follow [docs/setup.md](docs/setup.md). What this does and doesn't protect is in [docs/threat-model.md](docs/threat-model.md).

**Install:**
```bash
pip install "harpocrates[mcp]"   # requires Python 3.10+
```

**Run:**
```bash
harpocrates-mcp                  # listens on stdio, speaks JSON-RPC
```

**Tools exposed:**

| Tool | Arguments | Returns |
|------|-----------|---------|
| `scan_text` | `text: str`, `include_token: bool = False`, `include_contributions: bool = False` | List of finding dicts |
| `scan_file` | `path: str`, `include_token: bool = False`, `max_bytes: int\|None = None`, `include_contributions: bool = False` | List of finding dicts |
| `safe_read` | `path: str`, `start_line: int\|None = None`, `end_line: int\|None = None` | File text with secrets replaced by placeholders, numbered like `cat -n` |
| `safe_grep` | `pattern: str`, `path: str = "."`, `max_matches: int = 200` | `file:line:text` matches, searched after redaction, so a search can't reveal a secret |

How placeholders work:
- Each session uses its own random key, and the same secret gets the same placeholder throughout that session.
- Values are never restored.
- If the ML verifier fails, every candidate is redacted.

Tokens are redacted by default (`include_token=False`). Pass `include_contributions=True` to receive TreeSHAP explanations (requires `pip install harpocrates[ml]`). The MCP process has the same filesystem read scope as the user who launched it.

---

## Explainability

Harpocrates ships opt-in TreeSHAP explanations for ML-stage findings via `--explain`. The default scan path is unchanged — XGBoost is not imported unless `--explain` is passed.

**Install:**
```bash
pip install "harpocrates[ml]"
```

**CLI:**
```bash
# Emit JSON with per-feature SHAP contributions (implies --ml)
harpocrates scan ./my_project --explain

# Inspect the top features driving the first finding
harpocrates scan ./my_project --explain | jq '.findings[0].explanation.top_positive'
```

**Output shape:**
```json
{
  "findings": [
    {
      "finding": { "type": "ML_CANDIDATE", "severity": "high", "category": "api_token", ... },
      "explanation": {
        "finding_id": "a1b2c3d4e5f6a7b8",
        "category": "api_token",
        "base_log_odds": -1.4,
        "raw_log_odds": 2.8,
        "predicted_probability": 0.943,
        "top_positive": [
          { "name": "var_ngram_secret_score", "index": 12, "value": 0.873, "contribution": 1.85, "direction": "positive" },
          { "name": "token_entropy",          "index": 3,  "value": null,  "contribution": 1.10, "direction": "positive" }
        ],
        "top_negative": [ ... ],
        "contributions": [ ... ]
      }
    }
  ]
}
```

`explanation` is `null` for regex-tier findings (they have no model decision to explain). Token-derived feature values (`token_entropy`, `token_length`, etc.) are suppressed to `null` in the output to prevent token reconstruction.

**MCP:** pass `include_contributions=true` to `scan_text` or `scan_file` to receive explanations inline.

---

## Version history

### v0.4.0 — Redesigned 64-feature vector, v0.4 retrain, opt-in TreeSHAP explainability

**Feature engineering (Phase 7.0 / 7.0.11):**

- **Dropped 8 shortcut / collinear features** that caused the model to memorise narrow synthetic distributions rather than generalise. Dropped: `var_contains_secret` (25% importance but a direct mirror of the heuristic layer — label leak), `is_known_hash_length` (inverted signal on real credentials), `cryptographic_score`, `normalized_entropy` (collinear with `token_entropy` + char counts), `line_position_ratio` (train/serve skew from synthetic estimation), `hex_context_git_keywords`, `cross_line_entropy`, `contains_example_keyword` (collinear with their sibling features). Net feature count: 65 → 64.
- **Added 7 value-shape features** that distinguish credentials from file paths, enum constants, host configs, and template placeholders — the four false-positive classes from real-world scans: `value_starts_with_slash`, `value_contains_path_separator`, `value_ends_with_known_ext`, `value_is_lowercase_word`, `value_is_dotted_quad_or_host_literal`, `value_is_template_syntax`, `is_hex_with_no_alpha_mix`.
- **Tightened XGBoost regularisation** (`max_depth` 6→5, `n_estimators` 500→300, L1/L2 increased) to break reliance on shortcut features.

**v0.4 retrain (Phase 7):**

- Retrained on ~40k samples (v4 corpus) with the redesigned 64-feature vector.
- New negative-class generators covering the four v0.3.0 FP classes: file-path values, enum constants (lowercase words), host/port configs, template placeholders (`${...}`, `{{...}}`, `__X__`), file-extension values (`.pem`, `.jks`, `.p12`).
- New positive-class generators: APIM-style compound var names (`APIM_CLIENT_KEY`, `APIM_SECRET_KEY`), multiline structures (PEM blocks, YAML pipe-block secrets, k8s `Secret` manifests, `.env` multikey blocks, JSON arrays of keys, Terraform `for_each` secret maps) encoded via `context_before`/`context_after`.
- 300-sample hand-curated positive holdout fixture (`cli/tests/fixtures/positive_holdout_v4.jsonl`); golden OOD recall **97.33%** (gate: ≥97%).

**Opt-in explainability (Phase 8):**

- `--explain` flag on `harpocrates scan` emits structured JSON with per-feature TreeSHAP contributions. Implies `--ml`. Default scan path unchanged — xgboost is never imported without `--explain`.
- `include_contributions=True` on MCP `scan_text`/`scan_file` tools adds an `explanation` key to each finding dict.
- All float outputs rounded to 3 decimal places. Token-derived feature values (`token_entropy`, `token_length`, `char_class_count`, `digit_ratio`, `special_char_ratio`) suppressed to `null` in JSON output to prevent token reconstruction.
- Booster cached per process (first `--explain` call ~30ms; subsequent calls ~0.2ms each).
- Hot-path invariant enforced by `cli/tests/test_hot_path_no_xai.py` (subprocess isolation + AST static analysis).

**Severity calibration (v0.3.2, folded into this release train):**

- `_entropy_severity()` stub replaced with a classification-aware helper — entropy/ML findings now surface at MEDIUM or HIGH instead of always INFO.
- Suffix-style var-name lexicon added (`*_KEY`, `*_SECRET_KEY`, `*_CLIENT_KEY`, etc.) so `APIM_CLIENT_KEY`, `APIM_SECRET_KEY` classify as `api_token` at confidence 0.80 → MEDIUM.
- `OPENAI_API_KEY_LEGACY` regex added to `HIGH_SIGNATURES` — catches legacy/short OpenAI keys with hyphens (`sk-RQMJj8ELDjv7TRc-...`) missed by the strict 48-char CRITICAL pattern.

---

### v0.3.0 — ML model retrained on 62k samples, comment scanning, violation classification

**New ML capabilities (what the model can now detect that it couldn't before):**

- **Commented-out secrets** — all comment styles now scanned: `#` (Python/YAML/Ruby), `//` (JS/TS/Go/Java), `/* */` (C/CSS/Java), `<!--` (HTML), `--` (SQL/Lua). A credential commented out of active code is still in your git history forever.
- **AI-scaffolded credentials** — trained on LLM-generated code patterns where AI assistants hardcode credentials during integration scaffolding (OpenAI, Stripe, Twilio, Terraform, Helm, Dockerfile).
- **Runtime generation negatives** — the model now correctly ignores `secrets.token_hex(32)`, `crypto.randomBytes(32)`, `uuid.uuid4().hex`, `SecureRandom.getInstanceStrong()`. These are runtime generation calls, not stored secrets. v0.2 would flag them.
- **Dev/prod swap patterns** — `KEY = "sk_test_x"  # PROD: sk_live_y` — the trailing comment leaks the production key.
- **Duplicate-token lines** — `password = "password123"`, `token = "token-prod-abc..."`. The ML model now handles lines where the variable name appears as a substring of the value, which confused earlier models.
- **Env-var fallback leaks** — expanded coverage across Django (`config(..., default=)`), Pydantic Settings, Spring Boot `@Value`, Rails `credentials.dig`, Go, Node.js (`??`), TypeScript/Zod, Helm `| default`.
- **Violation classification** — every finding now includes a `category` field (`password`, `api_token`, `connection_string`, `jwt`, `private_key`, `crypto_key`, `oauth_secret`, `session_token`, `webhook_url`, `generic_secret`) and a `category_reason` string explaining which heuristic fired. This tells you what to rotate, not just that something was found.

**Model metrics (v5 corpus, 62k training samples):**

| Set | Recall | Precision | F1 | AUC-ROC |
|-----|--------|-----------|-----|---------|
| Synthetic test (internal) | 99.15% | 94.61% | 0.968 | 0.9957 |
| Held-out OOD (TruffleHog-validated) | **97.32%** | **87.14%** | 0.920 | 0.9823 |

Threshold low (recall gate): `0.15` — anything above this is flagged for review.
Threshold high (precision gate): `0.85` — anything above this is reported as a confirmed secret.

---

### v0.2.0 — Three-phase pipeline, MCP server, pre-commit hook

**ML capabilities added:**

- **XGBoost + ONNX Runtime** — single-stage classifier with 64 features. Replaces the v0.1 heuristic confidence scores with a calibrated probability (Platt-scaled).
- **64-feature context model** — captures token entropy, variable name semantics (`var_ngram_secret_score`), file type risk (`file_is_config`, `file_extension_risk`), surrounding context (`context_has_function_def`), and value structure (`is_hex_with_no_alpha_mix`, `value_starts_with_slash`, `value_is_dotted_quad_or_host_literal`, `value_is_template_syntax`).
- **Dual-threshold routing** — findings are routed to `SAFE` (below threshold\_low), `REVIEW` (between thresholds), or `SECRET` (above threshold\_high). The review zone is ~5% of lines on average.
- **Entropy candidates surfaced** — high-entropy tokens under any variable name now reach the ML stage, not just those matching known patterns.
- **MCP server** — `harpocrates-mcp` exposes `scan_text` and `scan_file` as JSON-RPC tools for agent-to-agent scanning (Claude, Cursor, Codeium, etc.).
- **TokenMatch infrastructure** — internal span tracking for exact token position within a line, required for the v0.3 feature-set retrain.

**Regex signatures added in v0.2:**

| Credential | Pattern |
|-----------|---------|
| OpenAI API key | `sk-...` / `sk-proj-...` |
| Anthropic Claude key | `sk-ant-api03-...` |
| GCP API key | `AIza...` |
| NPM token | `npm_...` |
| PyPI token | `pypi-...` |
| HashiCorp Vault token | `hvs....` |

---

### v0.1.0 — Initial release ("the wedge")

**What shipped:**

- Three-phase detection pipeline: Regex → Shannon Entropy → ML confidence scoring.
- 10 regex signatures: AWS Access Key ID, GitHub PAT, Slack token, Stripe key, private key PEM header, Slack webhook, Discord webhook, SendGrid key, Twilio SID, Databricks token.
- Pre-commit hook — scans every staged file before commit. Handles binary files and `UnicodeDecodeError` without crashing.
- CLI: `harpocrates scan`, `--json`, `--fail-on`, `--ml-threshold`, `--show-secrets`, `--ignore`.
- Packaged as `py3-none-any.whl` — no native extensions, installs on any platform without compilation.

---

![Contributing](docs/assets/divider-contributing.png)

## Contributing

**False negatives are the highest-priority reports.** If Harpocrates missed a real secret, [open an issue with the `false-negative` label](https://github.com/Skipa776/Harpocrates/issues/new?labels=false-negative) and include the variable name pattern and secret type. This is the most valuable feedback you can give.

For bugs, feature requests, and false positives, open an issue at [github.com/Skipa776/Harpocrates/issues](https://github.com/Skipa776/Harpocrates/issues).

---

![License](docs/assets/divider-license.png)

## License

MIT — see [LICENSE](LICENSE).
