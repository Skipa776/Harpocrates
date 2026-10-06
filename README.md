![Harpocrates](docs/assets/banner-a-classic-parchment.png)

<h1 align="center">Harpocrates</h1>

Harpocrates is a secrets scanner: regex patterns for known credential formats, Shannon entropy for high-randomness tokens, and an ML model that decides on the rest. It also ships a read tool for AI coding agents (Claude Code, Codex): the agent reads a file with every detected secret replaced by a placeholder such as `<<HARPO:api_token:3f9a>>`, so it can work with the file without ever seeing the credential.

The model is bundled in the package and scanning makes no network calls. The egress gate — a proxy that scans every provider-bound request — is the next milestone (M2) and is not shipped yet.

## Model card

- **Model:** v17 (detector v1.2)
- **Architecture:** XGBoost, 64 features, inference via ONNX Runtime
- **Distribution:** bundled in the package — no training needed, no network calls
- **Training data:** synthetic + LLM-written code + real open-source negatives (157,066 candidate rows)
- **Thresholds:** two operating points from one model:
  - commit `0.5559` — precision-first, used by `harpocrates scan`
  - gate `0.1904` — recall-first, used by the read tool

| Operating point | Benchmark recall | Benchmark precision | OSS recall | OSS precision | AUC (benchmark / OSS) |
|---|---|---|---|---|---|
| Gate | 95.0% | 90.4% | 99.3% | 94.5% | 0.986 / 1.000 |
| Commit | 93.9% | 94.4% | 98.7% | 97.1% | same model |

Reported confidence is calibrated; per-type recall and precision are in [docs/detector-report.md](docs/detector-report.md).

Known limits:

- The benchmark is synthetic; real repositories will differ.
- A 64-character hex HMAC signing key (`[]byte("7d4e9a1f…")` in Go) is missed.
- Recall ceiling (the share of secrets the scanner turns into a candidate at all): 95.5% benchmark / 99.5% OSS.

Evaluation data was never used for training. The benchmark has been scored for several model versions, so it is now retired for choosing models; future models are selected on leave-one-LLM-family-out validation and real-code alarm rates.

## Harpocrates vs other scanners

Five scanners, same files, full scans. Two test sets, never used for training or tuning: a 4,352-record benchmark written by LLMs that generated none of the training data (half the records carry a secret, half a harmless look-alike), and 3,000 records from 42 permissively licensed GitHub repos with fake secrets inserted and real high-entropy strings as negatives. Recall and precision columns are benchmark first, then open-source code.

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

**Two operating points, one model.** The gate threshold favors recall: it is for blocking secrets before they reach an AI model, where a missed secret is the costly mistake. The commit threshold favors precision for pre-commit hooks, where false alarms interrupt developers.

**Secrets only one scanner caught** (held-out benchmark): Harpocrates **202**, detect-secrets 12, CredSweeper 10, TruffleHog 3, gitleaks 0. Per-type recall and precision are in [docs/detector-report.md](docs/detector-report.md).

### Real-world cases none of the others catch

Each case below was scanned by all five tools; only Harpocrates flagged the secret. Values are fake and shortened here; the exact files are in `docs/samples/readme_examples.jsonl`.

| Situation | Code | TruffleHog | gitleaks | detect-secrets | CredSweeper | Harpocrates |
|---|---|---|---|---|---|---|
| Database password in a YAML config | `pass: "Summer2024!…"` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Password passed as a keyword argument | `psycopg2.connect(…, password="hunter…")` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Postgres URL with a percent-encoded password | `postgresql://orders_app:Wint3r%23…@db.internal/orders` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Twilio auth token passed positionally | `twilio('AC4f2a…', '9f2c4e1a…')` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Random session secret in a Rails config | `config.session_value = "vQ3#kL9$…"` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Telegram bot token inside an API URL | `…api.telegram.org/bot7318849210:AAHx9…/getMe` | ❌ | ❌ | ❌ | ❌ | ✅ |

Why these slip past the others: none has a provider prefix like `AKIA`, `ghp_` or `sk_live_`; several use neutral names (`pass`, `session_value`, a positional argument); two are embedded in a URL. Harpocrates extracts candidates from connection strings, URL parameters and password-shaped literals, and the model decides which are secrets.

### Where the others win

- **Precision at the extreme:** TruffleHog flags almost nothing that isn't a secret (97.6%). It also verifies credentials against provider APIs; Harpocrates never makes network calls.
- **Specific formats:** CredSweeper catches more JWTs (98.7% vs 94.7%) and Vault tokens (98.6% vs 93.2%); gitleaks and CredSweeper catch every GitHub fine-grained token (100% vs 94.3%).
- **Connection strings:** TruffleHog catches every JDBC URL (100% vs 96.8%) and more Azure storage connection strings (98.8% vs 97.6%).

Recommended setup: Harpocrates as the pre-commit hook and AI-agent gate (widest coverage), TruffleHog in CI (confirms which leaked credentials are live).

How these numbers were produced, and their known biases, is in [Methods](#methods). Reproduce the real-world cases with `bench/readme_examples.py`.

## Install

```bash
pip install harpocrates            # the scanner CLI (ML model included)
harpocrates scan .
```

Optional read tool for AI agents:

```bash
pip install "harpocrates[mcp]"
harpocrates setup                  # shows detected agent harnesses
harpocrates setup claude-code      # or: codex — prints the config to add
```

Agent setup steps are in [docs/setup.md](docs/setup.md); what the setup does and doesn't protect is in [docs/threat-model.md](docs/threat-model.md). The setup has three parts: deny rules for files named like secrets, the MCP registration, and a content-aware hook that refuses built-in and shell reads of any file holding a secret and points the agent to `safe_read`. In a live Claude Code run, 0 of 15 planted canaries reached the agent; without Harpocrates, 8 of 15 did.

Pre-commit hook — add to `.pre-commit-config.yaml`, then run `pre-commit install`:

```yaml
repos:
  - repo: https://github.com/Skipa776/Harpocrates
    rev: v1.2.0
    hooks:
      - id: harpocrates
```

Usage:

```bash
harpocrates scan ./my_project              # the ML model always runs
harpocrates scan ./my_project --json       # pipe-friendly JSON output
harpocrates scan ./my_project --fail-on high
harpocrates scan ./my_project --history
harpocrates scan ./my_project --engine rust   # needs cargo build --release --manifest-path core/Cargo.toml
```

The MCP server listens on stdio (`harpocrates-mcp`, requires Python 3.10+) and exposes four tools:

- `scan_text` / `scan_file` — scan a string or a file, return findings.
- `safe_read` — file contents with every detected secret replaced by a placeholder such as `<<HARPO:api_token:3f9a>>`.
- `safe_grep` — search results produced after redaction, so a search can't reveal a secret.

Placeholders: each session uses its own random key, the same secret gets the same placeholder throughout that session, and values are never restored. If the ML verifier fails, every candidate is redacted.

## Past versions

- **v1.2.0** (model v17): higher recall, read tool for AI agents, calibrated confidence. `--ml` was removed so the model always runs (drop `--ml` from scripts and pre-commit `args`; it is now an error); the REST API, LLM verifier, and legacy training CLI were removed.
- **v0.4:** redesigned 64-feature vector, TreeSHAP `--explain`, severity calibration.
- **v0.3:** retrained on 62k samples, comment scanning, violation categories.
- **v0.2:** XGBoost + ONNX model, MCP server, pre-commit hook.
- **v0.1:** initial regex + entropy pipeline.

Full notes for each release are in that tag's README, for example `git show v0.4.0:README.md`.

## Methods

### Test sets

Two held-out sets. Neither was used for training, and neither was used to choose the model or set thresholds.

**Open-source code (3,000 records).** No one hand-labeled real secrets, and no repository with known real secrets was used. The set is built by planting generated fake secrets into real code (`scripts/fetch_oss_corpus.py`, `scripts/build_eval_set.py`):

1. **Corpus.** 200 permissively licensed GitHub repositories (MIT, Apache-2.0, BSD-3-Clause), 20 per language, 500 to 20,000 stars, each pinned to a commit in `scripts/oss_repos.tsv`. Repositories are split into train, validation and test by a hash of their name, so the 42 test repositories never appear in training.
2. **Positives (1,500).** Fake values from the project's generators, in 24 provider formats plus passwords and connection strings. Each is inserted as a new line at a random position in a real file from a test repository. The variable name is randomly secret-sounding (`api_key`), neutral (`value`), or absent (`client.connect("…")`). The label is known because the value was generated.
3. **Generated negatives (750).** Look-alikes that are not secrets (git SHAs, UUIDs, checksums, documentation examples, public keys), inserted the same way.
4. **Real-code negatives (750).** Strings sampled from those that Harpocrates' own regex and entropy stages extract from the unmodified test repositories. They are labeled not-a-secret by assumption. A string that gitleaks flags, or that matches a Harpocrates provider regex, is excluded and sent to a review queue instead.

**Held-out benchmark (4,352 records).** Code written by two LLMs that generated none of the training data (Kimi K2.6, MiniMax M2.5), with typed placeholders that were afterwards filled with generated fake values or look-alikes. Labels are correct by construction.

### Scoring

`bench/compare_scanners.py` writes each record as its own file: up to three lines before, the line, and up to three lines after, with the original file extension. All five scanners run on those files.

- A scanner gets credit only for a finding on the record's line whose value overlaps the labeled token by at least 4 characters. detect-secrets reports only a hash of each secret, so any finding on the right line counts for it.
- Recall is the share of positives flagged. Precision is flagged positives divided by all flagged records.
- TruffleHog runs with `--no-verification`, because every value is fake.
- Tool versions: TruffleHog 3.95.2, gitleaks 8.30.1, detect-secrets 1.5.0, CredSweeper 1.18.5 at its default threshold.

### Known biases and gaps

- **The real-code negatives were never reviewed.** The planned check of at least 500 samples (DATA-05) has not been done, so some "negatives" may be real secrets, and the review queues are unread.
- **The negatives favor gitleaks.** Real strings gitleaks flagged were removed before scoring, so gitleaks can't produce a false positive on real code. Its 100.0% open-source precision partly reflects this.
- **The negatives are hard for Harpocrates.** The real-code negatives are strings Harpocrates' own extractor picks up, which other scanners may never flag. This penalizes Harpocrates' precision relative to the others.
- **The secrets are planted.** Each one is a new line inserted into code that has nothing to do with it. That makes it easier to spot than a secret a developer actually wrote.
- **Precision is measured on a set that is half secrets.** Real code contains far fewer secrets, so precision on real repositories is lower. The real-code alarm rate is the better guide: at the gate threshold, 62.7 alarms per 1,000 files across 42 repositories.
- **The benchmark is synthetic.** Its code was written by LLMs, not developers, and it has been scored for several model versions, so it is retired for choosing models.
- **Recall ceiling.** 95.5% of benchmark secrets and 99.5% of open-source secrets become candidates at all; no threshold can recover the rest.

These sets are not published yet: the open-source set contains third-party code and is rebuilt from the pinned list instead. Dataset provenance is in [docs/data.md](docs/data.md).

## Contributing

False negatives are the highest-priority reports. If Harpocrates missed a real secret, [open an issue with the `false-negative` label](https://github.com/Skipa776/Harpocrates/issues/new?labels=false-negative).
For bugs, feature requests, and false positives, open an issue at [github.com/Skipa776/Harpocrates/issues](https://github.com/Skipa776/Harpocrates/issues).

## License

MIT — see [LICENSE](LICENSE).
