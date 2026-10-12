![Harpocrates](docs/assets/banner-a-classic-parchment.png)

<h1 align="center">Harpocrates</h1>

Harpocrates is a secrets scanner. It combines regex patterns for known credential formats, Shannon entropy for high-randomness tokens, and a bundled ML model that decides on the rest, with no network calls.
It is also a guard that keeps secrets out of AI coding agents (Claude Code, Codex): the agent reads secret-bearing files through a tool that replaces each detected secret with a placeholder, so it can work with the file without the credential reaching it through that tool.
Detection misses and shell reads the hook doesn't cover are listed under [How agent protection works](#how-agent-protection-works). The egress gate, a proxy that scans every provider-bound request, is the next milestone (M2) and is not shipped yet.

## Model card

- **Model:** v17 (detector v1.2)
- **Architecture:** XGBoost (600 trees, depth 7), inference via ONNX Runtime. The input vector has 64 features, but the trees split on 53 of them. The 10 variable-name features are zeroed in training because they were a shortcut: removing them raised validation AUC. The model therefore judges the value and its surroundings, never the name (`pass`, `greeting` and `x` score alike). One more feature, `value_is_dotted_quad_or_host`, is never chosen.
- **Distribution:** bundled in the package — no training needed, no network calls. Harpocrates turns off ONNX Runtime's built-in Microsoft telemetry, which is on by default since ONNX Runtime 1.31. If you import onnxruntime yourself before Harpocrates in the same process, export ORT_DISABLE_TELEMETRY=1.
- **Training data:** synthetic + LLM-written code + real open-source negatives (157,066 candidate rows)
- **Thresholds:** two operating points from one model:
  - commit `0.5559` — precision-first, used by `harpocrates scan`
  - gate `0.1904` — recall-first, used by the read tool

| Operating point | Benchmark recall | Benchmark precision | OSS recall | OSS precision | AUC (benchmark / OSS) |
|---|---|---|---|---|---|
| Gate | 95.0% | 90.4% | 99.3% | 94.5% | 0.986 / 1.000 |
| Commit | 93.9% | 94.4% | 98.7% | 97.1% | same model |

Both test sets are about half secrets, so these precision figures are far higher than on real code. See [Precision on real code](#precision-on-real-code).

Reported confidence is calibrated; per-type recall and precision are in [docs/detector-report.md](docs/detector-report.md).

Known limits:

- The benchmark is synthetic; real repositories will differ.
- A 64-character hex HMAC signing key (`[]byte("7d4e9a1f…")` in Go) is missed.
- Passwords in test files are missed. `pass = "Summer2024!"` in `test_x.py` scores 0.008, below even the gate threshold; the same line in `app.yaml` scores 0.63. Being in a test file is the second-strongest signal in the model and outweighs the shape of the value.
- Human-style passwords such as `Summer2024!` only just clear the commit threshold (calibrated confidence 0.56).
- Regex hits skip the model. Documentation-style example keys that match a provider regex are always flagged. This sets a floor on the false-positive rate: 2.8% on the benchmark and 1.6% on open-source code, whatever the threshold.
- Recall ceiling (the share of secrets the scanner turns into a candidate at all): 95.5% benchmark / 99.5% OSS.

Evaluation data was never used for training. The benchmark has been scored for several model versions, so it is now retired for choosing models; future models are selected on leave-one-LLM-family-out validation and real-code alarm rates.

## Install

### 1. Scanner CLI

Install the scanner and run it on a folder. The ML model is included.

```bash
pip install harpocrates            # the scanner CLI (ML model included)
harpocrates scan .
```

`harpocrates --version` prints the installed version.

### 2. Protect your AI agent (Claude Code, Codex)

This adds the read tool and the hooks that keep the agent off secrets. Install the CLI (it includes the read tool and the hooks):

```bash
pip install "harpocrates[mcp]"
harpocrates install                  # every detected harness: Claude Code and/or Codex
harpocrates install --claude-code    # or name one: --codex
harpocrates install --dry-run        # show what would change; write and run nothing
harpocrates uninstall                # removes exactly what `install` added; backups stay
```

The first install takes a few minutes: it downloads ONNX Runtime and NumPy for the bundled model.

`harpocrates install` sets up three things: deny rules for files named like secrets, the MCP registration, and a content-aware hook that refuses built-in and shell reads of any file holding a secret and points the agent to `safe_read`.

Codex needs one manual step: start `codex`, type `/hooks`, and trust the Harpocrates hook once. `harpocrates setup` lists detected harnesses and prints the manual steps for each instead of writing them.

Setup steps are in [docs/setup.md](docs/setup.md). What the setup does and doesn't protect is in [docs/threat-model.md](docs/threat-model.md).

### 3. Claude Code plugin (optional)

In Claude Code, run:

```
/plugin marketplace add Skipa776/harpocrates
/plugin install harpocrates-prompt --marketplace Skipa776/harpocrates
```

The plugin shows the "edit without secret protection" question as a styled card and adds `/harpocrates` allowlist commands to the chat box. It needs the CLI hook from step 2.

### 4. Pre-commit hook (optional)

Add to `.pre-commit-config.yaml`, then run `pre-commit install`:

```yaml
repos:
  - repo: https://github.com/Skipa776/Harpocrates
    rev: v1.3.0
    hooks:
      - id: harpocrates
        args: ["--ignore", "graphify-out,*.lock"]   # optional
```

## How agent protection works

With step 2 in place, the agent is kept off secrets like this:

- A read of a file that holds a secret is refused, with no prompt. The agent uses `safe_read` or `safe_grep` instead, and the secrets come back as placeholders.
- Editing such a file asks once per file and session. Yes allows built-in Read and Edit of that file until the session ends.
- An edit that would write a placeholder is refused.
- Codex is refuse-only: its hooks can't ask, so a Codex edit to such a file is refused and you make that edit yourself.

### What has been tested

In three live Claude Code runs on one canary repo (seed 7), none of the 15 planted canaries reached the agent. In a control run without Harpocrates, 8 of 15 did. That result is narrower than it sounds:

- **The runs were cooperative.** The agent picked `Read`, `cat` and `nl`, which the hook blocks. In the task run, the hook made about 4 decisions, one per file the task asked for. The other 7 canaries were in files that were never requested.
- **This is too small to estimate a leak rate.** 0 of 15 bounds it below about 18% at 95% confidence only if the canaries were independent trials, and they aren't.
- **The shell hook covers a fixed list of read commands.** A program that opens files itself gets through: `python -c "print(open('app.yaml').read())"`, `perl`, `vim -es`, `find -exec cat`, and `xargs cat` fed by a pipe were all allowed in testing. An agent that reads files this way, by choice or because a prompt injection told it to, sees every secret.

The egress gate (M2) is the layer meant to catch what the hook misses. Until it ships, treat the read tool as protection against accidental exposure, not against an agent trying to read a secret.

## Allowlist

A file or folder that holds secrets but that you want the agent to read can be allowed. In a terminal:

- `harpocrates allow PATH` allows a file or folder. A folder covers everything under it.
- `harpocrates allow --name NAME` allows a folder or file name in every project, such as `graphify-out`.
- `--remove` takes one off the list and `--list` shows it.

With the Claude Code plugin, the same works from the chat box: `/harpocrates allow <path>`, `/harpocrates allow --name <name>`, `/harpocrates remove …` and `/harpocrates allowlist`. These run only when you type them there.

Only you can change the allowlist. The agent's own commands that call `harpocrates allow` or touch `~/.harpocrates` are refused. Names that look like secret files, credential folders, `.harpocrates`, wildcards and path separators are refused too. Every use of an allowlisted path is logged to `~/.harpocrates/allow.log`. See [docs/setup.md](docs/setup.md#allowlisting-paths-and-names).

## Using the scanner

Scan a folder:

```bash
harpocrates scan ./my_project              # the ML model always runs
harpocrates scan ./my_project --json       # pipe-friendly JSON output
harpocrates scan ./my_project --fail-on high
harpocrates scan ./my_project --history
harpocrates scan ./my_project --engine rust   # needs cargo build --release --manifest-path core/Cargo.toml
```

`--engine rust` changes only how fast regex and entropy candidates are found; the ML step is the same Python code. On this repository (295 files, Apple M3 Pro) the Python engine takes 9.4 s, Rust with one worker 144 ms, and Rust with 8 workers 100 ms. Nearly all of the speedup comes from Rust itself: 66x single-threaded, with parallel workers adding 1.4x. On this repository the two engines currently disagree by 2 findings (1,098 vs 1,096), and that mismatch hasn't been diagnosed yet.

Exclude files and directories with `--ignore`, a comma-separated list:

```bash
harpocrates scan . --ignore "graphify-out,tests"            # directories by name
harpocrates scan . --ignore "*.json,*.min.js"               # files by glob
harpocrates scan . --ignore "graphify-out,*.lock,fixtures"  # both
```

- A pattern without a wildcard matches a file or directory name exactly, at any depth. `graphify-out` skips that directory wherever it appears.
- A pattern with `*`, `?` or `[` is a glob matched against each name in the path.
- Patterns can't contain `/`: `app/static` matches nothing, so use `static`, which skips every directory with that name.
- Skipped by default: `.git`, `node_modules`, virtual environments, `dist`, `build`, `target`, caches, IDE folders, compiled files, archives and images. Symlinks are never followed.
- There is no ignore file yet, and `.gitignore` isn't read.

## MCP tools and placeholders

The MCP server listens on stdio (`harpocrates-mcp`, requires Python 3.10+) and exposes four tools:

- `scan_text` / `scan_file` — scan a string or a file, return findings.
- `safe_read` — file contents with every detected secret replaced by a placeholder such as `<<HARPO:api_token:3f9a>>`.
- `safe_grep` — search results produced after redaction, so a search can't reveal a secret.

Placeholder design:

- The tag is an HMAC of the value under a random 32-byte key that exists only in memory for the session. A plain hash would let anyone who sees the placeholder test guesses offline, which is cheap for a short password, and would give a secret the same tag across sessions.
- The same secret gets the same placeholder throughout a session, so the agent can tell repeated uses apart from different values. Values are never restored.
- The tag is 4 hex characters (16 bits). That keeps it short and reveals little about the value, but two different secrets of the same type can collide: about 1% likely with 36 such secrets in a session, 50% with 300. A collision makes the agent think two secrets are the same. Since values are never restored, it can't leak or misplace a secret.

Failure handling: if the model can't load, for example because its hash doesn't match, `safe_read` and `safe_grep` redact every candidate, the hook refuses files that contain any candidate, and `harpocrates scan` exits 2. There is no fallback to an older model.

## Benchmarks

Five scanners, same files, full scans. Two test sets, never used for training or tuning: a 4,352-record benchmark written by LLMs that generated none of the training data (half the records carry a secret, half a harmless look-alike), and 3,000 records from 42 permissively licensed GitHub repos with fake secrets inserted and real high-entropy strings as negatives. Recall and precision columns are benchmark first, then open-source code.

| Scanner | Recall | Precision | False-positive rate | Recall | Precision | AUC (benchmark / OSS) |
|---|---|---|---|---|---|---|
| TruffleHog 3.95 | 40.0% | 97.6% | 0.9% | 39.5% | 98.8% | n/a (yes/no only) |
| gitleaks 8.30 | 52.3% | 89.3% | 5.7% | 48.9% | 100.0% | n/a (yes/no only) |
| detect-secrets 1.5 ¹ | 55.3% | 63.8% | 28.3% | 74.0% | 78.7% | n/a (yes/no only) |
| CredSweeper 1.18 | 68.3% | 72.8% | 23.1% | 59.3% | 93.3% | 0.72 / 0.81 |
| Harpocrates (gate) | 95.0% | 90.4% | 9.1% | 99.3% | 94.5% | 0.986 / 1.000 ² |
| Harpocrates (commit) | 93.9% | 94.4% | 5.1% | 98.7% | 97.1% | same model |

¹ detect-secrets reports only a hash of each secret, so it is credited for any finding on the right line. That is lenient in its favor.
² Harpocrates AUC is measured over the candidates the scanner extracts. TruffleHog, gitleaks and detect-secrets output yes/no only, so they have one point on the ROC curve, not a curve. CredSweeper's AUC uses its ML probabilities (`--ml_threshold 0`).

**Compare recall only at matched false-positive rates.** The rows above run at different false-positive rates. The gate's 95.0% comes with 9.1% false positives, ten times TruffleHog's 0.9%. TruffleHog, gitleaks and detect-secrets have no threshold, so the fair question is what Harpocrates' recall is at *their* false-positive rate. Benchmark results for the shipped model at several thresholds:

| Harpocrates threshold | Recall | False-positive rate | Precision |
|---|---|---|---|
| 0.1904 (gate) | 95.0% | 9.1% | 90.4% |
| 0.5559 (commit) | 93.9% | 5.1% | 94.4% |
| 0.80 | 92.2% | 3.8% | 95.7% |
| 0.95 | 84.3% | 3.3% | 95.8% |
| regex hits only | 43.8% | 2.8% | 93.3% |

- **Against gitleaks (5.7% false positives):** the commit threshold is at 5.1%, with 93.9% recall against gitleaks' 52.3%.
- **Against TruffleHog (0.9%):** Harpocrates can't get that low. Regex hits bypass the model, and 58 of the 65 regex false positives are documentation example keys. That floors the rate at 2.8%, where recall is 43.8%, about TruffleHog's level at three times its false-positive rate.
- **The model itself is not the limit:** counting only the model's false positives, it reaches 91.9% recall at 0.9%. Sending regex hits on example keys through the model would remove the floor; that isn't done yet.

### Precision on real code

Both test sets are about half secrets, and real code is nowhere near that. Precision falls with the share of secrets, even when recall and the false-positive rate stay the same. Applying the benchmark's recall and false-positive rates at lower base rates gives:

| Share of scanned strings that are secrets | Harpocrates gate | Harpocrates commit | TruffleHog |
|---|---|---|---|
| 47% (benchmark) | 90% | 94% | 98% |
| 1% | 9.5% | 16% | 31% |
| 0.1% | 1.0% | 1.8% | 4.3% |

The benchmark's negatives are hard look-alikes, not ordinary code, so treat this table as a rough guide for every tool. The direct measurement is the real-code alarm rate: ML alarms per 1,000 unmodified files from the 16 validation repositories, 56.8 at the gate threshold and 14.9 at commit, with regex alarms on top. At one real secret per 1,000 files, that puts gate precision near 2% and commit precision near 6%. No other scanner has been run on those files yet, so there is no real-code comparison.

**Two operating points, one model.** The gate threshold favors recall. It is meant for blocking secrets before they reach an AI model, where a miss is the costly mistake and a false alarm costs one redacted string. The commit threshold is for pre-commit hooks, where false alarms interrupt developers. As the table shows, most of its alarms on real code will still be false.

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

Why these slip past the others: none has a provider prefix like `AKIA`, `ghp_` or `sk_live_`; several use neutral names (`pass`, `session_value`, a positional argument); two are embedded in a URL. Harpocrates extracts candidates from connection strings, URL parameters and password-shaped literals, and the model decides which are secrets. Some of these catches are narrow: the `Summer2024!` line only just clears the commit threshold, and the same line in a test file is missed (see Known limits).

### Where the others win

- **Precision at the extreme:** TruffleHog flags almost nothing that isn't a secret (97.6%, 0.9% false positives). Harpocrates can't operate below a 2.8% false-positive rate, and on real code TruffleHog's precision is likely higher than either Harpocrates threshold (see above). TruffleHog also verifies credentials against provider APIs; Harpocrates never makes network calls.
- **Specific formats:** CredSweeper catches more JWTs (98.7% vs 94.7%) and Vault tokens (98.6% vs 93.2%); gitleaks and CredSweeper catch every GitHub fine-grained token (100% vs 94.3%).
- **Connection strings:** TruffleHog catches every JDBC URL (100% vs 96.8%) and more Azure storage connection strings (98.8% vs 97.6%).

Recommended setup: Harpocrates as the pre-commit hook and AI-agent gate (widest coverage), TruffleHog in CI (confirms which leaked credentials are live).

How these numbers were produced, and their known biases, is in [Methods](#methods). Reproduce the real-world cases with `bench/readme_examples.py`.

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
- **Precision is measured on a set that is half secrets.** Real code contains far fewer secrets, so precision on real repositories is much lower; see [Precision on real code](#precision-on-real-code). The real-code alarm rate is the better guide: at the gate threshold, 56.8 alarms per 1,000 files on the 16 validation repositories, where the thresholds were set, and 62.7 on the 42 untouched test repositories.
- **Recall is compared at different false-positive rates.** The comparison table gives each scanner's own operating point. The matched-rate table above is the fair comparison, and Harpocrates has no operating point at TruffleHog's 0.9%.
- **The benchmark is synthetic.** Its code was written by LLMs, not developers, and it has been scored for several model versions, so it is retired for choosing models.
- **Recall ceiling.** 95.5% of benchmark secrets and 99.5% of open-source secrets become candidates at all; no threshold can recover the rest.

These sets are not published yet: the open-source set contains third-party code and is rebuilt from the pinned list instead. Dataset provenance is in [docs/data.md](docs/data.md).

## Past versions

- **v1.3.0:** agent protection reworked: reads of secret-bearing files are refused (use `safe_read`), edits ask once per file and session; `harpocrates install` / `uninstall` for Claude Code and Codex; allowlist names (`harpocrates allow --name`); Claude Code plugin `harpocrates-prompt` with `/harpocrates` allowlist commands; hook hardening.
- **v1.2.0** (model v17): higher recall, read tool for AI agents, calibrated confidence. `--ml` was removed so the model always runs (drop `--ml` from scripts and pre-commit `args`; it is now an error); the REST API, LLM verifier, and legacy training CLI were removed.
- **v0.4:** redesigned 64-feature vector, TreeSHAP `--explain`, severity calibration.
- **v0.3:** retrained on 62k samples, comment scanning, violation categories.
- **v0.2:** XGBoost + ONNX model, MCP server, pre-commit hook.
- **v0.1:** initial regex + entropy pipeline.

Full notes for each release are in that tag's README, for example `git show v0.4.0:README.md`.

## Contributing

False negatives are the highest-priority reports. If Harpocrates missed a real secret, [open an issue with the `false-negative` label](https://github.com/Skipa776/Harpocrates/issues/new?labels=false-negative).
For bugs, feature requests, and false positives, open an issue at [github.com/Skipa776/Harpocrates/issues](https://github.com/Skipa776/Harpocrates/issues).

## License

MIT — see [LICENSE](LICENSE).
