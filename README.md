![Harpo](https://raw.githubusercontent.com/Skipa776/Harpocrates/main/docs/assets/banner-a-classic-parchment.png)

<h1 align="center">Harpo</h1>

Harpo is a secrets scanner and a guard that keeps secrets out of AI coding agents (Claude Code and Codex). The scanner finds candidates with regex patterns and Shannon entropy, then a bundled ML model decides on the ones no pattern settles, with no network calls. The guard puts two layers around the agent: a read tool and hook that keep secrets out of what the agent reads, and an egress gate that scans every request the agent sends to its model provider.

**Status.** Version 1.4.1 (`pip install harpo-scan`) is complete: the scanner, the read tool and hook, the egress gate, `harpo guard`, `harpo doctor`, the audit logs and the Rust rules. Development has stopped; what I would build next is under [If I kept going](#if-i-kept-going). Harpo has had no outside security review, and no agent session through the gate has reached a real model provider yet; the gate's results come from a local mock (see [What is proven and what is not](#what-is-proven-and-what-is-not)). Requirements, decisions and progress are in [docs/design.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/design.md).

## At a glance

| Layer | What it does | Status | Evidence |
|---|---|---|---|
| Scanner (`harpo scan`, pre-commit) | Regex and Shannon entropy (in Rust with `harpo-scan[native]`, else the same rules in Python), then a bundled XGBoost model (ONNX, frozen at v17) decides. No network calls. | Released (v1.4.0) | 95.0% recall and 90.4% precision on a 4,352-record held-out benchmark that is half secrets, at the recall-first threshold. On real code most alarms are false (see [Precision on real code](#precision-on-real-code)). |
| Read tool (`safe_read`, `safe_grep`) and read hook | The agent's reads of secret-bearing files are refused. The agent reads them through an MCP tool that returns placeholders. | Released | Scripted sessions of the real Claude Code and Codex CLIs against a local mock provider: 0 of 22 known-format canaries received; 10 of 22 without Harpo. |
| Egress gate | A localhost proxy that parses every request to the model provider, replaces secrets with placeholders, and blocks what it cannot rewrite safely. Fails closed. | Released (v1.4.0) | 0 of 57 known-format canaries received in a corpus of real request shapes, on two seeds; 0 received in every gated agent session. |
| `harpo guard up` / `down`, `harpo doctor` | Routes both agents through the gate from their own settings files, and restores them byte-identical. | Released (v1.4.0) | Both agents routed by `guard up` alone: 0 of 11 known-format received, and `harpo doctor` passed. After `guard down`, the files are byte-identical and the same reads leak 5 of 11. |
| Audit logs (`harpo proof`) | A findings log from every layer, and a hash-chained proof log of what the gate forwarded. Hashes and counts only. | Released (v1.4.0) | Every forwarded request has a proof entry whose SHA-256 matches what the mock received. A sweep of every Harpo file found 0 canary values. |

## How it works

```mermaid
flowchart LR
    agent["Coding agent"]
    readtool["Read tool and hook"]
    gate["Egress gate on 127.0.0.1"]
    provider["Model provider"]
    core["Detection core: regex, entropy, ONNX model"]
    scan["harpo scan and pre-commit"]
    findings["Findings log"]
    proof["Proof log"]

    agent -- "file reads" --> readtool
    readtool --> agent
    agent -- "every API request" --> gate
    gate -- "redacted request" --> provider
    provider -- "response" --> gate
    gate --> agent
    readtool --> core
    gate --> core
    scan --> core
    readtool --> findings
    gate --> findings
    gate --> proof
```

- **Why two layers.** The read tool sees file context, but the agent can still `cat` a file through the shell. The gate sees every request, whatever tool produced it (complete mediation).
- **No TLS interception.** The agent sends plain HTTP to 127.0.0.1, and the gate opens its own TLS connection upstream.
- **Fail closed.** A scanner error, a body it cannot parse, an unknown content encoding or an audit log it cannot write blocks the request with a reason code (HARPO-BLOCK-*). Nothing is forwarded unscanned.

## What is proven and what is not

**Proven:**

- The agent-session results in At a glance, with Claude Code 2.1.306 and codex-cli 0.163.0, on Oct 21, 2026. Every run uses generated fake canaries, throwaway home folders and a local mock provider; no real credential and no real provider is involved. Commands and results are in [docs/testing.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/testing.md) and the design doc's Progress section.
- Fail closed: when the scanner errors, both agents are shown the block, and 0 requests reach the provider.
- The gate adds 12.2 ms p95 on follow-up turns and 347 ms p95 on first turns with a cold cache, over sessions that include 0.7 to 1 MB requests (measured locally on an Apple M3 Pro, Oct 20 to 21). CI now runs the same check, `bench/check_latency.py`, on agent sessions it records, and reports without blocking. Replaying 20 recorded sessions of about 45 KB per request on a 4-core Linux container (Oct 28), the gate adds 46.5 ms p95 on first turns with the Rust rules in process and 65.9 ms with the Python rules, and about 11 ms on follow-ups either way.
- 3,455 unit tests on both scan engines and on Python 3.10, 3.11 and 3.14, 93.8% line coverage (floor 90% in CI), 13 integration tests (three install the built wheel in a fresh venv and run it in a throwaway home; the agent suites drive the real Claude Code and Codex CLIs against the mock), and 23 Rust tests. CI runs all of them, plus the plugin tests and the unit suite on macOS, on every push to a pull request.

**Not proven or not done:**

- No session against a real provider has run through the gate yet. Streaming and subscription logins are tested against the mock only.
- The agent sessions are scripted: the mock provider tells each agent which 3 tools to call. They show the protection works inside the real CLIs, not how often a model would find a way around it. In three live Claude Code runs (Oct 19), none of 15 canaries reached the agent's context; that is too few to estimate a leak rate.
- Secrets with no known format (the model's job) do get through the gate: 12 to 13 of 23 such canaries in the request corpus.
- The read hook covers a fixed list of shell read commands. `python -c "print(open('.env').read())"` gets past it, and the gate then sees the request.
- Traffic the agent sends anywhere other than its model API (for example, `curl` to another server) is out of scope.
- The Rust and Python rules give identical candidates on this repository and on a fixture with a line for every rule, but outside ASCII they are not proven equal: a few character tests (what counts as a digit or a letter) are Rust's nearest equivalents of Python's and could differ on unusual characters.
- The model is frozen at v17. ML work stopped on Oct 28, 2026, so its known misses (below) will not be retrained away, and the planned review of the real-code negatives will not happen.
- Task success with the guard on, an evasion suite, and setup time measured by someone else are planned (M3), not done.

## Demos

Both demos run on generated canary secrets, never real ones, in a throwaway home directory. The tapes are in [docs/tapes/](https://github.com/Skipa776/Harpocrates/tree/main/docs/tapes/) and CI re-renders each GIF from its tape.

### Scene 1: `harpo scan` finds secrets a regex scanner misses

![Terminal recording: gitleaks reports two findings in a .env file, then harpo scan reports five findings in the same file, including the DATABASE_URL line.](https://raw.githubusercontent.com/Skipa776/Harpocrates/main/docs/assets/demo-scan.gif)

Measured on the seed-0 canary repo (15 planted secrets in 8 files, made by `bench/canary_repo.py`), with `harpo` from `main` before the 1.4.0 version bump at its default scan settings and gitleaks 8.30.1 at its default rules: harpo found 14 of 15 canaries and gitleaks found 11 of 15. Only harpo found the `DATABASE_URL` connection string; that is two of the canaries, the URL and the password inside it, because one finding covers both. Harpo also flagged the AWS access key id that gitleaks did not, but that one is not evidence: the generator draws the id from `A-Z0-9`, gitleaks's AWS rule accepts only the base32 characters `A-Z2-7` that real ids use, and gitleaks does find a base32-valid key. Gitleaks found the JWT, and harpo flagged that line through its entropy check. Neither found `POSTGRES_PASSWORD: ...` in the compose file. Per-secret results: [bench/results/demos/scan-vs-gitleaks.json](https://github.com/Skipa776/Harpocrates/blob/main/bench/results/demos/scan-vs-gitleaks.json); reproduce with `python bench/demos_measure.py`.

Transcript (secrets shown redacted, as the tools print them):

```text
$ gitleaks dir .env --no-banner --redact -v 2>&1 | grep -E 'RuleID|Line:|leaks found'
RuleID:      generic-api-key
Line:        2
RuleID:      generic-api-key
Line:        3
WRN leaks found: 2
$ harpo scan .env
Found 5 potential secrets
CRITICAL  AWS_ACCESS_KEY_ID   .env:1  AKIA...ISIG
CRITICAL  OPENAI_API_KEY      .env:3  sk-E...svk9
HIGH      ENV_ASSIGNMENT      .env:4  post...vice
INFO      ENTROPY_CANDIDATE   .env:2  AWS_...LpQA
MEDIUM    ENV_ASSIGNMENT      .env:2  GsNm...LpQA
$ # whole repo: harpo found 14 of 15 canaries, gitleaks 11 of 15
```

### Scene 2: an agent's read is blocked, and `safe_read` returns placeholders

![Terminal recording: a scripted Claude Code Read of .env is denied with a pointer to safe_read, then safe_read returns the file with each secret replaced by a HARPO placeholder.](https://raw.githubusercontent.com/Skipa776/Harpocrates/main/docs/assets/demo-safe-read.gif)

This scene is scripted, not a live session: a Claude Code `PreToolUse` event for reading `.env` is piped into the hook, then the MCP server's `safe_read` function is called directly. The placeholder hashes change on every run.

```text
$ python -m Harpocrates.hooks < event.json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": ".env (api_token, cloud_key, db_credential) contains secrets. Use mcp__harpo__safe_read (...) instead: it returns the same content with each secret replaced by a <<HARPO:type:hash>> placeholder."}}
$ python -c "from Harpocrates.mcp.server import safe_read; print(safe_read('.env'))"
     1  AWS_ACCESS_KEY_ID=<<HARPO:cloud_key:7fd7>>
     2  AWS_SECRET_ACCESS_KEY=<<HARPO:api_token:ff3f>>
     3  OPENAI_API_KEY=<<HARPO:api_token:9e29>>
     4  DATABASE_URL=postgresql:<<HARPO:db_credential:d151>>
     5  DEBUG=true
[harpo: 4 secret(s) redacted as <<HARPO:type:hash>>. They are not the real values; reference the variable or config key instead of copying them.]
```

## Challenges and what went wrong

These are the problems I ran into, what I changed for each, and what each one taught me.

- **Entropy and the model flagged every random-looking string, including images in web pages.** Early versions turned any long high-entropy token into a candidate, so inline base64 images (`data:` URIs) in HTML and CSS, webpack content hashes, CSS class fingerprints and SVG path data were all flagged, and the ML step did not filter them out. In v0.2.0 (Apr 2026) I stripped URLs and `data:` URIs before tokenizing and added filters for camelCase and lowercase-slug tokens, which cut false positives on a real codebase from 1,299 to 16; a May 2026 fix then skips the entropy step for .html, .css, .svg and .xml files while keeping the regex signatures, so an API key pasted into a `<script>` block is still caught. Lesson: entropy measures randomness, not secrecy, and the kind of file matters.
- **The model learned a shortcut from variable names.** With variable-name features, the model leaned on names like `password` instead of the value. Zeroing the 10 variable-name features in training raised validation AUC, so the shipped model judges the value and its surroundings, never the name. Lesson: a feature that makes the training data easy can make the real task harder.
- **I reused my benchmark until it stopped being a test.** The held-out benchmark was scored for several model versions, so it was no longer a clean test for choosing the next one. I retired it for model selection and kept it only as the published measure of the shipped model. Lesson: a held-out set is only held out the first time you look at it.
- **A large file was scanned only partly, and the scan still passed.** A single-file scan read only the first 10 MB of a larger file without saying so, so a secret after that point was missed with a clean exit. Now every file is scanned whole, in bounded pieces under a 64 MB memory cap, and a test plants a secret on the last line of a 100 MB file. Lesson: a size limit in a security tool must fail loudly, never quietly pass.
- **The Rust engine drifted from the Python one.** The Rust scanner had older copies of some patterns and lacked a whole rule phase, so on this repository it found 140 of the Python engine's 159 findings, while its parity test passed because it only compared seven sample files that none of the missing rules touched. Now Rust reports every pattern by name and a test compares the two lists, and both engines are compared field by field on a fixture with a line for every rule. Lesson: a parity test is only as good as the cases it covers.
- **Blocking reads in the agent was not enough.** The hook that refuses an agent's reads of secret-bearing files covers a fixed list of shell commands, and `python -c "print(open('app.yaml').read())"`, `perl`, `vim -es` and `xargs cat` all got through in testing. That is why I built the egress gate, which scans what actually leaves the machine, whatever tool produced it. Lesson: guard the boundary the data crosses, not each way of reaching it.

## If I kept going

Development has stopped, and these are the next steps I would take.

- **Learn from the user's "not a secret" answers, locally.** When Harpo flags a value the user knows is harmless, it should remember the answer so the same value is not flagged on the next run. The design constraints are already worked out: the hook cannot show a dialog inside the agent and has a time budget of about 25 seconds, so the read is refused at once and the question is asked out of band. The answer is stored as a keyed fingerprint (a hash, never the value), and only the user can answer, never the agent, so a prompt-injected agent cannot teach Harpo to ignore a real secret. A small local model could later learn the places where false alarms cluster. The goal is to learn your false alarms, not your secrets.
- **Protect more than credentials: high-risk personal data.** The egress gate already parses every request an agent sends and rewrites matching strings to placeholders, so the same machinery could keep the highest-risk class of data (P4 in a P1 to P4 data classification: Social Security numbers, health records, payment card and bank details) from reaching a model provider. This would likely matter to more people than API keys. The hard part is false alarms: a nine-digit number is usually not a Social Security number, so it would need validity checks (for example the Luhn check for card numbers) and context, and it was out of scope for this version.

## Model card

- **Model:** v17 (detector v1.2), frozen.
- **Architecture:** XGBoost (600 trees, depth 7), inference via ONNX Runtime. The input vector has 64 features, but the trees split on 53 of them. The 10 variable-name features are zeroed in training (see [Challenges](#challenges-and-what-went-wrong)), so the model judges the value and its surroundings, never the name (`pass`, `greeting` and `x` score alike). One more feature, `value_is_dotted_quad_or_host`, is never chosen.
- **Distribution:** bundled in the package — no training needed, no network calls. Harpo turns off ONNX Runtime's built-in Microsoft telemetry, which is on by default since ONNX Runtime 1.31. If you import onnxruntime yourself before Harpo in the same process, export ORT_DISABLE_TELEMETRY=1.
- **Training data:** synthetic + LLM-written code + real open-source negatives (157,066 candidate rows)
- **Thresholds:** two operating points from one model:
  - commit `0.5559` — precision-first, used by `harpo scan`
  - gate `0.1904` — recall-first, used by the read tool
- **Results:** recall, precision and AUC at both thresholds are under [Benchmarks](#benchmarks). Reported confidence is calibrated; per-type recall and precision are in [docs/detector-report.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/detector-report.md).

Known limits:

- A 64-character hex HMAC signing key (`[]byte("7d4e9a1f…")` in Go) is missed.
- Passwords in test files are missed. `pass = "Summer2024!"` in `test_x.py` scores 0.008, below even the gate threshold; the same line in `app.yaml` scores 0.63. Being in a test file is the second-strongest signal in the model and outweighs the shape of the value.
- Human-style passwords such as `Summer2024!` only just clear the commit threshold (calibrated confidence 0.56).
- Regex hits skip the model. Documentation-style example keys that match a provider regex are always flagged. This sets a floor on the false-positive rate: 2.8% on the benchmark and 1.6% on open-source code, whatever the threshold.
- Recall ceiling (the share of secrets the scanner turns into a candidate at all): 95.5% benchmark / 99.5% OSS. No threshold can recover the rest.

## Install

**Names.** Version 1.4.0 renamed the tool to Harpo: the command is `harpo` and the PyPI package is `harpo-scan`. The old names still work for two more minor releases; see the rename note near the end.

### 1. Scanner CLI

Install the scanner, with its ML model, and run it on a folder:

```bash
pip install harpo-scan
harpo scan .
```

Optional: `pip install "harpo-scan[native]"` adds the Rust rules, prebuilt for Linux and macOS (x86_64 and arm64). Every part of Harpo then runs its regex and entropy rules in Rust; without it, the same rules run in Python.

### 2. Protect your AI agent (Claude Code, Codex)

This adds the read tool and the hook that keep the agent off secrets:

```bash
pip install "harpo-scan[mcp]"
harpo install                  # every detected harness: Claude Code and/or Codex
harpo install --claude-code    # or name one: --codex
harpo install --dry-run        # show what would change; write and run nothing
harpo uninstall                # removes exactly what `install` added; backups stay
```

The first install takes a few minutes: it downloads ONNX Runtime and NumPy for the bundled model.

`harpo install` sets up three things: deny rules for files named like secrets, the MCP registration, and a content-aware hook that refuses built-in and shell reads of any file holding a secret and points the agent to `safe_read`.

Codex needs one manual step: start `codex`, type `/hooks`, and trust the Harpo hook once. `harpo setup` lists detected harnesses and prints the manual steps for each instead of writing them.

Setup steps are in [docs/setup.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/setup.md). What the setup does and doesn't protect is in [docs/threat-model.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/threat-model.md).

### 3. Egress gate

```bash
pip install "harpo-scan[mcp]"
harpo guard up        # start the gate on 127.0.0.1 with a new token; route Claude Code and Codex to it
harpo doctor          # check locally that each agent is routed to the running gate and nothing bypasses it
harpo proof           # the last forwarded requests, as hashes and placeholder counts; verifies the chain
harpo guard down      # restore both agents' settings files byte for byte and stop the gate
```

Restart running agent sessions after `guard up`. While the agents are routed, a stopped gate makes their requests fail to connect; nothing is sent unscanned. Details are in [docs/setup.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/setup.md#5-egress-gate-harpo-guard-up).

### 4. Claude Code plugin (optional)

In Claude Code, run:

```
/plugin marketplace add Skipa776/harpocrates
/plugin install harpocrates-prompt --marketplace Skipa776/harpocrates
```

The plugin shows the "edit without secret protection" question as a styled card and adds `/harpo` allowlist commands to the chat box. It needs the CLI hook from step 2.

### 5. Pre-commit hook (optional)

Add to `.pre-commit-config.yaml`, then run `pre-commit install`:

```yaml
repos:
  - repo: https://github.com/Skipa776/Harpocrates
    rev: v1.4.1
    hooks:
      - id: harpo
        args: ["--ignore", "graphify-out,*.lock"]   # optional
```

## How agent protection works

With step 2 in place, the agent is kept off secrets like this:

- A read of a file that holds a secret is refused, with no prompt. The agent uses `safe_read` or `safe_grep` instead, and the secrets come back as placeholders.
- Editing such a file asks once per file and session. Yes allows built-in Read and Edit of that file until the session ends.
- An edit that would write a placeholder is refused.
- Codex is refuse-only: its hooks can't ask, so a Codex edit to such a file is refused and you make that edit yourself.

The gate is the layer that sees what the hook misses, such as a program that opens a file itself. What has and has not been shown for each layer is under [What is proven and what is not](#what-is-proven-and-what-is-not).

## Allowlist

A file or folder that holds secrets but that you want the agent to read can be allowed. In a terminal:

- `harpo allow PATH` allows a file or folder. A folder covers everything under it.
- `harpo allow --name NAME` allows a folder or file name in every project, such as `graphify-out`.
- `--remove` takes one off the list and `--list` shows it.

With the Claude Code plugin, the same works from the chat box: `/harpo allow <path>`, `/harpo allow --name <name>`, `/harpo remove …` and `/harpo allowlist`. These run only when you type them there.

Only you can change the allowlist. The agent's own commands that call `harpo allow` or touch `~/.harpo` or `~/.harpocrates` are refused. Names that look like secret files, credential folders, `.harpo`, `.harpocrates`, wildcards and path separators are refused too. Every use of an allowlisted path is logged to `~/.harpo/allow.log`. See [docs/setup.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/setup.md#allowlisting-paths-and-names).

## Using the scanner

Scan a folder:

```bash
harpo scan ./my_project              # the ML model always runs
harpo scan ./my_project --json       # pipe-friendly JSON output
harpo scan ./my_project --fail-on high
harpo scan ./my_project --history
harpo scan ./my_project --engine rust   # needs cargo build --release --manifest-path core/Cargo.toml
```

**Python and Rust rules.** The regex and entropy rules exist twice: in Python, the reference, and in Rust (`core/`), which runs them about six times faster. Both carry the same pattern text, and a test compares the two lists. The Rust code ships two ways: the in-process module `harpo_native` (`pip install "harpo-scan[native]"`, or `pip install ./core` from a clone), which the read tool, the hook, the gate and the scanner use, and a scanner binary built from a clone. In auto mode, `harpo scan` uses the binary when one is built in the checkout or on the PATH, then `harpo_native`, then Python; `--engine rust` requires the binary. The ML step is the same Python code either way. On this repository (310 files, a 4-core Linux container, Oct 28) both engines report the same 2,236 raw candidates, field by field, and the same 159 findings after the model. Without the model, Rust takes 0.8 s and Python 5.0 s; with it, 1.9 s and 6.5 s.

Exclude files and directories with `--ignore`, a comma-separated list:

```bash
harpo scan . --ignore "graphify-out,tests"            # directories by name
harpo scan . --ignore "*.json,*.min.js"               # files by glob
harpo scan . --ignore "graphify-out,*.lock,fixtures"  # both
```

- A pattern without a wildcard matches a file or directory name exactly, at any depth. `graphify-out` skips that directory wherever it appears.
- A pattern with `*`, `?` or `[` is a glob matched against each name in the path.
- Patterns can't contain `/`: `app/static` matches nothing, so use `static`, which skips every directory with that name.
- Skipped by default: `.git`, `node_modules`, virtual environments, `dist`, `build`, `target`, caches, IDE folders, compiled files, archives and images. Symlinks are never followed.
- There is no ignore file yet, and `.gitignore` isn't read.

## MCP tools and placeholders

The MCP server listens on stdio (`harpo-mcp`) and exposes four tools:

- `scan_text` / `scan_file` — scan a string or a file, return findings.
- `safe_read` — file contents with every detected secret replaced by a placeholder such as `<<HARPO:api_token:3f9a>>`.
- `safe_grep` — search results produced after redaction, so a search can't reveal a secret.

Placeholder design:

- The tag is an HMAC of the value under a random 32-byte key that exists only in memory for the session. A plain hash would let anyone who sees the placeholder test guesses offline, which is cheap for a short password, and would give a secret the same tag across sessions.
- The same secret gets the same placeholder throughout a session, so the agent can tell repeated uses apart from different values. Values are never restored.
- The tag is 4 hex characters (16 bits). That keeps it short and reveals little about the value, but two different secrets of the same type can collide: about 1% likely with 36 such secrets in a session, 50% with 300. A collision makes the agent think two secrets are the same. Since values are never restored, it can't leak or misplace a secret.

Failure handling: if the model can't load, for example because its hash doesn't match, `safe_read` and `safe_grep` redact every candidate, the hook refuses files that contain any candidate, and `harpo scan` exits 2. There is no fallback to an older model.

## Benchmarks

Measured at v1.3.0 with model v17 and not re-run since. Each later change to the detector was checked to leave its findings on this repository's files unchanged; the one intended change, scanning files over 10 MB whole, does not touch these short records. Five scanners, same files, full scans, on two held-out test sets described under [Methods](#methods): a 4,352-record benchmark written by LLMs (half the records carry a secret, half a harmless look-alike), and 3,000 records from 42 open-source repositories with fake secrets planted. Recall and precision columns are benchmark first, then open-source code.

| Scanner | Recall | Precision | False-positive rate | Recall | Precision | AUC (benchmark / OSS) |
|---|---|---|---|---|---|---|
| TruffleHog 3.95 | 40.0% | 97.6% | 0.9% | 39.5% | 98.8% | n/a (yes/no only) |
| gitleaks 8.30 | 52.3% | 89.3% | 5.7% | 48.9% | 100.0% | n/a (yes/no only) |
| detect-secrets 1.5 ¹ | 55.3% | 63.8% | 28.3% | 74.0% | 78.7% | n/a (yes/no only) |
| CredSweeper 1.18 | 68.3% | 72.8% | 23.1% | 59.3% | 93.3% | 0.72 / 0.81 |
| Harpo (gate) | 95.0% | 90.4% | 9.1% | 99.3% | 94.5% | 0.986 / 1.000 ² |
| Harpo (commit) | 93.9% | 94.4% | 5.1% | 98.7% | 97.1% | same model |

¹ detect-secrets reports only a hash of each secret, so it is credited for any finding on the right line. That is lenient in its favor.
² Harpo AUC is measured over the candidates the scanner extracts. TruffleHog, gitleaks and detect-secrets output yes/no only, so they have one point on the ROC curve, not a curve. CredSweeper's AUC uses its ML probabilities (`--ml_threshold 0`).

**Compare recall only at matched false-positive rates.** The rows above run at different false-positive rates. The gate's 95.0% comes with 9.1% false positives, ten times TruffleHog's 0.9%. TruffleHog, gitleaks and detect-secrets have no threshold, so the fair question is what Harpo's recall is at *their* false-positive rate. Benchmark results for the shipped model at several thresholds:

| Harpo threshold | Recall | False-positive rate | Precision |
|---|---|---|---|
| 0.1904 (gate) | 95.0% | 9.1% | 90.4% |
| 0.5559 (commit) | 93.9% | 5.1% | 94.4% |
| 0.80 | 92.2% | 3.8% | 95.7% |
| 0.95 | 84.3% | 3.3% | 95.8% |
| regex hits only | 43.8% | 2.8% | 93.3% |

- **Against gitleaks (5.7% false positives):** the commit threshold is at 5.1%, with 93.9% recall against gitleaks' 52.3%.
- **Against TruffleHog (0.9%):** Harpo can't get that low. Regex hits bypass the model, and 58 of the 65 regex false positives are documentation example keys. That floors the rate at 2.8%, where recall is 43.8%, about TruffleHog's level at three times its false-positive rate.
- **The model itself is not the limit:** counting only the model's false positives, it reaches 91.9% recall at 0.9%. Sending regex hits on example keys through the model would remove the floor; with ML work stopped, that is not planned.

### Precision on real code

Both test sets are about half secrets, and real code is nowhere near that. Precision falls with the share of secrets, even when recall and the false-positive rate stay the same. Applying the benchmark's recall and false-positive rates at lower base rates gives:

| Share of scanned strings that are secrets | Harpo gate | Harpo commit | TruffleHog |
|---|---|---|---|
| 47% (benchmark) | 90% | 94% | 98% |
| 1% | 9.5% | 16% | 31% |
| 0.1% | 1.0% | 1.8% | 4.3% |

The benchmark's negatives are hard look-alikes, not ordinary code, so treat this table as a rough guide for every tool. The direct measurement is the real-code alarm rate: ML alarms per 1,000 unmodified files from the 16 validation repositories, 56.8 at the gate threshold and 14.9 at commit, with regex alarms on top. At one real secret per 1,000 files, that puts gate precision near 2% and commit precision near 6%. No other scanner has been run on those files yet, so there is no real-code comparison.

**Two operating points, one model.** The gate threshold favors recall. It is meant for blocking secrets before they reach an AI model, where a miss is the costly mistake and a false alarm costs one redacted string. The commit threshold is for pre-commit hooks, where false alarms interrupt developers. As the table shows, most of its alarms on real code will still be false.

**Secrets only one scanner caught** (held-out benchmark): Harpo **202**, detect-secrets 12, CredSweeper 10, TruffleHog 3, gitleaks 0. Per-type recall and precision are in [docs/detector-report.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/detector-report.md).

### Real-world cases none of the others catch

Each case below was scanned by all five tools; only Harpo flagged the secret. Values are fake and shortened here; the exact files are in `docs/samples/readme_examples.jsonl`.

| Situation | Code | TruffleHog | gitleaks | detect-secrets | CredSweeper | Harpo |
|---|---|---|---|---|---|---|
| Database password in a YAML config | `pass: "Summer2024!…"` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Password passed as a keyword argument | `psycopg2.connect(…, password="hunter…")` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Postgres URL with a percent-encoded password | `postgresql://orders_app:Wint3r%23…@db.internal/orders` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Twilio auth token passed positionally | `twilio('AC4f2a…', '9f2c4e1a…')` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Random session secret in a Rails config | `config.session_value = "vQ3#kL9$…"` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Telegram bot token inside an API URL | `…api.telegram.org/bot7318849210:AAHx9…/getMe` | ❌ | ❌ | ❌ | ❌ | ✅ |

Why these slip past the others: none has a provider prefix like `AKIA`, `ghp_` or `sk_live_`; several use neutral names (`pass`, `session_value`, a positional argument); two are embedded in a URL. Harpo extracts candidates from connection strings, URL parameters and password-shaped literals, and the model decides which are secrets. Some of these catches are narrow: the `Summer2024!` line only just clears the commit threshold, and the same line in a test file is missed (see Known limits).

### Where the others win

- **Precision at the extreme:** TruffleHog flags almost nothing that isn't a secret (97.6%, 0.9% false positives), below Harpo's 2.8% floor, and on real code its precision is likely higher than either Harpo threshold. TruffleHog also verifies credentials against provider APIs; Harpo never makes network calls.
- **Specific formats:** CredSweeper catches more JWTs (98.7% vs 94.7%) and Vault tokens (98.6% vs 93.2%); gitleaks and CredSweeper catch every GitHub fine-grained token (100% vs 94.3%).
- **Connection strings:** TruffleHog catches every JDBC URL (100% vs 96.8%) and more Azure storage connection strings (98.8% vs 97.6%).

Recommended setup: Harpo as the pre-commit hook and AI-agent gate (widest coverage), TruffleHog in CI (confirms which leaked credentials are live).

How these numbers were produced, and their known biases, is in [Methods](#methods). Reproduce the real-world cases with `bench/readme_examples.py`.

## Methods

### Test sets

Two held-out sets. Neither was used for training, and neither was used to choose the model or set thresholds.

**Open-source code (3,000 records).** No one hand-labeled real secrets, and no repository with known real secrets was used. The set is built by planting generated fake secrets into real code (reproduce at tag v1.3.0: `fetch_oss_corpus.py`, `build_eval_set.py`):

1. **Corpus.** 200 permissively licensed GitHub repositories (MIT, Apache-2.0, BSD-3-Clause), 20 per language, 500 to 20,000 stars, each pinned to a commit in `oss_repos.tsv` (at tag v1.3.0). Repositories are split into train, validation and test by a hash of their name, so the 42 test repositories never appear in training.
2. **Positives (1,500).** Fake values from the project's generators, in 24 provider formats plus passwords and connection strings. Each is inserted as a new line at a random position in a real file from a test repository. The variable name is randomly secret-sounding (`api_key`), neutral (`value`), or absent (`client.connect("…")`). The label is known because the value was generated.
3. **Generated negatives (750).** Look-alikes that are not secrets (git SHAs, UUIDs, checksums, documentation examples, public keys), inserted the same way.
4. **Real-code negatives (750).** Strings sampled from those that Harpo's own regex and entropy stages extract from the unmodified test repositories. They are labeled not-a-secret by assumption. A string that gitleaks flags, or that matches a Harpo provider regex, is excluded and sent to a review queue instead.

**Held-out benchmark (4,352 records).** Code written by two LLMs that generated none of the training data (Kimi K2.6, MiniMax M2.5), with typed placeholders that were afterwards filled with generated fake values or look-alikes. Labels are correct by construction.

### Scoring

`bench/compare_scanners.py` writes each record as its own file: up to three lines before, the line, and up to three lines after, with the original file extension. All five scanners run on those files.

- A scanner gets credit only for a finding on the record's line whose value overlaps the labeled token by at least 4 characters. detect-secrets reports only a hash of each secret, so any finding on the right line counts for it.
- Recall is the share of positives flagged. Precision is flagged positives divided by all flagged records.
- TruffleHog runs with `--no-verification`, because every value is fake.
- Tool versions: TruffleHog 3.95.2, gitleaks 8.30.1, detect-secrets 1.5.0, CredSweeper 1.18.5 at its default threshold.

### Known biases and gaps

- **The real-code negatives were never reviewed.** The check of at least 500 samples (DATA-05) was not done and, with ML work stopped, will not be, so some "negatives" may be real secrets, and the review queues are unread.
- **The negatives favor gitleaks.** Real strings gitleaks flagged were removed before scoring, so gitleaks can't produce a false positive on real code. Its 100.0% open-source precision partly reflects this.
- **The negatives are hard for Harpo.** The real-code negatives are strings Harpo's own extractor picks up, which other scanners may never flag. This penalizes Harpo's precision relative to the others.
- **The secrets are planted.** Each one is a new line inserted into code that has nothing to do with it. That makes it easier to spot than a secret a developer actually wrote.
- **Precision is measured on sets that are half secrets.** See [Precision on real code](#precision-on-real-code). On the 42 untouched test repositories, the gate threshold's real-code alarm rate is 62.7 per 1,000 files, against 56.8 on the 16 validation repositories where the thresholds were set.
- **The benchmark is synthetic.** Its code was written by LLMs, not developers, and it has been scored for several model versions, so it is retired for choosing models.

These sets are not published yet: the open-source set contains third-party code and is rebuilt from the pinned list instead. Dataset provenance is in [docs/data.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/data.md).

## Renamed from Harpocrates

v1.3.0 and earlier have only the old names: the commands `harpocrates` and `harpocrates-mcp`, the pre-commit id `harpocrates`, the MCP server `harpocrates` and the folder `~/.harpocrates`. Version 1.4.0 brought the new names below.

The project is now called Harpo. Everything you type changes with it:

| Before | Now |
|---|---|
| `harpocrates`, `harpocrates-mcp` commands | `harpo`, `harpo-mcp`. The old names still work for two more minor releases and print one deprecation line on stderr. |
| `~/.harpocrates` | `~/.harpo`. The first run copies the old folder there and leaves the original in place; once `~/.harpo` exists, the old folder is no longer read. Environment variables are `HARPO_*`; the `HARPOCRATES_*` names still work. |
| pre-commit `id: harpocrates` | `id: harpo`. The old id stays as an alias with the same command. |
| MCP server name `harpocrates` (tools `mcp__harpocrates__safe_read`, `mcp__harpocrates__safe_grep`) | `harpo` (`mcp__harpo__safe_read`, `mcp__harpo__safe_grep`). `safe_read`, `safe_grep` and the `<<HARPO:type:xxxx>>` placeholder are unchanged. |
| PyPI package `harpocrates` | `harpo-scan`. The old package is now a shim that installs `harpo-scan`. |

What to do: install the new package with `pip install harpo-scan`, then re-run `harpo install`. The install moves the MCP server, its allow rules, the notes in `CLAUDE.md` / `AGENTS.md` and the Codex profile over to the new names and takes the old ones out, without dropping any deny rule. `harpo uninstall` removes both the new and the old names. Until you re-run `harpo install`, an install from before the rename keeps working with its old names.

## Past versions

- **v1.4.1:** README fixes: the PyPI page now shows the images and its links work, and the package summary names the gate. No code changes.
- **v1.4.0:** the egress gate (Anthropic Messages and OpenAI Responses, including streaming, gzip and zstd bodies, media and base64 inside text); `harpo guard up` / `down`, `harpo doctor` and `harpo proof`; the findings and proof logs; whole-file scans under a 64 MB memory cap; every regex and entropy rule in the Rust core, matching the Python detector, as a binary and as an in-process module (`harpo_native`); the rename to Harpo.
- **v1.3.0:** agent protection reworked: reads of secret-bearing files are refused (use `safe_read`), edits ask once per file and session; `harpocrates install` / `uninstall` for Claude Code and Codex; allowlist names (`harpocrates allow --name`); Claude Code plugin `harpocrates-prompt` with `/harpocrates` allowlist commands; hook hardening.
- **v1.2.0** (model v17): higher recall, read tool for AI agents, calibrated confidence. `--ml` was removed so the model always runs (drop `--ml` from scripts and pre-commit `args`; it is now an error); the REST API, LLM verifier, and legacy training CLI were removed.
- **v0.4:** redesigned 64-feature vector, TreeSHAP `--explain`, severity calibration.
- **v0.3:** retrained on 62k samples, comment scanning, violation categories.
- **v0.2:** XGBoost + ONNX model, MCP server, pre-commit hook.
- **v0.1:** initial regex + entropy pipeline.

Full notes for each release are in that tag's README, for example `git show v0.4.0:README.md`.

## Contributing

False negatives are the highest-priority reports. If Harpo missed a real secret, [open an issue with the `false-negative` label](https://github.com/Skipa776/Harpocrates/issues/new?labels=false-negative).
For bugs, feature requests, and false positives, open an issue at [github.com/Skipa776/Harpocrates/issues](https://github.com/Skipa776/Harpocrates/issues).
To run the tests and the end-to-end suites, see [docs/testing.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/testing.md). Common problems are in [docs/troubleshooting.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/troubleshooting.md).

Where things are: the Python package (detector, read tool, hook, gate, installer, CLI) is in `cli/Harpocrates/`, with tests named after requirement IDs in `cli/tests/`; the Rust scanner is in `core/`; the mock provider, canary generator and end-to-end suites are in `bench/`; requirements, design decisions and progress are in [docs/design.md](https://github.com/Skipa776/Harpocrates/blob/main/docs/design.md).

## License

MIT — see [LICENSE](https://github.com/Skipa776/Harpocrates/blob/main/LICENSE).
