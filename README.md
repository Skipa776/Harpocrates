![Harpocrates](images/banner-a-classic-parchment.png)

<h1 align="center">Harpocrates</h1>

<p align="center">
  <strong>ML-powered secrets detection that catches what regex can't see</strong>
</p>

---

AI coding tools leak secrets at **2× the rate of human-written code** — 3.2% vs 1.6% of commits. 29 million secrets were exposed in 2025, up 34% year over year. One misconfigured environment variable cost a team **$87k in a single night**.

Regex scanners look for `AWS_ACCESS_KEY_ID` and `GITHUB_TOKEN`. They miss `client_secret`, `ENCRYPTION_KEY`, and `API_SECRET` — the names developers actually use.

Harpocrates catches what slips through.

---

## The gap

| Variable name | Secret type | TruffleHog | Harpocrates |
|--------------|-------------|-----------|-------------|
| `client_secret` | JWT | ❌ missed | ✅ caught (0.97) |
| `mock_secret` | JWT | ❌ missed | ✅ caught (0.97) |
| `ENCRYPTION_KEY` | High-entropy key | ❌ missed | ✅ caught (0.90) |
| `SECRET` | AWS Access Key ID | ❌ missed | ✅ caught (0.92) |
| `API_SECRET` | AWS Access Key ID | ❌ missed | ✅ caught (0.80) |

On a held-out evaluation set, Harpocrates caught **1,143 lines TruffleHog missed entirely** — all credentials stored under ambiguous variable names. TruffleHog caught 449 lines Harpocrates missed (live-credential API verification is its edge). They're complementary: run TruffleHog in CI, run Harpocrates before you commit.

---

![Installation](images/divider-installation.png)

## Install

```bash
pip install harpocrates
harpocrates scan .
```

With ML verification (recommended — ~95% precision, ~90% recall on real-world test set):

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

![Usage](images/divider-usage.png)

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

# Only fail CI on high or critical findings
harpocrates scan ./my_project --fail-on high

# Ignore specific patterns
harpocrates scan ./my_project --ignore "*.test.js,fixtures/*"
```

### Pre-commit hook

```yaml
repos:
  - repo: https://github.com/Skipa776/Harpocrates
    rev: v0.1.0
    hooks:
      - id: harpocrates
```

Add to `.pre-commit-config.yaml`, then run `pre-commit install`. Harpocrates scans every staged file before each commit.

---

![Configuration](images/divider-configuration.png)

## Configuration

### `harpocrates scan` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ml` | off | Enable ML verification to reduce false positives |
| `--ml-threshold FLOAT` | `0.19` | ML confidence threshold `0.0–1.0`. Lower = more recall, higher = more precision |
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
harpocrates serve            # Start the REST API server (requires harpocrates[api])
harpocrates --help           # Full command list
```

---

![How it scans](images/divider-how-it-scans.png)

## How it scans

Harpocrates runs a three-phase pipeline on every line of every file:

1. **Regex** — deterministic patterns for known credential formats (AWS, GitHub, Stripe, private keys, and more). No ML required. High-confidence, zero false positives on well-formed keys.

2. **Entropy analysis** — Shannon entropy flags high-randomness tokens that don't match any known pattern. Catches credentials stored under ambiguous variable names (`my_key`, `token`, `secret`) that regex scanners miss entirely.

3. **ML verification** (opt-in via `--ml`) — a single-stage XGBoost classifier extracts 65 features from the token, its variable name, and the surrounding code context. It learns to distinguish `api_secret = "AKIA..."` (secret) from `commit_sha = "a1b2c..."` (Git SHA) without relying on the variable name alone. Inference runs via ONNX Runtime when available, with native XGBoost as fallback.

**Ships pre-trained. No user training required.**

The ML model is bundled with the package. `pip install "harpocrates[ml]"` is all you need.

---

![Contributing](images/divider-contributing.png)

## Contributing

**False negatives are the highest-priority reports.** If Harpocrates missed a real secret, [open an issue with the `false-negative` label](https://github.com/Skipa776/Harpocrates/issues/new?labels=false-negative) and include the variable name pattern and secret type. This is the most valuable feedback you can give.

For bugs, feature requests, and false positives, open an issue at [github.com/Skipa776/Harpocrates/issues](https://github.com/Skipa776/Harpocrates/issues).

---

![License](images/divider-license.png)

## License

MIT — see [LICENSE](LICENSE).
