<p align="center">
  <img src="Image%20Background%20Remover.png" alt="Harpocrates - Silence" width="400"/>
</p>

<h1 align="center">🤫 Harpocrates</h1>

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

## Install

```bash
pip install harpocrates
harpocrates scan .
```

With ML verification (recommended — 88% precision, 97% recall):

```bash
pip install "harpocrates[ml]"
harpocrates scan . --ml
```

---

## Pre-commit hook

```yaml
repos:
  - repo: https://github.com/Skipa776/Harpocrates
    rev: v0.1.0
    hooks:
      - id: harpocrates
```

Add to `.pre-commit-config.yaml`, then run `pre-commit install`. Harpocrates will scan every staged file before each commit.

---

## Seeing something we miss?

Open an issue at [github.com/Skipa776/Harpocrates/issues](https://github.com/Skipa776/Harpocrates/issues). False negatives (secrets we miss) are the highest-priority reports — include the variable name pattern and secret type.

---

## How it works

Harpocrates runs a three-phase pipeline: regex patterns catch known formats (AWS, GitHub, Stripe); Shannon entropy analysis flags high-randomness tokens with unknown formats; the opt-in ML stage runs a two-stage XGBoost + LightGBM ensemble that extracts 58 features from the token, its variable name, and surrounding code context — learning to distinguish `api_secret = "AKIA..."` (secret) from `commit_sha = "a1b2c..."` (Git SHA) without relying on the variable name alone.

---

## CLI reference

```
harpocrates scan [OPTIONS] [PATHS]...

Options:
  --ml                     Enable ML-based verification (recommended)
  --ml-threshold FLOAT     Confidence threshold 0.0-1.0 (default: 0.19)
  --json                   Output results as JSON
  --show-secrets           Show full token values instead of redacted previews
  --fail-on LEVEL          Severity that triggers non-zero exit: critical|high|medium|low|info|none
  --max-size INTEGER       Max file size in MB (default: 10)
  --ignore TEXT            Comma-separated glob patterns to ignore
  --recursive/--no-recursive  Scan directories recursively (default: on)
```

**Exit codes:** `0` = no findings at or above `--fail-on` · `1` = findings detected · `2` = error

---

## Requirements

- Python 3.8+
- `typer`, `rich` (core — no ML deps required for basic scan)
- `xgboost`, `lightgbm`, `scikit-learn`, `numpy`, `pandas` (with `[ml]` extra)

## License

MIT — see [LICENSE](LICENSE).
