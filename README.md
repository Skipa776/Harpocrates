# Harpocrates

**ML-powered secrets detection for code repositories**

Harpocrates is a CLI-first secrets detection tool that combines pattern matching with machine learning to identify credentials, API keys, and tokens in your codebase while minimizing false positives.

## Features

- **Three-Phase Detection Engine**
  - **Regex Patterns**: High-precision detection for known secret formats (AWS, GitHub, Stripe, etc.)
  - **Entropy Analysis**: Shannon entropy scoring for unknown high-randomness tokens
  - **ML Verification** (opt-in): XGBoost/LightGBM ensemble reduces false positives by analyzing code context

- **Context-Aware Classification**: Distinguishes between real secrets and false positives like Git SHAs, UUIDs, and test fixtures by analyzing variable names and surrounding code

- **Anti-Shortcut Design**: ML model learns from context, not token format, preventing overfitting to simple patterns

- **Safety First**: Secrets are redacted by default to prevent leakage in logs

- **CI/CD Ready**: JSON output and exit codes for pipeline integration

## Installation

### Basic Installation

```bash
git clone https://github.com/Skipa776/Harpocrates.git
cd Harpocrates

python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# On Windows: .venv\Scripts\activate

pip install -e .
```

### With ML Support

```bash
pip install -e ".[ml]"
```

### Development Installation

```bash
pip install -e ".[dev,ml]"
```

## Quick Start

### Basic Scan

```bash
# Scan a single file
harpocrates scan config.env

# Scan a directory recursively
harpocrates scan ./my_project

# Output as JSON
harpocrates scan ./my_project --json
```

### ML-Enhanced Scan (Fewer False Positives)

```bash
# Enable ML verification (recommended)
harpocrates scan ./my_project --ml

# ML with custom confidence threshold (lower = more sensitive)
harpocrates scan ./my_project --ml --ml-threshold 0.3
```

## Usage Examples

### Create Test File

```bash
cat > test_secrets.txt << 'EOF'
# Example secrets for testing
GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U

# These should NOT trigger with --ml flag:
commit_sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
user_uuid = "550e8400-e29b-41d4-a716-446655440000"
EOF
```

### Run Scan

```bash
# Standard scan (may include false positives)
harpocrates scan test_secrets.txt

# ML-enhanced scan (filters false positives)
harpocrates scan test_secrets.txt --ml

# JSON output for CI/CD
harpocrates scan test_secrets.txt --json
```

## CLI Reference

```
harpocrates scan [OPTIONS] PATH

Arguments:
  PATH                      File or directory to scan

Options:
  -r, --recursive/--no-recursive
                           Scan directories recursively (default: True)
  --json                   Output results as JSON
  --max-size INTEGER       Max file size in MB (default: 10)
  --ignore TEXT            Comma-separated patterns to ignore
  --ml                     Enable ML-based verification
  --ml-threshold FLOAT     ML confidence threshold 0.0-1.0 (default: 0.19)
  --help                   Show help message
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0    | No secrets found |
| 1    | Secrets detected |
| 2    | Error (file not found, etc.) |

## Detected Secret Types

| Type | Pattern | Example |
|------|---------|---------|
| AWS Access Key | `AKIA...` | `AKIAIOSFODNN7EXAMPLE` |
| AWS Secret Key | 40-char base64 | `wJalrXUtnFEMI/K7MDENG...` |
| GitHub Token | `ghp_`, `gho_`, `github_pat_` | `ghp_xxxxxxxxxxxx` |
| Stripe Key | `sk_live_`, `sk_test_` | `sk_live_xxxxx` |
| OpenAI Key | `sk-` | `sk-xxxxxxxx` |
| Slack Token | `xoxb-`, `xoxp-` | `xoxb-xxx-xxx` |
| GCP API Key | `AIza` | `AIzaSyxxxxxxxxx` |
| JWT | `eyJ...` (3 parts) | `eyJhbGci...` |
| Azure Key Vault | URL pattern | `*.vault.azure.net` |

## ML Training (Advanced)

Train your own model for improved accuracy on your codebase.

### Quick Start (Optimal Settings)

```bash
# Generate training and test data
harpocrates generate-data -n 7000 -o train.jsonl --seed 1337
harpocrates generate-data -n 2000 -o test.jsonl --seed 42

# Train ensemble model with high recall (recommended)
harpocrates train -d train.jsonl \
  --val-data test.jsonl \
  --model-type ensemble \
  --target-precision 0.75

# Scan with the trained model
harpocrates scan ./my_project --ml
```

### 1. Generate Training Data

```bash
# Generate 10,000 training examples (CLI command)
harpocrates generate-data -n 10000 -o training_data.jsonl

# Generate with custom balance (60% secrets, 40% non-secrets)
harpocrates generate-data -n 10000 --balance 0.6 -o training_data.jsonl

# Include hard negatives for adversarial training (recommended)
harpocrates generate-data -n 10000 --adversarial -o training_data.jsonl

# Generate with train/val/test split (80/10/10)
harpocrates generate-data -n 10000 --split -o data/training.jsonl
```

### 2. Train Model

```bash
# Train XGBoost model (default)
harpocrates train -d training_data.jsonl

# Train with validation data (recommended)
harpocrates train -d training_data.jsonl --val-data test_data.jsonl

# Train with custom output path
harpocrates train -d training_data.jsonl -o models/my_model.json

# Train LightGBM model
harpocrates train -d training_data.jsonl --model-type lightgbm

# Train ensemble (XGBoost + LightGBM) - recommended for best accuracy
harpocrates train -d training_data.jsonl --model-type ensemble

# Use k-fold cross-validation for robust evaluation
harpocrates train -d training_data.jsonl --cross-validate --folds 5

# Custom training parameters
harpocrates train -d training_data.jsonl --max-depth 8 --n-estimators 200

# HIGH RECALL (recommended) - catches 98.6% of secrets
harpocrates train -d training_data.jsonl \
  --model-type ensemble \
  --target-precision 0.75 \
  --val-data test_data.jsonl

# BALANCED - good precision/recall tradeoff
harpocrates train -d training_data.jsonl \
  --model-type ensemble \
  --target-precision 0.85 \
  --val-data test_data.jsonl

# HIGH PRECISION - fewer false positives, may miss some secrets
harpocrates train -d training_data.jsonl \
  --model-type ensemble \
  --target-precision 0.95 \
  --val-data test_data.jsonl
```

### Precision-Recall Tradeoff

The `--target-precision` parameter controls the balance between catching secrets (recall) and avoiding false positives (precision):

| Target Precision | Precision | Recall | F1 Score | Threshold | Use Case |
|------------------|-----------|--------|----------|-----------|----------|
| 0.70 | 70% | 99%+ | 0.82 | ~0.10 | Maximum security - catch everything |
| **0.75** | **75%** | **98.6%** | **0.852** | **0.19** | **Recommended - high recall** |
| 0.85 | 85% | ~90% | 0.87 | ~0.35 | Balanced |
| 0.95 | 95% | ~65% | ~0.77 | ~0.60 | Minimal noise - may miss secrets |

**Recommendation**: Use `--target-precision 0.75` to catch 98.6% of secrets with the ensemble model. The default threshold (0.19) is optimized for this balance.

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data, -d` | Training data JSONL file | Required |
| `--val-data` | Validation data JSONL file | None |
| `--output, -o` | Model output path | Auto |
| `--model-type, -m` | Model type: xgboost, lightgbm, or ensemble | xgboost |
| `--cross-validate, -cv` | Use k-fold cross-validation | False |
| `--folds, -k` | Number of CV folds | 5 |
| `--target-precision` | Target precision for threshold tuning | 0.95 |
| `--max-depth` | Maximum tree depth | 6 |
| `--learning-rate` | Learning rate | 0.1 |
| `--n-estimators` | Number of trees | 100 |
| `--seed` | Random seed for reproducibility | 42 |

### 3. Evaluate Model

```bash
# Evaluate XGBoost model
python -m Harpocrates.training.evaluate \
  --model Harpocrates/ml/models/xgboost_v1.json \
  --data test_data.jsonl

# Evaluate LightGBM model
python -m Harpocrates.training.evaluate \
  --model Harpocrates/ml/models/lightgbm_v1.txt \
  --data test_data.jsonl

# Find optimal threshold for 90% precision
python -m Harpocrates.training.evaluate \
  --model Harpocrates/ml/models/xgboost_v1.json \
  --data test_data.jsonl \
  --find-threshold --target-precision 0.90

# Show more error examples
python -m Harpocrates.training.evaluate \
  --model Harpocrates/ml/models/xgboost_v1.json \
  --data test_data.jsonl \
  --error-examples 10
```

### Evaluation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Path to trained model file | Required |
| `--data, -d` | Path to test data JSONL file | Required |
| `--threshold, -t` | Classification threshold | From config |
| `--find-threshold` | Find optimal threshold for target precision | False |
| `--target-precision` | Target precision for threshold search | 0.95 |
| `--error-examples` | Number of error examples to show | 5 |
| `--output, -o` | Output JSON file for detailed results | None |

## How ML Verification Works

The ML verifier extracts **37 features** from each finding, designed to learn from **context rather than token format** to avoid shortcut learning:

**Token Features (14)**
- Length, entropy, character class composition
- Base64 pattern matching, padding detection
- Token structure score (random vs structured)
- Version pattern detection (v1.2.3)
- Normalized entropy, cryptographic score
- Vendor prefix boost

**Variable Name Features (10)**
- Contains secret keywords (`api_key`, `password`, `token`)
- Contains safe keywords (`commit`, `sha`, `uuid`, `hash`)
- N-gram secret/safe scoring
- Naming style (camelCase, CONSTANT_CASE)
- Assignment type detection

**Context Features (13)**
- File type (test file, config file, high-risk extensions)
- Surrounding code mentions (git, test, mock, hash)
- Import statements, function definitions
- Semantic context score (-1 safe to +1 risky)
- Line position ratio (secrets often at file top)
- Surrounding secret density

### Anti-Shortcut Design

The following features were **intentionally removed** to prevent the model from learning trivial patterns:
- `has_known_prefix` - Would allow model to simply match `AKIA*`, `ghp_*`, `sk-*`
- `prefix_type` - Would directly encode secret type from prefix
- `is_hex_like` - Would strongly correlate with non-secrets (git SHAs)

This forces the model to learn from **context**, enabling it to correctly distinguish:
- `api_secret = "a1b2c3d4..."` → likely a secret (secret variable name, config context)
- `commit_sha = "a1b2c3d4..."` → likely a Git SHA (safe variable name, git context)

### Ensemble Models

For best accuracy, use the ensemble option which combines XGBoost (60% weight) and LightGBM (40% weight):

```bash
harpocrates train training_data.jsonl --model-type ensemble
```

The ensemble leverages different learning strategies:
- **XGBoost**: Level-wise tree growth, strong regularization
- **LightGBM**: Leaf-wise tree growth, histogram-based splitting

## Development

### Run Tests

```bash
pytest -v
```

### Run Linting

```bash
ruff check .
```

### Project Structure

```
Harpocrates/
  cli/                    # Command-line interface (scan, train, generate-data)
  core/                   # Core detection engine
    detector.py           # Regex + entropy detection
    scanner.py            # File/directory scanning
    result.py             # Finding data structures
  detectors/              # Detection patterns
    regex_signatures.py   # Regex patterns for known secrets
    entropy_detector.py   # Shannon entropy analysis
  ml/                     # ML verification (opt-in)
    context.py            # Context extraction
    features.py           # Feature engineering (37 features)
    verifier.py           # XGBoost classifier
    lightgbm_verifier.py  # LightGBM classifier
    ensemble.py           # Ensemble verifier (XGBoost + LightGBM)
    models/               # Trained model files
  training/               # ML training pipeline
    generators/           # Synthetic data generation
    dataset.py            # JSONL dataset loader
    train.py              # Model training (XGBoost, LightGBM, ensemble)
    cross_validation.py   # K-fold cross-validation
    evaluate.py           # Model evaluation
```

## Requirements

- Python 3.8+
- typer, rich (core)
- xgboost, lightgbm, scikit-learn, numpy, pandas (ML features)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Named after [Harpocrates](https://en.wikipedia.org/wiki/Harpocrates), the Greek god of silence and secrets.
