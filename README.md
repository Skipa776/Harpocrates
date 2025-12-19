# Harpocrates

Harpocrates is a lightweight, CLI-first secrets detection tool designed to prevent secrets from being committed to codebases. It utilizes a combination of **regex signatures** and **entropy analysis** to identify potential credentials, API keys, and tokens.

## Features

- **Dual Detection Engine**: Combines high-precision regex patterns (for known secrets like AWS, GitHub) with Shannon entropy analysis (for unknown high-randomness tokens).
- **Safety First**: Secrets are redacted by default in outputs to prevent leakage in logs.
- **Smart Filtering**: Automatically skips comments and empty lines to reduce false positives.
- **Structured Output**: Supports JSON output for integration with CI/CD pipelines.

## Quickstart

### 1. Setup environment

```bash
git clone https://github.com/Skipa776/Harpocrates.git Harpocrates
cd Harpocrates

python -m venv .venv
source .venv/bin/activate # macOS / Linux
# On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Run linting and tests

```bash
ruff check .
pytest -q
```

## Usage

### Basic Scan

mkdir -p examples
cat > examples/fake_secrets.txt << 'EOF'
# Example fake secrets for Harpocrates demo
GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
AWS_ACCESS_KEY_ID=AKIA1234567890ABCD12
JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
EOF

Run the scanner:

```bash
harpocrates scan --path examples/fake_secrets.txt --json
