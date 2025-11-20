# Harpocrates

Harpocrates is a lightweight, CLI-first secrets dectection tool.
Tier 0 (repo hygine) + Tier 1 (core scan engine) are implemented here.

## Quickstart

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Skipa776/Harpocrates.git Harpocrates
cd Harpocrates

python -m venv .venv
source .venc/bin/activate # macOS / Linux
# On Windows: .venv\Scripts\activate

## Install dependencies

pip install --upgrade pip
pip install -r requirements.txt
pip install -e

## Run linting and tests

ruff check .
pytest -q

## Running the CLI

mkdir -p examples
cat > examples/fake_secrets.txt << 'EOF'
# Example fake secrets for Harpocrates demo
GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
AWS_ACCESS_KEY_ID=AKIA1234567890ABCD12
JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
EOF

### Scan

harpocrates scan --path examples/fake_secrets.txt --json
#Exit code 0: no findings
#Exit code 2: findings detected
#Exit code 1: erro