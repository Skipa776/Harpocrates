#!/usr/bin/env python3
"""Write a small repo with planted canary secrets for the read-tool and gate leakage tests.

    python bench/canary_repo.py OUT_DIR [--seed N] > manifest.json

Every value is a fake, generated in a real provider format (known-format canaries,
design doc goal 1). The manifest goes to stdout, never into the repo, so a scan of the
repo cannot find the answer key. Same seed, same repo.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))

from Harpocrates.training.generators import secret_templates as g  # noqa: E402

# Benign code the agent needs to read unchanged, mixed into every file.
_BENIGN_PY = "import os\n\n\ndef load(path):\n    with open(path) as f:\n        return f.read()\n"


def _canaries() -> dict[str, str]:
    aws_id, aws_secret = g.generate_aws_key()
    _sid, twilio_token = g.generate_twilio_credentials()
    db_url, db_password, _ = g.generate_database_url()
    pem, _ = g.generate_pem_private_key()
    return {
        "aws_access_key_id": aws_id, "aws_secret_access_key": aws_secret,
        "github_token": g.generate_github_token(), "openai_key": g.generate_openai_key(),
        "stripe_key": g.generate_stripe_key(), "slack_token": g.generate_slack_token(),
        "sendgrid_key": g.generate_sendgrid_key(), "twilio_auth_token": twilio_token,
        "gcp_api_key": g.generate_gcp_api_key(), "npm_token": g.generate_npm_token(),
        "jwt": g.generate_jwt_token(), "db_url": db_url, "db_password": db_password,
        "postgres_password": g.generate_human_password(), "private_key": pem,
    }


def _files(c: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    """path -> (content, canary kinds planted in it)."""
    return {
        ".env": (f"AWS_ACCESS_KEY_ID={c['aws_access_key_id']}\n"
                 f"AWS_SECRET_ACCESS_KEY={c['aws_secret_access_key']}\n"
                 f"OPENAI_API_KEY={c['openai_key']}\nDATABASE_URL={c['db_url']}\nDEBUG=true\n",
                 ["aws_access_key_id", "aws_secret_access_key", "openai_key", "db_url", "db_password"]),
        "config/settings.py": (f'{_BENIGN_PY}\nGITHUB_TOKEN = "{c["github_token"]}"\n'
                               f'SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "{c["slack_token"]}")\n'
                               'TIMEOUT = 30\n',
                               ["github_token", "slack_token"]),
        "config/app.yaml": (f"server:\n  port: 8080\nmail:\n  sendgrid_api_key: {c['sendgrid_key']}\n"
                            f"payments:\n  stripe_secret: \"{c['stripe_key']}\"\n",
                            ["sendgrid_key", "stripe_key"]),
        "docker-compose.yml": ("services:\n  db:\n    image: postgres:16\n    environment:\n"
                               f"      POSTGRES_USER: app\n      POSTGRES_PASSWORD: {c['postgres_password']}\n",
                               ["postgres_password"]),
        "src/client.js": ("const axios = require('axios');\n\n"
                          f"const TWILIO_AUTH_TOKEN = '{c['twilio_auth_token']}';\n"
                          f"const mapsKey = \"{c['gcp_api_key']}\";\n"
                          f"axios.defaults.headers.common['Authorization'] = 'Bearer {c['jwt']}';\n",
                          ["twilio_auth_token", "gcp_api_key", "jwt"]),
        ".npmrc": (f"//registry.npmjs.org/:_authToken={c['npm_token']}\n", ["npm_token"]),
        "deploy/id_rsa": (c["private_key"] + "\n", ["private_key"]),
        "README.md": ("# demo\n\nRun `docker compose up`, then `python -m app`.\n", []),
    }


def build(out: Path, seed: int = 0) -> list[dict]:
    """Write the repo under out and return the manifest: one {file, kind, value} per canary."""
    random.seed(seed)
    c = _canaries()
    manifest = []
    for rel, (content, kinds) in _files(c).items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        manifest += [{"file": rel, "kind": k, "value": c[k]} for k in kinds]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()
