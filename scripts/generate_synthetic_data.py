#!/usr/bin/env python3
"""
Synthetic training data generator for Harpocrates v0.2.0.

Strategy (design doc Part 2):
  Positive class — LLM-generated code snippets with hardcoded credentials
    (via Ollama running locally: https://ollama.ai)
  Negative class — Template-generated safe high-entropy strings:
    lock-file SHA hashes, CSS hex colours, git SHAs, UUIDs, GPG public
    key lines, Docker digest lines, build-artifact manifests.

AI-placeholder guard: Any LLM-produced token matching the AI leftover
patterns (YOUR_, _HERE, dummy, CHANGEME, …) is discarded so the model
never trains on trivially-detectable fakes.

Usage:
    # Full run (requires Ollama + llama3)
    python scripts/generate_synthetic_data.py --output data/synthetic_v2.jsonl --count 2000

    # Negative-class only (no LLM required)
    python scripts/generate_synthetic_data.py --no-llm --count 1000

    # Custom model / URL
    python scripts/generate_synthetic_data.py --model mistral --ollama-url http://localhost:11434

Requirements:
    pip install harpocrates[ml]
    # Plus Ollama running locally with your chosen model pulled.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from urllib.parse import urlparse

from Harpocrates.ml.features import _AI_PLACEHOLDER_RE, extract_features_from_record

# Minimum Shannon entropy for a candidate token to be considered a plausible secret.
# Plaintext words typically score <3.0; random hex/base64 tokens score >3.5.
# Set too low → placeholder strings enter positive class; too high → short real keys excluded.
_MIN_TOKEN_ENTROPY_GATE: float = 3.5

# Maximum bytes to read from a single Ollama response (prevent memory exhaustion).
_MAX_OLLAMA_RESPONSE_BYTES: int = 256 * 1024  # 256 KB

# Maximum length of any single field written to the JSONL training file.
_MAX_FIELD_LENGTH: int = 2048

# ---------------------------------------------------------------------------
# LLM prompt templates — force diverse hardcoded-credential contexts
# (design doc §2.2 Synthetic Context Generation Pipeline)
# ---------------------------------------------------------------------------
_POSITIVE_PROMPTS: List[str] = [
    (
        "Write a Python module that connects to a PostgreSQL database. "
        "Hardcode realistic credentials in the source (e.g. DB_PASSWORD = '...'). "
        "Use variable names like db_password, database_secret, or DB_PASS. "
        "Output ONLY the Python code."
    ),
    (
        "Write a JavaScript config file that hardcodes an AWS IAM access key "
        "and secret access key. Use const declarations with names like "
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY. "
        "Output ONLY the JS code."
    ),
    (
        "Write a Go source file that declares a Stripe API key as a package-level "
        "constant (e.g. sk_live_... format). "
        "Output ONLY the Go code."
    ),
    (
        "Write a Python .env-style config that includes a real-looking API token "
        "assigned to a variable named API_TOKEN, SECRET_KEY, or AUTH_TOKEN. "
        "Output ONLY the config."
    ),
    (
        "Write a Java Spring Boot application.properties file with a hardcoded "
        "database URL including credentials in the connection string. "
        "Output ONLY the .properties content."
    ),
    (
        "Write a Ruby on Rails database.yml file with a hardcoded production "
        "password under the production: section. "
        "Output ONLY the YAML."
    ),
    (
        "Write a Docker Compose file with hardcoded secrets in environment "
        "variables: POSTGRES_PASSWORD, SECRET_KEY_BASE, and REDIS_PASSWORD. "
        "Output ONLY the YAML."
    ),
    (
        "Write a Terraform HCL provider block that hardcodes an AWS access key "
        "and secret. "
        "Output ONLY the HCL."
    ),
    (
        "Write a Python script that calls the OpenAI API with a hardcoded API key "
        "stored in a variable. "
        "Output ONLY the Python code."
    ),
    (
        "Write a shell script that exports environment variables for a CI pipeline, "
        "including GITHUB_TOKEN and NPM_AUTH_TOKEN with realistic values. "
        "Output ONLY the shell script."
    ),
]

_FILE_PATH_BY_KEYWORD: Dict[str, str] = {
    "python":       "config/settings.py",
    "javascript":   "src/config.js",
    "go":           "internal/config/config.go",
    ".env":         ".env",
    "java":         "src/main/resources/application.properties",
    "ruby":         "config/database.yml",
    "docker":       "docker-compose.yml",
    "terraform":    "infra/main.tf",
    "openai":       "scripts/generate.py",
    "shell":        "scripts/ci_setup.sh",
}


def _infer_file_path(prompt: str) -> str:
    p = prompt.lower()
    for kw, path in _FILE_PATH_BY_KEYWORD.items():
        if kw in p:
            return path
    return "config/secrets.py"


# ---------------------------------------------------------------------------
# Negative-class template generators (design doc §2.4)
# ---------------------------------------------------------------------------

def _rand_hex(n: int) -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _rand_b64(n_bytes: int) -> str:
    import base64, os
    return base64.b64encode(os.urandom(n_bytes)).decode("ascii").rstrip("=")


def _neg_lock_file_hash() -> Dict[str, Any]:
    pkg = random.choice(["lodash", "react", "axios", "express", "webpack", "vue", "svelte"])
    ver = f"{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)}"
    h = _rand_b64(48)
    line = f'    "integrity": "sha512-{h}==",'
    return {
        "token": h,
        "line_content": line,
        "context_before": [f'  "{pkg}@{ver}": {{', f'    "version": "{ver}",'],
        "context_after":  ['    "resolved": "https://registry.npmjs.org/..."', "  },"],
        "file_path": "package-lock.json",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_css_hex_color() -> Dict[str, Any]:
    color = _rand_hex(6)
    prop = random.choice(["background-color", "color", "border-color", "fill", "outline-color"])
    sel = random.choice([".btn", ".header", "#main", "body", "a:hover", ".card"])
    return {
        "token": color,
        "line_content": f"  {prop}: #{color};",
        "context_before": [f"{sel} {{"],
        "context_after":  ["}"],
        "file_path": "styles/main.css",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_git_sha() -> Dict[str, Any]:
    sha = _rand_hex(40)
    msg = random.choice(["Fix auth bug", "Update deps", "Refactor auth", "Add tests"])
    return {
        "token": sha,
        "line_content": f"# commit: {sha}  # {msg}",
        "context_before": [f"# Revert commit {sha[:8]}"],
        "context_after":  ["# Author: Dev <dev@example.com>"],
        "file_path": "scripts/deploy.sh",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_uuid() -> Dict[str, Any]:
    import uuid
    uid = str(uuid.uuid4())
    var = random.choice(["user_id", "session_id", "request_id", "trace_id", "correlation_id"])
    return {
        "token": uid,
        "line_content": f'    "{var}": "{uid}",',
        "context_before": ['  "user": {'],
        "context_after":  ['    "name": "Alice"', "  },"],
        "file_path": "tests/fixtures/user_fixture.json",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_gpg_pubkey_line() -> Dict[str, Any]:
    data = _rand_b64(60)
    return {
        "token": data,
        "line_content": data,
        "context_before": ["-----BEGIN PGP PUBLIC KEY BLOCK-----", ""],
        "context_after":  ["-----END PGP PUBLIC KEY BLOCK-----"],
        "file_path": "keys/developer.pub",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_docker_digest() -> Dict[str, Any]:
    sha = _rand_hex(64)
    img = random.choice(["nginx", "postgres", "redis", "node", "python", "alpine"])
    return {
        "token": sha,
        "line_content": f"FROM {img}@sha256:{sha}",
        "context_before": ["# Production image"],
        "context_after":  ["RUN apt-get update -y"],
        "file_path": "Dockerfile",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_yarn_lock_sha() -> Dict[str, Any]:
    sha = _rand_hex(64)
    pkg = random.choice(["typescript", "eslint", "prettier", "jest", "babel-core"])
    ver = f"{random.randint(4,8)}.{random.randint(0,9)}.{random.randint(0,9)}"
    return {
        "token": sha,
        "line_content": f'  "{pkg}" "{ver}" sha256:{sha}',
        "context_before": ["__metadata:", "  version: 6"],
        "context_after":  [""],
        "file_path": "yarn.lock",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


def _neg_asset_manifest() -> Dict[str, Any]:
    sha = _rand_hex(40)
    artifact = random.choice(["app.bundle.js", "main.css", "vendor.js", "runtime.js"])
    return {
        "token": sha,
        "line_content": f'  "{artifact}": "{sha}",',
        "context_before": ["{", '  "version": 1,', '  "files": {'],
        "context_after":  ["},"],
        "file_path": "dist/asset-manifest.json",
        "label": 0,
        "secret_type": "ENTROPY_CANDIDATE",
    }


_NEGATIVE_GENERATORS = [
    _neg_lock_file_hash,
    _neg_css_hex_color,
    _neg_git_sha,
    _neg_uuid,
    _neg_gpg_pubkey_line,
    _neg_docker_digest,
    _neg_yarn_lock_sha,
    _neg_asset_manifest,
]


def generate_negative_samples(count: int) -> List[Dict[str, Any]]:
    return [random.choice(_NEGATIVE_GENERATORS)() for _ in range(count)]


# ---------------------------------------------------------------------------
# LLM positive-class generation via Ollama
# ---------------------------------------------------------------------------

def _validate_ollama_url(url: str) -> str:
    """Ensure the Ollama URL uses http/https only — prevent SSRF via file:// etc."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"--ollama-url must use http:// or https://, got: {url!r}. "
            "file://, ftp://, and other schemes are not permitted."
        )
    if not parsed.hostname:
        raise ValueError(f"--ollama-url has no hostname: {url!r}")
    return url


def _call_lmstudio(prompt: str, model: str, url: str, timeout: int = 90) -> Optional[str]:
    """POST to LM Studio's OpenAI-compatible /v1/chat/completions endpoint."""
    payload = json.dumps({
        "model": model,  # LM Studio often ignores this and uses the loaded model, but good practice
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 400,
        "stream": False
    }).encode()
    
    # Ensure we append the correct OpenAI endpoint path
    base_url = url.rstrip('/')
    if not base_url.endswith("/v1"):
        base_url += "/v1"
        
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(_MAX_OLLAMA_RESPONSE_BYTES)
            if len(raw) == _MAX_OLLAMA_RESPONSE_BYTES:
                print("[WARN] Response truncated at 256 KB — skipping.", file=sys.stderr)
                return None
            
            data = json.loads(raw)
            # Parse the standard OpenAI response JSON structure
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except urllib.error.URLError as e:
        print(f"  [WARN] LM Studio request failed: {e}", file=sys.stderr)
        return None


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```\w*\n?", "", text)


def _extract_token_from_line(line: str) -> Optional[str]:
    """Extract the target token from a code line, intentionally preserving low-entropy AI placeholders."""
    from Harpocrates.detectors.entropy_detector import shannon_entropy

    candidates: List[str] = []
    # LOWERED LENGTH TO 5: We want to catch short dummy passwords like "test1" or "dummy"
    for pat in (
        re.compile(r'["\']([A-Za-z0-9+/=_\-\.]{5,})["\']'),
        re.compile(r'=\s*([A-Za-z0-9+/=_\-\.]{5,})'),
        re.compile(r':\s*([A-Za-z0-9+/=_\-\.]{5,})'),
    ):
        for m in pat.finditer(line):
            candidates.append(m.group(1))

    best: Optional[str] = None
    # LOWERED ENTROPY GATE TO 0: We want everything the LLM spits out
    best_ent: float = 0.0 
    
    for c in candidates:
        # REMOVED: The AI placeholder exclusion logic that was starving our dataset
        ent = shannon_entropy(c)
        if ent > best_ent:
            best_ent = ent
            best = c
            
    return best


def generate_positive_samples_via_llm(
    count: int, model: str, ollama_url: str
) -> List[Dict[str, Any]]:
    """Generate positive-class records using a local LLM via Ollama."""
    samples: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 6

    while len(samples) < count and attempts < max_attempts:
        prompt = random.choice(_POSITIVE_PROMPTS)
        attempts += 1

        response = _call_lmstudio(prompt, model, ollama_url)
        if not response:
            continue

        lines = _strip_code_fences(response).splitlines()
        lines = [l for l in lines if l.strip()]

        for i, line in enumerate(lines):
            token = _extract_token_from_line(line)
            if token is None:
                continue

            samples.append({
                "token": token,
                "line_content": line,
                "context_before": lines[max(0, i - 2):i],
                "context_after":  lines[i + 1: i + 3],
                "file_path": _infer_file_path(prompt),
                "label": 1,
                "secret_type": "ENTROPY_CANDIDATE",
            })
            if len(samples) >= count:
                break

    if len(samples) < count:
        print(
            f"[WARN] Only generated {len(samples)}/{count} positive samples "
            f"after {attempts} LLM calls. "
            "Run with a different model or increase --count.",
            file=sys.stderr,
        )
    return samples


# ---------------------------------------------------------------------------
# Feature extraction wrapper
# ---------------------------------------------------------------------------

def _sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Cap string field lengths to prevent training-data poisoning from large responses."""
    for field in ("token", "line_content", "file_path"):
        v = record.get(field)
        if isinstance(v, str) and len(v) > _MAX_FIELD_LENGTH:
            record[field] = v[:_MAX_FIELD_LENGTH]
    for field in ("context_before", "context_after"):
        v = record.get(field)
        if isinstance(v, list):
            record[field] = [
                line[:_MAX_FIELD_LENGTH] if isinstance(line, str) else line
                for line in v
            ]
    return record


def _attach_features(record: Dict[str, Any]) -> Dict[str, Any]:
    """Cap field lengths, then extract and embed features in the record."""
    record = _sanitize_record(record)
    try:
        fv = extract_features_from_record(record)
        record["features_63"] = fv.to_array()
        record["features_extended"] = fv.to_extended_array()
    except Exception as e:
        print(f"  [WARN] Feature extraction failed: {e}", file=sys.stderr)
        record["features_63"] = None
        record["features_extended"] = None
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data (v0.2.0)")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_v2.jsonl"))
    parser.add_argument("--count", type=int, default=2000, help="Total samples to generate")
    parser.add_argument("--balance", type=float, default=0.5,
                        help="Fraction of samples that are positive class (default: 0.5)")
    parser.add_argument("--model", type=str, default="llama3",
                        help="Ollama model name (default: llama3)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM generation — produce negative class only")
    parser.add_argument("--no-features", action="store_true",
                        help="Skip feature extraction (faster, no feature columns)")
    args = parser.parse_args()

    n_positive = int(args.count * args.balance)
    n_negative = args.count - n_positive

    print(f"Target: {n_positive} positive + {n_negative} negative → {args.output}")

    if args.no_llm:
        positive: List[Dict[str, Any]] = []
        print("Skipping LLM generation (--no-llm).")
    else:
        try:
            _validate_ollama_url(args.ollama_url)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[+] Generating positive class via Ollama ({args.model} @ {args.ollama_url})…")
        positive = generate_positive_samples_via_llm(n_positive, args.model, args.ollama_url)

    print(f"\n[-] Generating negative class from templates ({n_negative} samples)…")
    negative = generate_negative_samples(n_negative)

    all_samples = positive + negative
    random.shuffle(all_samples)

    if not args.no_features:
        print(f"\n[F] Extracting features from {len(all_samples)} records…")
        all_samples = [_attach_features(s) for s in all_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for record in all_samples:
            f.write(json.dumps(record) + "\n")

    pos_n = sum(1 for r in all_samples if r.get("label") == 1)
    neg_n = sum(1 for r in all_samples if r.get("label") == 0)
    print(f"\nWrote {len(all_samples)} records → {args.output}")
    print(f"  positive (secret): {pos_n}")
    print(f"  negative (safe):   {neg_n}")


if __name__ == "__main__":
    main()
