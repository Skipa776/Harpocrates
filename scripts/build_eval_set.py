#!/usr/bin/env python3
"""Build a labeled secrets dataset from the pinned OSS corpus (DATA-02..07).

    python scripts/build_eval_set.py --split test --seed 1   # -> data/eval/test_v1.jsonl

Positives: generated fake secrets (provider formats + human-style passwords)
inserted into real files with language-appropriate syntax. Variable names are
secret-sounding, neutral, or absent, so a model can't shortcut on names.
Negatives: generated lookalikes (hashes, UUIDs, doc examples) inserted the same
way, plus strings in the same real files that the scanner raises as candidates. Real strings that
gitleaks or Harpocrates' provider regexes flag are excluded and queued for review (DATA-05).

Repos are assigned to train/val/test by a hash of their name, so no repo spans
two splits (DATA-07). Output is git-ignored; rebuild with the same seed and the
pinned scripts/oss_repos.tsv to reproduce it exactly. Values are all fake.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))

from Harpocrates.core.detector import _collect_text_findings
from Harpocrates.training.generators import secret_templates as t

GENERATOR_VERSION = "3"
CORPUS = ROOT / "data" / "oss"
OUT = ROOT / "data" / "eval"
MAX_FILE_BYTES = 200_000
SKIP_DIRS = {".git", "node_modules", "vendor", "dist", "build", "target", "third_party"}

# ext -> (language, template). {i}=indent, {n}=name, {v}=value.
TEMPLATES = {
    ".py": ("python", '{i}{n} = "{v}"'), ".rb": ("ruby", '{i}{n} = "{v}"'),
    ".js": ("javascript", '{i}const {n} = "{v}";'), ".ts": ("typescript", '{i}const {n} = "{v}";'),
    ".go": ("go", '{i}{n} := "{v}"'), ".java": ("java", '{i}String {n} = "{v}";'),
    ".cs": ("csharp", '{i}var {n} = "{v}";'), ".php": ("php", '{i}${n} = "{v}";'),
    ".rs": ("rust", '{i}let {n} = "{v}";'), ".sh": ("shell", '{i}export {n}="{v}"'),
    ".yml": ("yaml", "{i}{n}: {v}"), ".yaml": ("yaml", "{i}{n}: {v}"),
    ".json": ("json", '{i}"{n}": "{v}",'), ".toml": ("toml", '{i}{n} = "{v}"'),
}
CALL_TEMPLATE = '{i}client.connect("{v}")'  # no variable name at all
CALL_LANGS = {"python", "ruby", "javascript", "typescript", "go", "java", "csharp", "php", "rust"}

POSITIVES = {
    "aws_access_key": lambda: t.generate_aws_key()[0], "aws_secret_key": lambda: t.generate_aws_key()[1],
    "github_token": t.generate_github_token, "github_fine_grained": t.generate_github_fine_grained_token,
    "stripe_key": t.generate_stripe_key, "openai_key": t.generate_openai_key,
    "slack_token": t.generate_slack_token, "gcp_api_key": t.generate_gcp_api_key,
    "digitalocean_token": t.generate_digitalocean_token, "vault_token": t.generate_hashicorp_vault_token,
    "sendgrid_key": t.generate_sendgrid_key, "discord_token": t.generate_discord_token,
    "telegram_token": t.generate_telegram_token, "jwt": t.generate_jwt_token,
    "npm_token": t.generate_npm_token, "pypi_token": t.generate_pypi_token,
    "twilio_sid": lambda: t.generate_twilio_credentials()[0],
    "azure_connection_string": t.generate_azure_connection_string,
    "generic_random": t.generate_random_secret, "password": t.generate_human_password,
    "connection_uri": t.generate_connection_uri, "ado_connection_string": t.generate_ado_connection_string,
    "jdbc_url": t.generate_jdbc_url, "token_url": t.generate_token_url,
}
NEGATIVES = {
    "git_sha": t.generate_git_sha, "uuid": t.generate_uuid, "checksum": t.generate_checksum,
    "base64_data": t.generate_base64_data, "doc_example": lambda: t.generate_documentation_example()[0],
    "encoded_non_secret": lambda: t.generate_encoded_non_secret()[0],
    "connection_placeholder": t.generate_connection_placeholder, "identifier": t.generate_identifier,
    "file_path": t.generate_file_path, "template_placeholder": t.generate_template_placeholder,
    "public_key": t.generate_public_key, "integrity_hash": t.generate_integrity_hash,
}
SECRET_NAMES = ["api_key", "secret_key", "access_token", "db_password", "auth_token", "client_secret"]
NEUTRAL_NAMES = ["value", "data", "config_value", "entry", "ref", "item_id"]
NEGATIVE_NAMES = ["checksum", "commit_sha", "request_id", "digest", "etag", "integrity"]


def split_of(repo: str) -> str:
    bucket = int(hashlib.sha256(repo.encode()).hexdigest(), 16) % 10
    return "train" if bucket < 7 else "val" if bucket == 7 else "test"


def _name(raw: str, language: str) -> str:
    if language == "shell":
        return raw.upper()
    if language in {"javascript", "typescript", "java", "csharp", "go"}:
        head, *rest = raw.split("_")
        return head + "".join(w.capitalize() for w in rest)
    return raw


def _files(repo_dir: Path) -> list[Path]:
    return sorted(
        p for p in repo_dir.rglob("*")
        if p.suffix in TEMPLATES and p.is_file() and not SKIP_DIRS & set(p.relative_to(repo_dir).parts)
        and p.stat().st_size <= MAX_FILE_BYTES
    )


def _gitleaks_hits(repo_dir: Path) -> set[tuple[str, int]]:
    with tempfile.NamedTemporaryFile(suffix=".json") as report:
        subprocess.run(
            ["gitleaks", "dir", str(repo_dir), "--report-format", "json", "--report-path", report.name,
             "--no-banner", "--exit-code", "0", "--log-level", "error"],
            check=True,
        )
        findings = json.loads(Path(report.name).read_text() or "[]")
    return {(str(Path(f["File"]).resolve().relative_to(repo_dir.resolve())), f["StartLine"]) for f in findings}


def _record(token, label, kind, source, lines, idx, line, rel, repo, language, style, name_style, seed):
    return {
        "token": token, "label": label, "secret_type": kind, "line_content": line,
        "context_before": lines[max(0, idx - 3):idx], "context_after": lines[idx:idx + 3],
        "file_path": rel, "source": source, "generator": "build_eval_set",
        "generator_version": GENERATOR_VERSION, "seed": seed, "repo": repo,
        "file_type": Path(rel).suffix, "language": language,
        "insertion_style": style, "name_style": name_style,
    }


def _insert(rng, value, kind, label, source, names, name_style, path, repo_dir, repo, seed):
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 7:
        return None
    language, template = TEMPLATES[path.suffix]
    idx = rng.randrange(3, len(lines) - 3)
    indent = re.match(r"\s*", lines[idx]).group(0)
    if name_style == "none" and language in CALL_LANGS:
        line, style = CALL_TEMPLATE.format(i=indent, v=value), "call_argument"
    else:
        name_style = "neutral" if name_style == "none" else name_style
        line = template.format(i=indent, n=_name(rng.choice(names), language), v=value)
        style = "assignment"
    rel = str(path.relative_to(repo_dir))
    return _record(value, label, kind, source, lines, idx, line, rel, repo, language, style, name_style, seed)


def _real_negatives(path, repo_dir, repo, flagged, review, seed):
    """Strings in real code that the scanner raises as candidates (regex, entropy, or ML-pending)."""
    rel = str(path.relative_to(repo_dir))
    language = TEMPLATES[path.suffix][0]
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    out, seen = [], set()
    for finding in _collect_text_findings(text):
        token, lineno = finding.token, finding.line
        if not token or not lineno or (token, lineno) in seen:
            continue
        seen.add((token, lineno))
        # A provider-format regex hit or a gitleaks hit may be a real key: review, don't label.
        if (rel, lineno) in flagged or finding.evidence.value == "regex":
            review.append({"repo": repo, "file_path": rel, "line": lineno})
            continue
        rec = _record(token, 0, "real_code_candidate", "real_code", lines, lineno - 1,
                      lines[lineno - 1], rel, repo, language, "existing", "existing", seed)
        # The token's own line is line_content, so context starts after it (as for inserted records).
        out.append({**rec, "context_after": lines[lineno:lineno + 3]})
    return out


def build(split: str, seed: int, n_pos: int, n_gen_neg: int, n_real_neg: int) -> None:
    rng = random.Random(seed)
    random.seed(seed)  # the provider generators use the global RNG
    with (ROOT / "scripts" / "oss_repos.tsv").open() as f:
        repos = [r["repo"] for r in csv.DictReader(f, delimiter="\t") if split_of(r["repo"]) == split]
    pool = []  # (repo, repo_dir, file)
    for repo in repos:
        repo_dir = CORPUS / repo.replace("/", "__")
        if repo_dir.exists():
            pool.extend((repo, repo_dir, p) for p in _files(repo_dir))
    if not pool:
        sys.exit(f"no files for split {split}; run scripts/fetch_oss_corpus.py clone first")

    records, review = [], []
    while sum(r["label"] == 1 for r in records) < n_pos:
        kind = rng.choice(sorted(POSITIVES))
        name_style = rng.choice(["secret", "neutral", "none"])
        names = SECRET_NAMES if name_style == "secret" else NEUTRAL_NAMES
        repo, repo_dir, path = rng.choice(pool)
        rec = _insert(rng, POSITIVES[kind](), kind, 1, "generated_insert", names, name_style,
                      path, repo_dir, repo, seed)
        if rec:
            records.append(rec)
    generated_neg = 0
    while generated_neg < n_gen_neg:
        kind = rng.choice(sorted(NEGATIVES))
        name_style = rng.choice(["negative", "neutral", "none"])
        names = NEGATIVE_NAMES if name_style == "negative" else NEUTRAL_NAMES
        repo, repo_dir, path = rng.choice(pool)
        rec = _insert(rng, NEGATIVES[kind](), kind, 0, "generated_negative_insert", names, name_style,
                      path, repo_dir, repo, seed)
        if rec:
            records.append(rec)
            generated_neg += 1

    real = []
    for repo in sorted({r for r, _, _ in pool}):
        repo_dir = CORPUS / repo.replace("/", "__")
        flagged = _gitleaks_hits(repo_dir)
        for path in _files(repo_dir):
            real.extend(_real_negatives(path, repo_dir, repo, flagged, review, seed))
    records.extend(rng.sample(real, min(n_real_neg, len(real))))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{split}_v{GENERATOR_VERSION}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    (OUT / f"{split}_v{GENERATOR_VERSION}_review_queue.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in review))
    print(json.dumps({
        "out": str(out.relative_to(ROOT)), "repos": len({r for r, _, _ in pool}), "files": len(pool),
        "positives": n_pos, "generated_negatives": generated_neg,
        "real_negatives": min(n_real_neg, len(real)), "real_candidates": len(real),
        "gitleaks_review_queue": len(review),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--positives", type=int, default=1500)
    parser.add_argument("--generated-negatives", type=int, default=750)
    parser.add_argument("--real-negatives", type=int, default=750)
    args = parser.parse_args()
    build(args.split, args.seed, args.positives, args.generated_negatives, args.real_negatives)


if __name__ == "__main__":
    main()
