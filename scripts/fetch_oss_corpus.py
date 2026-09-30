#!/usr/bin/env python3
"""Select and shallow-clone permissively licensed OSS repos for data generation (DATA-04, DATA-10).

    python scripts/fetch_oss_corpus.py select   # query GitHub, write scripts/oss_repos.tsv
    python scripts/fetch_oss_corpus.py clone    # clone every repo in the list into data/oss/

The committed list pins each repo to a commit SHA, so the corpus is reproducible
without redistributing third-party code. data/oss/ is git-ignored.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "scripts" / "oss_repos.tsv"
DEST = ROOT / "data" / "oss"

LANGUAGES = ["python", "javascript", "typescript", "go", "java", "ruby", "php", "rust", "c#", "shell"]
LICENSES = ["mit", "apache-2.0", "bsd-3-clause"]
PER_LANGUAGE = 20
# Repos whose purpose is secrets or lists would skew the corpus or carry real-looking keys.
EXCLUDE = ("secret", "leak", "gitleaks", "trufflehog", "awesome", "tutorial", "interview", "public-api", "cheatsheet")


def _gh_search(language: str, license_key: str) -> list[dict]:
    out = subprocess.run(
        ["gh", "search", "repos", "--language", language, "--license", license_key,
         "--stars", "500..20000", "--size", "<50000", "--archived=false", "--limit", "40",
         "--json", "fullName,license,defaultBranch"],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def select() -> None:
    rows = []
    for language in LANGUAGES:
        seen: dict[str, dict] = {}
        for license_key in LICENSES:
            for repo in _gh_search(language, license_key):
                name = repo["fullName"]
                if not any(word in name.lower() for word in EXCLUDE):
                    seen.setdefault(name, repo)
        for name in sorted(seen)[:PER_LANGUAGE]:
            sha = subprocess.run(
                ["gh", "api", f"repos/{name}/commits/{seen[name]['defaultBranch']}", "-q", ".sha"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            rows.append((name, language, seen[name]["license"]["key"], sha))
        print(f"{language}: {min(len(seen), PER_LANGUAGE)} repos", file=sys.stderr)
    with LIST.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["repo", "language", "license", "commit"])
        writer.writerows(rows)


def clone() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with LIST.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            target = DEST / row["repo"].replace("/", "__")
            if target.exists():
                continue
            # Fetch exactly the pinned commit, no history.
            subprocess.run(["git", "init", "-q", str(target)], check=True)
            git = ["git", "-C", str(target)]
            subprocess.run([*git, "remote", "add", "origin", f"https://github.com/{row['repo']}.git"], check=True)
            fetched = subprocess.run([*git, "fetch", "-q", "--depth", "1", "origin", row["commit"]])
            if fetched.returncode != 0:
                print(f"skip {row['repo']}: fetch failed", file=sys.stderr)
                continue
            subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)


if __name__ == "__main__":
    {"select": select, "clone": clone}[sys.argv[1]]()
