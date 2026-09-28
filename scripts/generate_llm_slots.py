#!/usr/bin/env python3
"""Generate code files with typed value slots from any OpenAI-compatible LLM.

    python scripts/generate_llm_slots.py --model deepseek-coder-6.7b-instruct --count 1000
    python scripts/generate_llm_slots.py --cli codex --model gpt-6-luna --count 1000
    python scripts/generate_llm_slots.py --cli claude --model claude-sonnet-5-5 --effort low --count 500
    python scripts/generate_llm_slots.py --cli agy --model gemini-3.8-flash-low --count 500
    python scripts/generate_llm_slots.py --cli opencode --model tritonai/deepseek-v4-flash --count 500

Files land in data/llm_slots/<model>/ with a manifest.jsonl. Then:
    python scripts/build_slot_set.py data/llm_slots/<model> data/llm_<model>.jsonl
The LLM writes code only; scripts/build_slot_set.py fills each {{SECRET:kind}} /
{{NONSECRET:kind}} slot with a fake value, so labels are correct by construction.
Resumable: existing files are skipped. Keep training and benchmark models disjoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_set import NEGATIVES, POSITIVES
from build_slot_set import ANY_MARKER, SLOT

LANGUAGES = {  # language -> file suffix (or full name)
    "Python": ".py", "JavaScript": ".js", "TypeScript": ".ts", "Go": ".go", "Java": ".java",
    "C#": ".cs", "Ruby": ".rb", "PHP": ".php", "Rust": ".rs", "Kotlin": ".kt", "Scala": ".scala",
    "Swift": ".swift", "C": ".c", "C++": ".cpp", "Bash": ".sh", "PowerShell": ".ps1",
    "YAML": ".yaml", "JSON": ".json", "TOML": ".toml", "INI": ".ini", "dotenv": ".env",
    "Dockerfile": "Dockerfile", "Terraform": ".tf", "SQL": ".sql", "XML": ".xml",
    "Java properties": ".properties", "Gradle (Groovy)": ".gradle",
}
FILE_KINDS = ["application module", "configuration file", "unit test", "deployment script",
              "CI workflow", "database migration", "API client", "CLI entry point"]
PROJECT_KINDS = ["web API", "CLI tool", "data pipeline", "mobile backend", "infrastructure repo",
                 "e-commerce site", "chat bot", "internal admin tool"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FENCE = re.compile(r"^```[\w+-]*\n(.*?)\n?```\s*$", re.S)

PROMPT = """Write a realistic {language} {file_kind} (20-40 lines) from a {project_kind} project.
Mark every place a string value goes with exactly one of these markers, verbatim, including braces:
  {{{{SECRET:<kind>}}}}     where a real app would hardcode a credential
  {{{{NONSECRET:<kind>}}}}  where a real app would put a high-entropy but harmless value
Use exactly these markers, once each: {markers}
Vary how values appear: assignment, dict/map literal, keyword argument, function call,
config block, env-var default, HTTP header, URL query parameter, or comment.
Use ordinary, varied variable names; don't always name secrets "api_key" or "secret".
Give at least one NONSECRET marker a credential-sounding name (e.g. token_url, secret_name).
Write each marker exactly where the value would be, inside quotes if the value is quoted.
Never add comments or names that say whether a value is sensitive, secret, safe, or public.
Output only the file contents, no explanation."""


def jobs(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        language = rng.choice(sorted(LANGUAGES))
        out.append({
            "id": i, "language": language, "file_kind": rng.choice(FILE_KINDS),
            "project_kind": rng.choice(PROJECT_KINDS),
            "secret_kinds": rng.sample(sorted(POSITIVES), rng.randint(1, 2)),
            "nonsecret_kinds": rng.sample(sorted(NEGATIVES), rng.randint(1, 2)),
        })
    return out


def clean_output(text: str) -> str | None:
    """Strip a markdown fence; return None unless every marker is a valid typed slot."""
    text = text.strip()
    if fenced := FENCE.match(text):
        text = fenced.group(1)
    markers = ANY_MARKER.findall(text)
    if not markers:
        return None
    for marker in markers:
        slot = SLOT.fullmatch(marker)
        if not slot or slot.group(2) not in (POSITIVES if slot.group(1) == "SECRET" else NEGATIVES):
            return None
    return text + "\n"


def _filename(job: dict) -> str:
    suffix = LANGUAGES[job["language"]]
    return f"{job['id']:05d}_{suffix}" if not suffix.startswith(".") else f"{job['id']:05d}{suffix}"


# opencode's default agent can write files, run shell commands, and use the user's MCP servers
# (e.g. Playwright); an unrestricted run once wrote ~90 files into this repo. Generation must be
# text-only, so every tool and MCP server is disabled via an inline config.
_OPENCODE_TEXT_ONLY = json.dumps({
    "permission": {"edit": "deny", "bash": "deny", "webfetch": "deny"},
    "tools": {t: False for t in ("write", "edit", "bash", "patch", "apply_patch", "webfetch", "websearch",
                                 "read", "glob", "grep", "list", "todowrite", "task", "skill", "question")},
    "mcp": {"playwright": {"type": "local", "command": ["true"], "enabled": False}},
})


def _cli_command(args, prompt: str, out_file: Path) -> list[str]:
    # Each CLI must run text-only: no built-in tools, no user MCP servers, plugins, or hooks.
    if args.cli == "codex":
        return ["codex", "exec", "-m", args.model, "--ignore-user-config", "--sandbox", "read-only",
                "--skip-git-repo-check", "--ephemeral", "-o", str(out_file), prompt]
    if args.cli == "claude":
        # --tools "" removes built-ins only; user MCP servers need --strict-mcp-config + empty config.
        return ["claude", "-p", "--model", args.model, "--effort", args.effort, "--tools", "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
                "--settings", '{"disableAllHooks": true}', "--no-session-persistence", prompt]
    if args.cli == "agy":
        # Headless agy auto-denies write tools; plan mode + sandbox restrict it further.
        return ["agy", "--model", args.model, "--mode", "plan", "--sandbox",
                "--print-timeout", f"{args.timeout}s", "--print", prompt]
    return ["opencode", "run", "--pure", "--dir", str(out_file.parent), "-m", args.model, prompt]


async def _ask_cli(args, prompt: str) -> str:
    # Empty working dir: no project CLAUDE.md/AGENTS.md or repo files leak into the prompt.
    with tempfile.TemporaryDirectory() as cwd:
        out_file = Path(cwd) / "last_message.txt"
        env = {**os.environ, "OPENCODE_CONFIG_CONTENT": _OPENCODE_TEXT_ONLY}
        proc = await asyncio.create_subprocess_exec(
            *_cli_command(args, prompt, out_file), cwd=cwd, env=env,
            # Empty stdin: codex/opencode wait on an inherited stdin when launched in the background.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()  # reap the killed child
            raise ValueError(f"{args.cli} timed out after {args.timeout}s")
        if proc.returncode != 0:
            raise ValueError(f"{args.cli} exited {proc.returncode}")
        if args.cli == "codex":
            return out_file.read_text()
        text = ANSI.sub("", stdout.decode(errors="replace"))
        # opencode prints a "> build · <model>" status line before the answer.
        return "\n".join(line for line in text.splitlines() if not line.startswith("> build"))


async def _ask_http(client, args, prompt: str) -> str:
    resp = await client.post("/chat/completions", json={
        "model": args.model, "temperature": 0.8, "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}]})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _generate(client, sem, job, args, out_dir, manifest) -> str:
    path = out_dir / _filename(job)
    if path.exists():
        return "skipped"
    markers = ", ".join([f"{{{{SECRET:{k}}}}}" for k in job["secret_kinds"]]
                        + [f"{{{{NONSECRET:{k}}}}}" for k in job["nonsecret_kinds"]])
    prompt = PROMPT.format(language=job["language"], file_kind=job["file_kind"],
                           project_kind=job["project_kind"], markers=markers)
    async with sem:
        for _attempt in range(2):
            try:
                raw = await (_ask_cli(args, prompt) if args.cli else _ask_http(client, args, prompt))
                text = clean_output(raw)
            except (httpx.HTTPError, KeyError, ValueError, OSError) as e:
                print(f"job {job['id']}: {type(e).__name__}: {e}", file=sys.stderr)
                text = None
            if text:
                path.write_text(text)
                manifest.write(json.dumps({**job, "file": path.name, "model": args.model,
                                           "cli": args.cli or "http"}) + "\n")
                return "ok"
    return "rejected"


async def _main(args) -> None:
    out_dir = args.out or ROOT / "data" / "llm_slots" / re.sub(r"[^\w.-]", "_", args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if args.api_key_env else {}
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers, timeout=180) as client:
        with (out_dir / "manifest.jsonl").open("a") as manifest:
            results = await asyncio.gather(*(
                _generate(client, sem, job, args, out_dir, manifest) for job in jobs(args.count, args.seed)))
    print(json.dumps({"out": str(out_dir), **{k: results.count(k) for k in ("ok", "skipped", "rejected")}}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", choices=["codex", "claude", "opencode", "agy"],
                        help="run a headless CLI instead of calling an OpenAI-compatible HTTP API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="low", help="claude --effort level")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per CLI call")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key-env", help="name of the env var holding the API key (never the key itself)")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1,
                        help="one seed per model; keep it fixed for an output dir so resuming works")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", type=Path)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
