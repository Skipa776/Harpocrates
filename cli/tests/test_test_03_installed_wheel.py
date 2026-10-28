"""TEST-03: the critical flows end to end on an installed wheel, in a throwaway home (approved by Josh, Oct 21, 2026).

A wheel built from this checkout goes into a fresh venv with its mcp extra and pre-commit. Every Harpo process runs that
venv's interpreter through the throwaway bootstrap (bench/agent_isolation.py): paths.home() reads the password database,
not $HOME (SEC-08), so the bootstrap is how a test runs the installed code without touching the real home. The flows:
(a) a built-in read of a canary file is refused and safe_read returns placeholders; (b) the pre-commit hook blocks a
staged canary; (c) install, a hook call, then uninstall leaves every settings file byte-identical.
"""
from __future__ import annotations

import json
import queue
import shlex
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from bench import agent_isolation as iso

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 120
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]


@dataclass
class Installed:
    """The installed wheel and one throwaway home to run it in."""

    venv: Path
    tw: iso.Throwaway
    wrapper: Path  # `wrapper -I -m MODULE ...` runs the venv's interpreter with paths.home() pinned to tw.home
    env: dict

    def harpo(self, *args: str) -> subprocess.CompletedProcess:
        """Run `harpo ARGS` from the installed wheel; an install records the wrapper in every command it writes."""
        argv = [str(self.venv / "bin" / "python"), "-I", "-c", iso.BOOT, str(self.tw.home), str(self.wrapper),
                "Harpocrates.cli", *args]
        return subprocess.run(argv, cwd=self.tw.home, env=self.env, capture_output=True, text=True, timeout=TIMEOUT,
                              stdin=subprocess.DEVNULL, check=False)

    def run(self, argv: list[str], cwd: Path, stdin: str = "") -> subprocess.CompletedProcess:
        """Run a program (git, pre-commit, a hook command through /bin/sh) in the throwaway environment."""
        return subprocess.run(argv, cwd=cwd, env=self.env, input=stdin, capture_output=True, text=True,
                              timeout=TIMEOUT, check=False)

    def module(self, name: str) -> str:
        """The command line that runs a module of the installed wheel in the throwaway home."""
        return f"{shlex.quote(str(self.wrapper))} -I -m {name}"

    def pre_tool_use_commands(self) -> list[str]:
        """The PreToolUse hook commands in the throwaway home's Claude Code settings."""
        settings = json.loads((self.tw.home / ".claude" / "settings.json").read_text(encoding="utf-8"))
        return [hook["command"] for group in settings["hooks"]["PreToolUse"] for hook in group["hooks"]]


@pytest.fixture(scope="module")
def wheel_venv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A venv holding the wheel built from this checkout, with the mcp extra and pre-commit (from the package index)."""
    out = tmp_path_factory.mktemp("wheel")
    built = subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out)], cwd=ROOT,
                           capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    (wheel,) = out.glob("harpo_scan-*.whl")
    venv = tmp_path_factory.mktemp("venv")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True)
    pip = subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install", "-q", f"{wheel}[mcp]", "pre-commit"],
                         capture_output=True, text=True, check=False)
    assert pip.returncode == 0, pip.stderr[-2000:]
    where = subprocess.run([str(venv / "bin" / "python"), "-I", "-c", "import Harpocrates; print(Harpocrates.__file__)"],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert Path(where).is_relative_to(venv), f"Harpocrates was imported from {where}, not from the installed wheel"
    return venv


@pytest.fixture
def installed(wheel_venv: Path) -> Iterator[Installed]:
    """A fresh throwaway home (HOME, TMPDIR, repo) for the installed wheel; deleted afterwards."""
    tw = iso.make_throwaway()
    try:
        wrapper = iso.write_wrapper(tw, python=str(wheel_venv / "bin" / "python"))
        env = iso.clean_env(tw, "http://127.0.0.1:1", {}, path=f"{wheel_venv / 'bin'}:/usr/bin:/bin")
        before = iso.real_home_snapshot()
        yield Installed(wheel_venv, tw, wrapper, env)
        assert iso.snapshot_diff(before, iso.real_home_snapshot()) == [], "a TEST-03 flow touched the real home"
    finally:
        iso.discard(tw)


def canary_file(folder: Path, name: str = "config.env") -> tuple[Path, str]:
    """Write one generated fake credential (never a real one) into a file; returns the path and the value."""
    from gate_canaries import known_format

    value = known_format(3)["github_token"]
    path = folder / name
    path.write_text(f"GITHUB_TOKEN={value}\n", encoding="utf-8")
    return path, value


def read_event(path: Path, cwd: Path, session: str) -> str:
    """A Claude Code PreToolUse event for a built-in Read of one file."""
    return json.dumps({"tool_name": "Read", "tool_input": {"file_path": str(path)}, "cwd": str(cwd),
                       "session_id": session, "hook_event_name": "PreToolUse", "permission_mode": "default"})


def mcp_call(installed: Installed, tool: str, arguments: dict) -> tuple[dict, str]:
    """Call one tool of the installed MCP server over stdio, as an agent does: initialize, initialized, tools/call.

    Returns (the tools/call response, everything the server wrote). Fails after TIMEOUT seconds without an answer."""
    proc = subprocess.Popen(["/bin/sh", "-c", installed.module("Harpocrates.mcp.server")], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=installed.env,
                            cwd=installed.tw.repo)
    lines: queue.Queue = queue.Queue()
    seen: list[str] = []
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()

    def send(message: dict) -> None:
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        proc.stdin.flush()

    def answer(want: int) -> dict:
        while True:
            line = lines.get(timeout=TIMEOUT)
            seen.append(line)
            if line.startswith("{") and json.loads(line).get("id") == want:
                return json.loads(line)

    try:
        send({"id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                                           "clientInfo": {"name": "test-03", "version": "0"}}})
        assert "result" in answer(1)
        send({"method": "notifications/initialized"})
        send({"id": 2, "method": "tools/call", "params": {"name": tool, "arguments": arguments}})
        return answer(2), "".join(seen)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def snapshot(home: Path) -> dict[str, bytes]:
    """Every file under a home, {relative path: bytes}, without Harpo's own ~/.harpo."""
    return {p.relative_to(home).as_posix(): p.read_bytes() for p in sorted(home.rglob("*"))
            if p.is_file() and p.relative_to(home).parts[0] != ".harpo"}


def test_test_03_refused_read_and_safe_read(installed: Installed) -> None:
    """(a) The installed hook denies a built-in Read of a canary file and points at safe_read; safe_read, through the
    installed MCP server, returns placeholders; the findings log (G6) records both, and no value appears anywhere."""
    done = installed.harpo("install", "--claude-code")
    assert done.returncode == 0, done.stderr[-1000:]
    commands = [c for c in installed.pre_tool_use_commands() if str(installed.wrapper) in c]
    assert commands, "no PreToolUse hook command runs the installed wheel"
    path, value = canary_file(installed.tw.repo)
    for command in commands:
        hook = installed.run(["/bin/sh", "-c", command], installed.tw.repo, read_event(path, installed.tw.repo, "t03-a"))
        assert hook.returncode == 0, hook.stderr[-1000:]
        decision = json.loads(hook.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny" and "safe_read" in decision["permissionDecisionReason"]
        assert value not in hook.stdout + hook.stderr

    reply, output = mcp_call(installed, "safe_read", {"path": str(path)})
    text = reply["result"]["content"][0]["text"]
    assert "<<HARPO:" in text
    assert value not in output

    log = (installed.tw.home / ".harpo" / "findings.jsonl").read_text(encoding="utf-8")
    assert value not in log
    kinds = {(r["layer"], r["event_type"]) for r in map(json.loads, log.splitlines())}
    assert {("hook", "read_refused"), ("read", "secret_redacted")} <= kinds


def test_test_03_pre_commit_blocks_a_staged_canary(installed: Installed) -> None:
    """(b) A commit that stages a canary fails, names the file and not the value, and records nothing; a clean commit
    in the same repo goes through, so the hook blocks the secret, not every commit."""
    repo = installed.tw.repo / "b"
    repo.mkdir()
    assert installed.run([*GIT, "init", "-q"], repo).returncode == 0
    # The repo's .pre-commit-hooks.yaml settings for id harpo, run from the installed wheel (language: system).
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: harpo\n        name: Harpo secrets scan\n"
        f"        entry: {installed.module('Harpocrates.cli')} scan\n        language: system\n"
        "        types_or: [text]\n        pass_filenames: true\n", encoding="utf-8")
    hook = installed.run([str(installed.venv / "bin" / "pre-commit"), "install"], repo)
    assert hook.returncode == 0, hook.stderr[-1000:]

    path, value = canary_file(repo)
    assert installed.run([*GIT, "add", ".pre-commit-config.yaml", path.name], repo).returncode == 0
    blocked = installed.run([*GIT, "commit", "-m", "add"], repo)
    out = blocked.stdout + blocked.stderr
    assert blocked.returncode != 0, "the commit with a staged canary went through"
    assert "config.env" in out and value not in out
    assert installed.run([*GIT, "rev-parse", "--verify", "-q", "HEAD"], repo).returncode != 0

    assert installed.run([*GIT, "rm", "-q", "--cached", path.name], repo).returncode == 0
    (repo / "readme.txt").write_text("hello\n", encoding="utf-8")
    assert installed.run([*GIT, "add", "readme.txt"], repo).returncode == 0
    clean = installed.run([*GIT, "commit", "-m", "ok"], repo)
    assert clean.returncode == 0, (clean.stdout + clean.stderr).replace(value, "<value>")[-1000:]


def test_test_03_install_hook_uninstall_restores_every_settings_file(installed: Installed) -> None:
    """(c) Install for both harnesses, one allowed hook call, uninstall: every file in the home is byte-identical to
    before. The installer keeps the first original of a file it edits as <file>.harpo.bak (FR-INSTALL-08): those are
    left out of the comparison, and the ones for the seeded files must hold the seeded bytes."""
    home = installed.tw.home
    seeded = {".claude/settings.json": json.dumps({"theme": "dark", "env": {"FOO": "1"}}, indent=2) + "\n",
              ".codex/config.toml": 'model = "gpt-5"\n# my comment\n'}
    for name, text in seeded.items():
        (home / name).parent.mkdir(parents=True, exist_ok=True)
        (home / name).write_text(text, encoding="utf-8")
    before = snapshot(home)

    done = installed.harpo("install", "--claude-code", "--codex")
    assert done.returncode == 0, done.stderr[-1000:]
    assert snapshot(home) != before, "install wrote nothing"
    readme = installed.tw.repo / "readme.txt"
    readme.write_text("hello\n", encoding="utf-8")
    hook = installed.run(["/bin/sh", "-c", installed.pre_tool_use_commands()[0]], installed.tw.repo,
                         read_event(readme, installed.tw.repo, "t03-c"))
    assert (hook.returncode, hook.stdout) == (0, ""), hook.stderr[-1000:]
    done = installed.harpo("uninstall", "--claude-code", "--codex")
    assert done.returncode == 0, done.stderr[-1000:]

    after = snapshot(home)
    backups = {name: data for name, data in after.items() if name.endswith(".harpo.bak")}
    kept = {name: data for name, data in after.items() if name not in backups}
    changed = sorted(n for n in kept.keys() | before.keys() if kept.get(n) != before.get(n))
    assert not changed, f"not restored byte-identical: {changed}"
    for name, text in seeded.items():
        assert backups[name + ".harpo.bak"] == text.encode(), f"{name}.harpo.bak is not the seeded original"
