"""Agent mode of the canary harness: run `claude -p` and `codex exec` against the mock provider (FR-PROOF-04, G2).

For each agent: a throwaway HOME and a seeded canary repo (bench/agent_isolation.py, bench/canary_repo.py), Harpo's read
hook installed into that home, a mock provider on 127.0.0.1 that answers with scripted tool calls (Read .env, cat a canary
file, grep for TOKEN, then "done"), and one non-interactive agent run with a fake API key. The recording is then searched
for the planted canaries. Before the egress gate exists this measures the read-tool path alone: what the read hook let
through to the provider.

Nothing here reaches a real provider or the real home: the environment is built from scratch, the base URL is loopback, the
key is fake, and the real ~/.harpo, ~/.harpocrates, ~/.claude/settings.json, ~/.codex/config.toml and ~/.codex/auth.json are
snapshotted before and after.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))

from bench import agent_isolation as iso  # noqa: E402
from bench import canary_repo, mock_provider  # noqa: E402
from bench import canary_report as report  # noqa: E402
from bench.canary_harness import HARNESS_VERSION, HarnessResult  # noqa: E402

SCRIPTED_CALLS = 3
PROMPT = "Review this repository's configuration: read .env, show config/settings.py, and search for TOKEN."
AGENT_TIMEOUT = 240
MODEL_ROUTES = ("anthropic.messages", "openai.responses")
# Codex background features that would call out from a throwaway home (a plugin catalog sync runs `git ls-remote` against
# github.com, an app-server daemon may start, analytics may post): switched off, so the only traffic is to the mock.
CODEX_QUIET = ("\n[features]\nplugins = false\nremote_plugin = false\nplugin_sharing = false\napps = false\n"
               "daemon_auto_start = false\nin_app_updates = false\n\n[analytics]\nenabled = false\n")
GIT_ENV = {"GIT_AUTHOR_NAME": "canary", "GIT_AUTHOR_EMAIL": "canary@example.test", "GIT_COMMITTER_NAME": "canary",
           "GIT_COMMITTER_EMAIL": "canary@example.test"}


def claude_script(repo: Path) -> list[dict]:
    """
    Description: The scripted turns for Claude Code: read .env with the Read tool, cat a canary file, grep for TOKEN.
    Parameters:
        repo: the canary repo, whose absolute path the Read tool needs.
    Output: Four turns: three tool calls, then {"text": "done"}. This build of Claude Code has no Grep tool (its tool list
        was captured in G2), so the search is a Bash grep.
    """
    return [{"tool_calls": [{"name": "Read", "input": {"file_path": str(repo / ".env")}}]},
            {"tool_calls": [{"name": "Bash", "input": {"command": "cat config/settings.py", "description": "Show settings"}}]},
            {"tool_calls": [{"name": "Bash", "input": {"command": "grep -rn TOKEN .", "description": "Search for TOKEN"}}]},
            {"text": "done"}]


def codex_script(repo: Path) -> list[dict]:
    """
    Description: The scripted turns for Codex: the same three reads through its exec_command tool, then "done".
    Parameters:
        repo: the canary repo (Codex runs in it, so the commands use relative paths).
    Output: Four turns: three exec_command calls, then {"text": "done"}. Codex has no separate Read tool.
    """
    del repo
    return [{"tool_calls": [{"name": "exec_command", "input": {"cmd": cmd}}]}
            for cmd in ("cat .env", "cat config/settings.py", "grep -rn TOKEN .")] + [{"text": "done"}]


def claude_argv(prompt: str = PROMPT) -> list[str]:
    """
    Description: The `claude` command line for one scripted run.
    Parameters:
        prompt: the user prompt (the mock ignores its content).
    Output: An argv: print mode, stream-json output, default permission mode, and an explicit allowed-tools list of Read,
        cat and grep. No permission bypass of any kind. Hooks and settings come from the throwaway home.
    """
    return ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--permission-mode", "default",
            "--allowedTools", "Read", "Bash(cat *)", "Bash(grep *)", "--setting-sources", "user,project,local"]


def codex_argv(repo: Path, prompt: str = PROMPT) -> list[str]:
    """
    Description: The `codex exec` command line for one scripted run.
    Parameters:
        repo: the canary repo, used as Codex's working root.
        prompt: the user prompt (the mock ignores its content).
    Output: An argv: JSON events, a read-only sandbox, the repo as root. --dangerously-bypass-hook-trust is passed because
        the only hooks that exist are those Harpo's installer wrote into the throwaway CODEX_HOME, and Codex otherwise runs
        a hook only after an interactive /hooks trust step. No sandbox or approval bypass.
    """
    return ["codex", "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(repo),
            "--dangerously-bypass-hook-trust", prompt]


def claude_env(tw: iso.Throwaway, url: str) -> dict:
    """
    Description: The environment for `claude -p`: throwaway HOME, loopback base URL, fake API key.
    Parameters:
        tw: the throwaway session.
        url: the mock provider's URL.
    Output: A from-scratch environment (see agent_isolation.clean_env). ANTHROPIC_API_KEY makes print mode use the key path,
        so no stored login is used; ANTHROPIC_BASE_URL points at the mock; nonessential traffic and the updater are off.
    """
    return iso.clean_env(tw, url, {
        "ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": iso.FAKE_API_KEY, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", **GIT_ENV})


def codex_env(tw: iso.Throwaway, url: str) -> dict:
    """
    Description: The environment for `codex exec`: throwaway HOME and CODEX_HOME, fake API key.
    Parameters:
        tw: the throwaway session.
        url: the mock provider's URL (checked to be loopback; Codex reads it from config.toml).
    Output: A from-scratch environment with CODEX_HOME set to tw.home/.codex and CODEX_API_KEY set to a fake key.
    """
    return iso.clean_env(tw, url, {"CODEX_HOME": str(tw.home / ".codex"), "CODEX_API_KEY": iso.FAKE_OPENAI_KEY, **GIT_ENV})


def write_codex_config(tw: iso.Throwaway, url: str, extra_provider: str = "") -> Path:
    """
    Description: Write the Codex config for route A: a custom provider that points at the mock over plain HTTP.
    Parameters:
        tw: the throwaway session.
        url: the mock provider's URL.
        extra_provider: extra lines for the provider table.
    Output: The config.toml path. The provider is named mockprov, never "openai", with wire_api responses and
        requires_openai_auth true (docs/gate-transport.md route A); supports_websockets is left off, so Codex uses HTTP.
        CODEX_QUIET switches off the background features that would call out.
    """
    path = tw.home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'model = "gpt-5"\nmodel_provider = "mockprov"\n\n[model_providers.mockprov]\nname = "MockProv"\n'
                    f'base_url = "{url}/v1"\nwire_api = "responses"\nrequires_openai_auth = true\n{extra_provider}{CODEX_QUIET}',
                    encoding="utf-8")
    tomllib.loads(path.read_text(encoding="utf-8"))
    return path


def results_seen(records: list[dict]) -> int:
    """
    Description: Count the tool results the agent sent back, as the largest count in any one model request.
    Parameters:
        records: the recording entries.
    Output: The most tool results (tool_result blocks, or *_call_output items) in a single /v1/messages or /v1/responses
        body; 0 for none. This shows how far through the script the agent got.
    """
    best = 0
    for rec in records:
        if rec.get("route") not in MODEL_ROUTES or not isinstance(rec.get("body"), str):
            continue
        try:
            body = json.loads(rec["body"])
        except ValueError:
            continue
        fmt = "anthropic" if rec["route"] == "anthropic.messages" else "responses"
        best = max(best, mock_provider.count_tool_results(body, fmt))
    return best


def credential_header_names(records: list[dict]) -> list[str]:
    """
    Description: List the names of credential-class headers the agent sent, which shows which auth path it used.
    Parameters:
        records: the recording entries (values are already <present>).
    Output: Sorted, lower-cased, de-duplicated header names that mock_provider treats as credential-bearing.
    """
    names = {name.lower() for rec in records for name, _ in rec.get("headers", []) if mock_provider.is_credential_header(name)}
    return sorted(names)


def seed_repo(tw: iso.Throwaway, seed: int, agent: str) -> list[dict]:
    """
    Description: Build the seeded canary repo in the throwaway session and commit it.
    Parameters:
        tw: the throwaway session.
        seed: the canary seed.
        agent: the agent the repo is for (unused except for the commit message).
    Output: The canary manifest (file, kind, value per canary; values stay in memory). The repo is a git repository with
        everything committed, using a global-config-free git.
    """
    manifest = canary_repo.build(tw.repo, seed)
    env = {**os.environ, **GIT_ENV, "HOME": str(tw.home), "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    for argv in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", f"canary repo for {agent}"]):
        subprocess.run(argv, cwd=tw.repo, env=env, check=True, capture_output=True, stdin=subprocess.DEVNULL)
    return manifest


def canaries_from_manifest(manifest: list[dict], agent: str) -> list[report.Canary]:
    """
    Description: Turn the repo manifest into canaries for matching.
    Parameters:
        manifest: canary_repo.build's list of {file, kind, value}.
        agent: the agent session the repo was made for.
    Output: One Canary per entry with ident agent:file:kind (safe to print), the value held in memory only.
    """
    return [report.Canary(f"{agent}:{m['file']}:{m['kind']}", m["kind"], m["value"], agent) for m in manifest]


@dataclass
class AgentRun:
    """
    Description: What happened to one agent in one harness run.
    Parameters:
        agent: claude or codex.
        status: ok, skipped (CLI missing), failed (install or process error), timeout, or unfinished (the script did not run through).
        reason: a short explanation without values.
        version: the agent's --version text, or None.
        exit_code: the agent process's exit code, or None.
        seconds: wall time of the agent process.
        records: the recording entries (in memory; the file holds them too).
        canaries: the canaries planted for this agent.
        extra: further facts for the report (install summary, header names, counts).
    Output: A mutable record filled in as the run proceeds.
    """

    agent: str
    status: str = "skipped"
    reason: str = ""
    version: Optional[str] = None
    exit_code: Optional[int] = None
    seconds: float = 0.0
    records: list = field(default_factory=list)
    canaries: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def run_process_captured(argv: list[str], env: dict, cwd: Path, timeout: int,
                         stdin_data: Optional[bytes] = None) -> tuple[Optional[int], bool, float, str, str]:
    """
    Description: Run a command to completion in its own process group, killing the group on timeout, and keep its output.
    Parameters:
        argv: the command.
        env: its whole environment.
        cwd: its working directory.
        timeout: seconds before it is killed.
        stdin_data: bytes to send on stdin (a very long prompt), or None for /dev/null.
    Output: (exit code or None, timed out, seconds, stdout, stderr), the output decoded with replacement characters. Use it
        only where the output cannot hold a planted canary (a session that reads no canary file).
    """
    start = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL if stdin_data is None else subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = proc.communicate(input=stdin_data, timeout=timeout)
        return proc.returncode, False, time.monotonic() - start, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        return None, True, time.monotonic() - start, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def run_process(argv: list[str], env: dict, cwd: Path, timeout: int,
                stdin_data: Optional[bytes] = None) -> tuple[Optional[int], bool, float]:
    """
    Description: Run a command to completion in its own process group, killing the group on timeout; its output is dropped.
    Parameters:
        argv: the command.
        env: its whole environment.
        cwd: its working directory.
        timeout: seconds before it is killed.
        stdin_data: bytes to send on stdin (a very long prompt), or None for /dev/null.
    Output: (exit code or None, timed out, seconds). stdin is /dev/null unless stdin_data is given (codex exec waits for end
        of input otherwise); stdout and stderr are captured and dropped, since they hold the model's view of the files.
    """
    return run_process_captured(argv, env, cwd, timeout, stdin_data)[:3]


def _version(binary: str) -> Optional[str]:
    """
    Description: Ask a CLI for its version with a scrubbed environment.
    Parameters:
        binary: the command name.
    Output: The first line of `binary --version`, or None if it fails.
    """
    try:
        done = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
                              env={"PATH": os.environ.get("PATH", ""), "HOME": os.devnull}, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip().splitlines()[0] if done.stdout.strip() else None


AGENTS: dict[str, dict] = {
    "claude": {"binary": "claude", "target": "claude-code", "script": claude_script, "env": claude_env,
               "argv": lambda tw: claude_argv()},
    "codex": {"binary": "codex", "target": "codex", "script": codex_script, "env": codex_env,
              "argv": lambda tw: codex_argv(tw.repo)},
}


@dataclass
class Session:
    """
    Description: One agent's throwaway home, canary repo and mock provider, ready to run the agent.
    Parameters:
        agent: claude or codex.
        tw: the throwaway session folders.
        server: the running mock provider (loopback).
        canaries: the planted canaries (values in memory only).
        install: ok, "none (control run)" or a failure summary.
        install_error: the installer's stderr tail when it failed, else "".
    Output: A record handed out by agent_session; run() starts the agent.
    """

    agent: str
    tw: iso.Throwaway
    server: mock_provider.MockServer
    canaries: list
    install: str
    install_error: str = ""

    def argv(self) -> list[str]:
        """
        Description: The default command line for this agent.
        Parameters: None
        Output: claude_argv() or codex_argv(repo).
        """
        return AGENTS[self.agent]["argv"](self.tw)

    def env(self) -> dict:
        """
        Description: The default environment for this agent.
        Parameters: None
        Output: claude_env or codex_env for this session's throwaway home and mock URL.
        """
        return AGENTS[self.agent]["env"](self.tw, self.server.url)

    def run(self, argv: Optional[list[str]] = None, env: Optional[dict] = None, timeout: int = AGENT_TIMEOUT,
            stdin_data: Optional[bytes] = None) -> tuple[Optional[int], bool, float]:
        """
        Description: Run the agent in the canary repo.
        Parameters:
            argv: a command line instead of the default.
            env: an environment instead of the default (it must still be built from scratch).
            timeout: seconds before the process group is killed.
            stdin_data: bytes for stdin, or None.
        Output: (exit code or None, timed out, seconds), as run_process.
        """
        return run_process(argv or self.argv(), env or self.env(), self.tw.repo, timeout, stdin_data)

    def run_captured(self, argv: Optional[list[str]] = None, env: Optional[dict] = None, timeout: int = AGENT_TIMEOUT,
                     stdin_data: Optional[bytes] = None) -> tuple[Optional[int], bool, float, str, str]:
        """
        Description: Run the agent and keep its output, for sessions that read no canary file.
        Parameters:
            argv: a command line instead of the default.
            env: an environment instead of the default.
            timeout: seconds before the process group is killed.
            stdin_data: bytes for stdin, or None.
        Output: (exit code or None, timed out, seconds, stdout, stderr), as run_process_captured.
        """
        return run_process_captured(argv or self.argv(), env or self.env(), self.tw.repo, timeout, stdin_data)


@contextlib.contextmanager
def agent_session(agent: str, seed: int, out_root: Path, run_id: str, label: Optional[str] = None,
                  script: Optional[object] = None, control: bool = False, keep_work: bool = False,
                  mock_kwargs: Optional[dict] = None, codex_config: Optional[Callable[..., Path]] = None,
                  before_install: Optional[Callable[[iso.Throwaway, mock_provider.MockServer], None]] = None):
    """
    Description: Set up one agent's throwaway home, seeded canary repo, mock provider and Harpo install.
    Parameters:
        agent: claude or codex.
        seed: the canary seed.
        out_root: the results folder; the recording goes to mock/<run_id>/<label>/.
        run_id: the run's id.
        label: the recording sub-folder, defaulting to the agent name.
        script: scripted turns (or a function repo -> turns) instead of the agent's default three reads.
        control: install nothing into the throwaway home.
        keep_work: keep the throwaway folder instead of deleting it.
        mock_kwargs: extra start_server arguments (reply_headers, count_tokens_status).
        codex_config: a function (tw, url) -> Path that writes the Codex config, instead of route A's.
        before_install: a function (tw, server) run after the config is written and before Harpo is installed.
    Output: Yields a Session. The mock is stopped and the throwaway folder deleted on exit (unless keep_work), also when the
        body raises.
    """
    spec = AGENTS[agent]
    tw = iso.make_throwaway()
    try:
        canaries = canaries_from_manifest(seed_repo(tw, seed, agent), agent)
        turns = script(tw.repo) if callable(script) else script if script is not None else spec["script"](tw.repo)
        script_path = tw.root / "script.json"
        script_path.write_text(json.dumps(turns), encoding="utf-8")
        record_dir = Path(out_root) / "mock" / run_id / (label or agent)
        with mock_provider.start_server(record_dir=record_dir, turns=mock_provider.load_script(script_path),
                                        **(mock_kwargs or {})) as server:
            if agent == "codex":
                (codex_config or write_codex_config)(tw, server.url)
            if before_install is not None:
                before_install(tw, server)
            installed = None if control else iso.install_harpo(tw, spec["target"])
            summary = "none (control run)" if installed is None else "ok" if installed.returncode == 0 else f"exit {installed.returncode}"
            yield Session(agent, tw, server, canaries, summary,
                          installed.stderr.strip()[-200:] if installed is not None and installed.returncode else "")
    finally:
        if not keep_work:
            iso.discard(tw)


def run_one(agent: str, seed: int, out_root: Path, run_id: str, keep_work: bool = False,
            timeout: int = AGENT_TIMEOUT, control: bool = False) -> AgentRun:
    """
    Description: Run one agent through the scripted session and record what reached the mock.
    Parameters:
        agent: claude or codex.
        seed: the canary seed.
        out_root: the results folder; the recording goes to mock/<run_id>/<agent>/.
        run_id: the run's id.
        keep_work: keep the throwaway folder instead of deleting it.
        timeout: seconds allowed for the agent process.
        control: install nothing into the throwaway home, so the same scripted reads run unguarded.
    Output: An AgentRun. A missing CLI gives status skipped; an installer failure, a non-zero exit or a script that did not
        run through is reported, never passed. The throwaway folder is deleted unless keep_work.
    """
    spec = AGENTS[agent]
    run = AgentRun(agent)
    if shutil.which(spec["binary"]) is None:
        run.reason = f"`{spec['binary']}` is not on PATH"
        return run
    run.version = _version(spec["binary"])
    with agent_session(agent, seed, out_root, run_id, keep_work=keep_work, control=control) as session:
        run.canaries = session.canaries
        run.extra["install"] = session.install
        if session.install_error:
            run.status, run.reason = "failed", "Harpo install into the throwaway home failed: " + session.install_error
            return run
        code, timed_out, seconds = session.run(timeout=timeout)
        run.exit_code, run.seconds = code, round(seconds, 1)
        run.records = session.server.records()
    seen = results_seen(run.records)
    run.extra.update({"scripted_tool_calls": SCRIPTED_CALLS, "tool_results_seen": seen, "requests": len(run.records),
                      "credential_header_names": credential_header_names(run.records),
                      "recording": f"mock/{run_id}/{agent}/requests.jsonl"})
    if timed_out:
        run.status, run.reason = "timeout", f"the agent did not finish in {timeout} s"
    elif code != 0:
        run.status, run.reason = "failed", f"the agent exited with status {code}"
    elif seen < SCRIPTED_CALLS:
        run.status, run.reason = "unfinished", f"only {seen} of {SCRIPTED_CALLS} scripted tool calls returned a result"
    else:
        run.status = "ok"
    return run


def _agent_report(run: AgentRun) -> dict:
    """
    Description: Build one agent's entry for the report's runs list.
    Parameters:
        run: the agent's AgentRun.
    Output: A dict with the agent, status, reason, version, exit code, seconds and the extra facts, plus the planted and
        received counts for that agent. No values.
    """
    findings = report.find_received(run.records, run.canaries)
    return {"agent": run.agent, "status": run.status, "reason": run.reason, "version": run.version,
            "exit_code": run.exit_code, "seconds": run.seconds, **run.extra, "planted": report.counts(run.canaries),
            "received": report.received_counts(run.canaries, findings)}


def run_agents(seed: int, out_root: Path, run_id: str, which: str = "all", keep_work: bool = False,
               timeout: int = AGENT_TIMEOUT, runner: Callable[..., AgentRun] = run_one, control: bool = False) -> HarnessResult:
    """
    Description: Run the requested agents and publish the canary counts.
    Parameters:
        seed: the canary seed.
        out_root: the results folder; the report goes to proof/<run_id>.json.
        run_id: the run's id.
        which: claude, codex or all.
        keep_work: keep the throwaway folders.
        timeout: seconds allowed per agent.
        runner: the per-agent function, replaceable for tests.
        control: run the same script with Harpo not installed. No protection is claimed, so a leak is a count, not a failure;
            the run fails instead when it received no known-format canary at all, because then the counter could not see a leak.
    Output: A HarnessResult. The verdict is 1 when a known-format canary reached the mock through the read-tool path, the
        real home changed, or (control) nothing arrived; 2 when an agent was skipped or did not finish; else 0. The real home
        is snapshotted before and after; the difference is in the report (empty when nothing changed).
    """
    names = ["claude", "codex"] if which == "all" else [which]
    before = iso.real_home_snapshot()
    runs: list[AgentRun] = []
    for name in names:
        runs.append(runner(name, seed, out_root, run_id, keep_work, timeout, control))
        if iso.snapshot_diff(before, iso.real_home_snapshot()):
            break  # stop at once; the report says what changed
    changed = iso.snapshot_diff(before, iso.real_home_snapshot())
    canaries = [c for r in runs for c in r.canaries]
    findings = [f for r in runs for f in report.find_received(r.records, r.canaries)]
    planted, received = report.counts(canaries), report.received_counts(canaries, findings)
    all_ran = len(runs) == len(names) and all(r.status == "ok" for r in runs)
    verdict = report.verdict("agent", "none" if control else "read-tool", received["known_format"], all_ran=all_ran)
    if control and all_ran and received["known_format"] == 0:
        verdict = report.Verdict(1, "counter check failed: the control run (no Harpo) received no known-format canary")
    if changed:
        verdict = report.Verdict(1, f"the real home changed ({len(changed)} paths); see isolation.real_home_changed")
    data = {
        "harness_version": HARNESS_VERSION, "run_id": run_id, "mode": "agent", "seed": seed, "target": "mock",
        "variant": "control" if control else "read-hook",
        "protection": "none" if control else "read-tool", "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "mock_provider": "1", **{r.agent: r.version for r in runs}},
        "planted": planted, "received": received,
        "by_agent": {r.agent: {"planted": report.counts(r.canaries),
                               "received": report.received_counts(r.canaries, report.find_received(r.records, r.canaries))}
                     for r in runs},
        "findings": [f._asdict() for f in findings], "runs": [_agent_report(r) for r in runs],
        "isolation": {"real_home_paths_snapshotted": len(before), "real_home_changed": changed,
                      "method": "throwaway HOME and CODEX_HOME, from-scratch environment, fake API keys, loopback mock, "
                                "paths.home pinned by bootstrap"},
        "verdict": {"exit_code": verdict.exit_code, "reason": verdict.reason},
    }
    path = report.write_report(Path(out_root) / "proof" / f"{run_id}.json", data, canaries)
    return HarnessResult(data, verdict, path, canaries)
