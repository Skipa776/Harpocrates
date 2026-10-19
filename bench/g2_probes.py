#!/usr/bin/env python3
"""G2 capture probes: settle the "to verify" items of the gate parser spec and transport notes (FR-PROOF-04, FR-GATE-02).

    python bench/g2_probes.py [--only NAME[,NAME]] [--run-id ID] [--out-dir bench/results]

Each probe runs the real `claude` or `codex` against the mock provider in a throwaway home (bench/agent_isolation.py), with
fake keys and a loopback base URL, and reads the recording (bench/g2_analysis.py). The report,
bench/results/proof/g2-probes-<run>.json, holds header and field names, shapes, counts and fixed texts. It never holds a
credential or a canary; the writer checks. The recordings, which hold request bodies, go to bench/results/mock/<run>/.

Probes (the items each settles):
  claude-corpus            V1 V3 V4 V6 V7 T1 T10   a scripted Claude Code session with thinking, images, MCP results
  claude-long              V3 V7 T1 T10            a 700 KB prompt: compression and framing of a long request
  claude-settings          T2                      project settings against user settings for ANTHROPIC_BASE_URL
  claude-block-400         T3                      the agent's reaction to a 400 on /v1/messages
  codex-corpus             V3 V4 V5 V6 V7 T4 T8 T9 T10   a scripted Codex session, route A, API key
  codex-long               V3 V7 T10               a 1 MB prompt
  codex-block-400          T3                      the agent's reaction to a 400 on /v1/responses
  codex-chatgpt-route-a    T5                      route A with a fake ChatGPT login
  codex-route-b-apikey     T6                      the built-in provider pointed at the mock, API key: WebSocket then HTTP
  codex-route-b-chatgpt    T6 T7                   the same with a fake ChatGPT login: zstd and permessage-deflate
  codex-compaction         T9                      forcing a compaction to see its endpoint
Nothing here reaches a real provider or the real home; the real home is snapshotted before and after.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))

from bench import agent_isolation as iso  # noqa: E402
from bench import agent_runner as runner  # noqa: E402
from bench import g2_analysis as g2  # noqa: E402
from bench import mock_provider  # noqa: E402

BLOCK_MESSAGE = ("Harpo blocked this request before it left your machine (reason HARPO-BLOCK-UNKNOWN-TYPE). "
                 "Run `harpo doctor` for details.")
PROBE_TIMEOUT = 180
COMPACTION_SECONDS = 15
FAKE_ACCOUNT = "acct-fake-0000"


class ProbeReportError(Exception):
    """
    Description: A probe report was about to be written with a fake credential or canary value in it.
    Parameters:
        None: the message names no value.
    Output: Raised by write_probe_report before anything is written.
    """


def _b64(data: dict) -> str:
    """
    Description: Encode a dict as an unpadded base64url JSON segment of a JWT.
    Parameters:
        data: the JSON object.
    Output: The segment text.
    """
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")


def _fake_jwt(claims: dict) -> str:
    """
    Description: Build an unsigned JWT-shaped string that is plainly fake.
    Parameters:
        claims: the payload claims.
    Output: header.payload.fakesignature, with alg none.
    """
    return f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(claims)}.fakesignature"


def fake_chatgpt_auth() -> dict:
    """
    Description: Build the content of a Codex auth.json for a ChatGPT login, from made-up values.
    Parameters: None
    Output: A dict with auth_mode chatgpt, JWT-shaped id and access tokens (unsigned, expiring in 2100, so Codex never
        tries to refresh them), a refresh token and an account id that say "fake", and a last_refresh of now. Nothing in it
        is a credential for any service.
    """
    claims = {"email": "fake@example.test", "iat": 1760000000, "exp": 4102444800, "https://api.openai.com/auth": {
        "chatgpt_plan_type": "plus", "chatgpt_account_id": FAKE_ACCOUNT, "chatgpt_user_id": "user-fake-0000"}}
    return {"auth_mode": "chatgpt", "OPENAI_API_KEY": None, "last_refresh": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "tokens": {"id_token": _fake_jwt(claims), "access_token": _fake_jwt({"iat": 1760000000, "exp": 4102444800}),
                       "refresh_token": "fake-refresh-token-not-real", "account_id": FAKE_ACCOUNT}}


def write_chatgpt_auth(tw: iso.Throwaway, url: str) -> Path:
    """
    Description: Give a throwaway CODEX_HOME a fake ChatGPT login and point Codex's ChatGPT backend at the mock.
    Parameters:
        tw: the throwaway session.
        url: the mock's URL.
    Output: The auth.json path (mode 0600). Also puts chatgpt_base_url at the top of config.toml, so no request for a plugin
        list, account check or analytics can go to chatgpt.com. Raises IsolationError for a non-loopback URL.
    """
    iso.assert_loopback_url(url)
    codex = tw.home / ".codex"
    codex.mkdir(parents=True, exist_ok=True)
    config = codex / "config.toml"
    old = config.read_text(encoding="utf-8") if config.exists() else ""
    config.write_text(f'chatgpt_base_url = "{url}/backend-api/"\n{old}', encoding="utf-8")
    auth = codex / "auth.json"
    fd = os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(fake_chatgpt_auth(), handle)
    return auth


def chatgpt_env(tw: iso.Throwaway, url: str) -> dict:
    """
    Description: The Codex environment for a fake ChatGPT login: no API key, and the token refresh URL pointed at the mock.
    Parameters:
        tw: the throwaway session.
        url: the mock's URL.
    Output: codex_env without CODEX_API_KEY, plus CODEX_REFRESH_TOKEN_URL_OVERRIDE set to the mock, so even a refresh attempt
        stays on loopback.
    """
    env = {k: v for k, v in runner.codex_env(tw, url).items() if k != "CODEX_API_KEY"}
    env["CODEX_REFRESH_TOKEN_URL_OVERRIDE"] = f"{url}/oauth/token"
    return env


def write_route_b_config(tw: iso.Throwaway, url: str, chatgpt: bool) -> Path:
    """
    Description: Write the Codex config for route B: the built-in OpenAI provider with its base URL moved to the mock.
    Parameters:
        tw: the throwaway session.
        url: the mock's URL.
        chatgpt: also install a fake ChatGPT login (and chatgpt_base_url) instead of using the API key.
    Output: The config.toml path. No custom provider is defined, so Codex treats the provider as OpenAI and may try a
        WebSocket first (docs/gate-transport.md, route B).
    """
    iso.assert_loopback_url(url)
    path = tw.home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'model = "gpt-5"\nopenai_base_url = "{url}/v1"\n{runner.CODEX_QUIET}', encoding="utf-8")
    if chatgpt:
        write_chatgpt_auth(tw, url)
    return path


def write_png(path: Path) -> Path:
    """
    Description: Write a valid 1 x 1 PNG, so a Read of it makes the agent send an image block.
    Parameters:
        path: where to write it.
    Output: The path.
    """
    def chunk(kind: bytes, data: bytes) -> bytes:
        """
        Description: Encode one PNG chunk.
        Parameters:
            kind: the 4-byte chunk type.
            data: the chunk data.
        Output: Length, type, data and CRC.
        """
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))
    path.write_bytes(png)
    return path


def long_prompt(size: int) -> str:
    """
    Description: Make a long, plain prompt to see how an agent frames a large request.
    Parameters:
        size: the length in characters.
    Output: Exactly size characters of repeated pangrams. Plain ASCII; no secret-looking content.
    """
    unit = "The quick brown fox jumps over the lazy dog. "
    return (unit * (size // len(unit) + 1))[:size]


def claude_corpus_script(repo: Path) -> list[dict]:
    """
    Description: The scripted turns for the Claude Code corpus session.
    Parameters:
        repo: the canary repo (absolute paths are needed by Read).
    Output: Five turns: thinking, text and a Read of README.md; redacted_thinking and a cat; a Read of an image; the Harpo MCP
        safe_read of a file that holds canaries (returns placeholders); then "done". Together they make the agent send
        tool_result strings, text and image blocks, MCP results, and thinking blocks back.
    """
    return [{"thinking": "Plan: read the README first.", "text": "Reading the README.",
             "tool_calls": [{"name": "Read", "input": {"file_path": str(repo / "README.md")}}]},
            {"redacted_thinking": "mock-redacted-0001",
             "tool_calls": [{"name": "Bash", "input": {"command": "cat README.md", "description": "Show README"}}]},
            {"tool_calls": [{"name": "Read", "input": {"file_path": str(repo / "logo.png")}}]},
            {"tool_calls": [{"name": "mcp__harpo__safe_read", "input": {"path": str(repo / "config" / "settings.py")}}]},
            {"text": "done"}]


def codex_corpus_script() -> list[dict]:
    """
    Description: The scripted turns for the Codex corpus session.
    Parameters: None
    Output: Four turns: thinking, text and exec_command cat README.md; redacted_thinking and exec_command ls; exec_command
        git log; then "done". The agent sends reasoning items, function calls and their outputs back.
    """
    return [{"thinking": "Plan: read the README first.", "text": "Reading the README.",
             "tool_calls": [{"name": "exec_command", "input": {"cmd": "cat README.md"}}]},
            {"redacted_thinking": "mock-redacted-0002", "tool_calls": [{"name": "exec_command", "input": {"cmd": "ls"}}]},
            {"tool_calls": [{"name": "exec_command", "input": {"cmd": "git log --oneline"}}]}, {"text": "done"}]


def write_probe_report(path: Path, data: dict, secrets: list[str]) -> Path:
    """
    Description: Write a probe report after checking it holds none of the fake credentials or canaries used.
    Parameters:
        path: the file to write; its folder is created.
        data: the report.
        secrets: every fake key, token and canary value in play.
    Output: The path. Raises ProbeReportError, writing nothing, if any of them appears in the serialized report.
    """
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    for secret in secrets:
        if secret and secret in text:
            raise ProbeReportError("a probe report would contain a fake credential or canary value")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return Path(path)


@dataclass
class Context:
    """
    Description: What every probe needs and collects.
    Parameters:
        seed: the canary seed.
        out_root: the results folder.
        run_id: the run's id.
        secrets: every fake credential and canary value used (grows as probes run).
    Output: A mutable record passed to each probe.
    """

    seed: int
    out_root: Path
    run_id: str
    secrets: list = field(default_factory=lambda: [iso.FAKE_API_KEY, iso.FAKE_OPENAI_KEY])


def _parse_events(text: str) -> list[dict]:
    """
    Description: Parse the JSON lines an agent printed, skipping anything else.
    Parameters:
        text: stdout of `claude --output-format stream-json` or `codex exec --json`.
    Output: The list of event dicts.
    """
    events = []
    for line in text.splitlines():
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _result(name: str, verifies: list[str], agent: str, session: runner.Session, code: Optional[int], timed_out: bool,
            seconds: float, observations: dict) -> dict:
    """
    Description: Assemble one probe's report entry.
    Parameters:
        name: the probe name.
        verifies: the verify items it settles.
        agent: claude or codex.
        session: the session, for the recording.
        code: the agent's exit code.
        timed_out: whether it was killed.
        seconds: its wall time.
        observations: the facts found.
    Output: A dict with probe, verifies, agent, version, status, exit code, request count and observations.
    """
    records = session.server.records()
    return {"probe": name, "verifies": verifies, "agent": agent, "agent_version": runner._version(runner.AGENTS[agent]["binary"]),
            "status": "timeout" if timed_out else "ok" if code == 0 else f"exit {code}", "exit_code": code,
            "seconds": round(seconds, 1), "requests": len(records), "observations": observations}


def _need(agent: str) -> Optional[dict]:
    """
    Description: Report a probe as skipped when its CLI is missing.
    Parameters:
        agent: claude or codex.
    Output: A skipped entry, or None when the CLI is on PATH.
    """
    import shutil  # noqa: PLC0415

    if shutil.which(runner.AGENTS[agent]["binary"]) is None:
        return {"agent": agent, "status": "skipped", "reason": f"`{runner.AGENTS[agent]['binary']}` is not on PATH"}
    return None


def probe_claude_corpus(ctx: Context) -> dict:
    """
    Description: Run the Claude Code corpus session and read V1, V3, V4, V6, V7, T1 and T10 from it.
    Parameters:
        ctx: the probe context.
    Output: The probe entry: attribution block, block types, transport, whether thinking blocks came back, and whether the
        session id in the request body names the transcript file.
    """
    skip = _need("claude")
    if skip:
        return {"probe": "claude-corpus", **skip}
    with runner.agent_session("claude", ctx.seed, ctx.out_root, ctx.run_id, "claude-corpus", script=claude_corpus_script,
                              before_install=lambda tw, server: write_png(tw.repo / "logo.png")) as s:
        ctx.secrets += [c.value for c in s.canaries]
        code, timed_out, seconds = s.run(timeout=PROBE_TIMEOUT)
        records = s.server.records()
        sessions = list((s.tw.home / ".claude" / "projects").glob("*/*.jsonl"))
        body = next((b for _, b in g2._bodies(records, g2.MESSAGES)), {})
        try:
            session_id = json.loads(body.get("metadata", {}).get("user_id", "{}")).get("session_id")
        except (ValueError, AttributeError):
            session_id = None
        obs = {"attribution": g2.attribution(records), "block_types": g2.block_types(records), "transport": g2.transport(records),
               "thinking": g2.thinking_resent(records, {"thinking": ["mocksig"], "redacted": ["mock-redacted-0001"], "reasoning": []}),
               "results_seen": runner.results_seen(records), "install": s.install,
               "session_id_in_body_names_transcript": bool(session_id) and any(f.stem == session_id for f in sessions),
               "transcript_path_pattern": "~/.claude/projects/<project-slug>/<session-id>.jsonl" if sessions else None,
               "session_header_present": any(n == "x-claude-code-session-id" for n in g2.transport(records)["header_names"])}
        return _result("claude-corpus", ["V1", "V3", "V4", "V6", "V7", "T1", "T10"], "claude", s, code, timed_out, seconds, obs)


def probe_claude_long(ctx: Context) -> dict:
    """
    Description: Send Claude Code a 700 KB prompt and look at compression and framing (V3, V7, T1, T10).
    Parameters:
        ctx: the probe context.
    Output: The probe entry with transport(), the largest body size and the Content-Encoding values seen.
    """
    skip = _need("claude")
    if skip:
        return {"probe": "claude-long", **skip}
    argv = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--permission-mode", "default"]
    with runner.agent_session("claude", ctx.seed, ctx.out_root, ctx.run_id, "claude-long", script=[{"text": "done"}], control=True) as s:
        code, timed_out, seconds = s.run(argv=argv, timeout=PROBE_TIMEOUT, stdin_data=long_prompt(700_000).encode())
        records = s.server.records()
        biggest = max((r["body_size"] for r in records if r["route"] == g2.MESSAGES), default=0)
        obs = {"transport": g2.transport(records), "largest_model_request_bytes": biggest, "prompt_bytes_sent": 700_000}
        return _result("claude-long", ["V3", "V7", "T1", "T10"], "claude", s, code, timed_out, seconds, obs)


def probe_claude_settings(ctx: Context) -> dict:
    """
    Description: Set ANTHROPIC_BASE_URL in user settings and in project settings, and see which mock gets the request (T2).
    Parameters:
        ctx: the probe context.
    Output: The probe entry: for each case (settings only; settings plus a shell variable) which of the three loopback mocks
        received the model request.
    """
    skip = _need("claude")
    if skip:
        return {"probe": "claude-settings", **skip}
    base = Path(ctx.out_root) / "mock" / ctx.run_id
    cases = {}
    with mock_provider.start_server(record_dir=base / "claude-settings-project") as project, \
            mock_provider.start_server(record_dir=base / "claude-settings-shell") as shell:
        def settings(tw: iso.Throwaway, user: mock_provider.MockServer) -> None:
            """
            Description: Write the conflicting settings files into the throwaway home and repo.
            Parameters:
                tw: the throwaway session.
                user: the session's own mock, named in the user settings.
            Output: None.
            """
            (tw.home / ".claude").mkdir(exist_ok=True)
            (tw.home / ".claude" / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": user.url}}))
            (tw.repo / ".claude").mkdir(exist_ok=True)
            (tw.repo / ".claude" / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": project.url}}))

        with runner.agent_session("claude", ctx.seed, ctx.out_root, ctx.run_id, "claude-settings-user", script=[{"text": "done"}],
                                  control=True, before_install=settings) as s:
            for case, shell_url in (("user_and_project_settings", None), ("user_and_project_settings_and_shell", shell.url)):
                before = [len(x.records()) for x in (s.server, project, shell)]
                env = {k: v for k, v in s.env().items() if k != "ANTHROPIC_BASE_URL"}
                if shell_url:
                    env["ANTHROPIC_BASE_URL"] = shell_url
                code, timed_out, seconds = s.run(env=env, timeout=PROBE_TIMEOUT)
                after = [len(x.records()) for x in (s.server, project, shell)]
                got = {name: after[i] - before[i] for i, name in enumerate(("user_settings_url", "project_settings_url", "shell_url"))}
                winner = next((n for n, c in got.items() if c), None)
                cases[case] = {"requests_received": got, "winner": winner, "exit_code": code, "timed_out": timed_out}
            return _result("claude-settings", ["T2"], "claude", s, 0, False, 0.0, {"cases": cases, "install": s.install})


def _claude_result(events: list[dict]) -> dict:
    """
    Description: Pull what matters out of Claude Code's stream-json events after an API error.
    Parameters:
        events: the parsed stdout events.
    Output: {"result_event": {is_error, api_error_status, subtype, terminal_reason}, "message_shown": bool} where message_shown
        is whether the mock's block message text appears anywhere in the output.
    """
    final = next((e for e in reversed(events) if e.get("type") == "result"), {})
    shown = BLOCK_MESSAGE in json.dumps(events)
    return {"result_event": {k: final.get(k) for k in ("is_error", "api_error_status", "subtype", "terminal_reason")},
            "message_shown": shown, "event_types": sorted({str(e.get("type")) for e in events})}


def probe_claude_block(ctx: Context) -> dict:
    """
    Description: Answer Claude Code's request with the planned block response, first and after a tool result (T3).
    Parameters:
        ctx: the probe context.
    Output: The probe entry: per case the exit code, the result event fields, whether the message was shown to the user, and
        how many times the agent sent the request (retries). Also whether count_tokens was ever called.
    """
    skip = _need("claude")
    if skip:
        return {"probe": "claude-block-400", **skip}
    error = {"error": {"status": 400, "message": BLOCK_MESSAGE}}
    cases = {}
    for case, script in (("first_request", [error]),
                         ("after_a_tool_result", lambda repo: [{"tool_calls": [{"name": "Read", "input": {"file_path": str(repo / "README.md")}}]}, error])):
        with runner.agent_session("claude", ctx.seed, ctx.out_root, ctx.run_id, f"claude-block-{case}", script=script, control=True) as s:
            code, timed_out, seconds, out, err = s.run_captured(timeout=PROBE_TIMEOUT)
            records = s.server.records()
            cases[case] = {"exit_code": code, "timed_out": timed_out, **_claude_result(_parse_events(out)),
                           "model_requests": sum(1 for r in records if r["route"] == g2.MESSAGES),
                           "statuses": g2.transport(records)["response_statuses"],
                           "count_tokens_requests": sum(1 for r in records if r["route"] == "anthropic.count_tokens"),
                           "stderr_mentions_message": BLOCK_MESSAGE in err}
    return {"probe": "claude-block-400", "verifies": ["T3"], "agent": "claude", "status": "ok",
            "agent_version": runner._version("claude"), "observations": {"cases": cases}}


def probe_codex_corpus(ctx: Context) -> dict:
    """
    Description: Run the Codex corpus session and read V3, V4, V5, V6, V7, T4, T8, T9 and T10 from it.
    Parameters:
        ctx: the probe context.
    Output: The probe entry: block types, transport (endpoints called), reasoning items sent back, x-codex-turn-metadata shape
        and whether a git remote credential appears in it, whether x-codex-turn-state is echoed, and how the session id
        header relates to the rollout file name.
    """
    skip = _need("codex")
    if skip:
        return {"probe": "codex-corpus", **skip}
    fake_pw = "probe-pw-" + base64.b16encode(os.urandom(6)).decode().lower()
    ctx.secrets.append(fake_pw)
    remote = f"https://probeuser:{fake_pw}@example.test/r.git"

    def add_remote(tw: iso.Throwaway, server: mock_provider.MockServer) -> None:
        """
        Description: Add a git remote whose URL carries a made-up password.
        Parameters:
            tw: the throwaway session.
            server: the mock (unused).
        Output: None.
        """
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=tw.repo, check=True, capture_output=True,
                       env={**os.environ, "HOME": str(tw.home), "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})

    with runner.agent_session("codex", ctx.seed, ctx.out_root, ctx.run_id, "codex-corpus", script=codex_corpus_script(),
                              mock_kwargs={"reply_headers": {"x-codex-turn-state": "mock-turn-state-0001"}},
                              before_install=add_remote) as s:
        ctx.secrets += [c.value for c in s.canaries]
        code, timed_out, seconds, out, err = s.run_captured(timeout=PROBE_TIMEOUT)
        records = s.server.records()
        thread_ids = set(g2.header_values(records, "thread-id").values())
        rollouts = sorted(p.name for p in (s.tw.home / ".codex" / "sessions").glob("*/*/*/rollout-*.jsonl"))
        events = _parse_events(out)
        obs = {"block_types": g2.block_types(records), "transport": g2.transport(records), "install": s.install,
               "thinking": g2.thinking_resent(records, {"thinking": [], "redacted": [], "reasoning": ["mockcipher", "mock-redacted-0002"]}),
               "turn_metadata_shape": g2.header_shape(records, "x-codex-turn-metadata"),
               "turn_metadata_facts": g2.turn_metadata_facts(records),
               "git_remote_credential_found_in": g2.where_found(records, fake_pw),
               "turn_state_sent_by_mock": "mock-turn-state-0001",
               "turn_state_echoed_in_requests": g2.header_echoed(records, "x-codex-turn-state", "mock-turn-state-0001"),
               "model_requests": [r["index"] for r in records if r["route"] == g2.RESPONSES],
               "thread_id_header_values": len(thread_ids), "rollout_files": len(rollouts),
               "rollout_name_contains_thread_id": bool(thread_ids) and all(any(t in name for name in rollouts) for t in thread_ids),
               "rollout_path_pattern": "$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread-id>.jsonl" if rollouts else None,
               "event_types": sorted({str(e.get("type")) for e in events}),
               "error_items": sorted({str(e["item"].get("message", ""))[:90] for e in events
                                      if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "error"}),
               "results_seen": runner.results_seen(records)}
        return _result("codex-corpus", ["V3", "V4", "V5", "V6", "V7", "T4", "T8", "T9", "T10"], "codex", s, code, timed_out, seconds, obs)


def probe_codex_long(ctx: Context) -> dict:
    """
    Description: Send Codex a 1 MB prompt on stdin and look at compression and framing (V3, V7, T10).
    Parameters:
        ctx: the probe context.
    Output: The probe entry with transport() and the largest body size.
    """
    skip = _need("codex")
    if skip:
        return {"probe": "codex-long", **skip}
    with runner.agent_session("codex", ctx.seed, ctx.out_root, ctx.run_id, "codex-long", script=[{"text": "done"}], control=True) as s:
        argv = runner.codex_argv(s.tw.repo)[:-1] + ["-"]
        code, timed_out, seconds = s.run(argv=argv, timeout=PROBE_TIMEOUT, stdin_data=long_prompt(1_000_000).encode())
        records = s.server.records()
        biggest = max((r["body_size"] for r in records if r["route"] == g2.RESPONSES), default=0)
        obs = {"transport": g2.transport(records), "largest_model_request_bytes": biggest, "prompt_bytes_sent": 1_000_000}
        return _result("codex-long", ["V3", "V7", "T10"], "codex", s, code, timed_out, seconds, obs)


def probe_codex_block(ctx: Context) -> dict:
    """
    Description: Answer Codex's request with the planned block response (T3, and open question 8 of the parser spec).
    Parameters:
        ctx: the probe context.
    Output: The probe entry: exit code, the events Codex printed, whether the message was shown, and how often it retried.
    """
    skip = _need("codex")
    if skip:
        return {"probe": "codex-block-400", **skip}
    with runner.agent_session("codex", ctx.seed, ctx.out_root, ctx.run_id, "codex-block-400",
                              script=[{"error": {"status": 400, "message": BLOCK_MESSAGE}}], control=True) as s:
        code, timed_out, seconds, out, err = s.run_captured(timeout=PROBE_TIMEOUT)
        records = s.server.records()
        events = _parse_events(out)
        obs = {"event_types": sorted({str(e.get("type")) for e in events}), "message_shown_in_events": BLOCK_MESSAGE in out,
               "message_shown_on_stderr": BLOCK_MESSAGE in err, "model_requests": sum(1 for r in records if r["route"] == g2.RESPONSES),
               "statuses": g2.transport(records)["response_statuses"], "timed_out": timed_out}
        return _result("codex-block-400", ["T3"], "codex", s, code, timed_out, seconds, obs)


def _codex_endpoints(records: list[dict]) -> dict:
    """
    Description: Summarize what a Codex run called: endpoints, statuses and which credential headers each model request had.
    Parameters:
        records: the recording entries.
    Output: {"transport", "model_request": {content_encoding, credential_header_names, chatgpt_account_id_header}}.
    """
    model = [r for r in records if r["route"] == g2.RESPONSES]
    names = sorted({n.lower() for r in model for n, _ in r["headers"] if mock_provider.is_credential_header(n)})
    return {"transport": g2.transport(records),
            "model_request": {"count": len(model), "content_encodings": sorted({r.get("content_encoding") or "none" for r in model}),
                              "body_statuses": sorted({r["body_status"] for r in model}),
                              "credential_header_names": names}}


def probe_codex_chatgpt_route_a(ctx: Context) -> dict:
    """
    Description: Run Codex on route A (custom provider) with a fake ChatGPT login (T5).
    Parameters:
        ctx: the probe context.
    Output: The probe entry: every endpoint called (the ChatGPT-backend ones included), and the headers of the model request.
    """
    skip = _need("codex")
    if skip:
        return {"probe": "codex-chatgpt-route-a", **skip}
    with runner.agent_session("codex", ctx.seed, ctx.out_root, ctx.run_id, "codex-chatgpt-route-a", script=[{"text": "done"}],
                              control=True, before_install=lambda tw, server: write_chatgpt_auth(tw, server.url)) as s:
        code, timed_out, seconds = s.run(env=chatgpt_env(s.tw, s.server.url), timeout=PROBE_TIMEOUT)
        return _result("codex-chatgpt-route-a", ["T5"], "codex", s, code, timed_out, seconds, _codex_endpoints(s.server.records()))


def _route_b(ctx: Context, name: str, verifies: list[str], chatgpt: bool) -> dict:
    """
    Description: Run Codex on route B (built-in provider, base URL moved to the mock), with an API key or a fake ChatGPT login.
    Parameters:
        ctx: the probe context.
        name: the probe name.
        verifies: the items it settles.
        chatgpt: use a fake ChatGPT login instead of the API key.
    Output: The probe entry: the request sequence (WebSocket upgrade, its answer, the HTTP request that follows), the
        extensions offered, the Content-Encoding of the model request, and what Codex printed about the fallback.
    """
    skip = _need("codex")
    if skip:
        return {"probe": name, **skip}
    with runner.agent_session("codex", ctx.seed, ctx.out_root, ctx.run_id, name, script=[{"text": "done"}], control=True,
                              codex_config=lambda tw, url: write_route_b_config(tw, url, chatgpt)) as s:
        env = chatgpt_env(s.tw, s.server.url) if chatgpt else s.env()
        code, timed_out, seconds, out, err = s.run_captured(env=env, timeout=PROBE_TIMEOUT)
        records = s.server.records()
        sequence = [f"{r['method']} {r['path'].partition('?')[0]} -> {r['response_status']}" for r in records
                    if r["route"] in ("websocket-upgrade", g2.RESPONSES)]
        obs = {**_codex_endpoints(records), "model_request_sequence": sequence,
               "stderr_mentions_426": "426" in err, "stderr_mentions_websocket": "websocket" in err.lower(),
               "turn_completed": any(e.get("type") == "turn.completed" for e in _parse_events(out))}
        return _result(name, verifies, "codex", s, code, timed_out, seconds, obs)


def probe_codex_route_b_apikey(ctx: Context) -> dict:
    """
    Description: Route B with an API key (T6).
    Parameters:
        ctx: the probe context.
    Output: See _route_b.
    """
    return _route_b(ctx, "codex-route-b-apikey", ["T6"], False)


def probe_codex_route_b_chatgpt(ctx: Context) -> dict:
    """
    Description: Route B with a fake ChatGPT login (T6, T7).
    Parameters:
        ctx: the probe context.
    Output: See _route_b.
    """
    return _route_b(ctx, "codex-route-b-chatgpt", ["T6", "T7"], True)


def probe_codex_compaction(ctx: Context) -> dict:
    """
    Description: Set a tiny auto-compact limit, so Codex compacts between turns, and see which endpoint it uses (T9).
    Parameters:
        ctx: the probe context.
    Output: The probe entry: every endpoint called and the Content-Encoding and statuses; compaction shows as a path other than
        /v1/responses, or as extra /v1/responses requests when Codex compacts locally.
    """
    skip = _need("codex")
    if skip:
        return {"probe": "codex-compaction", **skip}
    with runner.agent_session("codex", ctx.seed, ctx.out_root, ctx.run_id, "codex-compaction", control=True) as s:
        argv = runner.codex_argv(s.tw.repo)
        argv[2:2] = ["-c", "model_auto_compact_token_limit=2000"]
        code, timed_out, seconds, out, err = s.run_captured(argv=argv, timeout=COMPACTION_SECONDS)
        records = s.server.records()
        compaction = [r for r in records if g2.request_kind(r) == "compaction"]
        facts = g2.turn_metadata_facts(records)
        obs = {"requests_before_the_kill": len(records), "request_kinds": facts["request_kinds"],
               "results_seen": runner.results_seen(records), "turn_completed": any(e.get("type") == "turn.completed" for e in _parse_events(out)),
               "compaction_requests": len(compaction),
               "compaction_endpoint": sorted({f"{r['method']} {r['path'].partition('?')[0]}" for r in compaction}),
               "compaction_request_tools": sorted({len(json.loads(r["body"]).get("tools", [])) for r in compaction if r.get("body")}),
               "non_responses_paths": sorted({r["path"].partition("?")[0] for r in records
                                              if r["method"] == "POST" and r["route"] != g2.RESPONSES}),
               "note": "the mock's usage numbers never shrink, so Codex compacts in a loop until the probe kills it"}
        return _result("codex-compaction", ["T9"], "codex", s, code, timed_out, seconds, obs)


PROBES: dict[str, Callable[[Context], dict]] = {
    "claude-corpus": probe_claude_corpus, "claude-long": probe_claude_long, "claude-settings": probe_claude_settings,
    "claude-block-400": probe_claude_block, "codex-corpus": probe_codex_corpus, "codex-long": probe_codex_long,
    "codex-block-400": probe_codex_block, "codex-chatgpt-route-a": probe_codex_chatgpt_route_a,
    "codex-route-b-apikey": probe_codex_route_b_apikey, "codex-route-b-chatgpt": probe_codex_route_b_chatgpt,
    "codex-compaction": probe_codex_compaction,
}


def run_probes(seed: int, out_root: Path, run_id: str, only: Optional[list[str]] = None) -> tuple[dict, Path]:
    """
    Description: Run the chosen probes and write the report.
    Parameters:
        seed: the canary seed.
        out_root: the results folder; the report goes to proof/g2-probes-<run_id>.json.
        run_id: the run's id.
        only: probe names to run, or None for all.
    Output: (report, path). A probe that raises is recorded as an error with its message (no values). The real home is
        snapshotted before and after; the report lists any change (expected: none).
    """
    names = only or list(PROBES)
    unknown = [n for n in names if n not in PROBES]
    if unknown:
        raise ValueError(f"unknown probes: {unknown}")
    ctx = Context(seed, Path(out_root), run_id)
    before = iso.real_home_snapshot()
    results = []
    for name in names:
        try:
            results.append(PROBES[name](ctx))
        except Exception as exc:  # noqa: BLE001 - one broken probe must not hide the others
            results.append({"probe": name, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    changed = iso.snapshot_diff(before, iso.real_home_snapshot())
    report = {"run_id": run_id, "kind": "g2-probes", "seed": seed, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "python": sys.version.split()[0], "probes": results,
              "isolation": {"real_home_changed": changed, "method": "throwaway HOME and CODEX_HOME, from-scratch environment, "
                            "fake keys, loopback mock; fake ChatGPT login also has its refresh URL and chatgpt_base_url on loopback"}}
    path = write_probe_report(Path(out_root) / "proof" / f"g2-probes-{run_id}.json", report, ctx.secrets)
    return report, path


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Run the G2 probes from the command line.
    Parameters:
        argv: arguments, or None for sys.argv.
    Output: Exit status 0 when every probe ran and the real home is unchanged, else 1; prints one line per probe (status only).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="comma-separated probe names")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "bench" / "results")
    args = parser.parse_args(argv)
    report, path = run_probes(args.seed, args.out_dir, args.run_id, args.only.split(",") if args.only else None)
    for entry in report["probes"]:
        print(f"{entry['probe']}: {entry.get('status')}")
    print(f"real home changed: {report['isolation']['real_home_changed']}\nreport: {path}")
    bad = [e for e in report["probes"] if e.get("status") in ("error",)] or report["isolation"]["real_home_changed"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
