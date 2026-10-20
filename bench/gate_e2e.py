#!/usr/bin/env python3
"""End-to-end test of the egress gate with real agents (AGENTS.md rule 7; FR-GATE-01, FR-GATE-03, SEC-10, FR-GATE-04).

    python -m bench.gate_e2e --suite g3 [--out-dir bench/results]
    python -m bench.gate_e2e --suite g4 [--out-dir bench/results]

Each scenario runs a real `claude -p` or `codex exec` in a throwaway HOME (bench/agent_isolation.py) with a fake API key,
pointed at the gate on 127.0.0.1, whose upstream is bench/mock_provider.py on 127.0.0.1. The gate token goes where an
install would put it: ANTHROPIC_CUSTOM_HEADERS for Claude Code, http_headers in the Codex provider. Nothing reaches a real
provider, and the real ~/.harpo, ~/.harpocrates, ~/.claude/settings.json, ~/.codex/config.toml and ~/.codex/auth.json are
snapshotted before and after the suite.

g3  The gate with the G3 scan step, which blocked everything (reproduced here as the explicit "stub" step, since the gate no
    longer ships one): both agents must show the gate's 400 message and the mock must record nothing. Then two Claude Code variants with a test-only pass-through scan step, so the mock can
    script tool calls: the install record says the gate is on the port Claude Code uses (SEC-10 stays quiet) and on another
    port (SEC-10 refuses every tool call).
g4  The gate with its real scanner, in front of the scripted reads (Read .env, cat, grep): alone, and with Harpo's read hook
    installed too; plus a control with the gate off (a pass-through step), which must leak, to show the counter works. The
    mock's recording is searched for the planted canaries and for placeholders.

Output: bench/results/proof/<run>.json holds counts only (checked for canary values). The mock's recording, which holds
request bodies, goes to bench/results/mock/<run>/ and is never committed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))

from bench import agent_isolation as iso  # noqa: E402
from bench import agent_runner as runner  # noqa: E402
from bench import canary_report as report  # noqa: E402
from Harpocrates.gate.config import GateConfig  # noqa: E402
from Harpocrates.gate.scan import Forward  # noqa: E402
from Harpocrates.gate.server import RunningGate, start_gate  # noqa: E402
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token  # noqa: E402

E2E_VERSION = "1"
DEFAULT_OUT = ROOT / "bench" / "results"
BLOCK_MESSAGE = re.compile(r"Harpo blocked this request before it left your machine \(reason (HARPO-BLOCK-[A-Z0-9-]+)\)")
SEC10_PHRASE = "Harpo gate bypass"
PLACEHOLDER = re.compile(r"<<HARPO:([a-z0-9_]+):[0-9a-f]{4}>>")


class Scenario(NamedTuple):
    """
    Description: One agent session through the gate.
    Parameters:
        label: a short unique name, used for the recording folder.
        agent: claude or codex.
        scan: "default" for the scan step the gate ships (the real scanner), "passthrough" for a test-only step that forwards
            everything, "stub" for a test-only step that raises, as the G3 stub did (every request blocked).
        hook: whether Harpo is installed into the throwaway home (its read hook, and for Claude Code the SEC-10 check).
        record: how the install record's gate entry relates to the gate: "match" (its port is the gate's), "mismatch"
            (another port), or "none" (no entry written).
        capture: keep the agent's output to look for the block message. Only for sessions that read no canary file.
        script: "reads" for the agent runner's scripted reads of the canary repo (Read .env, cat, grep), or "sec10" for
            three calls that read no canary file: Read README.md, cat config/settings.py, grep, as Claude Code names them.
    Output: An immutable record.
    """

    label: str
    agent: str
    scan: str
    hook: bool
    record: str
    capture: bool = False
    script: str = "reads"


SUITES: dict[str, list[Scenario]] = {
    "g3": [Scenario("claude-default-scan", "claude", "stub", True, "match", True),
           Scenario("codex-default-scan", "codex", "stub", True, "none", True),
           Scenario("claude-sec10-match", "claude", "passthrough", True, "match", False, "sec10"),
           Scenario("claude-sec10-mismatch", "claude", "passthrough", True, "mismatch", True, "sec10")],
    "g4": [Scenario("claude-gate-only", "claude", "default", False, "none"),
           Scenario("codex-gate-only", "codex", "default", False, "none"),
           Scenario("claude-gate-and-hook", "claude", "default", True, "match"),
           Scenario("codex-gate-and-hook", "codex", "default", True, "none"),
           Scenario("claude-gate-off-control", "claude", "passthrough", False, "none"),
           Scenario("codex-gate-off-control", "codex", "passthrough", False, "none")],
}


def sec10_script(repo: Path) -> list[dict]:
    """
    Description: Three tool calls for the SEC-10 scenarios, none of which the install's own deny rules stop first.
    Parameters:
        repo: the canary repo.
    Output: Four turns: Read README.md, Bash cat config/settings.py, Bash grep -rn TOKEN ., then {"text": "done"}. (The
        installer denies the built-in Read of .env, which would hide whether SEC-10 refused that call.)
    """
    calls = [("Read", {"file_path": str(repo / "README.md")}),
             ("Bash", {"command": "cat config/settings.py", "description": "Show settings"}),
             ("Bash", {"command": "grep -rn TOKEN .", "description": "Search for TOKEN"})]
    return [{"tool_calls": [{"name": name, "input": data}]} for name, data in calls] + [{"text": "done"}]


def stub_scan(body: bytes, route: object, *rest: object) -> Forward:
    """
    Description: A scan step that fails every time, as the G3 stub did, so every request is blocked.
    Parameters:
        body: the decoded request body (ignored).
        route: the route (ignored).
        *rest: any further arguments the gate passes (ignored).
    Output: Never returns. Raises RuntimeError, which the gate turns into a block.
    """
    raise RuntimeError("the G3 stub: no scanner")


def passthrough_scan(body: bytes, route: object, *rest: object) -> Forward:
    """
    Description: A scan step that forwards every request unchanged. Only these tests use it; the gate ships none.
    Parameters:
        body: the decoded request body.
        route: the route (ignored).
        *rest: any further arguments the gate passes (ignored).
    Output: Forward(body).
    """
    return Forward(body)


def merge_json_file(path: Path, patch: dict) -> None:
    """
    Description: Merge top-level keys into a JSON file in the throwaway home, creating it when missing.
    Parameters:
        path: the file.
        patch: keys to set; a dict value is merged one level deep with what is there.
    Output: None. Raises ValueError for a file that is not a JSON object.
    """
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    for key, value in patch.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) and isinstance(data.get(key), dict) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_gate_install(tw: iso.Throwaway, gate: RunningGate, token: str, record: str) -> None:
    """
    Description: Make the throwaway Claude Code install say what `harpo guard up` will say once the gate is on.
    Parameters:
        tw: the throwaway session.
        gate: the running gate.
        token: the gate token.
        record: "match" (the record names the gate's port), "mismatch" (it names another port) or "none" (no gate entry).
    Output: None. Sets ANTHROPIC_BASE_URL and ANTHROPIC_CUSTOM_HEADERS in tw.home/.claude/settings.json and, unless record
        is "none", the "gate" entry {"enabled", "port", "token_header"} of tw.home/.harpo/install.json (SEC-10).
    """
    merge_json_file(tw.home / ".claude" / "settings.json", {"env": {
        "ANTHROPIC_BASE_URL": gate.url, "ANTHROPIC_CUSTOM_HEADERS": f"{TOKEN_HEADER}: {token}"}})
    if record != "none":
        port = gate.port if record == "match" else (gate.port % 60000) + 1024 + 1
        merge_json_file(tw.home / ".harpo" / "install.json",
                        {"gate": {"enabled": True, "port": port, "token_header": TOKEN_HEADER}})


def sec10_refusals(records: Iterable[dict]) -> int:
    """
    Description: Count the tool calls SEC-10 refused, as seen in the tool results the agent sent back to the provider.
    Parameters:
        records: the mock's recording entries.
    Output: The number of distinct tool_use ids whose tool_result carries the SEC-10 refusal text, across all requests.
    """
    ids: set[str] = set()
    for rec in records:
        if rec.get("route") != "anthropic.messages" or not isinstance(rec.get("body"), str) or SEC10_PHRASE not in rec["body"]:
            continue
        for message in json.loads(rec["body"]).get("messages", []):
            for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
                if block.get("type") == "tool_result" and SEC10_PHRASE in json.dumps(block.get("content")):
                    ids.add(block.get("tool_use_id", ""))
    return len(ids)


def placeholder_counts(records: Iterable[dict]) -> dict:
    """
    Description: Count the <<HARPO:type:hash4>> placeholders in the model requests the mock recorded.
    Parameters:
        records: the mock's recording entries.
    Output: {"total": n, "by_type": {type: n}} over the last model request of the recording, which holds the whole
        conversation (earlier requests are prefixes of it, so counting all would count a placeholder several times).
    """
    bodies = [r["body"] for r in records if r.get("route") in runner.MODEL_ROUTES and isinstance(r.get("body"), str)]
    found = PLACEHOLDER.findall(bodies[-1]) if bodies else []
    by_type: dict[str, int] = {}
    for kind in found:
        by_type[kind] = by_type.get(kind, 0) + 1
    return {"total": len(found), "by_type": dict(sorted(by_type.items()))}


def tool_results_with_placeholder(records: Iterable[dict]) -> tuple[int, int]:
    """
    Description: Count the tool results in the last model request that carry a placeholder, out of all tool results in it.
    Parameters:
        records: the mock's recording entries.
    Output: (results with a <<HARPO:type:hash4>> placeholder, results). A result is a tool_result block of a Messages request
        or a function_call_output item of a Responses request, in the last model request (the whole conversation). (0, 0) when
        there is no model request.
    """
    bodies = [r["body"] for r in records if r.get("route") in runner.MODEL_ROUTES and isinstance(r.get("body"), str)]
    if not bodies:
        return 0, 0
    body = json.loads(bodies[-1])
    results = [json.dumps(block.get("content")) for message in body.get("messages", []) if isinstance(message.get("content"), list)
               for block in message["content"] if block.get("type") == "tool_result"]
    results += [json.dumps(item.get("output")) for item in body.get("input", []) if isinstance(item, dict)
                and item.get("type") == "function_call_output"]
    return sum(bool(PLACEHOLDER.search(text)) for text in results), len(results)


def gate_log_summary(lines: list[str]) -> dict:
    """
    Description: Summarise the gate's request log without any content.
    Parameters:
        lines: the JSONL lines the gate wrote.
    Output: {"lines": n, "by_decision": {...}, "by_reason": {...}, "by_route": {...}}.
    """
    summary: dict = {"lines": len(lines), "by_decision": {}, "by_reason": {}, "by_route": {}}
    for line in lines:
        entry = json.loads(line)
        for key, name in (("decision", "by_decision"), ("reason", "by_reason"), ("route", "by_route")):
            if key in entry:
                summary[name][entry[key]] = summary[name].get(entry[key], 0) + 1
    return summary


class _Rig:
    """
    Description: The gate and its token, started once the mock's address is known.
    Parameters:
        scan: "default", "passthrough" or "stub".
    Output: An object with start(mock_url), gate, token and log_lines.
    """

    def __init__(self, scan: str) -> None:
        """
        Description: Remember the scan mode; nothing runs yet.
        Parameters:
            scan: "default", "passthrough" or "stub".
        Output: None.
        """
        self.scan, self.gate, self.token, self.log_lines = scan, None, new_token(), []

    def start(self, mock_url: str) -> RunningGate:
        """
        Description: Start the gate once, pointed at the mock for both providers.
        Parameters:
            mock_url: the mock provider's loopback URL.
        Output: The running gate (the same one on a second call).
        """
        if self.gate is None:
            extra = {} if self.scan == "default" else {"scan_step": stub_scan if self.scan == "stub" else passthrough_scan}
            config = GateConfig(anthropic_upstream=mock_url, openai_upstream=mock_url)
            self.gate = start_gate(config, self.token, log_sink=self.log_lines.append, **extra)
        return self.gate


def run_scenario(sc: Scenario, seed: int, out_root: Path, run_id: str, timeout: int) -> dict:
    """
    Description: Run one agent through the gate and describe what the mock received.
    Parameters:
        sc: the scenario.
        seed: the canary seed.
        out_root: the results folder; the recording goes to mock/<run_id>/<label>/.
        run_id: the run's id.
        timeout: seconds allowed for the agent process.
    Output: A dict of counts and facts, no values: agent, version, exit code, mock request counts, the gate log summary,
        whether the block message was shown, SEC-10 refusals, tool results seen, planted and received canaries,
        placeholders. The result's "_canaries" key holds the Canary list for the report's leak check (not serialized).
    """
    rig = _Rig(sc.scan)
    out: dict = {"scenario": sc.label, "agent": sc.agent, "scan": sc.scan, "hook_installed": sc.hook, "record": sc.record,
                 "version": runner._version(sc.agent)}
    if out["version"] is None:
        return {**out, "status": "skipped", "reason": f"`{sc.agent}` is not on PATH", "_canaries": []}
    token_line = f'http_headers = {{ "{TOKEN_HEADER}" = "{rig.token}" }}\n'
    try:
        with runner.agent_session(sc.agent, seed, out_root, run_id, label=sc.label, control=not sc.hook,
                                  script=sec10_script if sc.script == "sec10" else None,
                                  codex_config=lambda tw, url: runner.write_codex_config(tw, rig.start(url).url, token_line),
                                  before_install=lambda tw, server: rig.start(server.url)) as session:
            gate = rig.start(session.server.url)
            if sc.agent == "claude" and sc.hook:
                write_gate_install(session.tw, gate, rig.token, sc.record)
            env = session.env()
            if sc.agent == "claude":  # Codex reads the gate address and token from its config.toml
                env.update({"ANTHROPIC_BASE_URL": gate.url, "ANTHROPIC_CUSTOM_HEADERS": f"{TOKEN_HEADER}: {rig.token}"})
            argv = session.argv()
            if sc.capture:
                code, timed_out, seconds, stdout, stderr = session.run_captured(argv, env, timeout)
            else:
                code, timed_out, seconds = session.run(argv, env, timeout)
                stdout = stderr = ""
            records = session.server.records()
            canaries = session.canaries
            install = session.install
    finally:
        if rig.gate is not None:
            rig.gate.close()
    shown = BLOCK_MESSAGE.search(stdout + stderr)
    findings = report.find_received(records, canaries)
    seen = runner.results_seen(records)
    out.update({
        "status": "timeout" if timed_out else "ok", "exit_code": code, "seconds": round(seconds, 1), "install": install,
        "mock_requests": len(records), "mock_model_requests": sum(r.get("route") in runner.MODEL_ROUTES for r in records),
        "gate_log": gate_log_summary(rig.log_lines),
        "block_message_shown": bool(shown) if sc.capture else None, "block_reason_shown": shown.group(1) if shown else None,
        "tool_results_seen": seen, "sec10_refusals": sec10_refusals(records),
        "planted": report.counts(canaries), "received": report.received_counts(canaries, findings),
        "placeholders": placeholder_counts(records), "tool_results_with_placeholder": tool_results_with_placeholder(records)[0],
        "_canaries": canaries})
    return out


def run_suite(suite: str, seed: int, out_root: Path, run_id: str, timeout: int = runner.AGENT_TIMEOUT,
              runner_fn: Callable[..., dict] = run_scenario) -> tuple[dict, Path]:
    """
    Description: Run every scenario of a suite and write the counts-only report.
    Parameters:
        suite: g3 or g4.
        seed: the canary seed.
        out_root: the results folder; the report goes to proof/<run_id>.json.
        run_id: the run's id.
        timeout: seconds allowed per agent process.
        runner_fn: the per-scenario function, replaceable for tests.
    Output: (the report dict, its path). The real home is snapshotted before and after; the difference is in the report
        under isolation.real_home_changed. The report holds no canary value (write_report checks).
    """
    before = iso.real_home_snapshot()
    results = [runner_fn(sc, seed, out_root, run_id, timeout) for sc in SUITES[suite]]
    changed = iso.snapshot_diff(before, iso.real_home_snapshot())
    canaries = [c for r in results for c in r.pop("_canaries")]
    data = {"e2e_version": E2E_VERSION, "run_id": run_id, "suite": suite, "seed": seed, "target": "gate -> mock",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"python": sys.version.split()[0], **{r["agent"]: r["version"] for r in results}},
            "scenarios": results,
            "isolation": {"real_home_paths_snapshotted": len(before), "real_home_changed": changed,
                          "method": "throwaway HOME and CODEX_HOME, from-scratch environment, fake API keys, loopback "
                                    "gate and mock, paths.home pinned by bootstrap"}}
    return data, report.write_report(Path(out_root) / "proof" / f"{run_id}.json", data, canaries)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Run a suite from the command line.
    Parameters:
        argv: arguments, or None for sys.argv.
    Output: Exit status 0 after writing the report (the report, not the status, says whether the checks held); 1 when the
        real home changed. Prints the report path and one counts-only line per scenario.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout", type=int, default=runner.AGENT_TIMEOUT)
    args = parser.parse_args(argv)
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-gate-e2e-{args.suite}-s{args.seed}"
    data, path = run_suite(args.suite, args.seed, args.out_dir, run_id, args.timeout)
    for r in data["scenarios"]:
        print(f"{r['scenario']}: {r.get('status')} exit={r.get('exit_code')} mock_requests={r.get('mock_requests')} "
              f"blocked_msg={r.get('block_message_shown')} sec10_refusals={r.get('sec10_refusals')} "
              f"results_seen={r.get('tool_results_seen')} received_known={r.get('received', {}).get('known_format')} "
              f"placeholders={r.get('placeholders', {}).get('total')}")
    print(f"report: {path}")
    return 1 if data["isolation"]["real_home_changed"] else 0


if __name__ == "__main__":
    sys.exit(main())
