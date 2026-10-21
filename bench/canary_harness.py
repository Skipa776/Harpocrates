#!/usr/bin/env python3
"""Canary harness: count the planted canaries a mock LLM provider received (FR-PROOF-04, G2).

    python bench/canary_harness.py --mode direct [--target mock|gate] [--seed N] [--out-dir bench/results]
    python bench/canary_harness.py --mode agent  [--seed N] [--agent claude|codex|all] [--out-dir bench/results]

direct   Builds one synthetic request per block, item and tool type covered by FR-GATE-02 (docs/design.md), in both provider
         formats, each carrying a generated canary (bench/gate_corpus.py), and sends them to a mock provider on
         127.0.0.1. Against the bare mock every canary must arrive, which shows that the counter works. The corpus
         is also the test data for the test_fr_gate_02_* parser tests. With --target gate the requests go to the real gate
         (its real scanner) in front of the mock, and no known-format canary may arrive.
agent    Builds the seeded canary repo (bench/canary_repo.py) and runs `claude -p` and `codex exec` in it against the
         mock, driven by scripted tool calls (bench/agent_runner.py), each in a throwaway HOME with a fake API key. Agents
         through the gate are run by bench/gate_e2e.py.

Output: bench/results/proof/<run>.json holds counts, kinds, known-format versus ML-only, and where each received canary
sat (request, field path, block type), plus the harness and agent versions. It never holds a value; the harness checks.
The mock's recording, which does hold request bodies, goes to bench/results/mock/<run>/ and is never committed.

Exit status: 0 passed; 1 a known-format canary reached the provider through a path the target claims to protect (or, in
direct mode, a planted canary was not counted); 2 inconclusive (an agent is missing or did not finish). Every run
publishes its counts, failing or not.
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))

from bench import canary_report as report  # noqa: E402
from bench import gate_corpus, mock_provider  # noqa: E402
from Harpocrates.gate.config import GateConfig  # noqa: E402
from Harpocrates.gate.server import start_gate  # noqa: E402
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token  # noqa: E402

HARNESS_VERSION = "1"
DEFAULT_OUT = ROOT / "bench" / "results"
CASE_HEADER = "X-Canary-Case"


class HarnessResult(NamedTuple):
    """
    Description: The outcome of one harness run.
    Parameters:
        report: the report dict that was written.
        verdict: the exit decision.
        report_path: where the report was written.
        canaries: every planted canary, in memory only.
    Output: An immutable record; print report, never canaries.
    """

    report: dict
    verdict: report.Verdict
    report_path: Path
    canaries: list


def new_run_id(mode: str, seed: int, now: Optional[datetime] = None) -> str:
    """
    Description: Name a run so its report and recording share an id.
    Parameters:
        mode: direct or agent.
        seed: the seed.
        now: the time to use, or None for the current UTC time.
    Output: A string like 20261019T153000Z-direct-s0.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{mode}-s{seed}"


def post_case(url: str, case: gate_corpus.Case, body: Optional[dict] = None, headers: Optional[dict] = None) -> int:
    """
    Description: Send one corpus case to the mock provider (or to the gate in front of it) with http.client, which never
        uses a proxy.
    Parameters:
        url: the base URL of the mock or the gate.
        case: the corpus case; its route and body are used.
        body: a body to send instead of the case's, for tests.
        headers: further headers (the gate token).
    Output: The HTTP status. The request carries an X-Canary-Case header with the case id, so the recording can be matched
        to the case. Raises OSError when the server cannot be reached.
    """
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=30)
    try:
        conn.connect()
        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # headers and body are two writes
        conn.request("POST", case.route, json.dumps(case.body if body is None else body).encode(), {
            "Content-Type": "application/json", "anthropic-version": "2023-06-01", CASE_HEADER: case.case_id,
            **(headers or {})})
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


def _case_requests(records: list[dict]) -> dict[str, int]:
    """
    Description: Map each corpus case id to the index of the recorded request that carried it.
    Parameters:
        records: the recording entries.
    Output: {case id: request index}, read from the X-Canary-Case header of each entry.
    """
    out: dict[str, int] = {}
    for rec in records:
        for name, value in rec.get("headers", []):
            if name.lower() == CASE_HEADER.lower():
                out[value] = rec["index"]
    return out


def run_direct(seed: int, out_root: Path, run_id: str, target: str = "mock",
               gate_scan_step: Optional[Callable] = None) -> HarnessResult:
    """
    Description: Run direct mode: send the corpus to a mock provider, bare or behind the gate, and count what arrived.
    Parameters:
        seed: the corpus seed.
        out_root: the results folder; the report goes to proof/<run_id>.json and the recording to mock/<run_id>/.
        run_id: the run's id.
        target: "mock" (the bare mock: every canary must arrive, which shows the counter works) or "gate" (the real gate in
            front of the mock: no known-format canary may arrive).
        gate_scan_step: a scan step to run the gate with instead of its own scanner, for tests of the harness.
    Output: A HarnessResult. Against the mock the verdict is 1 when a planted canary is missing from the recording (the
        counter is broken). Against the gate it is 1 when a known-format canary arrived. Starts and stops a mock provider
        (and a gate) on loopback ports; writes the report and the recording. The report of a gate run adds each case's HTTP
        status and the counts forwarded and blocked.
    """
    cases = gate_corpus.build_corpus(seed)
    canaries = [report.Canary(f"case:{c.case_id}", c.canary_kind, c.canary_value) for c in cases if c.canary_value]
    statuses: dict[str, int] = {}
    with mock_provider.start_server(record_dir=Path(out_root) / "mock" / run_id) as server:
        if target == "gate":
            token, extra = new_token(), {"scan_step": gate_scan_step} if gate_scan_step else {}
            with start_gate(GateConfig(anthropic_upstream=server.url, openai_upstream=server.url), token, **extra) as gate:
                for case in cases:
                    statuses[case.case_id] = post_case(gate.url, case, headers={TOKEN_HEADER: token})
        else:
            for case in cases:
                post_case(server.url, case)
        records = server.records()
    findings = report.find_received(records, canaries)
    # A connector credential the user configured on purpose (decision 6, class K) is meant to arrive: listed, not a leak.
    by_design = {c.case_id for c in cases if c.expect == "pass" and c.canary_value} if target == "gate" else set()
    where = _case_requests(records)
    rows = []
    for case in cases:
        own = [f for f in findings if f.ident == f"case:{case.case_id}" and f.request == where.get(case.case_id)]
        rows.append({**case.summary(), "planted_path": case.field_path, "received": bool(own),
                     "found_paths": list(dict.fromkeys(f.path for f in own)),
                     **({"gate_status": statuses[case.case_id], "by_design": case.case_id in by_design}
                        if target == "gate" else {})})
    planted = report.counts(canaries)
    received = report.received_counts(canaries, [f for f in findings if f.ident.removeprefix("case:") not in by_design])
    routes: dict[str, int] = {}
    for rec in records:
        routes[rec["route"]] = routes.get(rec["route"], 0) + 1
    gated = target == "gate"
    verdict = (report.verdict("direct", "gate", received["known_format"], all_ran=True) if gated
               else report.direct_verdict(planted["total"], received["total"]))
    data = {
        "harness_version": HARNESS_VERSION, "run_id": run_id, "mode": "direct", "seed": seed,
        "target": "gate -> mock" if gated else "mock",
        "protection": "gate" if gated else "none", "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "mock_provider": "1"},
        "planted": planted, "received": received, "requests": {"total": len(records), "by_route": dict(sorted(routes.items()))},
        "findings": [f._asdict() for f in findings], "direct": {"cases": rows},
        **({"gate": {"forwarded": sum(s < 400 for s in statuses.values()), "blocked": sum(s >= 400 for s in statuses.values()),
                     "received_by_design": sorted(by_design)}} if gated else {}),
        "verdict": {"exit_code": verdict.exit_code, "reason": verdict.reason},
    }
    path = report.write_report(Path(out_root) / "proof" / f"{run_id}.json", data, canaries)
    return HarnessResult(data, verdict, path, canaries)


def _summary(result: HarnessResult) -> str:
    """
    Description: Format a run for the console: counts only.
    Parameters:
        result: the run's result.
    Output: A few lines with the run id, planted and received counts by class, the verdict and the report path.
    """
    data = result.report
    lines = [f"run {data['run_id']} mode={data['mode']} seed={data['seed']} target={data['target']} "
             f"protection={data['protection']}"]
    for label in ("planted", "received"):
        by_class = data[label]["by_class"]
        lines.append(f"{label}: {data[label]['total']} (known-format {by_class['known-format']}, ml-only {by_class['ml-only']})")
    lines.append(f"verdict: {result.verdict.reason} (exit {result.verdict.exit_code})")
    lines.append(f"report: {result.report_path}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Run the harness from the command line.
    Parameters:
        argv: arguments, or None for sys.argv.
    Output: The exit status (see the module docstring); prints a counts-only summary. An invalid option exits with status 2.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("direct", "agent"), required=True)
    parser.add_argument("--target", choices=("mock", "gate"), default="mock",
                        help="mock: the bare mock provider. gate: the gate in front of it (direct mode only; agents behind the "
                             "gate are run by bench/gate_e2e.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", choices=("claude", "codex", "all"), default="all", help="agent mode only")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--control", action="store_true", help="agent mode: install nothing, to show the counter sees leaks")
    parser.add_argument("--keep-work", action="store_true", help="agent mode: keep the throwaway folders for inspection")
    parser.add_argument("--timeout", type=int, default=240, help="agent mode: seconds allowed per agent")
    args = parser.parse_args(argv)
    if args.mode == "agent" and args.target == "gate":
        parser.error("agent mode through the gate is run by bench/gate_e2e.py")
    run_id = args.run_id or new_run_id(args.mode, args.seed)
    if args.mode == "direct":
        result = run_direct(args.seed, args.out_dir, run_id, args.target)
    else:
        from bench import (
            agent_runner,  # noqa: PLC0415 - agent mode pulls in the throwaway-home machinery
        )

        result = agent_runner.run_agents(args.seed, args.out_dir, run_id, args.agent, args.keep_work, args.timeout,
                                         control=args.control)
    print(_summary(result))
    return result.verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
