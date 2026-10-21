#!/usr/bin/env python3
"""End-to-end test of G7 and G6 with real agents (FR-INSTALL-01, -02, -06, -08, FR-UX-01, FR-LOG-01, FR-PROOF-01, -02,
SEC-01).

    python -m bench.guard_e2e [--seed N] [--out-dir DIR]

Each scenario runs `harpo guard up` through the real CLI in a throwaway home (bench/agent_isolation.py), then a real
`claude -p` or `codex exec` whose environment names no base URL and no token: its only routing is what guard up wrote.
"routed": no known-format canary may reach the mock, and `harpo doctor` must pass. "after-down": `harpo guard down` runs
before the agent; the files must be byte-identical to before guard up, and the same reads must then reach the mock,
which shows that guard down took the routing out and that the counter works.

G6 and SEC-01, read after each session: every request the mock received has a proof entry with the same SHA-256, the
chain verifies, the gate logged findings, and no canary value is in any file under the throwaway ~/.harpo or TMPDIR.

Only here is the gate's upstream the mock: the throwaway bootstrap swaps in a GateConfig whose upstream defaults are the
mock, for the gate process, when HARPO_E2E_UPSTREAM is set. The product has no such switch. Output: DIR/proof/<run>.json,
counts only; the recordings go to DIR/mock/<run>/ and are never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT))

from bench import agent_isolation as iso  # noqa: E402
from bench import agent_runner as runner  # noqa: E402
from bench import canary_report as report  # noqa: E402
from Harpocrates import audit  # noqa: E402

DEFAULT_OUT = ROOT / "bench" / "results"
ROUTING_VARS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS", "OPENAI_BASE_URL")
HARPO_TIMEOUT = 120


class Scenario(NamedTuple):
    """
    Description: One agent session behind a gate that `harpo guard up` started.
    Parameters:
        agent: claude or codex.
        mode: "routed" (guard up, then the agent) or "after-down" (guard up, guard down, then the agent).
    Output: An immutable record.
    """

    agent: str
    mode: str

    @property
    def label(self) -> str:
        """
        Description: The scenario's name, used for its recording folder.
        Parameters: None
        Output: <agent>-guard-<mode>.
        """
        return f"{self.agent}-guard-{self.mode}"


SCENARIOS = [Scenario("claude", "routed"), Scenario("codex", "routed"),
             Scenario("claude", "after-down"), Scenario("codex", "after-down")]
FILES = {"claude": ".claude/settings.json", "codex": ".codex/config.toml"}


def codex_config(tw: iso.Throwaway, _url: str) -> Path:
    """
    Description: A Codex config with no provider, so guard up adds its own (it refuses one that is already set).
    Parameters:
        tw: the throwaway session.
        _url: the mock's URL (unused: the gate is Codex's provider).
    Output: The config path.
    """
    path = tw.home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('model = "gpt-5"\n' + runner.CODEX_QUIET, encoding="utf-8")
    return path


def harpo(tw: iso.Throwaway, args: list[str], mock_url: str) -> subprocess.CompletedProcess:
    """
    Description: Run the real `harpo` CLI in the throwaway home through the bootstrap.
    Parameters:
        tw: the throwaway session.
        args: the CLI arguments.
        mock_url: the mock's URL: the upstream of a gate this command starts.
    Output: The CompletedProcess, with stdout and stderr as text.
    """
    iso.assert_outside_real_home(tw.home)
    env = iso.clean_env(tw, "http://127.0.0.1:1", {"HARPO_E2E_UPSTREAM": mock_url})
    argv = [sys.executable, "-I", "-c", iso.BOOT, str(tw.home), str(iso.write_wrapper(tw)), "Harpocrates.cli", *args]
    return subprocess.run(argv, cwd=tw.repo, env=env, capture_output=True, text=True, timeout=HARPO_TIMEOUT,
                          stdin=subprocess.DEVNULL, check=False)


def _digest(path: Path) -> Optional[str]:
    """
    Description: Fingerprint a file, to compare it before and after guard up and guard down.
    Parameters:
        path: the file.
    Output: Its SHA-256 in hex, or None when it does not exist.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def sweep(roots: list[Path], canaries: list) -> dict:
    """
    Description: SEC-01: look for every canary value in every file under some folders.
    Parameters:
        roots: the folders.
        canaries: the planted Canary records.
    Output: {"files": the files read, "hits": [{"file", "canaries": how many}]}, each file named relative to its root's
        parent. A value counts as found as is or JSON-escaped (a private key's newlines). Never the values themselves.
    """
    needles = [{c.value.encode(), json.dumps(c.value)[1:-1].encode()} for c in canaries]
    files, hits = [], []
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
            name = str(path.relative_to(root.parent))
            files.append(name)
            found = sum(any(n in path.read_bytes() for n in forms) for forms in needles)
            if found:
                hits.append({"file": name, "canaries": found})
    return {"files": files, "hits": hits}


def audit_facts(state: Path, records: list[dict]) -> dict:
    """
    Description: What the gate's G6 logs say about a session, next to what the mock received.
    Parameters:
        state: the throwaway ~/.harpo.
        records: the mock's records.
    Output: {"proof_entries", "proof_problem", "proof_matches_mock" (the same SHA-256 values, as many times each),
        "findings" ({"layer:event_type": count})}.
    """
    proof = audit.verify(state)
    lines = [json.loads(line) for line in audit._lines(state / audit.FINDINGS)]
    return {"proof_entries": len(proof.entries), "proof_problem": proof.problem,
            "proof_matches_mock": sorted(e["sha256"] for e in proof.entries) == sorted(r["body_sha256"] for r in records),
            "findings": dict(Counter(f"{f['layer']}:{f['event_type']}" for f in lines))}


def run_scenario(sc: Scenario, seed: int, out_root: Path, run_id: str, timeout: int) -> dict:
    """
    Description: Run one scenario and describe what the mock received.
    Parameters:
        sc: the scenario.
        seed: the canary seed.
        out_root: the results folder.
        run_id: the run's id.
        timeout: seconds allowed for the agent process.
    Output: A dict of counts and facts, no values. "_canaries" holds the Canary list for the report's leak check.
    """
    out: dict = {"scenario": sc.label, "agent": sc.agent, "mode": sc.mode, "version": runner._version(sc.agent)}
    if out["version"] is None:
        return {**out, "status": "skipped", "reason": f"`{sc.agent}` is not on PATH", "_canaries": []}
    with runner.agent_session(sc.agent, seed, out_root, run_id, label=sc.label, control=True,
                              codex_config=codex_config) as session:
        tw, url = session.tw, session.server.url
        routed = tw.home / FILES[sc.agent]
        before = _digest(routed)
        target = "--claude-code" if sc.agent == "claude" else "--codex"
        up = harpo(tw, ["guard", "up", "--port", "0", target], url)
        out.update({"guard_up_exit": up.returncode, "routing_written": _digest(routed) != before})
        if sc.mode == "routed":
            doctor = harpo(tw, ["doctor", "--json"], url)
            checks = json.loads(doctor.stdout) if doctor.stdout.strip().startswith("[") else []
            out.update({"doctor_exit": doctor.returncode,
                        "doctor_failed": sorted({c["name"] for c in checks if c["status"] == "fail"})})
            env = {k: v for k, v in session.env().items() if k not in ROUTING_VARS}  # only guard up's routing is left
        else:
            down = harpo(tw, ["guard", "down"], url)
            out.update({"guard_down_exit": down.returncode, "restored_exactly": _digest(routed) == before})
            # The control talks to the mock directly. Claude Code's environment names the mock, and a settings env
            # that guard down had left behind would win over it, sending the reads to the stopped gate instead.
            # Codex gets the harness's own mock provider, written after the byte-identical check above.
            env = session.env()
            if sc.agent == "codex":
                runner.write_codex_config(tw, url)
        code, timed_out, seconds = session.run(session.argv(), env, timeout)
        records = session.server.records()
        out["audit"] = audit_facts(tw.home / ".harpo", records)
        out["sec01"] = sweep([tw.home / ".harpo", tw.tmp], session.canaries)  # before guard down removes gate.out
        if sc.mode == "routed":
            out["guard_down_exit"] = harpo(tw, ["guard", "down"], url).returncode
            out["restored_exactly"] = _digest(routed) == before
        findings = report.find_received(records, session.canaries)
        out.update({"status": "timeout" if timed_out else "ok", "exit_code": code, "seconds": round(seconds, 1),
                    "mock_model_requests": sum(r.get("route") in runner.MODEL_ROUTES for r in records),
                    "tool_results_seen": runner.results_seen(records),
                    "planted": report.counts(session.canaries),
                    "received": report.received_counts(session.canaries, findings),
                    "_canaries": session.canaries})
    return out


def run_suite(seed: int, out_root: Path, run_id: str, timeout: int = runner.AGENT_TIMEOUT) -> tuple[dict, Path]:
    """
    Description: Run every scenario and write the counts-only report.
    Parameters:
        seed: the canary seed.
        out_root: the results folder; the report goes to proof/<run_id>.json.
        run_id: the run's id.
        timeout: seconds allowed per agent process.
    Output: (the report, its path). The real home is snapshotted before and after (isolation.real_home_changed).
    """
    before = iso.real_home_snapshot()
    results = [run_scenario(sc, seed, out_root, run_id, timeout) for sc in SCENARIOS]
    changed = iso.snapshot_diff(before, iso.real_home_snapshot())
    canaries = [c for r in results for c in r.pop("_canaries")]
    data = {"run_id": run_id, "suite": "g7", "seed": seed, "target": "harpo guard up -> gate -> mock",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"python": sys.version.split()[0], **{r["agent"]: r["version"] for r in results}},
            "scenarios": results,
            "isolation": {"real_home_paths_snapshotted": len(before), "real_home_changed": changed}}
    return data, report.write_report(Path(out_root) / "proof" / f"{run_id}.json", data, canaries)


def passed(data: dict) -> bool:
    """
    Description: Judge a report by its contents, not by exit codes alone.
    Parameters:
        data: the report.
    Output: True when nothing in the real home changed and every scenario that ran did what it must.
    """
    if data["isolation"]["real_home_changed"]:
        return False
    for s in data["scenarios"]:
        if s.get("status") == "skipped":
            continue
        ran = s["status"] == "ok" and s["guard_up_exit"] == 0 and s["routing_written"] and s["tool_results_seen"] >= 1
        restored = s.get("guard_down_exit") == 0 and s.get("restored_exactly")
        logs = s["audit"]["proof_problem"] is None and not s["sec01"]["hits"]
        if s["mode"] == "routed":
            ok = (ran and restored and logs and s["doctor_exit"] == 0 and s["received"]["known_format"] == 0
                  and s["audit"]["proof_entries"] == s["mock_model_requests"] > 0 and s["audit"]["proof_matches_mock"]
                  and s["audit"]["findings"].get("gate:secret_redacted", 0) > 0)
        else:
            ok = ran and restored and logs and s["received"]["known_format"] > 0 and s["audit"]["proof_entries"] == 0
        if not ok:
            return False
    return True


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Run the suite from the command line.
    Parameters:
        argv: the arguments, or None for sys.argv.
    Output: 0 when the report passes (passed()), 1 when it does not, 2 when no agent was installed.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout", type=int, default=runner.AGENT_TIMEOUT)
    args = parser.parse_args(argv)
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-guard-e2e-s{args.seed}"
    data, path = run_suite(args.seed, args.out_dir, run_id, args.timeout)
    for s in data["scenarios"]:
        print(f"{s['scenario']}: {s.get('status')} up={s.get('guard_up_exit')} doctor={s.get('doctor_exit')} "
              f"down={s.get('guard_down_exit')} restored={s.get('restored_exactly')} "
              f"received_kf={(s.get('received') or {}).get('known_format')} audit={s.get('audit')} "
              f"sec01_hits={len((s.get('sec01') or {}).get('hits', []))}")
    print(f"report: {path}")
    if all(s.get("status") == "skipped" for s in data["scenarios"]):
        return 2
    return 0 if passed(data) else 1


if __name__ == "__main__":
    sys.exit(main())
