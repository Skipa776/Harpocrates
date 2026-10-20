"""Latency the real scanner adds (NFR-01; G4 spec, "Performance"): replay recorded sessions through the gate.

    python -m bench.gate_scan_latency [--recordings bench/results/mock] [--repeats 5] [--out bench/results/gate]

Every recorded model request (bench/results/mock/<run>/<session>/requests.jsonl, local and never committed) is read into
memory and sent twice to the mock provider on 127.0.0.1: directly, and through a gate with the real scanner. A session is
replayed in order against a fresh gate (a new session key and an empty cache), so its first request is a "first turn" and
the rest are "follow-up turns" with the cache warm. The recordings of the G2 probes hold one request for each of the two
large sessions (764 KB and 1.04 MB), so a follow-up of that size is made by appending one assistant turn and one user turn to
the recorded first request ("large follow-up", synthesized). Added latency is the gate's time minus the direct time for the
same request; the scan step's own time comes from the gate's request log. The record holds numbers, session labels and the
machine; never a body, header or value. Target: follow-up p95 added latency under 50 ms.
"""
from __future__ import annotations

import argparse
import cProfile
import http.client
import json
import pstats
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

from bench import mock_provider as mock
from bench.gate_latency import describe_hardware, summarize
from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.finder import strict_findings
from Harpocrates.gate.request_scan import RequestScanner
from Harpocrates.gate.routes import classify
from Harpocrates.gate.scan import RequestHead
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token

DEFAULT_RECORDINGS = Path(__file__).resolve().parent / "results" / "mock"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "gate"
MODEL_ROUTES = ("anthropic.messages", "openai.responses")
TARGET_MS = 50
LARGE_BYTES = 500_000
_DROPPED = frozenset({"host", "content-length", "accept-encoding", "connection", "content-type", "transfer-encoding", "expect"})
_ENTROPY_SAMPLE = "api_token = 'Zx9qLmT2vBn4Zw8Yc1RkPa7Ue'"  # makes the detector hand a candidate to the model


class Req(NamedTuple):
    """
    Description: One recorded model request, in memory.
    Parameters:
        route: anthropic.messages or openai.responses.
        path: the request target, with its query.
        headers: the recorded (name, value) pairs without the ones the client library sets again and without credentials.
        body: the decoded request body.
    Output: An immutable record; never printed or written out.
    """

    route: str
    path: str
    headers: tuple
    body: bytes


class Session(NamedTuple):
    """
    Description: The model requests of one recorded session, in order.
    Parameters:
        label: "<run>/<session>", a folder name.
        requests: the requests.
    Output: An immutable record.
    """

    label: str
    requests: list


def load_sessions(root: Path) -> list[Session]:
    """
    Description: Read the recorded model requests of every session under a recordings folder.
    Parameters:
        root: the folder holding <run>/<session>/requests.jsonl.
    Output: A Session for each file with at least one POST model request whose body was recorded in full. Headers whose value
        was stored as <present> (credentials) and the headers a client library sets again are left out. Runs named *-direct-*
        (the synthetic corpus, one request per case, many of them meant to be blocked) are not sessions and are skipped.
    """
    sessions = []
    for path in sorted(Path(root).glob("*/*/requests.jsonl")):
        if "-direct-" in path.parent.parent.name:
            continue
        requests = []
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("method") != "POST" or rec.get("route") not in MODEL_ROUTES or not isinstance(rec.get("body"), str):
                continue
            headers = tuple((n, v) for n, v in rec.get("headers", []) if n.lower() not in _DROPPED and v != mock.PRESENT)
            requests.append(Req(rec["route"], rec.get("path") or ("/v1/messages" if rec["route"].startswith("anthropic")
                                                                  else "/v1/responses"), headers, rec["body"].encode("utf-8")))
        if requests:
            sessions.append(Session(f"{path.parent.parent.name}/{path.parent.name}", requests))
    return sessions


def grow_follow_up(route: str, body: bytes) -> bytes:
    """
    Description: Make the request that would follow a recorded one: the same history plus one assistant and one user turn.
    Parameters:
        route: anthropic.messages or openai.responses.
        body: the recorded request body.
    Output: The new body (JSON). Raises ValueError when the body has no message list or input list to extend.
    """
    data = json.loads(body)
    if route.startswith("anthropic"):
        if not isinstance(data.get("messages"), list):
            raise ValueError("no messages to extend")
        data["messages"] += [{"role": "assistant", "content": [{"type": "text", "text": "Done. Anything else?"}]},
                             {"role": "user", "content": [{"type": "text", "text": "Please continue with the next step."}]}]
    else:
        if not isinstance(data.get("input"), list):
            raise ValueError("no input to extend")
        data["input"] += [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}]},
                          {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Please continue."}]}]
    return json.dumps(data).encode("utf-8")


def _post(conn: http.client.HTTPConnection, req: Req, body: bytes, extra: dict) -> tuple[float, int]:
    """
    Description: Send one request and read the whole reply.
    Parameters:
        conn: a connection to the mock or the gate.
        req: the recorded request (path, headers).
        body: the body to send.
        extra: further headers (the gate token).
    Output: (the elapsed milliseconds, the HTTP status).
    """
    started = time.perf_counter()
    conn.request("POST", req.path, body=body, headers={"Content-Type": "application/json", **dict(req.headers), **extra})
    response = conn.getresponse()
    response.read()
    return (time.perf_counter() - started) * 1000, response.status


def _scan_ms(lines: list, count: int) -> float:
    """
    Description: Read the scan time of the latest request from the gate's request log.
    Parameters:
        lines: the log lines so far.
        count: how many lines there should be (the gate logs just after it answers).
    Output: ms_scan of the last line, waiting up to a second for it to appear.
    """
    deadline = time.monotonic() + 1.0
    while len(lines) < count and time.monotonic() < deadline:
        time.sleep(0.002)
    return json.loads(lines[count - 1])["ms_scan"]


def _group(samples: list[tuple[float, float, int]]) -> dict:
    """
    Description: Reduce (added ms, scan ms, body bytes) samples to the figures the record keeps.
    Parameters:
        samples: the samples of one class.
    Output: {"n", "body_bytes": {"min", "p50", "max"}, "added_ms": summary, "scan_ms": summary} or {"n": 0}.
    """
    if not samples:
        return {"n": 0}
    sizes = sorted(s[2] for s in samples)
    return {"n": len(samples), "body_bytes": {"min": sizes[0], "p50": sizes[len(sizes) // 2], "max": sizes[-1]},
            "added_ms": summarize([s[0] for s in samples]), "scan_ms": summarize([s[1] for s in samples])}


def _top_cost(sessions: list[Session], max_follow_ups: int) -> list[dict]:
    """
    Description: Profile the scan step on follow-up turns to name what costs the most.
    Parameters:
        sessions: the recorded sessions.
        max_follow_ups: how many follow-ups of a session to profile.
    Output: The eight functions with the most own time, as {"function", "share_pct"} (a function name and file, no content),
        over the scan of every follow-up turn with the cache warm.
    """
    profiler, head = cProfile.Profile(), RequestHead((), "/v1/messages", "")
    for session in sessions:
        scanner = RequestScanner()
        for position, req in enumerate(session.requests[:1 + max_follow_ups]):
            route = classify("POST", req.path.partition("?")[0], {"anthropic-version": "x"} if req.route.startswith("anthropic") else {})
            if position == 0:
                scanner(req.body, route, head)
            else:
                profiler.runcall(scanner, req.body, route, head)
    stats = pstats.Stats(profiler).stats
    total = sum(entry[2] for entry in stats.values()) or 1.0
    top = sorted(stats.items(), key=lambda kv: kv[1][2], reverse=True)[:8]
    return [{"function": f"{Path(key[0]).name}:{key[2]}", "share_pct": round(100 * entry[2] / total, 1)} for key, entry in top]


def measure(recordings: Path, repeats: int = 5, max_follow_ups: int = 30, large_bytes: int = LARGE_BYTES) -> dict:
    """
    Description: Replay the recorded sessions and measure what the scanner adds.
    Parameters:
        recordings: the folder holding <run>/<session>/requests.jsonl.
        repeats: how many times each session is replayed against a fresh gate.
        max_follow_ups: how many recorded follow-ups of a session are replayed.
        large_bytes: the size from which a follow-up counts as large.
    Output: The record: date, hardware, per-class figures (first_turn, first_turn_large, follow_up, follow_up_large: n,
        body sizes, added_ms, scan_ms), the target and whether it was met, the scan step's first-call time, how many replayed requests the gate
        blocked or the mock refused (they give no sample), the sessions (labels and counts), and top_cost. Starts and stops a mock and gates on 127.0.0.1.
    """
    sessions = load_sessions(recordings)
    started = time.perf_counter()
    strict_findings(_ENTROPY_SAMPLE)  # in a fresh process this includes loading the model
    first_scan_ms = (time.perf_counter() - started) * 1000
    classes: dict[str, list] = {"first": [], "first_large": [], "follow": [], "large": []}
    blocked = refused_by_mock = 0
    upstream = mock.start_server()
    direct = http.client.HTTPConnection(upstream.host, upstream.port, timeout=60)
    try:
        for repeat in range(repeats):
            for session in sessions:
                token, lines = new_token(), []
                gate = start_gate(GateConfig(anthropic_upstream=upstream.url, openai_upstream=upstream.url), token,
                                  log_sink=lines.append)
                via = http.client.HTTPConnection(gate.host, gate.port, timeout=60)
                plan = [(r, r.body, "first" if i == 0 else "follow") for i, r in enumerate(session.requests[:1 + max_follow_ups])]
                if len(session.requests[0].body) >= large_bytes:
                    plan.append((session.requests[0], grow_follow_up(session.requests[0].route, session.requests[0].body), "follow"))
                try:
                    for position, (req, body, kind) in enumerate(plan):
                        if (repeat + position) % 2:
                            (d, direct_status), (g, status) = _post(direct, req, body, {}), _post(via, req, body, {TOKEN_HEADER: token})
                        else:
                            (g, status), (d, direct_status) = _post(via, req, body, {TOKEN_HEADER: token}), _post(direct, req, body, {})
                        if direct_status != 200:  # the mock itself refuses this recorded request: nothing to compare
                            refused_by_mock += 1
                            continue
                        if status != 200:  # the gate blocked it: no upstream time to compare, so no sample; counted
                            blocked += 1
                            continue
                        sample = (g - d, _scan_ms(lines, position + 1), len(body))
                        classes[kind].append(sample)
                        if len(body) >= large_bytes:
                            classes["large" if kind == "follow" else "first_large"].append(sample)
                finally:
                    via.close()
                    gate.close()
    finally:
        direct.close()
        upstream.close()
    follow = _group(classes["follow"])
    return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "hardware": describe_hardware(), "scan_step": "real scanner",
            "repeats": repeats, "client_connection": "keep-alive", "requests_blocked_by_the_gate": blocked,
            "requests_refused_by_the_mock": refused_by_mock, "first_scan_in_process_ms": round(first_scan_ms, 1),
            "first_turn": _group(classes["first"]), "first_turn_large": _group(classes["first_large"]), "follow_up": follow,
            "follow_up_large": _group(classes["large"]),
            "target": {"follow_up_p95_ms": TARGET_MS, "met": follow.get("n", 0) > 0 and follow["added_ms"]["p95"] < TARGET_MS},
            "sessions": [{"label": s.label, "requests": len(s.requests), "max_body_bytes": max(len(r.body) for r in s.requests)}
                         for s in sessions],
            "top_cost": _top_cost(sessions, max_follow_ups)}


def write_record(record: dict, out_dir: Path) -> Path:
    """
    Description: Save a record as scan-latency-DATE.json.
    Parameters:
        record: the dict from measure.
        out_dir: the folder, created if missing.
    Output: The path written. An earlier record of the same date is replaced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scan-latency-{record['date']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Measure and save the record.
    Parameters:
        argv: command-line arguments, or None to use sys.argv.
    Output: Exit status 0; prints the path written, then first-turn and follow-up added p50 and p95 and whether the target held.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-follow-ups", type=int, default=30)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    record = measure(args.recordings, args.repeats, args.max_follow_ups)
    path = write_record(record, args.out)
    print(path)
    for name in ("first_turn", "first_turn_large", "follow_up", "follow_up_large"):
        part = record[name]
        if part["n"]:
            print(f"{name}: n={part['n']} body p50 {part['body_bytes']['p50']} B  added p50 {part['added_ms']['p50']} ms "
                  f"p95 {part['added_ms']['p95']} ms  (scan p50 {part['scan_ms']['p50']} p95 {part['scan_ms']['p95']})")
    print(f"target follow-up p95 < {TARGET_MS} ms: {'met' if record['target']['met'] else 'MISSED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
