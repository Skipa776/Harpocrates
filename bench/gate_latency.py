"""Timing reference for the gate (G3, FR-GATE-01): latency added over direct-to-mock for a 100 KB body.

    python -m bench.gate_latency [--calls 200] [--warmup 20] [--body-bytes 102400] [--out bench/results/gate]

Everything runs on 127.0.0.1: bench/mock_provider.py is the upstream, and the gate runs with a pass-through scan step
defined here (the gate itself ships none; G4 replaces its stub with the real scanner, so these numbers leave the scan
out). The calls alternate direct, then through the gate, so drift in the machine hits both. The client keeps one
connection to each; the gate opens a new one to the mock per request, as it will to a provider. The record holds
numbers and the machine and date only, never a request. A reference for NFR-01 (M3), not a gate on G3.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bench import mock_provider as mock
from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.scan import Forward
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "gate"


def percentile(sorted_values: list[float], q: float) -> float:
    """
    Description: Take a percentile of ordered samples by linear interpolation between neighbours.
    Parameters:
        sorted_values: the samples, smallest first.
        q: the percentile, 0 to 100.
    Output: The interpolated value. Raises ValueError for no samples.
    """
    if not sorted_values:
        raise ValueError("no samples")
    position = (len(sorted_values) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """
    Description: Reduce timing samples to the figures the record keeps.
    Parameters:
        samples_ms: durations in milliseconds, any order.
    Output: {"p50", "p95", "mean", "min", "max"} in milliseconds, each rounded to 3 places.
    """
    ordered = sorted(samples_ms)
    return {"p50": round(percentile(ordered, 50), 3), "p95": round(percentile(ordered, 95), 3),
            "mean": round(sum(ordered) / len(ordered), 3), "min": round(ordered[0], 3), "max": round(ordered[-1], 3)}


def describe_hardware() -> dict[str, object]:
    """
    Description: Name the machine the numbers came from.
    Parameters: None
    Output: {"machine", "cpu", "cores", "os", "python"}: the architecture, the CPU model (from sysctl on macOS,
        otherwise the platform string), the logical core count, the OS release and the Python version.
    """
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip() or cpu
    except (OSError, subprocess.SubprocessError):
        pass
    return {"machine": platform.machine(), "cpu": cpu, "cores": os.cpu_count() or 0,
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()}


def make_body(size: int) -> bytes:
    """
    Description: Build a Messages request body of an exact size.
    Parameters:
        size: the body length in bytes.
    Output: JSON bytes of exactly that length: a user message padded with one repeated letter.
    """
    def build(pad: int) -> bytes:
        message = {"role": "user", "content": "x" * pad}
        return json.dumps({"model": "bench", "max_tokens": 1, "messages": [message]}).encode()

    return build(size - len(build(0)))


def _timed_post(conn: http.client.HTTPConnection, body: bytes, headers: dict) -> float:
    """
    Description: Send one POST and read the whole reply.
    Parameters:
        conn: a connection to the mock or the gate.
        body: the request body.
        headers: the request headers.
    Output: The elapsed milliseconds. Raises AssertionError if the status is not 200.
    """
    started = time.perf_counter()
    conn.request("POST", "/v1/messages", body=body, headers=headers)
    response = conn.getresponse()
    response.read()
    elapsed = (time.perf_counter() - started) * 1000
    assert response.status == 200, response.status
    return elapsed


def measure(calls: int = 200, warmup: int = 20, body_bytes: int = 100 * 1024) -> dict[str, object]:
    """
    Description: Time the same request direct to the mock and through the gate.
    Parameters:
        calls: how many timed calls per path.
        warmup: how many untimed calls first, per path.
        body_bytes: the request body size.
    Output: The record: date, hardware, sizes, scan step, client connection kind, and direct_ms, via_gate_ms and
        added_ms (the gate's p50 and p95 minus direct's). Starts and stops a mock and a gate on 127.0.0.1.
    """
    body, token = make_body(body_bytes), new_token()
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    upstream = mock.start_server()
    config = GateConfig(anthropic_upstream=upstream.url, openai_upstream=upstream.url)
    gate = start_gate(config, token, scan_step=lambda data, route, head: Forward(data))
    direct_conn = http.client.HTTPConnection(upstream.host, upstream.port, timeout=30)
    gate_conn = http.client.HTTPConnection(gate.host, gate.port, timeout=30)
    direct, via = [], []
    try:
        for index in range(warmup + calls):
            d = _timed_post(direct_conn, body, headers)
            g = _timed_post(gate_conn, body, {**headers, TOKEN_HEADER: token})
            if index >= warmup:
                direct.append(d)
                via.append(g)
    finally:
        direct_conn.close()
        gate_conn.close()
        gate.close()
        upstream.close()
    direct_s, via_s = summarize(direct), summarize(via)
    return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "hardware": describe_hardware(),
            "body_bytes": body_bytes, "calls": calls, "warmup": warmup, "scan_step": "pass-through",
            "client_connection": "keep-alive", "direct_ms": direct_s, "via_gate_ms": via_s,
            "added_ms": {"p50": round(via_s["p50"] - direct_s["p50"], 3), "p95": round(via_s["p95"] - direct_s["p95"], 3)}}


def write_record(record: dict[str, object], out_dir: Path) -> Path:
    """
    Description: Save a record as bench/results/gate/latency-DATE.json.
    Parameters:
        record: the dict from measure.
        out_dir: the folder, created if missing.
    Output: The path written. An earlier record of the same date is replaced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"latency-{record['date']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Measure and save the record.
    Parameters:
        argv: command-line arguments, or None to use sys.argv.
    Output: Exit status 0; prints the path written and the added p50 and p95.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--body-bytes", type=int, default=100 * 1024)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    record = measure(args.calls, args.warmup, args.body_bytes)
    path = write_record(record, args.out)
    print(f"{path}  added p50 {record['added_ms']['p50']} ms, p95 {record['added_ms']['p95']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
