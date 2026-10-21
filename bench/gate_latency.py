"""Timing reference for the gate (G3, FR-GATE-01; G5 connection reuse, NFR-01): latency added over direct-to-mock.

    python -m bench.gate_latency [--calls 200] [--warmup 20] [--body-bytes 102400] [--tls] [--no-pool] [--out DIR]

Everything runs on 127.0.0.1: bench/mock_provider.py is the upstream, and the gate runs with a pass-through scan step
defined here (the gate itself ships none), so these numbers leave the scan out; bench/gate_scan_latency.py measures the
scan. The calls alternate direct, then through the gate, so drift in the machine hits both. The client keeps one
connection to each. --tls puts the mock behind TLS with a throwaway certificate (made with the openssl command and
trusted only by this run), as a provider is; --no-pool makes the gate open a new upstream connection per request, as
it did before G5. The record holds numbers and the machine and date only, never a request. On loopback a TLS handshake
costs a few milliseconds; over a real network it costs one or two round trips more, so these numbers understate what
the pool saves.
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import platform
import ssl
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import Harpocrates.gate.upstream as upstream_module
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


def _throwaway_tls() -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """
    Description: Make a one-day certificate for 127.0.0.1 and the server and client TLS settings that use it.
    Parameters: None
    Output: (server context, client context that trusts only that certificate and checks the name). The key and
        certificate files are deleted before returning; the contexts keep what they loaded. Raises OSError when the
        openssl command is missing.
    """
    with tempfile.TemporaryDirectory() as folder:
        cert, key = Path(folder) / "cert.pem", Path(folder) / "key.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out",
                        str(cert), "-days", "1", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
                       check=True, capture_output=True)
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.load_cert_chain(str(cert), str(key))
        client = ssl.create_default_context(cafile=str(cert))
    return server, client


@contextlib.contextmanager
def _pool_size(size: int):
    """
    Description: Set how many idle upstream connections the gate keeps, for one measurement.
    Parameters:
        size: the limit; 0 turns reuse off.
    Output: A context manager that restores the gate's limit on exit.
    """
    saved = upstream_module.POOL_MAX_IDLE
    upstream_module.POOL_MAX_IDLE = size
    try:
        yield
    finally:
        upstream_module.POOL_MAX_IDLE = saved


def measure(calls: int = 200, warmup: int = 20, body_bytes: int = 100 * 1024, tls: bool = False,
            pool: bool = True) -> dict[str, object]:
    """
    Description: Time the same request direct to the mock and through the gate.
    Parameters:
        calls: how many timed calls per path.
        warmup: how many untimed calls first, per path.
        body_bytes: the request body size.
        tls: put the mock behind TLS (a throwaway certificate) for both paths.
        pool: let the gate reuse upstream connections (the default); False opens one per request.
    Output: The record: date, hardware, sizes, scan step, client connection kind, upstream TLS, pool, and direct_ms,
        via_gate_ms and added_ms (the gate's p50 and p95 minus direct's). Starts and stops a mock and a gate on 127.0.0.1.
    """
    body, token = make_body(body_bytes), new_token()
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    upstream = mock.start_server()
    client_ctx = None
    if tls:
        server_ctx, client_ctx = _throwaway_tls()
        # ponytail: wraps the listening socket in place while serving; fine for this serial bench (each TLS handshake runs
        # in accept), a TLS-aware mock if a bench ever needs parallel clients.
        upstream.server.socket = server_ctx.wrap_socket(upstream.server.socket, server_side=True)
    base = f"{'https' if tls else 'http'}://127.0.0.1:{upstream.port}"
    config = GateConfig(anthropic_upstream=base, openai_upstream=base)
    gate = start_gate(config, token, scan_step=lambda data, route, head: Forward(data))
    if tls:  # trust the throwaway certificate on the gate's side too
        gate.server.pool.close()
        gate.server.pool = upstream_module.UpstreamPool(client_ctx)
    direct_conn = (http.client.HTTPSConnection(upstream.host, upstream.port, timeout=30, context=client_ctx) if tls
                   else http.client.HTTPConnection(upstream.host, upstream.port, timeout=30))
    gate_conn = http.client.HTTPConnection(gate.host, gate.port, timeout=30)
    direct, via = [], []
    try:
        with _pool_size(upstream_module.POOL_MAX_IDLE if pool else 0):
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
            "client_connection": "keep-alive", "upstream_tls": tls, "pool": pool, "direct_ms": direct_s,
            "via_gate_ms": via_s,
            "added_ms": {"p50": round(via_s["p50"] - direct_s["p50"], 3), "p95": round(via_s["p95"] - direct_s["p95"], 3)}}


def write_record(record: dict[str, object], out_dir: Path) -> Path:
    """
    Description: Save a record as bench/results/gate/latency-DATE[-tls][-nopool].json.
    Parameters:
        record: the dict from measure.
        out_dir: the folder, created if missing.
    Output: The path written. An earlier record of the same date and settings is replaced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("-tls" if record.get("upstream_tls") else "") + ("" if record.get("pool", True) else "-nopool")
    path = out_dir / f"latency-{record['date']}{suffix}.json"
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
    parser.add_argument("--tls", action="store_true", help="put the mock behind TLS with a throwaway certificate")
    parser.add_argument("--no-pool", action="store_true", help="open a new upstream connection per request")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    record = measure(args.calls, args.warmup, args.body_bytes, tls=args.tls, pool=not args.no_pool)
    path = write_record(record, args.out)
    print(f"{path}  added p50 {record['added_ms']['p50']} ms, p95 {record['added_ms']['p95']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
