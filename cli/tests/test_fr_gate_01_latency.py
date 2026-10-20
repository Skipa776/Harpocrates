"""FR-GATE-01 timing reference (G3 spec, "Test plan"): added latency of the gate against direct-to-mock for a 100 KB
body, p50 and p95 over 200 calls, kept in bench/results/gate/. A reference for NFR-01 (M3), not a gate on G3."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import gate_latency

RESULTS = Path(__file__).resolve().parents[2] / "bench" / "results" / "gate"


def test_fr_gate_01_percentile_interpolates_between_samples() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert gate_latency.percentile(data, 0) == 1.0 and gate_latency.percentile(data, 100) == 5.0
    assert gate_latency.percentile(data, 50) == 3.0
    assert gate_latency.percentile(data, 95) == pytest.approx(4.8)
    assert gate_latency.percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        gate_latency.percentile([], 50)


def test_fr_gate_01_summary_holds_p50_p95_mean_min_max_in_milliseconds() -> None:
    summary = gate_latency.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert summary == {"p50": 3.0, "p95": 4.8, "mean": 3.0, "min": 1.0, "max": 5.0}


def test_fr_gate_01_a_measurement_runs_the_gate_and_the_mock_and_records_numbers(tmp_path: Path) -> None:
    record = gate_latency.measure(calls=6, warmup=2, body_bytes=4096)
    assert record["calls"] == 6 and record["warmup"] == 2 and record["body_bytes"] == 4096
    assert record["scan_step"] == "pass-through" and record["client_connection"] == "keep-alive"
    for key in ("direct_ms", "via_gate_ms"):
        assert set(record[key]) == {"p50", "p95", "mean", "min", "max"} and record[key]["min"] > 0
    assert record["added_ms"] == {
        "p50": round(record["via_gate_ms"]["p50"] - record["direct_ms"]["p50"], 3),
        "p95": round(record["via_gate_ms"]["p95"] - record["direct_ms"]["p95"], 3)}
    assert record["hardware"] and all(isinstance(v, (str, int)) for v in record["hardware"].values())
    assert len(record["date"]) == 10
    path = gate_latency.write_record(record, tmp_path)
    assert path.name == f"latency-{record['date']}.json" and json.loads(path.read_text()) == record


def test_fr_gate_01_the_recorded_reference_is_a_100_kb_body_over_200_calls() -> None:
    files = sorted(RESULTS.glob("latency-*.json"))
    assert files, "run: python -m bench.gate_latency"
    record = json.loads(files[-1].read_text())
    assert record["body_bytes"] == 100 * 1024 and record["calls"] == 200
    assert {"p50", "p95"} <= set(record["added_ms"]) and record["hardware"]["machine"] and record["date"]
