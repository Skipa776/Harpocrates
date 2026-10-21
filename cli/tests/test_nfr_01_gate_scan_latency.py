"""NFR-01 / G4 spec "Performance": the latency the real scanner adds, measured by replaying recorded G2 sessions through the
gate (bench/gate_scan_latency.py). Follow-up turns (cache warm) and first turns are reported apart, in bench/results/gate/.

The unit tests here use small made-up recordings; the recorded reference is checked at the end."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import gate_scan_latency as bench

RESULTS = Path(__file__).resolve().parents[2] / "bench" / "results" / "gate"
MARKER = "MARKER-FOR-NO-CONTENT-IN-RECORD"


def record(index: int, route: str, body: dict, path: str = "") -> dict:
    return {"index": index, "method": "POST", "route": route, "path": path or ("/v1/messages?beta=true" if route.startswith("anthropic") else "/v1/responses"),
            "headers": [["User-Agent", "agent/1"], ["x-api-key", "<present>"], ["anthropic-version", "2023-06-01"], ["Host", "h"],
                        ["Content-Length", "1"], ["Accept-Encoding", "gzip"]],
            "body": json.dumps(body), "body_status": "ok"}


def write_session(root: Path, run: str, label: str, records: list) -> None:
    folder = root / run / label
    folder.mkdir(parents=True)
    (folder / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def anthropic_body(turns: int, pad: int = 0) -> dict:
    messages = [{"role": "user", "content": f"{MARKER} " + "x" * pad}]
    for n in range(turns):
        messages += [{"role": "assistant", "content": [{"type": "text", "text": f"reply {n}"}]}, {"role": "user", "content": f"q {n}"}]
    return {"model": "m", "max_tokens": 1, "messages": messages}


def test_nfr_01_grow_follow_up_appends_an_assistant_and_a_user_turn_in_either_format() -> None:
    grown = json.loads(bench.grow_follow_up("anthropic.messages", json.dumps(anthropic_body(1))))
    assert len(grown["messages"]) == 5 and [m["role"] for m in grown["messages"][-2:]] == ["assistant", "user"]
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    out = json.loads(bench.grow_follow_up("openai.responses", json.dumps({"model": "m", "input": items, "store": False})))
    assert len(out["input"]) == 3 and out["input"][-1]["role"] == "user" and out["store"] is False
    with pytest.raises(ValueError):
        bench.grow_follow_up("anthropic.messages", json.dumps({"messages": "nope"}))


def test_nfr_01_load_sessions_keeps_model_requests_only_and_drops_credential_valued_headers(tmp_path: Path) -> None:
    write_session(tmp_path, "run1", "claude-a", [record(0, "anthropic.messages", anthropic_body(0)),
                                                {**record(1, "unknown", {}), "route": "unknown"},
                                                record(2, "anthropic.messages", anthropic_body(1))])
    write_session(tmp_path, "run1", "empty", [{"index": 0, "route": "unknown", "method": "GET", "body": None}])
    sessions = bench.load_sessions(tmp_path)
    assert [s.label for s in sessions] == ["run1/claude-a"] and len(sessions[0].requests) == 2
    request = sessions[0].requests[0]
    names = [n.lower() for n, _ in request.headers]
    assert "x-api-key" not in names and "host" not in names and "content-length" not in names and "user-agent" in names
    assert request.route == "anthropic.messages" and request.path == "/v1/messages?beta=true" and isinstance(request.body, bytes)


def test_nfr_01_replay_reports_first_turns_and_follow_ups_apart_without_any_content(tmp_path: Path) -> None:
    write_session(tmp_path, "run1", "claude-a", [record(i, "anthropic.messages", anthropic_body(i)) for i in range(4)])
    write_session(tmp_path, "run1", "big", [record(0, "anthropic.messages", anthropic_body(0, pad=600_000))])
    data = bench.measure(tmp_path, repeats=2, max_follow_ups=5, large_bytes=500_000)
    assert data["first_turn"]["n"] == 2 * 2 and data["follow_up"]["n"] == 2 * 3 + 2 * 1  # three recorded + one synthesized
    for part in ("first_turn", "follow_up"):
        assert set(data[part]["added_ms"]) == {"p50", "p95", "mean", "min", "max"}
        assert set(data[part]["scan_ms"]) == {"p50", "p95", "mean", "min", "max"} and data[part]["body_bytes"]["max"] > 0
    assert data["follow_up_large"]["n"] == 2 and data["follow_up_large"]["body_bytes"]["min"] >= 500_000
    assert data["first_turn_large"]["n"] == 2 and data["first_turn_large"]["body_bytes"]["min"] >= 500_000  # reported apart too
    assert data["target"] == {"follow_up_p95_ms": 50, "met": data["follow_up"]["added_ms"]["p95"] < 50}
    assert data["hardware"]["machine"] and len(data["date"]) == 10 and data["scan_step"] == "real scanner"
    assert MARKER not in json.dumps(data) and data["sessions"][0]["requests"] >= 1


def test_nfr_01_write_record_names_the_file_by_date(tmp_path: Path) -> None:
    path = bench.write_record({"date": "2026-10-20", "x": 1}, tmp_path)
    assert path.name == "scan-latency-2026-10-20.json" and json.loads(path.read_text()) == {"date": "2026-10-20", "x": 1}


def test_nfr_01_the_recorded_scan_latency_reference_separates_first_turns_from_follow_ups() -> None:
    files = sorted(RESULTS.glob("scan-latency-*.json"))
    assert files, "run: python -m bench.gate_scan_latency"
    data = json.loads(files[-1].read_text())
    assert data["target"]["follow_up_p95_ms"] == 50 and isinstance(data["target"]["met"], bool)
    assert data["follow_up"]["n"] >= 40 and data["first_turn"]["n"] >= 10 and data["first_turn_large"]["n"] >= 2
    assert {"p50", "p95"} <= set(data["follow_up"]["added_ms"]) and data["hardware"]["cpu"] and data["date"]
    assert "top_cost" in data


def test_nfr_01_load_sessions_can_be_limited_to_the_runs_of_one_day(tmp_path: Path) -> None:
    write_session(tmp_path, "20261019T1-g2", "a", [record(0, "anthropic.messages", anthropic_body(0))])
    write_session(tmp_path, "20261020-later", "b", [record(0, "anthropic.messages", anthropic_body(0))])
    write_session(tmp_path, "20261019T2-direct-s0", "c", [record(0, "anthropic.messages", anthropic_body(0))])
    assert [s.label for s in bench.load_sessions(tmp_path)] == ["20261019T1-g2/a", "20261020-later/b"]  # *-direct-* is never a session
    assert [s.label for s in bench.load_sessions(tmp_path, "20261019")] == ["20261019T1-g2/a"]
