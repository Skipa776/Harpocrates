"""FR-GATE-01 (G3 spec, "Log"): one JSONL line per request with the time, route, status, decision, sizes and
timings, and never a header value or any body content. The audit log proper is FR-LOG-01 (G6)."""
from __future__ import annotations

import json
import re
import threading

import pytest

from Harpocrates.gate.request_log import RequestLog, format_entry

FULL = {"method": "POST", "route": "anthropic.messages", "status": 200, "decision": "forwarded",
        "bytes_in": 1200, "bytes_out": 1200, "bytes_back": 4096, "ms_scan": 1.234567, "ms_upstream": 80.5,
        "ms_total": 95.0}


def test_fr_gate_01_a_line_holds_the_listed_fields_as_json() -> None:
    line = format_entry({**FULL, "time": "2026-10-19T12:00:00.000Z"})
    assert line.endswith("\n") and line.count("\n") == 1
    assert json.loads(line) == {**FULL, "ms_scan": 1.235, "time": "2026-10-19T12:00:00.000Z"}


def test_fr_gate_01_fields_the_log_does_not_list_are_dropped() -> None:
    sneaky = {**FULL, "headers": {"x-api-key": "FAKE-KEY-1"}, "authorization": "Bearer FAKE-2", "body": "FAKE-3",
              "path": "/v1/messages?key=FAKE-4", "x-api-key": "FAKE-5", "client": ("127.0.0.1", 5)}
    line = format_entry(sneaky)
    assert not [leak for leak in ("FAKE-", "headers", "authorization", "/v1/") if leak in line]
    assert set(json.loads(line)) == set(FULL)


@pytest.mark.parametrize("method,expected", [("POST", "POST"), ("GET", "GET"), ("HEAD", "HEAD"), ("OPTIONS", "OPTIONS"),
                                             ("sk-FAKE-SECRET", "OTHER"), ("post", "OTHER"), ("", "OTHER")])
def test_fr_gate_01_an_odd_method_is_logged_as_other(method: str, expected: str) -> None:
    assert json.loads(format_entry({"method": method}))["method"] == expected


@pytest.mark.parametrize("route,expected", [
    ("anthropic.messages", "anthropic.messages"), ("openai.responses", "openai.responses"),
    ("websocket-upgrade", "websocket-upgrade"), ("unknown", "unknown"), ("/v1/secret-path", "unknown"),
    ("anthropic.messages?key=FAKE", "unknown"),
])
def test_fr_gate_01_a_route_is_a_fixed_name_never_a_path(route: str, expected: str) -> None:
    assert json.loads(format_entry({"route": route}))["route"] == expected


def test_fr_gate_01_decision_and_reason_are_fixed_vocabulary() -> None:
    ok = json.loads(format_entry({"decision": "blocked", "reason": "HARPO-BLOCK-TOKEN"}))
    assert ok == {"decision": "blocked", "reason": "HARPO-BLOCK-TOKEN"}
    odd = json.loads(format_entry({"decision": "FAKE-SECRET", "reason": "text with FAKE-SECRET"}))
    assert odd == {}


@pytest.mark.parametrize("field,value", [("status", 99), ("status", 600), ("status", "200"), ("status", True),
                                         ("bytes_in", -1), ("bytes_in", "5"), ("bytes_in", None), ("ms_total", -0.5),
                                         ("ms_total", float("nan")), ("ms_total", float("inf")), ("ms_scan", True)])
def test_fr_gate_01_a_number_out_of_range_or_of_the_wrong_kind_is_dropped(field: str, value: object) -> None:
    assert field not in json.loads(format_entry({field: value}))


def test_fr_gate_01_request_log_stamps_the_time_and_writes_to_the_sink() -> None:
    lines: list[str] = []
    RequestLog(lines.append).write(FULL)
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", entry["time"])
    assert entry["route"] == "anthropic.messages"


def test_fr_gate_01_request_log_without_a_sink_writes_nothing_and_does_not_fail() -> None:
    RequestLog(None).write(FULL)


def test_fr_gate_01_concurrent_writes_stay_one_whole_line_each() -> None:
    lines: list[str] = []
    log = RequestLog(lines.append)
    threads = [threading.Thread(target=lambda n=n: [log.write({**FULL, "bytes_in": n}) for _ in range(40)])
               for n in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(lines) == 800
    assert all(json.loads(line)["route"] == "anthropic.messages" for line in lines)


def test_fr_gate_01_every_route_name_the_gate_can_produce_is_in_the_log_vocabulary() -> None:
    from Harpocrates.gate.request_log import ROUTE_NAMES
    from Harpocrates.gate.routes import classify

    cases = [("POST", "/v1/messages", {}), ("POST", "/v1/messages/count_tokens", {}), ("POST", "/v1/responses", {}),
             ("GET", "/v1/models", {}), ("GET", "/v1/models", {"anthropic-version": "x"}), ("GET", "/x", {"upgrade": "websocket"}),
             ("POST", "/nope", {})]
    assert {classify(*case).name for case in cases} <= ROUTE_NAMES
