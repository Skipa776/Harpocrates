"""FR-GATE-01 (G3 spec, "Log") and the fail-closed rule: one JSONL line per request with the route, status, decision,
sizes and timings, never a path, header value or body; and an unexpected error is a block, not a pass."""
from __future__ import annotations

import http.client
import json
import time

import pytest
from gate_rig import ANTHROPIC, JSON, MESSAGES, RESPONSES, passthrough_scan, running_rig
from gate_support import ScriptedUpstream, free_closed_port

import Harpocrates.gate.server as gate_server
from Harpocrates.gate.session_token import TOKEN_HEADER


def entries(rig, count: int = 0) -> list[dict]:
    """The log lines so far, as dicts. The gate logs after it answers, so wait (up to 3 s) for `count` lines."""
    deadline = time.monotonic() + 3
    while len(rig.log_lines) < count and time.monotonic() < deadline:
        time.sleep(0.01)
    return [json.loads(line) for line in rig.log_lines]


def test_fr_gate_01_one_log_line_per_request_with_route_status_decision_sizes_and_timings() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        rig.call("POST", "/v1/messages?beta=true", MESSAGES, {**JSON, **ANTHROPIC})
        rig.call("GET", "/v1/models")
        rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC}, token=False)
        rig.call("POST", "/v1/chat/completions", RESPONSES, JSON)
        rig.call("GET", "/v1/responses", None, {"Upgrade": "websocket"})
        rig.call("POST", "/v1/responses", b"{not json but forwarded", JSON)
        got = entries(rig, 6)
    assert [(e["route"], e["status"], e["decision"], e.get("reason")) for e in got] == [
        ("anthropic.messages", 200, "forwarded", None), ("openai.models", 200, "forwarded", None),
        ("anthropic.messages", 401, "blocked", "HARPO-BLOCK-TOKEN"),
        ("unknown", 400, "blocked", "HARPO-BLOCK-ROUTE"), ("websocket-upgrade", 426, "upgrade-refused", "HARPO-BLOCK-UPGRADE"),
        ("openai.responses", 400, "forwarded", None),  # the mock refused the body; the gate forwards the provider's answer
    ]
    first = got[0]
    assert first["method"] == "POST" and first["bytes_in"] == len(MESSAGES) == first["bytes_out"]
    assert first["bytes_back"] > 0
    assert 0 <= first["ms_scan"] <= first["ms_total"] and 0 <= first["ms_upstream"] <= first["ms_total"]
    assert got[1]["method"] == "GET" and got[1]["bytes_in"] == 0
    assert "bytes_in" not in got[2] and "ms_upstream" not in got[2]  # nothing was read or sent after the token failed
    assert all("time" in e and "ms_total" in e for e in got)


def test_fr_gate_01_a_blocked_by_the_scan_step_request_logs_its_sizes_and_the_reason() -> None:
    with running_rig() as rig:  # the default stub
        rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        (entry,) = entries(rig, 1)
    assert (entry["decision"], entry["reason"], entry["status"]) == ("blocked", "HARPO-BLOCK-SCAN-ERROR", 400)
    assert entry["bytes_in"] == len(MESSAGES) and "bytes_out" not in entry and entry["ms_scan"] >= 0


def test_fr_gate_01_an_unreachable_provider_logs_an_upstream_error() -> None:
    url = f"http://127.0.0.1:{free_closed_port()}"
    with running_rig(scan_step=passthrough_scan, anthropic_upstream=url, openai_upstream=url) as rig:
        rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        (entry,) = entries(rig, 1)
    assert (entry["decision"], entry["reason"], entry["status"]) == ("upstream-error", "HARPO-UPSTREAM-FAILED", 502)


def test_fr_gate_01_the_log_holds_no_path_query_header_or_body() -> None:
    secrets = ["FAKE-PATH-77", "FAKE-QUERY-88", "FAKE-HEADER-99", "FAKE-BODY-11"]
    body = json.dumps({"x": secrets[3]}).encode()
    with running_rig(scan_step=passthrough_scan) as rig:
        rig.call("POST", f"/{secrets[0]}?k={secrets[1]}", body, {**JSON, "x-api-key": secrets[2]})
        rig.call("POST", f"/v1/messages?k={secrets[1]}", body, {**JSON, "x-api-key": secrets[2]}, token="wrong")
        rig.call("POST", f"/v1/messages?k={secrets[1]}", body, {**JSON, "x-api-key": secrets[2], **ANTHROPIC})
        entries(rig, 3)
        text = "".join(rig.log_lines)
    assert len(rig.log_lines) == 3 and not [s for s in secrets if s in text]


def test_fr_gate_01_a_request_line_with_an_odd_method_is_logged_as_other() -> None:
    with running_rig() as rig:
        rig.raw(f"sk-FAKE-METHOD /v1/messages HTTP/1.1\r\nHost: g\r\n{TOKEN_HEADER}: {rig.token}\r\n\r\n".encode())
        rig.call("POST", "/v1/nothing", b"{}", JSON)
        entries(rig, 1)
        text = "".join(rig.log_lines)
    assert "sk-FAKE" not in text


@pytest.mark.parametrize("patched", ["encode_body", "decode_body", "classify"])
def test_fr_gate_03_an_unexpected_error_inside_the_gate_is_a_block_and_nothing_is_forwarded(
        patched: str, monkeypatch: pytest.MonkeyPatch, capfd) -> None:
    def explode(*args: object, **kwargs: object):
        raise RuntimeError("internal failure carrying FAKE-SECRET-IN-ERROR")

    monkeypatch.setattr(gate_server, patched, explode)
    with running_rig(scan_step=passthrough_scan) as rig:
        reply = rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        assert reply.status == 400 and "HARPO-BLOCK-INTERNAL" in reply.body.decode()
        assert rig.mock.records() == []
        (entry,) = entries(rig, 1)
        assert (entry["decision"], entry["reason"]) == ("blocked", "HARPO-BLOCK-INTERNAL")
        seen = reply.body.decode() + "".join(rig.log_lines) + capfd.readouterr().err
        assert "FAKE-SECRET" not in seen


def test_fr_gate_03_an_error_after_the_reply_began_closes_the_connection_and_adds_nothing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object):
        raise RuntimeError("relay failed")

    monkeypatch.setattr(gate_server, "relay_response", explode)
    with running_rig(scan_step=passthrough_scan) as rig:
        conn = http.client.HTTPConnection(rig.gate.host, rig.gate.port, timeout=10)
        conn.request("POST", "/v1/messages", body=MESSAGES, headers={**JSON, **ANTHROPIC, TOKEN_HEADER: rig.token})
        with pytest.raises((http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionError)):
            conn.getresponse()  # no half-written or second answer: the connection just ends
        conn.close()
        assert entries(rig, 1)[0]["reason"] == "HARPO-BLOCK-INTERNAL"


def test_fr_gate_03_the_error_hook_prints_only_the_exception_class(capfd) -> None:
    with running_rig() as rig:
        try:
            raise ValueError("text with FAKE-SECRET-IN-HOOK")
        except ValueError:
            rig.gate.server.handle_error(None, None)
    err = capfd.readouterr().err
    assert "ValueError" in err and "FAKE-SECRET" not in err and "Traceback" not in err


def test_fr_gate_01_silent_upstream_keeps_the_log_line_even_when_the_agent_gives_up() -> None:
    with ScriptedUpstream(lambda conn, head, body: conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nab")) as up:
        with running_rig(scan_step=passthrough_scan, anthropic_upstream=up.url) as rig:
            conn = http.client.HTTPConnection(rig.gate.host, rig.gate.port, timeout=10)
            conn.request("POST", "/v1/messages", body=MESSAGES,
                         headers={**JSON, **ANTHROPIC, TOKEN_HEADER: rig.token})
            with pytest.raises(http.client.IncompleteRead):
                conn.getresponse().read()
            conn.close()
            (entry,) = entries(rig, 1)
    assert entry["decision"] == "forwarded" and entry["bytes_back"] == 2  # what was relayed before the cut
