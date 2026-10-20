"""FR-GATE-01, SEC-03, FR-INSTALL-04, FR-INSTALL-05, FR-GATE-03, FR-GATE-06: the gate end to end, in front of
bench/mock_provider.py on 127.0.0.1. These are the thirteen tests of the G3 spec "Test plan", plus a few that pin
what the plan implies. Every credential and canary in them is a made-up string."""
from __future__ import annotations

import dataclasses
import json
import socket

import pytest
from gate_rig import passthrough_scan, running_rig  # noqa: F401

from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token


def test_fr_gate_01_binds_loopback_only() -> None:
    with running_rig() as rig:
        assert rig.gate.host == "127.0.0.1" and rig.gate.server.server_address[0] == "127.0.0.1"
        assert rig.gate.port > 0 and rig.gate.url == f"http://127.0.0.1:{rig.gate.port}"
        # Not reachable on this machine's other addresses.
        try:
            other = socket.gethostbyname(socket.gethostname())
        except OSError:
            other = "127.0.0.1"
        if other != "127.0.0.1" and not other.startswith("127."):
            with pytest.raises(OSError):
                socket.create_connection((other, rig.gate.port), timeout=2).close()
    # The settings refuse any other address, and start_gate re-checks the settings it is given.
    for host in ("0.0.0.0", "::", "::1", "localhost", "192.0.2.1", ""):
        with pytest.raises(ValueError, match="127.0.0.1"):
            GateConfig(host=host)
    forged = GateConfig()
    object.__setattr__(forged, "host", "0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        start_gate(forged, new_token())
    assert dataclasses.is_dataclass(forged)


# --- SEC-03: the gate token --------------------------------------------------------------------------------------

JSON = {"Content-Type": "application/json"}
MESSAGES = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}).encode()
RESPONSES = json.dumps({"model": "m", "input": "hi"}).encode()
ROUTES = [  # (method, target, body, extra headers)
    ("POST", "/v1/messages", MESSAGES, {"anthropic-version": "2023-06-01"}),
    ("POST", "/v1/messages/count_tokens", MESSAGES, {"anthropic-version": "2023-06-01"}),
    ("POST", "/v1/responses", RESPONSES, {}),
    ("GET", "/v1/models", None, {}),
    ("POST", "/v1/chat/completions", RESPONSES, {}),
    ("GET", "/v1/responses", None, {"Upgrade": "websocket", "Connection": "Upgrade"}),
]


@pytest.mark.parametrize("method,target,body,extra", ROUTES)
@pytest.mark.parametrize("token", [False, "wrong-token", "", "x" * 400])
def test_sec_03_missing_or_wrong_token_rejected(method, target, body, extra, token) -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        reply = rig.call(method, target, body, {**JSON, **extra}, token=token)
        assert reply.status == 401
        error = json.loads(reply.body)
        assert "HARPO-BLOCK-TOKEN" in json.dumps(error)
        assert ("type" in error and error["type"] == "error") if "anthropic-version" in extra else "error" in error
        assert rig.token not in reply.body.decode() and rig.mock.records() == []  # nothing was forwarded


def test_sec_03_the_right_token_is_not_a_401_and_the_header_name_is_case_blind() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        for name in (TOKEN_HEADER, "X-Harpo-Gate-Token", "X-HARPO-GATE-TOKEN"):
            assert rig.call("POST", "/v1/messages", MESSAGES, {**JSON, name: rig.token}, token=False).status != 401


def test_sec_03_two_token_headers_a_token_in_the_url_or_in_another_header_are_not_accepted() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        both = rig.raw(
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
            + f"{TOKEN_HEADER}: {rig.token}\r\n{TOKEN_HEADER}: {rig.token}\r\n\r\n{{}}".encode())
        assert both.startswith(b"HTTP/1.1 401")
        assert rig.call("POST", f"/v1/messages?{TOKEN_HEADER}={rig.token}", MESSAGES, JSON, token=False).status == 401
        for name in ("Authorization", "x-api-key", "X-Harpo-Token"):
            assert rig.call("POST", "/v1/messages", MESSAGES, {**JSON, name: rig.token}, token=False).status == 401
        assert rig.mock.records() == []


def test_sec_03_a_rejected_request_with_a_large_body_still_gets_its_answer() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        big = b'{"pad":"' + b"a" * 600_000 + b'"}'
        reply = rig.call("POST", "/v1/messages", big, JSON, token=False)
        assert reply.status == 401 and rig.mock.records() == []


# --- routes ------------------------------------------------------------------------------------------------------

RECOVERY_PHRASES = ("thinking", "signature", "bound to a different conversation", "extra inputs are not permitted",
                    "prompt is too long")


@pytest.mark.parametrize("method,target", [
    ("POST", "/v1/chat/completions"), ("POST", "/v1/images/generations"), ("POST", "/v1/realtime/calls"),
    ("POST", "/v1/memories/trace_summarize"), ("GET", "/v1/messages"), ("DELETE", "/v1/messages"),
    ("PUT", "/v1/responses"), ("PATCH", "/v1/responses"), ("OPTIONS", "/v1/messages"), ("HEAD", "/api/hello"),
    ("POST", "/v1/messages/"), ("POST", "/"), ("GET", "/v1/models/gpt-5"), ("POST", "/thinking/signature?key=FAKE-Q-1"),
])
def test_fr_gate_01_each_unknown_route_is_blocked_without_echoing_the_path(method: str, target: str) -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        for headers in ({}, {"anthropic-version": "2023-06-01"}):
            reply = rig.call(method, target, MESSAGES if method in ("POST", "PUT", "PATCH") else None, headers)
            assert reply.status == 400
            if method != "HEAD":
                text = reply.body.decode()
                assert "HARPO-BLOCK-ROUTE" in text
                assert not [p for p in RECOVERY_PHRASES if p in text.lower()]
                assert "FAKE-Q-1" not in text and "/v1/" not in text and "/api/" not in text  # no path is echoed
                assert ("type" in json.loads(text)) == bool(headers)  # Anthropic shape only for an Anthropic caller
        assert rig.mock.records() == []


def test_fr_gate_01_unknown_route_blocked_and_websocket_426() -> None:
    """The plan's test: a route the gate does not know is a 400 and nothing is forwarded; a WebSocket upgrade is a 426."""
    with running_rig(scan_step=passthrough_scan) as rig:
        blocked = rig.call("POST", "/v1/chat/completions", RESPONSES, JSON)
        assert blocked.status == 400 and "HARPO-BLOCK-ROUTE" in blocked.body.decode()
        upgrade = rig.call("GET", "/v1/responses", None, {"Upgrade": "websocket", "Connection": "Upgrade"})
        assert upgrade.status == 426 and "HARPO-BLOCK-UPGRADE" in upgrade.body.decode()
        assert rig.mock.records() == []


@pytest.mark.parametrize("target", ["/v1/responses", "/v1/messages", "/v1/models", "/anything"])
def test_fr_gate_01_a_websocket_upgrade_is_a_426_on_any_path_and_a_401_without_the_token(target: str) -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        upgrade = {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13",
                   "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}
        reply = rig.call("GET", target, None, upgrade)
        assert reply.status == 426
        body = json.loads(reply.body)
        assert "HARPO-BLOCK-UPGRADE" in body["error"]["message"]
        assert not [p for p in RECOVERY_PHRASES if p in reply.body.decode().lower()]
        assert rig.mock.records() == []
        # Without the right token an upgrade is a 401 first (token check, then route).
        assert rig.call("GET", target, None, upgrade, token=False).status == 401
