"""FR-GATE-01: which requests the gate scans, passes, refuses with 426 or blocks (G3 spec, "Route", decisions 3 and 13)."""
from __future__ import annotations

import pytest

from Harpocrates.gate.routes import Route, classify


@pytest.mark.parametrize("method,target,expected", [
    ("POST", "/v1/messages", Route("anthropic.messages", "anthropic", "scan")),
    ("POST", "/v1/messages?beta=true", Route("anthropic.messages", "anthropic", "scan")),
    ("POST", "/v1/messages/count_tokens", Route("anthropic.count_tokens", "anthropic", "scan")),
    ("POST", "/v1/messages/count_tokens?beta=true", Route("anthropic.count_tokens", "anthropic", "scan")),
    ("POST", "/v1/responses", Route("openai.responses", "openai", "scan")),
])
def test_fr_gate_01_content_routes_go_to_the_scan_step(method: str, target: str, expected: Route) -> None:
    assert classify(method, target, {}) == expected


def test_fr_gate_01_models_is_passed_and_follows_the_caller_s_provider() -> None:
    assert classify("GET", "/v1/models", {}) == Route("openai.models", "openai", "pass")
    assert classify("GET", "/v1/models?limit=1000", {"anthropic-version": "2023-06-01"}) == Route(
        "anthropic.models", "anthropic", "pass")


@pytest.mark.parametrize("target", ["/v1/responses", "/v1/messages", "/", "/anything"])
@pytest.mark.parametrize("value", ["websocket", "WebSocket", "h2c, websocket"])
def test_fr_gate_01_websocket_upgrade_is_refused_whatever_the_path(target: str, value: str) -> None:
    route = classify("GET", target, {"upgrade": value})
    assert route.action == "upgrade" and route.name == "websocket-upgrade"


@pytest.mark.parametrize("method,target", [
    ("GET", "/v1/messages"), ("PUT", "/v1/messages"), ("DELETE", "/v1/messages"), ("POST", "/v1/models"),
    ("POST", "/v1/chat/completions"), ("POST", "/v1/images/generations"), ("POST", "/v1/realtime/calls"),
    ("POST", "/v1/messages/"), ("POST", "//v1/messages"), ("POST", "/v1/messages/../responses"),
    ("POST", "/v1/%6Dessages"), ("POST", "/V1/messages"), ("POST", "v1/messages"),
    ("POST", "http://example.test/v1/messages"), ("HEAD", "/api/hello"), ("OPTIONS", "/v1/messages"),
    ("post", "/v1/messages"), ("GET", "/v1/models/gpt-5"), ("HEAD", "/v1/models"), ("POST", "*"), ("POST", ""),
    ("POST", "/v1/messages#frag"), ("POST", "/v1/messages?x=1#frag"), ("POST", "/v1/messages?a=\x01"),
    ("POST", "/v1/messages?a=é"), ("POST", "/v1/messages?a=b c"), ("POST", "/v1/messages\x7f"),
])
def test_fr_gate_01_everything_else_is_blocked(method: str, target: str) -> None:
    route = classify(method, target, {})
    assert route.action == "block" and route.name == "unknown"


def test_fr_gate_01_a_blocked_route_still_gets_the_caller_s_provider_error_shape() -> None:
    assert classify("POST", "/nope", {"anthropic-version": "2023-06-01"}).provider == "anthropic"
    assert classify("POST", "/nope", {}).provider == "openai"
    assert classify("GET", "/nope", {"upgrade": "websocket"}).provider == "openai"
