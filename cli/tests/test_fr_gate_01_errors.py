"""FR-GATE-01 / decision 10: a gate block is HTTP 400 (401 token, 426 upgrade) in the provider's own error shape,
with a plain message that avoids the phrases Claude Code uses to start its own recovery."""
from __future__ import annotations

import json

import pytest

from Harpocrates.gate.errors import REASONS, block_message, error_response, is_block_code

RECOVERY_PHRASES = ("thinking", "signature", "bound to a different conversation", "extra inputs are not permitted",
                    "prompt is too long")


def test_fr_gate_01_message_template_is_the_one_checked_in_g2() -> None:
    assert block_message("HARPO-BLOCK-ROUTE") == (
        "Harpo blocked this request before it left your machine (reason HARPO-BLOCK-ROUTE). "
        "Run `harpo doctor` for details.")


@pytest.mark.parametrize("code", sorted(REASONS))
def test_fr_gate_01_block_messages_avoid_the_recovery_phrases(code: str) -> None:
    for provider in ("anthropic", "openai"):
        _, raw = error_response(provider, code)
        assert not [p for p in RECOVERY_PHRASES if p in raw.decode("ascii").lower()]


@pytest.mark.parametrize("code,status", [
    ("HARPO-BLOCK-TOKEN", 401), ("HARPO-BLOCK-UPGRADE", 426), ("HARPO-BLOCK-ROUTE", 400),
    ("HARPO-BLOCK-SCAN-ERROR", 400), ("HARPO-BLOCK-UNKNOWN-TYPE", 400), ("HARPO-UPSTREAM-FAILED", 502),
])
def test_fr_gate_01_status_per_reason(code: str, status: int) -> None:
    assert error_response("openai", code)[0] == status


def test_fr_gate_01_every_listed_reason_is_a_block_code_or_the_upstream_failure() -> None:
    for code, status in REASONS.items():
        assert is_block_code(code) or code == "HARPO-UPSTREAM-FAILED"
        assert status in (400, 401, 426, 502)


def test_fr_gate_01_anthropic_error_shape() -> None:
    _, raw = error_response("anthropic", "HARPO-BLOCK-ROUTE")
    body = json.loads(raw)
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "HARPO-BLOCK-ROUTE" in body["error"]["message"]
    assert json.loads(error_response("anthropic", "HARPO-BLOCK-TOKEN")[1])["error"]["type"] == "authentication_error"
    assert json.loads(error_response("anthropic", "HARPO-UPSTREAM-FAILED")[1])["error"]["type"] == "api_error"


def test_fr_gate_01_openai_error_shape() -> None:
    _, raw = error_response("openai", "HARPO-BLOCK-ROUTE")
    body = json.loads(raw)
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert body["error"]["type"] == "invalid_request_error" and "HARPO-BLOCK-ROUTE" in body["error"]["message"]
    assert json.loads(error_response("openai", "HARPO-UPSTREAM-FAILED")[1])["error"]["type"] == "server_error"


@pytest.mark.parametrize("code", ["HARPO-BLOCK-UNKNOWN-TYPE", "HARPO-BLOCK-S1", "HARPO-BLOCK-" + "A" * 36])
def test_fr_gate_01_is_block_code_accepts_reason_codes(code: str) -> None:
    assert is_block_code(code)


@pytest.mark.parametrize("code", [
    "", "HARPO-BLOCK-", "harpo-block-x", "HARPO-BLOCK-x", "HARPO-BLOCK-A B", "HARPO-BLOCK-A\nB", "HARPO-BLOCK--A",
    "HARPO-BLOCK-A-", "HARPO-BLOCK-" + "A" * 37, "HARPO-UPSTREAM-FAILED", "BLOCK-X", "HARPO-BLOCK-THINKING",
    "HARPO-BLOCK-SIGNATURE-BAD", "HARPO-BLOCK-PROMPT-IS-TOO-LONG", "HARPO-BLOCK-É", "HARPO-BLOCK-A_B",
])
def test_fr_gate_01_is_block_code_refuses_anything_else(code: str) -> None:
    assert not is_block_code(code)


def test_fr_gate_01_error_response_refuses_a_code_that_is_not_a_reason() -> None:
    for code in ("nope", "HARPO-BLOCK-THINKING", "HARPO-BLOCK-A\nB", "HARPO-UPSTREAM-OTHER"):
        with pytest.raises(ValueError):
            error_response("anthropic", code)


def test_fr_gate_01_error_response_is_ascii_json_bytes() -> None:
    _, raw = error_response("openai", "HARPO-BLOCK-ROUTE")
    assert isinstance(raw, bytes) and raw.decode("ascii") and json.loads(raw)
