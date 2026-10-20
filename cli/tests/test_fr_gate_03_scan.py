"""FR-GATE-03: running a scan step. An error, a timeout or an answer the gate cannot trust is a block. Unit tests of
run_scan."""
from __future__ import annotations

import threading
import time

import pytest

from Harpocrates.gate.routes import Route
from Harpocrates.gate.scan import (
    Block,
    Forward,
    RequestHead,
    run_scan,
)

ROUTE = Route("anthropic.messages", "anthropic", "scan")
BODY = b'{"messages":[]}'
HEAD = RequestHead((('user-agent', 'agent/1'), ('x-api-key', 'k')), '/v1/messages', 'beta=true')


def test_fr_gate_03_a_step_gets_the_decoded_body_the_route_and_the_head() -> None:
    seen: list[tuple] = []

    def step(body: bytes, route: Route, head: RequestHead) -> Forward:
        seen.append((body, route, head))
        return Forward(body)

    assert run_scan(step, BODY, ROUTE, HEAD, 5.0) == Forward(BODY)
    assert seen == [(BODY, ROUTE, HEAD)]


def test_fr_gate_03_a_step_may_rewrite_the_body() -> None:
    assert run_scan(lambda body, route, head: Forward(b"rewritten"), BODY, ROUTE, HEAD, 5.0) == Forward(b"rewritten")


def test_fr_gate_03_a_step_may_block_with_a_reason_code() -> None:
    assert run_scan(lambda body, route, head: Block("HARPO-BLOCK-UNKNOWN-TYPE"), BODY, ROUTE, HEAD, 5.0) == Block(
        "HARPO-BLOCK-UNKNOWN-TYPE")


@pytest.mark.parametrize("reason", ["", "blocked: secret sk-abc found", "HARPO-BLOCK-THINKING", "harpo-block-x",
                                    "HARPO-BLOCK-A\nB", "HARPO-UPSTREAM-FAILED", "x" * 500])
def test_fr_gate_03_a_block_reason_that_is_not_a_safe_code_becomes_a_scan_error(reason: str) -> None:
    assert run_scan(lambda body, route, head: Block(reason), BODY, ROUTE, HEAD, 5.0) == Block("HARPO-BLOCK-SCAN-ERROR")


@pytest.mark.parametrize("answer", [None, b"bytes", "text", {"body": b"x"}, (b"x",), Forward("not bytes"),
                                    Forward(bytearray(b"x")), 3])
def test_fr_gate_03_an_answer_of_the_wrong_kind_is_a_scan_error(answer: object) -> None:
    assert run_scan(lambda body, route, head: answer, BODY, ROUTE, HEAD, 5.0) == Block("HARPO-BLOCK-SCAN-ERROR")


@pytest.mark.parametrize("error", [RuntimeError("boom"), ValueError("x"), KeyError("k"), MemoryError(),
                                   RecursionError(), SystemExit(1), KeyboardInterrupt()])
def test_fr_gate_03_a_step_that_raises_is_a_scan_error(error: BaseException) -> None:
    def step(body: bytes, route: Route, head: RequestHead) -> Forward:
        raise error

    assert run_scan(step, BODY, ROUTE, HEAD, 5.0) == Block("HARPO-BLOCK-SCAN-ERROR")


def test_fr_gate_03_a_step_over_the_budget_is_a_timeout_and_is_not_waited_for() -> None:
    release = threading.Event()
    finished: list[bool] = []

    def slow(body: bytes, route: Route, head: RequestHead) -> Forward:
        release.wait(10)
        finished.append(True)
        return Forward(body)

    started = time.monotonic()
    result = run_scan(slow, BODY, ROUTE, HEAD, 0.2)
    elapsed = time.monotonic() - started
    release.set()
    assert result == Block("HARPO-BLOCK-SCAN-TIMEOUT")
    assert 0.15 <= elapsed < 2
    time.sleep(0.2)
    assert finished == [True]  # the late answer arrived, and changed nothing: the result above stands


def test_fr_gate_03_a_step_inside_the_budget_is_not_a_timeout() -> None:
    def quick(body: bytes, route: Route, head: RequestHead) -> Forward:
        time.sleep(0.05)
        return Forward(body)

    assert run_scan(quick, BODY, ROUTE, HEAD, 2.0) == Forward(BODY)
