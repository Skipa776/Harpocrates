"""FR-GATE-01 / decision 13: a request the HTTP parser itself refuses (an unknown method, a bad request line, an
oversize header, an unsupported version) is blocked like any other unknown request: 400 in the provider's error shape
with a reason code, one log line, nothing forwarded. Without this the standard library answers with an HTML page."""
from __future__ import annotations

import json

import pytest
from gate_rig import passthrough_scan, running_rig

from Harpocrates.gate.session_token import TOKEN_HEADER

RECOVERY_PHRASES = ("thinking", "signature", "bound to a different conversation", "extra inputs are not permitted",
                    "prompt is too long")


def split(raw: bytes) -> tuple[bytes, dict | None]:
    head, _, body = raw.partition(b"\r\n\r\n")
    return head.split(b"\r\n", 1)[0], (json.loads(body) if body else None)


@pytest.mark.parametrize("line", [b"FOO /v1/messages HTTP/1.1", b"TRACE / HTTP/1.1", b"CONNECT example.test:443 HTTP/1.1",
                                  b"sk-FAKE-METHOD-123 /v1/messages HTTP/1.1", b"post /v1/messages HTTP/1.1"])
def test_fr_gate_01_an_unknown_method_is_a_400_in_the_provider_shape_and_is_logged(line: bytes) -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        raw = rig.raw(line + f"\r\nHost: g\r\n{TOKEN_HEADER}: {rig.token}\r\n\r\n".encode())
        status, body = split(raw)
        assert status == b"HTTP/1.1 400 Bad Request" and "HARPO-BLOCK-ROUTE" in body["error"]["message"]
        assert b"<html" not in raw.lower() and b"FAKE-METHOD" not in raw
        raw = rig.raw(line + b"\r\nHost: g\r\nanthropic-version: 2023-06-01\r\n\r\n")
        assert split(raw)[1]["type"] == "error"  # an Anthropic caller gets the Anthropic shape
        lines = rig.wait_log(2)
        assert [json.loads(entry)["decision"] for entry in lines] == ["blocked", "blocked"]
        assert "FAKE-METHOD" not in "".join(lines) and rig.mock.records() == []


@pytest.mark.parametrize("raw_request", [
    b"POST /v1/messages HTTP/2.0\r\nHost: g\r\n\r\n",
    b"POST /v1/messages HTTP/1.1\r\nHost: g\r\nX-Big: " + b"a" * 70000 + b"\r\n\r\n",
    b"POST /v1/messages HTTP/1.1\r\n" + b"X-H: 1\r\n" * 150 + b"\r\n",
    b"POST /" + b"a" * 70000 + b" HTTP/1.1\r\nHost: g\r\n\r\n",
])
def test_fr_gate_01_a_malformed_request_is_a_400_with_a_reason_code(raw_request: bytes) -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        status, body = split(rig.raw(raw_request))
        assert status == b"HTTP/1.1 400 Bad Request" and "HARPO-BLOCK-REQUEST" in body["error"]["message"]
        assert not [p for p in RECOVERY_PHRASES if p in json.dumps(body).lower()]
        assert rig.mock.records() == []
        assert len(rig.wait_log(1)) == 1


@pytest.mark.parametrize("raw_request", [b"garbage\r\n\r\n", b"\x00\x01\x02\r\n\r\n",
                                         b"POST /v1/messages HTTP/x.y\r\n\r\n"])
def test_fr_gate_01_a_request_line_that_is_not_http_still_gets_a_status_line_and_is_logged(raw_request: bytes) -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        raw = rig.raw(raw_request)
        assert raw.startswith(b"HTTP/1.1 400 Bad Request\r\n") and b"HARPO-BLOCK-REQUEST" in raw
        assert b"<html" not in raw.lower()
        assert len(rig.wait_log(1)) == 1 and rig.mock.records() == []


def test_fr_gate_01_an_http_0_9_request_has_no_headers_so_it_has_no_token_and_gets_the_401_body_only() -> None:
    with running_rig(scan_step=passthrough_scan) as rig:
        raw = rig.raw(b"GET /v1/models\r\n")
        assert b"HARPO-BLOCK-TOKEN" in raw and not raw.startswith(b"HTTP/") and rig.mock.records() == []
