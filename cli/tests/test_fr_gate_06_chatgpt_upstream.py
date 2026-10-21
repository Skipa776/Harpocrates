"""FR-GATE-06, G5 spec part 2 (docs/gate-media-upstream-spec.md): an OpenAI request that carries chatgpt-account-id goes
to the ChatGPT upstream with /v1 replaced by /backend-api/codex; without the header it goes to the OpenAI upstream
unchanged; an Anthropic route never looks at the header. Credentials pass byte-identical.

Every upstream is a ScriptedUpstream on 127.0.0.1 that records the raw request head. The credential values are fakes.
"""
from __future__ import annotations

import http.client
import json

import pytest
from gate_rig import passthrough_scan
from gate_support import ScriptedUpstream

from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import TOKEN_HEADER, new_token

FAKE_BEARER = "Bearer fake-chatgpt-access-token-0000"
FAKE_ACCOUNT = "acct-fake-0000-1111"
BODY = json.dumps({"model": "m", "input": "hi", "store": False}).encode()
OK = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")


def answer_ok(conn, head: bytes, body: bytes) -> None:
    conn.sendall(OK)


class Gate:
    """A gate with three separate scripted upstreams (Anthropic, OpenAI, ChatGPT), all on 127.0.0.1."""

    def __enter__(self) -> "Gate":
        self.anthropic, self.openai, self.chatgpt = (ScriptedUpstream(answer_ok) for _ in range(3))
        self.token = new_token()
        config = GateConfig(anthropic_upstream=self.anthropic.url, openai_upstream=self.openai.url,
                            chatgpt_upstream=self.chatgpt.url)
        self.gate = start_gate(config, self.token, scan_step=passthrough_scan, log_sink=lambda line: None)
        return self

    def __exit__(self, *exc: object) -> None:
        self.gate.close()
        for upstream in (self.anthropic, self.openai, self.chatgpt):
            upstream.close()

    def call(self, method: str, target: str, headers: dict, body: bytes = None) -> int:
        conn = http.client.HTTPConnection(self.gate.host, self.gate.port, timeout=15)
        try:
            conn.request(method, target, body=body, headers={TOKEN_HEADER: self.token, **headers})
            response = conn.getresponse()
            response.read()
            return response.status
        finally:
            conn.close()


def request_line(head: bytes) -> bytes:
    return head.split(b"\r\n", 1)[0]


def header_lines(head: bytes) -> list[bytes]:
    return head.split(b"\r\n")[1:]


CHATGPT = {"Content-Type": "application/json", "Authorization": FAKE_BEARER, "chatgpt-account-id": FAKE_ACCOUNT}
API_KEY = {"Content-Type": "application/json", "Authorization": "Bearer sk-fake-api-key-0000"}


def test_fr_gate_06_chatgpt_account_routes_to_chatgpt_upstream() -> None:
    with Gate() as g:
        assert g.call("POST", "/v1/responses", CHATGPT, BODY) == 200
    assert g.openai.heads == [] and g.anthropic.heads == []
    (head,) = g.chatgpt.heads
    assert request_line(head) == b"POST /backend-api/codex/responses HTTP/1.1"
    lines = header_lines(head)
    assert f"Authorization: {FAKE_BEARER}".encode() in lines
    assert f"chatgpt-account-id: {FAKE_ACCOUNT}".encode() in lines  # byte-identical, case kept (FR-INSTALL-05)
    assert f"Host: 127.0.0.1:{g.chatgpt.port}".encode() in lines
    assert g.chatgpt.bodies == [BODY]


def test_fr_gate_06_chatgpt_route_keeps_the_query_string() -> None:
    with Gate() as g:
        assert g.call("POST", "/v1/responses?client=codex&v=2", CHATGPT, BODY) == 200
    assert request_line(g.chatgpt.heads[0]) == b"POST /backend-api/codex/responses?client=codex&v=2 HTTP/1.1"


def test_fr_gate_06_chatgpt_models_is_mapped_the_same_way() -> None:
    with Gate() as g:
        assert g.call("GET", "/v1/models", {"Authorization": FAKE_BEARER, "chatgpt-account-id": FAKE_ACCOUNT}) == 200
    assert request_line(g.chatgpt.heads[0]) == b"GET /backend-api/codex/models HTTP/1.1"


@pytest.mark.parametrize("target,method", [("/v1/responses", "POST"), ("/v1/models", "GET")])
def test_fr_gate_06_api_key_request_routes_to_openai_upstream(target, method) -> None:
    with Gate() as g:
        assert g.call(method, target, API_KEY, BODY if method == "POST" else None) == 200
    assert g.chatgpt.heads == [] and g.anthropic.heads == []
    assert request_line(g.openai.heads[0]) == f"{method} {target} HTTP/1.1".encode()


def test_fr_gate_06_header_name_case_does_not_matter_for_the_choice() -> None:
    with Gate() as g:
        assert g.call("POST", "/v1/responses", {**API_KEY, "ChatGPT-Account-ID": FAKE_ACCOUNT}, BODY) == 200
    assert g.openai.heads == [] and request_line(g.chatgpt.heads[0]).startswith(b"POST /backend-api/codex/responses ")


def test_fr_gate_06_anthropic_route_ignores_chatgpt_account_id() -> None:
    body = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}).encode()
    with Gate() as g:
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01", "chatgpt-account-id": FAKE_ACCOUNT}
        assert g.call("POST", "/v1/messages", headers, body) == 200
    assert g.chatgpt.heads == [] and g.openai.heads == []
    assert request_line(g.anthropic.heads[0]) == b"POST /v1/messages HTTP/1.1"


def test_fr_gate_06_default_chatgpt_upstream_is_chatgpt_com() -> None:
    upstream = GateConfig().upstream_for("chatgpt")
    assert (upstream.scheme, upstream.host, upstream.port) == ("https", "chatgpt.com", 443)


@pytest.mark.parametrize("url", ["https://chatgpt.com/backend-api", "https://chatgpt.com/backend-api/codex",
                                 "http://chatgpt.com", "https://user:pw@chatgpt.com", "ftp://chatgpt.com"])
def test_fr_gate_06_chatgpt_upstream_url_with_path_refused(url) -> None:
    """The path prefix is a constant; the URL is scheme://host[:port] only, as for every upstream."""
    with pytest.raises(ValueError) as error:
        GateConfig(chatgpt_upstream=url)
    assert "pw" not in str(error.value)
