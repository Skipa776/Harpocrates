"""Helpers for the scan-step tests: scanners with the real detector or an oracle, routes and heads."""
from __future__ import annotations

import json
from typing import Iterable, Optional

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.request_scan import RequestScanner
from Harpocrates.gate.routes import Route, classify
from Harpocrates.gate.scan import Block, Forward, RequestHead

MESSAGES = Route("anthropic.messages", "anthropic", "scan")
RESPONSES = Route("openai.responses", "openai", "scan")
COUNT_TOKENS = Route("anthropic.count_tokens", "anthropic", "scan")
HEAD = RequestHead((("content-type", "application/json"), ("anthropic-version", "2023-06-01")), "/v1/messages", "")


def finding(token: str) -> Finding:
    """A finding that names a value exactly (the oracle's answer)."""
    return Finding(type="X", file=None, line=1, snippet="", entropy=0.0, evidence=EvidenceType.REGEX, severity=None,
                   confidence=1.0, token=token, category="generic_secret")


def oracle(values: Iterable[str]):
    """A detector that finds exactly the given values wherever they occur: it tests what the gate does with a finding, not
    what the detector can find."""
    wanted = [v for v in values if v]

    def detect(text: str) -> list:
        return [finding(v) for v in wanted if v in text]

    return detect


def scanner(values: Optional[Iterable[str]] = None, finder: Optional[Finder] = None) -> RequestScanner:
    """A request scanner with an oracle for the given values, or with the real detector when none are given."""
    if finder is None:
        finder = Finder(detector=oracle(values)) if values is not None else Finder()
    return RequestScanner(finder)


def run(scan: RequestScanner, body: object, route: Route = MESSAGES, head: RequestHead = HEAD):
    """Scan a body (bytes, or a JSON-serializable value) and return Forward or Block."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return scan(data, route, head)


def forwarded(result) -> bytes:
    """The bytes of a Forward, failing the test with the reason when it is a Block."""
    assert isinstance(result, Forward), result
    return result.body


def blocked(result) -> str:
    """The reason code of a Block, failing the test when the request was forwarded."""
    assert isinstance(result, Block), "the request was forwarded"
    return result.reason


def head_for(route_target: str, *headers: tuple[str, str]) -> RequestHead:
    """A head built the way the server builds one, from a target and headers."""
    path, _, query = route_target.partition("?")
    return RequestHead(tuple(headers), path, query)


def route_of(method: str, target: str, **headers: str) -> Route:
    """The route the gate would choose."""
    return classify(method, target, {k.lower(): v for k, v in headers.items()})
