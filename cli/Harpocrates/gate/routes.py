"""Route policy (FR-GATE-01): the five request kinds the gate knows, and a closed rule for everything else."""
from __future__ import annotations

import re
from typing import Mapping, NamedTuple

_VISIBLE_ASCII = re.compile(r"/[\x21-\x7e]*")
_SCAN_ROUTES = {
    ("POST", "/v1/messages"): ("anthropic.messages", "anthropic"),
    ("POST", "/v1/messages/count_tokens"): ("anthropic.count_tokens", "anthropic"),
    ("POST", "/v1/responses"): ("openai.responses", "openai"),
}


class Route(NamedTuple):
    """
    Description: What the gate does with one request.
    Parameters:
        name: a fixed label such as "anthropic.messages", "websocket-upgrade" or "unknown". Never a request path.
        provider: "anthropic" or "openai": whose error shape and upstream apply.
        action: "scan" (read a body, run the scan step, forward), "pass" (no body; the scan step still runs),
            "upgrade" (answer 426) or "block" (answer 400).
    Output: An immutable record built by classify.
    """

    name: str
    provider: str
    action: str


def classify(method: str, target: str, headers: Mapping[str, str]) -> Route:
    """
    Description: Name the route of a request from its method, target and headers.
    Parameters:
        method: the HTTP method, exactly as received (methods are case-sensitive).
        target: the request target with its query string, exactly as received.
        headers: the request headers with lower-case names; only upgrade and anthropic-version are read.
    Output: A Route. A request that asks for a WebSocket upgrade is "upgrade" on any path. A known (method, path) pair
        is "scan" (POST /v1/messages, /v1/messages/count_tokens, /v1/responses) or "pass" (GET /v1/models). Everything
        else is "block": a path that differs in any way (a trailing slash, an encoded character, a different case), a
        target with a fragment or a byte outside printable ASCII, and every other method. The provider is anthropic when
        the request has an anthropic-version header (or the route is an Anthropic one), otherwise openai.
    """
    caller = "anthropic" if "anthropic-version" in headers else "openai"
    if "websocket" in headers.get("upgrade", "").lower():
        return Route("websocket-upgrade", caller, "upgrade")
    if not _VISIBLE_ASCII.fullmatch(target) or "#" in target:
        return Route("unknown", caller, "block")
    path = target.partition("?")[0]
    if (method, path) in _SCAN_ROUTES:
        name, provider = _SCAN_ROUTES[(method, path)]
        return Route(name, provider, "scan")
    if (method, path) == ("GET", "/v1/models"):
        return Route(f"{caller}.models", caller, "pass")
    return Route("unknown", caller, "block")
