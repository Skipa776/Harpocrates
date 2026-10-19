"""Read a mock recording for the G2 verify items (FR-PROOF-04): names and shapes, never values.

Each function answers one item of docs/gate-parser-spec.md ("To verify": V1, V3 to V7) or docs/gate-transport.md ("To verify
in G2": T1 to T10) from the recording entries bench/mock_provider.py writes. What comes out is a header or field name, a type,
a count, or a short fixed text such as the attribution prefix of a system block. A credential-named header's value is
already "<present>" in the recording; a JSON header is described by its shape, not its content.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from bench import canary_match as match

MESSAGES, RESPONSES = "anthropic.messages", "openai.responses"
BILLING_PREFIX = "x-anthropic-billing-header:"
BILLING_SHAPE = re.compile(r"^(x-anthropic-billing-header: cc_version=)([0-9.]+?)\.([0-9a-f]{3}); cc_entrypoint=([A-Za-z0-9_-]+);$")
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,48}$")


def _bodies(records: list[dict], route: str) -> list[tuple[dict, dict]]:
    """
    Description: Pair each recorded request of a route with its parsed JSON body.
    Parameters:
        records: the recording entries.
        route: anthropic.messages or openai.responses.
    Output: (entry, body) pairs for entries whose body is a JSON object, in request order.
    """
    out = []
    for rec in records:
        if rec.get("route") == route and isinstance(rec.get("body"), str):
            try:
                body = json.loads(rec["body"])
            except ValueError:
                continue
            if isinstance(body, dict):
                out.append((rec, body))
    return out


def json_shape(value: object) -> object:
    """
    Description: Describe a JSON value by structure: key names and value types, none of the content.
    Parameters:
        value: a parsed JSON value.
    Output: A dict (key -> shape; keys that are not plain identifiers collapse to "<key>"), a one-element list of the first
        element's shape (empty for an empty list), or one of the strings str, int, float, bool, null.
    """
    if isinstance(value, dict):
        return {(k if _KEY.match(k) else "<key>"): json_shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_shape(value[0])] if value else []
    if value is None:
        return "null"
    return {bool: "bool", int: "int", float: "float", str: "str"}.get(type(value), type(value).__name__)


def attribution(records: list[dict]) -> dict:
    """
    Description: Read the attribution block Claude Code puts first in the system prompt (verify item V1).
    Parameters:
        records: the recording entries.
    Output: {"found", "position", "own_block", "has_cache_control", "leading_text", "template", "example",
        "stable_across_requests", "requests_checked"}. The block is found when system is a list whose first block starts
        with x-anthropic-billing-header:. Version, fingerprint and entrypoint are not credentials.
    """
    texts, first = [], None
    for _, body in _bodies(records, MESSAGES):
        system = body.get("system")
        block = system[0] if isinstance(system, list) and system and isinstance(system[0], dict) else None
        if block and isinstance(block.get("text"), str) and block["text"].startswith(BILLING_PREFIX):
            texts.append(block["text"])
            first = first or block
    if not texts:
        return {"found": False, "requests_checked": len(_bodies(records, MESSAGES))}
    shaped = BILLING_SHAPE.match(texts[0])
    return {"found": True, "position": 0, "own_block": "\n" not in texts[0], "has_cache_control": "cache_control" in first,
            "leading_text": shaped.group(1) if shaped else BILLING_PREFIX,
            "template": "x-anthropic-billing-header: cc_version=<version>.<hex3>; cc_entrypoint=<entrypoint>;" if shaped else None,
            "example": texts[0], "stable_across_requests": len(set(texts)) == 1, "requests_checked": len(texts)}


def _anthropic_types(bodies: list[dict]) -> dict:
    """
    Description: Summarize the Anthropic request fields seen in a list of bodies.
    Parameters:
        bodies: parsed /v1/messages bodies.
    Output: The "anthropic" dict of block_types().
    """
    content, nested, forms, string_roles, tool_kinds, tool_names, roles = set(), set(), set(), set(), set(), set(), set()
    system_forms, system_counts, keys, thinking = set(), set(), set(), []
    for body in bodies:
        keys |= set(body)
        system = body.get("system")
        system_forms.add("blocks" if isinstance(system, list) else "string" if isinstance(system, str) else "none")
        system_counts.add(len(system) if isinstance(system, list) else 1 if system else 0)
        if "thinking" in body:
            thinking.append(json_shape(body["thinking"]))
        for tool in body.get("tools", []):
            tool_names.add(tool.get("name", ""))
            tool_kinds.add(f"tools[].type={tool['type']}" if "type" in tool else "tools[].custom")
        for message in body.get("messages", []):
            roles.add(message.get("role"))
            blocks = message.get("content")
            if isinstance(blocks, str):
                string_roles.add(message.get("role"))
                continue
            for block in blocks or []:
                content.add(block.get("type"))
                if block.get("type") == "tool_result":
                    inner = block.get("content")
                    forms.add("string" if isinstance(inner, str) else "blocks")
                    nested |= {b.get("type") for b in inner} if isinstance(inner, list) else set()
    return {"top_level_keys": sorted(keys), "system_form": sorted(system_forms)[0] if len(system_forms) == 1 else sorted(system_forms),
            "system_blocks": max(system_counts, default=0), "message_roles": sorted(roles), "message_string_content_roles": sorted(string_roles),
            "content_block_types": sorted(content), "tool_result_content_forms": sorted(forms),
            "tool_result_block_types": sorted(nested), "tool_names": sorted(tool_names), "tool_kinds": sorted(tool_kinds),
            "thinking_param": thinking[0] if thinking else None}


def _responses_types(bodies: list[dict]) -> dict:
    """
    Description: Summarize the OpenAI Responses request fields seen in a list of bodies.
    Parameters:
        bodies: parsed /v1/responses bodies.
    Output: The "responses" dict of block_types().
    """
    items, roles, parts, tool_types, tool_names, keys = set(), set(), set(), set(), set(), set()
    store, include, forms = [], [], set()
    for body in bodies:
        keys |= set(body)
        for key, sink in (("store", store), ("include", include)):
            if key in body and body[key] not in sink:
                sink.append(body[key])
        for tool in body.get("tools", []):
            tool_types.add(tool.get("type"))
            if tool.get("name"):
                tool_names.add(tool["name"])
        raw = body.get("input")
        forms.add("string" if isinstance(raw, str) else "items")
        for item in raw if isinstance(raw, list) else []:
            items.add(item.get("type"))
            if item.get("role"):
                roles.add(item["role"])
            content = item.get("content")
            parts |= {c.get("type") for c in content} if isinstance(content, list) else set()
    return {"top_level_keys": sorted(keys), "input_form": sorted(forms)[0] if len(forms) == 1 else sorted(forms),
            "input_item_types": sorted(items), "message_roles": sorted(roles), "content_part_types": sorted(parts),
            "tool_types": sorted(tool_types), "tool_names": sorted(tool_names), "store": store, "include": include,
            "has_previous_response_id": "previous_response_id" in keys, "has_conversation": "conversation" in keys}


def block_types(records: list[dict]) -> dict:
    """
    Description: List the block, item and tool types the agents actually sent (verify item V6, and V5 for previous_response_id).
    Parameters:
        records: the recording entries.
    Output: {"anthropic": {...}, "responses": {...}} with the sorted type names found across all requests of each format:
        top-level keys, content block types, tool_result forms and nested block types, tool names and kinds, system form;
        item types, roles, content part types, tool types, store/include values and whether previous_response_id or
        conversation ever appeared. Empty dicts for a format with no requests.
    """
    anth = [b for _, b in _bodies(records, MESSAGES)]
    resp = [b for _, b in _bodies(records, RESPONSES)]
    return {"anthropic": _anthropic_types(anth) if anth else {}, "responses": _responses_types(resp) if resp else {}}


def transport(records: list[dict]) -> dict:
    """
    Description: Summarize how requests arrived (verify items V3, V7, T1, T6, T7, T10).
    Parameters:
        records: the recording entries.
    Output: {"requests", "by_route", "endpoints" (METHOD /path, queries dropped), "content_encodings", "framing"
        (content_length, chunked, none), "upgrade_requests", "websocket_extensions_offered", "header_names" (lower-cased,
        no values), "response_statuses"}.
    """
    by_route, encodings, framing, statuses, headers, endpoints, extensions = {}, {}, {"content_length": 0, "chunked": 0, "none": 0}, {}, set(), set(), []
    upgrades = 0
    for rec in records:
        by_route[rec["route"]] = by_route.get(rec["route"], 0) + 1
        enc = rec.get("content_encoding") or "none"
        encodings[enc] = encodings.get(enc, 0) + 1
        framing["chunked" if rec.get("chunked") else "content_length" if rec.get("content_length_header") is not None else "none"] += 1
        statuses[str(rec.get("response_status"))] = statuses.get(str(rec.get("response_status")), 0) + 1
        endpoints.add(f"{rec['method']} {rec['path'].partition('?')[0]}")
        names = {str(n).lower(): v for n, v in rec.get("headers", [])}
        headers |= set(names)
        if "websocket" in str(names.get("upgrade", "")).lower():
            upgrades += 1
            if names.get("sec-websocket-extensions"):
                extensions.append(names["sec-websocket-extensions"])
    return {"requests": len(records), "by_route": dict(sorted(by_route.items())), "endpoints": sorted(endpoints),
            "content_encodings": dict(sorted(encodings.items())), "framing": framing, "upgrade_requests": upgrades,
            "websocket_extensions_offered": sorted(set(extensions)), "header_names": sorted(headers),
            "response_statuses": dict(sorted(statuses.items()))}


def thinking_resent(records: list[dict], produced: dict) -> dict:
    """
    Description: Check whether the agent sent back the thinking the mock produced (verify item V4).
    Parameters:
        records: the recording entries.
        produced: {"thinking": [signatures], "redacted": [data strings], "reasoning": [encrypted_content strings]} the
            scripted replies carried.
    Output: {"thinking_resent", "redacted_thinking_resent", "reasoning_items_resent", "first_request_with_it"}: True or
        False when something of that kind was produced, None when nothing was; first_request_with_it is the recording index
        of the first request that carried any of it, or None.
    """
    seen = {"thinking": set(), "redacted": set(), "reasoning": set()}
    first: Optional[int] = None
    for rec, body in _bodies(records, MESSAGES):
        for message in body.get("messages", []):
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "thinking" and block.get("signature") in produced.get("thinking", []):
                    seen["thinking"].add(block["signature"])
                elif block.get("type") == "redacted_thinking" and block.get("data") in produced.get("redacted", []):
                    seen["redacted"].add(block["data"])
                else:
                    continue
                first = rec["index"] if first is None else first
    for rec, body in _bodies(records, RESPONSES):
        for item in body.get("input") if isinstance(body.get("input"), list) else []:
            if item.get("type") == "reasoning" and item.get("encrypted_content") in produced.get("reasoning", []):
                seen["reasoning"].add(item["encrypted_content"])
                first = rec["index"] if first is None else first

    def verdict(kind: str) -> Optional[bool]:
        """
        Description: Whether any produced value of this kind came back.
        Parameters:
            kind: thinking, redacted or reasoning.
        Output: None when none was produced, else True or False.
        """
        return (bool(seen[kind]) if produced.get(kind) else None)

    return {"thinking_resent": verdict("thinking"), "redacted_thinking_resent": verdict("redacted"),
            "reasoning_items_resent": verdict("reasoning"), "first_request_with_it": first}


def header_values(records: list[dict], name: str) -> dict[int, str]:
    """
    Description: Collect one header's recorded value per request.
    Parameters:
        records: the recording entries.
        name: the header name, any case.
    Output: {recording index: value} for requests that carried it. Use only for headers whose values are not credentials
        (a credential-named header is "<present>" in the recording anyway).
    """
    out = {}
    for rec in records:
        for header, value in rec.get("headers", []):
            if header.lower() == name.lower():
                out[rec["index"]] = value
    return out


def header_echoed(records: list[dict], name: str, value: str) -> list[int]:
    """
    Description: Find the requests that sent back a header value the mock had set.
    Parameters:
        records: the recording entries.
        name: the header name, any case.
        value: the value the mock sent in a response.
    Output: The recording indices whose request carried that header with exactly that value.
    """
    return [index for index, seen in header_values(records, name).items() if seen == value]


def header_shape(records: list[dict], name: str) -> object:
    """
    Description: Describe a header's value by shape (verify item T8: what x-codex-turn-metadata holds).
    Parameters:
        records: the recording entries.
        name: the header name, any case.
    Output: json_shape of the first request's value when it parses as JSON, "str" when it is plain text, None when no request
        carried the header. No content is returned.
    """
    values = header_values(records, name)
    if not values:
        return None
    raw = values[min(values)]
    try:
        return json_shape(json.loads(raw))
    except ValueError:
        return "str"


def where_found(records: list[dict], needle: str) -> list[dict]:
    """
    Description: Say where a planted string appeared in the requests, without printing it.
    Parameters:
        records: the recording entries.
        needle: the planted value (held in memory only).
    Output: A list of {"request", "where": "header"|"body", "name"} in request order: the header name, or the body field
        path. Empty when the string never appeared.
    """
    out = []
    for rec in records:
        for header, value in rec.get("headers", []):
            if match.needles(needle) and any(text in value for _, text in match.needles(needle)):
                out.append({"request": rec["index"], "where": "header", "name": header.lower()})
        if isinstance(rec.get("body"), str):
            out += [{"request": rec["index"], "where": "body", "name": hit.path} for hit in match.locate(rec["body"], needle)]
    return out
