"""FR-GATE-03, FR-GATE-04, FR-GATE-05, FR-GATE-10, FR-GATE-02, SEC-02: the request scan step as a whole.

Every secret is a generated canary (gate_canaries.py). The real detector is used unless a test says otherwise.
"""
from __future__ import annotations

import json
import random
import time

from gate_canaries import PLACEHOLDER, known_format
from gate_scan_support import (
    COUNT_TOKENS,
    HEAD,
    MESSAGES,
    RESPONSES,
    blocked,
    forwarded,
    head_for,
    route_of,
    run,
    scanner,
)

from Harpocrates.gate import blocks
from Harpocrates.gate.errors import block_message, is_block_code
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.request_scan import RequestScanner
from Harpocrates.gate.scan import Block, Forward, run_scan

KNOWN = known_format()
GITHUB, OPENAI, AWS_ID = KNOWN["github_token"], KNOWN["openai_key"], KNOWN["aws_access_key_id"]
REAL = Finder()


def real() -> RequestScanner:
    return scanner(finder=REAL)


def anthropic_history(turns: int, secret_in_turn_one: str) -> dict:
    messages = [{"role": "user", "content": f"turn one: my key is {secret_in_turn_one}, remember it"}]
    for turn in range(2, turns + 1):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"reply {turn}"}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": f"turn {turn}: next question"}]})
    return {"model": "m", "max_tokens": 8, "system": "You are helpful.", "messages": messages}


def responses_history(turns: int, secret_in_turn_one: str) -> dict:
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"turn one: key {secret_in_turn_one}"}]}]
    for turn in range(2, turns + 1):
        items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"reply {turn}"}]})
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"turn {turn}"}]})
    return {"model": "m", "input": items, "store": False}


def test_fr_gate_04_turn1_secret_redacted_in_turn5() -> None:
    """The agent resends the whole history; a secret in the first turn is still in the fifth request, and must be gone."""
    for build, route in ((anthropic_history, MESSAGES), (responses_history, RESPONSES)):
        for turns in range(1, 6):
            out = forwarded(run(real(), build(turns, GITHUB), route)).decode()
            assert GITHUB not in out and PLACEHOLDER.search(out), (build.__name__, turns)
        assert "reply 5" in out and "turn 5" in out  # the later turns went through untouched


def test_fr_gate_05_same_secret_same_placeholder_across_turns() -> None:
    scan = real()
    first = forwarded(run(scan, anthropic_history(1, OPENAI))).decode()
    fifth = forwarded(run(scan, anthropic_history(5, OPENAI))).decode()
    placeholder = PLACEHOLDER.search(first).group()
    assert placeholder in fifth and PLACEHOLDER.findall(first) == PLACEHOLDER.findall(fifth) == [placeholder]
    one, five = json.loads(first), json.loads(fifth)
    assert five["messages"][0] == one["messages"][0]  # the unchanged start of each request stays identical (FR-GATE-05)


def test_fr_gate_05_clean_body_forwarded_byte_identical() -> None:
    rng = random.Random(3)
    scan = real()
    for _ in range(40):
        body = {"model": "m", "max_tokens": rng.choice([1, 8, 1024, 1e3, 1.5e1]), "temperature": rng.choice([0, 0.0, 1, 0.70, 1e-1]),
                "messages": [{"role": "user", "content": [{"type": "text", "text": "héllo ☃ \n tab\t 😀", "cache_control": {"type": "ephemeral"}}]},
                             {"role": "assistant", "content": "plain \\ and \" quotes"}],
                "tools": [{"name": "t", "description": "does a thing", "input_schema": {"type": "object", "properties": {}}}],
                "stream": True}
        keys = list(body)
        rng.shuffle(keys)
        body = {k: body[k] for k in keys}
        data = json.dumps(body, indent=rng.choice([None, 0, 1, 4]), separators=rng.choice([None, (",", ":"), (" , ", " :  ")]),
                          ensure_ascii=rng.choice([True, False])).encode()
        data = rng.choice([b"", b" ", b"\n", b"\t\r\n "]) + data + rng.choice([b"", b"\n", b"  \n"])
        assert forwarded(run(scan, data)) == data
    data = json.dumps({"model": "m", "input": "hi", "store": False}).encode()
    assert forwarded(run(scan, data, RESPONSES)) == data


def test_fr_gate_05_only_the_leaf_with_the_secret_changes_in_a_pretty_printed_body() -> None:
    body = json.dumps({"z": 1.0, "a": [1, 2, {"k": "v"}], "messages": [{"role": "user", "content": f"key {GITHUB} here"}],
                       "tail": "x"}, indent=2).replace("1.0", "1.00")
    out = forwarded(run(real(), body.encode())).decode()
    assert GITHUB not in out
    assert PLACEHOLDER.split(out) == body.split(GITHUB)  # byte for byte, key order, spacing and 1.00 included


def test_fr_gate_02_determinism_cache_control_untouched() -> None:
    body = {"model": "m", "max_tokens": 8, "system": [{"type": "text", "text": f"rules {OPENAI}", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]}],
            "tools": [{"name": "t", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}]}
    data = json.dumps(body).encode()
    out = forwarded(run(real(), data))
    assert out.count(b'"cache_control": {"type": "ephemeral"') == data.count(b'"cache_control": {"type": "ephemeral"') == 3
    assert b'"cache_control": {"type": "ephemeral", "ttl": "1h"}' in out and OPENAI.encode() not in out


def test_fr_gate_02_determinism_prefix_identical_across_turns() -> None:
    scan = real()
    outputs = [forwarded(run(scan, anthropic_history(n, AWS_ID))) for n in range(1, 6)]
    for earlier, later in zip(outputs, outputs[1:]):
        head = earlier.rstrip()[:-2]  # the closing "]}" of the shorter request
        assert later.startswith(head), "the start of each request is the same bytes as the request before"


def test_fr_gate_10_thinking_dropped_request_succeeds() -> None:
    thinking = {"type": "thinking", "thinking": f"I will use {GITHUB}", "signature": "SIG0123456789"}
    redacted = {"type": "redacted_thinking", "data": "DATA0123456789"}
    for indent in (None, 2):
        for blocks_ in ([thinking, redacted, {"type": "text", "text": "answer"}], [{"type": "text", "text": "answer"}, thinking],
                        [thinking, {"type": "text", "text": "a"}, redacted, {"type": "tool_use", "id": "t", "name": "n", "input": {}}]):
            body = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "go"},
                                                                 {"role": "assistant", "content": blocks_},
                                                                 {"role": "user", "content": "next"}]}
            out = forwarded(run(real(), json.dumps(body, indent=indent).encode()))
            assert GITHUB.encode() not in out and b"SIG0123456789" not in out and b"DATA0123456789" not in out
            kept = [b for b in blocks_ if b["type"] not in ("thinking", "redacted_thinking")]
            assert json.loads(out)["messages"][1]["content"] == kept
    only = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "go"},
                                                        {"role": "assistant", "content": [thinking]},
                                                        {"role": "user", "content": "next"}]}
    assert [m["content"] for m in json.loads(forwarded(run(real(), only)))["messages"]] == ["go", "next"]


def test_fr_gate_10_secret_in_encrypted_reasoning_blocks() -> None:
    cipher = "gAAAAAB" + "x" * 80
    item = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": f"I will use {GITHUB}"}],
            "encrypted_content": cipher}
    body = {"model": "m", "input": [{"type": "message", "role": "user", "content": "go"}, item], "store": False}
    assert blocked(run(real(), body, RESPONSES)) == blocks.PROVIDER_TEXT
    clean = {**body, "input": [body["input"][0], {**item, "summary": [{"type": "summary_text", "text": "thinking about it"}]}]}
    data = json.dumps(clean).encode()
    assert forwarded(run(real(), data, RESPONSES)) == data  # ciphertext and clean summaries pass untouched
    anthropic = {"model": "m", "max_tokens": 1, "messages": [{"role": "assistant", "content": [
        {"type": "compaction", "content": f"summary {GITHUB}", "signature": "SIG0123456789"}]}]}
    assert blocked(run(real(), anthropic)) == blocks.PROVIDER_TEXT
    assert "start a new session" in block_message(blocks.PROVIDER_TEXT).lower()


def test_fr_gate_03_parse_error_blocks() -> None:
    scan = real()
    for bad in (b"", b"not json", b"[1, 2]", b'"text"', b"7", b"null", b'{"messages": [', b'{"a": 1, "a": 2}',
                b"\xff\xfe\x00{", b'{"messages": [{"role": "user", "content": "x"}]} trailing', b"[" * 200 + b"]" * 200):
        assert blocked(run(scan, bad)) == blocks.PARSE, bad[:20]


def test_fr_gate_02_malformed_blocked_nesting_and_non_object() -> None:
    def with_arrays(depth: int) -> bytes:
        return b'{"model": "m", "metadata": {"a": ' + b"[" * depth + b"]" * depth + b"}}"

    assert isinstance(run(real(), with_arrays(126)), Forward)  # object, object and 126 arrays: 128 deep
    assert blocked(run(real(), with_arrays(127))) == blocks.PARSE


def test_fr_gate_02_duplicate_key_blocked() -> None:
    assert blocked(run(real(), b'{"model": "m", "messages": [], "model": "n"}')) == blocks.PARSE
    assert blocked(run(real(), b'{"messages": [{"role": "user", "content": "a", "content": "b"}]}')) == blocks.PARSE


def test_fr_gate_03_detector_error_blocks() -> None:
    def broken(text: str) -> list:
        raise RuntimeError("detector crashed: " + GITHUB)

    scan = RequestScanner(Finder(detector=broken))
    body = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hello there"}]}).encode()
    result = run_scan(scan, body, MESSAGES, HEAD, 5.0)
    assert result == Block("HARPO-BLOCK-SCAN-ERROR")  # the exception text, which holds a secret, goes nowhere


def test_fr_gate_03_scan_timeout_blocks() -> None:
    def slow(text: str) -> list:
        time.sleep(3)
        return []

    scan = RequestScanner(Finder(detector=slow))
    body = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hello there"}]}).encode()
    started = time.monotonic()
    assert run_scan(scan, body, MESSAGES, HEAD, 0.3) == Block("HARPO-BLOCK-SCAN-TIMEOUT")
    assert time.monotonic() - started < 2


def test_fr_gate_02_secret_in_json_key_blocks() -> None:
    use = {"type": "tool_use", "id": "t", "name": "n", "input": {GITHUB: 1}}
    body = {"model": "m", "max_tokens": 1, "messages": [{"role": "assistant", "content": [use]}]}
    assert blocked(run(real(), body)) == blocks.KEY_SECRET
    args = {"model": "m", "input": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": json.dumps({GITHUB: 1})}]}
    assert blocked(run(real(), args, RESPONSES)) == blocks.KEY_SECRET


def test_fr_gate_02_secret_in_header_blocks() -> None:
    body = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}).encode()
    for headers in ((("x-custom", f"token {GITHUB}"),), (("x-note", f"a {OPENAI} b"),), (("referer", f"https://u:p@h/?k={AWS_ID}"),)):
        head = head_for("/v1/messages", ("content-type", "application/json"), *headers)
        assert blocked(real()(body, MESSAGES, head)) == blocks.HEADER_SECRET
    for target in (f"/v1/messages?key={GITHUB}", f"/v1/messages?beta=true&x={OPENAI}", f"/v1/messages?x=%73{GITHUB[1:]}",
                   "/v1/messages?x=" + "".join(f"%{ord(c):02x}" for c in GITHUB)):
        assert blocked(real()(body, MESSAGES, head_for(target))) == blocks.HEADER_SECRET, target
    assert blocked(real()(body, MESSAGES, head_for(f"/v1/{GITHUB}/messages"))) == blocks.HEADER_SECRET
    clean = head_for("/v1/messages?beta=true", ("user-agent", "claude-cli/2.1.303 (external, sdk-cli)"),
                     ("x-claude-code-session-id", "01a1554f-7dc8-7d81-af86-b58e45c41d91"))
    assert real()(body, MESSAGES, clean) == Forward(body)


def test_fr_gate_02_credential_headers_are_the_users_own_and_pass_unscanned() -> None:
    body = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}).encode()
    head = head_for("/v1/messages", ("x-api-key", OPENAI), ("Authorization", f"Bearer {GITHUB}"), ("anthropic-beta", "a,b"),
                    ("ChatGPT-Account-ID", AWS_ID), ("cookie", f"s={GITHUB}"), ("x-custom-token", GITHUB))
    assert real()(body, MESSAGES, head) == Forward(body)


def test_fr_gate_02_a_route_without_a_body_still_scans_its_headers_and_query() -> None:
    models = head_for("/v1/models", ("anthropic-version", "2023-06-01"))
    route = route_of("GET", "/v1/models", **{"anthropic-version": "2023-06-01"})
    assert real()(b"", route, models) == Forward(b"")
    assert blocked(real()(b"", route, head_for(f"/v1/models?x={GITHUB}"))) == blocks.HEADER_SECRET


def test_fr_gate_02_count_tokens_is_parsed_like_messages() -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": f"key {GITHUB}"}]}
    out = forwarded(run(real(), body, COUNT_TOKENS))
    assert GITHUB.encode() not in out and PLACEHOLDER.search(out.decode())


def test_fr_gate_02_block_message_has_no_recovery_phrases() -> None:
    phrases = ("thinking", "signature", "bound to a different conversation", "extra inputs are not permitted", "prompt is too long")
    for code in [blocks.PARSE, blocks.UNKNOWN_TYPE, blocks.PROVIDER_TEXT, blocks.KEY_SECRET, blocks.HEADER_SECRET, blocks.HISTORY,
                 blocks.JSON_ARGS]:
        assert is_block_code(code)
        text = block_message(code).lower()
        assert code.lower() in text and not any(p in text for p in phrases), code


def test_sec_02_the_scanner_makes_a_new_key_per_gate_and_the_cache_holds_no_values() -> None:
    one, two = RequestScanner(), RequestScanner()
    body = anthropic_history(1, GITHUB)
    a, b = forwarded(run(one, body)), forwarded(run(two, body))
    assert PLACEHOLDER.search(a.decode()) and PLACEHOLDER.search(b.decode())
    cache = one.finder.cache
    assert GITHUB not in repr(cache._entries) and all(len(k) == 32 for k in cache._entries) and len(cache) > 0
    assert one.finder.redactor is not two.finder.redactor


def test_fr_gate_02_a_secret_the_oracle_names_in_every_kind_of_field_is_replaced_once_per_occurrence() -> None:
    body = {"model": "m", "max_tokens": 1, "system": f"s {GITHUB}", "metadata": {"user_id": GITHUB},
            "messages": [{"role": "user", "content": f"a {GITHUB} b {GITHUB}"}]}
    out = forwarded(run(scanner([GITHUB]), body)).decode()
    assert GITHUB not in out and len(PLACEHOLDER.findall(out)) == 4
    assert len(set(PLACEHOLDER.findall(out))) == 1


def test_fr_gate_02_free_form_fields_named_like_ciphertext_are_still_scanned() -> None:
    """Only the block types the spec names carry a signature or ciphertext. A user's own metadata, or a tool's arguments, may
    use the same words as keys, and a secret there must not slip through."""
    for key in ("signature", "encrypted_content", "encrypted_index", "encrypted_stdout"):
        body = {"model": "m", "max_tokens": 1, "metadata": {key: f"v {GITHUB}"}, "messages": [{"role": "user", "content": "x"}]}
        out = forwarded(run(real(), body)).decode()
        assert GITHUB not in out and PLACEHOLDER.search(out), key
        responses = {"model": "m", "input": "x", "client_metadata": {key: f"v {GITHUB}"}, "prompt": {"id": "p", "variables": {key: GITHUB}}}
        out = forwarded(run(real(), responses, RESPONSES)).decode()
        assert GITHUB not in out and PLACEHOLDER.search(out), key
    cipher = {"type": "web_search_tool_result", "tool_use_id": "s", "content": [{"type": "web_search_result", "title": "t",
                                                                                 "url": "https://x.test", "encrypted_content": "gAAAA" + "x" * 90}]}
    data = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "assistant", "content": [cipher]}]}).encode()
    assert forwarded(run(real(), data)) == data  # while the ciphertext inside a provider block is left alone
