"""FR-GATE-11 (docs/gate-base64-text-spec.md): a secret base64-encoded inside a class R or S text field is found by
decoding long base64 runs, at every alignment, and blocks the request.

Every secret is a generated canary (gate_canaries.py). Tests that check a request is *not* blocked use an oracle detector
that finds only the given values: the real detector's entropy stage may redact a random run on its own, which would hide
what this rule does.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import random
import struct
import zlib

import pytest
from gate_canaries import known_format
from gate_scan_support import HEAD, MESSAGES, RESPONSES, blocked, forwarded, run, scanner

from Harpocrates.gate import blocks
from Harpocrates.gate.errors import block_message, is_block_code
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.request_scan import RequestScanner
from Harpocrates.gate.scan import Block, run_scan
from Harpocrates.gate.spancache import SpanCache, cache_key

OPENAI = known_format()["openai_key"]
SECRET_FILE = f"# local settings\nOPENAI_API_KEY={OPENAI}\nDEBUG=1\n".encode()
CLEAN_FILE = b"# local settings\nDEBUG=1\nPORT=8080\nNAME=demo-service\n"
REAL = Finder()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def real() -> RequestScanner:
    return scanner(finder=REAL)


def oracle(*values: str) -> RequestScanner:
    return scanner(list(values) or [OPENAI])


def anthropic(*blocks_: dict) -> dict:
    return {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": list(blocks_)}]}


def responses(*items: dict) -> dict:
    return {"model": "m", "input": list(items), "store": False}


def text_fields(text: str) -> list[tuple[str, object, dict]]:
    """Every kind of class R and S field the rule covers, holding the given text."""
    return [
        ("anthropic_text_block", MESSAGES, anthropic({"type": "text", "text": text})),
        ("anthropic_tool_result_string", MESSAGES,
         anthropic({"type": "tool_result", "tool_use_id": "toolu_01", "content": text})),
        ("anthropic_tool_use_input", MESSAGES,
         {"model": "m", "max_tokens": 8, "messages": [
             {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "w", "input": {"content": text}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}]}]}),
        ("responses_input_text", RESPONSES,
         responses({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})),
        ("responses_function_call_output", RESPONSES,
         responses({"type": "function_call_output", "call_id": "call_01", "output": text})),
        ("responses_function_call_arguments", RESPONSES,
         responses({"type": "function_call", "call_id": "call_01", "name": "w", "arguments": json.dumps({"content": text})})),
        ("responses_reasoning_summary_class_s", RESPONSES,
         responses({"type": "reasoning", "id": "rs_01", "summary": [{"type": "summary_text", "text": text}]})),
    ]


ENCODED = f"Here is the file you asked for: {b64(SECRET_FILE)} (base64)."
FIELDS = text_fields(ENCODED)


@pytest.mark.parametrize("name,route,body", FIELDS, ids=[f[0] for f in FIELDS])
def test_fr_gate_11_encoded_secret_in_text_blocks(name, route, body) -> None:
    """The canary is not in the raw text, only in the decoded run: the oracle cannot see it unless the gate decodes."""
    assert OPENAI not in json.dumps(body)
    assert blocked(run(oracle(), body, route)) == blocks.ENCODED_SECRET


@pytest.mark.parametrize("name,route,body", FIELDS, ids=[f[0] for f in FIELDS])
def test_fr_gate_11_encoded_secret_blocks_with_the_real_detector(name, route, body) -> None:
    """In a class S field the raw scan may block first (the entropy stage can flag the run itself): either code blocks."""
    allowed = {blocks.ENCODED_SECRET, blocks.PROVIDER_TEXT} if name.endswith("class_s") else {blocks.ENCODED_SECRET}
    assert blocked(run(real(), body, route)) in allowed


def test_fr_gate_11_codex_mcp_resource_blob_blocks() -> None:
    """The exact shape found in the G5 end-to-end run: Codex sends an MCP resource as JSON text in an input_text part."""
    resource = json.dumps({"type": "resource", "resource": {"uri": "file:///attachment.pdf",
                                                            "mimeType": "application/pdf", "blob": b64(SECRET_FILE)}},
                          separators=(",", ":"))
    output = [{"type": "input_text", "text": "Wall time: 0.0018 seconds\nOutput:"}, {"type": "input_text", "text": resource}]
    body = responses({"type": "function_call", "call_id": "call_01", "name": "show_file", "namespace": "mcp__media",
                      "arguments": "{}"},
                     {"type": "function_call_output", "call_id": "call_01", "output": output})
    assert blocked(run(real(), body, RESPONSES)) == blocks.ENCODED_SECRET
    assert blocked(run(oracle(), body, RESPONSES)) == blocks.ENCODED_SECRET


def _wrapped(text: str, width: int) -> str:
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


SHAPES = {
    "after_an_escaped_newline": "x\n" + b64(SECRET_FILE),
    "glued_to_one_letter": "a" + b64(SECRET_FILE),
    "glued_to_three_letters": "abc" + b64(SECRET_FILE),
    "mime_wrapped_at_76": _wrapped(b64(SECRET_FILE), 76),
    "mime_wrapped_at_64": _wrapped(b64(SECRET_FILE), 64),
    "crlf_wrapped": _wrapped(b64(SECRET_FILE), 76).replace("\n", "\r\n"),
    "base64url": base64.urlsafe_b64encode(SECRET_FILE + b"???~~~").decode(),
    "base64url_unpadded": base64.urlsafe_b64encode(SECRET_FILE + b"???~~~x").decode().rstrip("="),
    "standard_unpadded": b64(SECRET_FILE + b"x").rstrip("="),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_fr_gate_11_misaligned_wrapped_and_base64url_runs_are_decoded(shape) -> None:
    text = SHAPES[shape]
    if shape.startswith("base64url"):
        assert "-" in text or "_" in text  # the URL-safe alphabet is really exercised
    for name, route, body in text_fields(text):
        assert blocked(run(oracle(), body, route)) == blocks.ENCODED_SECRET, name


def png(seed: int = 1, size: int = 32) -> bytes:
    """A real PNG of random pixels: its compressed data is what an image's base64 looks like."""
    rng = random.Random(seed)
    rows = b"".join(b"\x00" + bytes(rng.randrange(256) for _ in range(size * 3)) for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    head = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


NOT_TEXT = {
    "png": b64(png()),
    "large_png": b64(png(2, 128)),
    "gzip_of_the_secret_file": b64(gzip.compress(SECRET_FILE)),
    "sha256": b64(hashlib.sha256(b"x").digest()),
    "random_bytes": b64(random.Random(5).randbytes(4000)),
}


@pytest.mark.parametrize("kind", sorted(NOT_TEXT))
def test_fr_gate_11_binary_and_random_runs_are_not_scanned(kind) -> None:
    """A decoding that is not UTF-8 text (90% printable) is not scanned: binary passes as before (an accepted limit,
    like G5's), and garbage alignments never reach the detector."""
    for name, route, body in text_fields(f"blob: {NOT_TEXT[kind]}"):
        data = json.dumps(body).encode()
        assert forwarded(run(oracle(), data, route)) == data, name
    from Harpocrates.gate import encoded
    assert encoded.decoded_texts(NOT_TEXT[kind]) == []


def test_fr_gate_11_short_run_is_not_decoded() -> None:
    """23 base64 characters are not a run; 24 are."""
    short, exact = b"a=SECRETVALUE1abc", b"a=SECRETVALUE1abcd"  # 17 bytes -> 23 characters and "=", 18 bytes -> 24
    assert len(b64(short).rstrip("=")) == 23 and len(b64(exact)) == 24
    for name, route, body in text_fields(f"value {b64(short)} end"):
        data = json.dumps(body).encode()
        assert forwarded(run(oracle("SECRETVALUE1"), data, route)) == data, name
    for name, route, body in text_fields(f"value {b64(exact)} end"):
        assert blocked(run(oracle("SECRETVALUE1"), body, route)) == blocks.ENCODED_SECRET, name


def test_fr_gate_11_clean_encoded_text_passes_byte_identical() -> None:
    for name, route, body in text_fields(f"config: {b64(CLEAN_FILE)} and {_wrapped(b64(CLEAN_FILE * 3), 76)}"):
        data = json.dumps(body).encode()
        assert forwarded(run(oracle(), data, route)) == data, name


def test_fr_gate_11_a_placeholder_glued_to_a_run_does_not_hide_it() -> None:
    body = anthropic({"type": "text", "text": f"earlier: <<HARPO:openai_key:1a2b>>{b64(SECRET_FILE)}"})
    assert blocked(run(oracle(), body)) == blocks.ENCODED_SECRET


def test_fr_gate_11_one_level_only() -> None:
    """A secret encoded twice is not found (an accepted limit, named in the spec)."""
    body = anthropic({"type": "text", "text": f"twice: {b64(b64(SECRET_FILE).encode())}"})
    data = json.dumps(body).encode()
    assert forwarded(run(oracle(), data)) == data


def test_fr_gate_11_decoder_error_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(text: str) -> list:
        raise RuntimeError("decoder crashed: " + OPENAI)

    from Harpocrates.gate import encoded
    monkeypatch.setattr(encoded, "texts_of_run", broken)
    body = json.dumps(anthropic({"type": "text", "text": f"see {b64(CLEAN_FILE)}"})).encode()
    assert run_scan(oracle(), body, MESSAGES, HEAD, 5.0) == Block("HARPO-BLOCK-SCAN-ERROR")


def test_fr_gate_11_block_code_and_message_hold_no_request_text() -> None:
    assert is_block_code(blocks.ENCODED_SECRET) and blocks.ENCODED_SECRET == "HARPO-BLOCK-ENCODED-SECRET"
    message = block_message(blocks.ENCODED_SECRET)
    assert "base64" in message and "decoded text" in message and OPENAI not in message


def test_fr_gate_11_cache_holds_no_values() -> None:
    """SEC-02: the decoded scan is cached by hash; neither the run, its decoding nor the secret is held."""
    cache = SpanCache()
    scan = RequestScanner(Finder(cache=cache))
    run_ = b64(SECRET_FILE)
    body = anthropic({"type": "text", "text": f"file: {run_}"})
    assert blocked(run(scan, body)) == blocks.ENCODED_SECRET
    held = repr(cache._entries) + repr(vars(cache))
    for part in (OPENAI, run_[:32], SECRET_FILE.decode().splitlines()[1][:24]):
        assert part not in held
    assert all(isinstance(key, bytes) and len(key) == 32 for key in cache._entries)


def test_fr_gate_11_a_seen_run_is_not_decoded_or_scanned_again() -> None:
    """Follow-up turns resend the history: an encoded run already scanned is a cache hit, not a new detector call."""
    calls: list[str] = []

    def detect(text: str) -> list:
        calls.append(text)
        return []

    scan = RequestScanner(Finder(detector=detect))
    body = json.dumps(anthropic({"type": "text", "text": f"file: {b64(CLEAN_FILE * 4)}"})).encode()
    forwarded(run(scan, body))
    first = len(calls)
    assert any("PORT=8080" in text for text in calls)  # the run was decoded and its text scanned
    forwarded(run(scan, body))
    assert len(calls) == first


DIRTY = {
    "utf_16_le_with_bom": "\ufeff".encode("utf-16-le") + SECRET_FILE.decode().encode("utf-16-le"),
    "utf_16_le": SECRET_FILE.decode().encode("utf-16-le"),
    "utf_16_be": SECRET_FILE.decode().encode("utf-16-be"),
    "five_stray_bytes_before": b"\x01" * 5 + SECRET_FILE,
    "five_stray_bytes_after": SECRET_FILE + b"\x01" * 5,
    "mostly_control_bytes_around_the_secret": b"\x01\x02\x03\x04\x05\x06" * 40 + f"K={OPENAI}\n".encode(),
    "latin_1_accents": "# réglages café crème à côté élève\n".encode("latin-1") * 4 + SECRET_FILE,
    "nul_padded": SECRET_FILE + b"\x00" * 64,
}


@pytest.mark.parametrize("kind", sorted(DIRTY))
def test_fr_gate_11_text_next_to_stray_bytes_or_in_utf16_blocks(kind) -> None:
    """A decoding that is not 90% text is still searched for printable stretches of 32 or more characters, and UTF-16
    is read as the text it is: a .env written by PowerShell, or a file with a few stray or Latin-1 bytes, still blocks."""
    for name, route, body in text_fields(f"file: {b64(DIRTY[kind])}"):
        assert blocked(run(oracle(), body, route)) == blocks.ENCODED_SECRET, name


def test_fr_gate_11_a_long_garbage_prefix_does_not_dilute_the_secret() -> None:
    text = "".join(random.Random(9).choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") for _ in range(300))
    body = anthropic({"type": "text", "text": text + b64(SECRET_FILE)})
    assert blocked(run(oracle(), body)) == blocks.ENCODED_SECRET


def _split(text: str, every: int, sep: str) -> str:
    return sep.join(text[i:i + every] for i in range(0, len(text), every))


SPLITS = {"space": " ", "tab": "\t", "escaped_newline": "\\n", "escaped_crlf": "\\r\\n", "spaces_and_newline": "  \n"}


@pytest.mark.parametrize("sep", sorted(SPLITS))
def test_fr_gate_11_run_split_by_spaces_tabs_or_escapes_is_joined(sep) -> None:
    """Pieces of 16 or more characters split by whitespace or a literal \\n (a JSON or notebook file printed as text)
    are one run; a secret across a split is found."""
    text = _split(b64(SECRET_FILE), 30, SPLITS[sep])
    for name, route, body in text_fields(text):
        assert blocked(run(oracle(), body, route)) == blocks.ENCODED_SECRET, name


def test_fr_gate_11_prose_is_not_joined_into_runs() -> None:
    from Harpocrates.gate import encoded
    prose = "The quick brown fox jumps over the lazy dog while configuration documentation mentions authentication."
    assert list(encoded.runs(prose * 20)) == []


def test_fr_gate_11_encoded_secret_in_a_json_key_or_nested_value_blocks() -> None:
    blob = b64(SECRET_FILE)
    for arguments in (json.dumps({blob: 1}), json.dumps({"a": [{"b": blob}]}), json.dumps([[["x", blob]]])):
        body = responses({"type": "function_call", "call_id": "call_01", "name": "w", "arguments": arguments})
        assert blocked(run(oracle(), body, RESPONSES)) == blocks.ENCODED_SECRET, arguments[:10]
    tool_use = {"model": "m", "max_tokens": 8, "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "w", "input": {"a": {"b": [blob]}}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}]}]}
    assert blocked(run(oracle(), tool_use)) == blocks.ENCODED_SECRET


def test_fr_gate_11_arguments_that_are_not_json_are_still_searched() -> None:
    from Harpocrates.gate import encoded
    for broken in ("{not json " + b64(SECRET_FILE), "[" * 100000):
        assert encoded.strings_of(broken, True) == [broken]  # no crash, the raw text is still searched
    body = responses({"type": "function_call", "call_id": "call_01", "name": "w", "arguments": "{not json " + b64(SECRET_FILE)})
    result = run(oracle(), body, RESPONSES)
    assert blocked(result) in {blocks.ENCODED_SECRET, blocks.PARSE}


def test_fr_gate_11_text_seen_plain_then_as_json_arguments_still_blocks() -> None:
    """The clean-leaf memo is kept apart for plain and JSON readings. With every character written as a \\u escape, the
    raw text holds no run, but the parsed arguments do: seen first as plain text (clean), the same text as tool-call
    arguments must still block."""
    escaped = "".join(f"\\u{ord(ch):04x}" for ch in b64(SECRET_FILE))
    arguments = '{"content": "' + escaped + '"}'
    scan = oracle()
    plain = json.dumps(anthropic({"type": "text", "text": arguments})).encode()
    assert forwarded(run(scan, plain)) == plain
    call = responses({"type": "function_call", "call_id": "c", "name": "w", "arguments": arguments})
    assert blocked(run(scan, call, RESPONSES)) == blocks.ENCODED_SECRET


def test_fr_gate_11_memo_keys_differ_for_plain_and_json_leaves() -> None:
    cache = SpanCache()
    scan = RequestScanner(Finder(cache=cache, detector=lambda text: []))
    text = json.dumps({"k": "v"})
    forwarded(run(scan, responses({"type": "function_call", "call_id": "c", "name": "w", "arguments": text}), RESPONSES))
    forwarded(run(scan, anthropic({"type": "text", "text": text})))
    assert cache.get(cache_key("J", text)) == () and cache.get(cache_key("E", text)) == ()


def test_fr_gate_11_clean_leaf_is_not_decoded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    from Harpocrates.gate import encoded
    calls: list[str] = []
    real_decode = encoded.texts_of_run
    monkeypatch.setattr(encoded, "texts_of_run", lambda run_: calls.append(run_) or real_decode(run_))
    scan = oracle()
    body = json.dumps(anthropic({"type": "text", "text": f"file: {b64(CLEAN_FILE)}"})).encode()
    forwarded(run(scan, body))
    first = len(calls)
    forwarded(run(scan, body))
    assert first > 0 and len(calls) == first


def test_fr_gate_11_a_leaf_with_a_secret_is_not_remembered_clean() -> None:
    scan = oracle()
    body = anthropic({"type": "text", "text": f"file: {b64(SECRET_FILE)}"})
    assert blocked(run(scan, body)) == blocks.ENCODED_SECRET
    assert blocked(run(scan, body)) == blocks.ENCODED_SECRET


def test_fr_gate_11_decodings_are_cached_under_class_b() -> None:
    cache = SpanCache()
    scan = RequestScanner(Finder(cache=cache, detector=lambda text: []))
    forwarded(run(scan, anthropic({"type": "text", "text": f"file: {b64(CLEAN_FILE)}"})))
    assert cache.get(cache_key("B", CLEAN_FILE.decode())) == ()


def test_fr_gate_11_run_with_one_leftover_character_still_decodes() -> None:
    run_ = "Q" + b64(SECRET_FILE).rstrip("=")
    run_ = run_ if len(run_) % 4 == 1 else run_ + "A" * ((1 - len(run_)) % 4)
    assert len(run_) % 4 == 1
    assert blocked(run(oracle(), anthropic({"type": "text", "text": run_}))) == blocks.ENCODED_SECRET


def test_fr_gate_11_text_with_many_line_breaks_and_tabs_is_text() -> None:
    data = b"\n\t" * 40 + f"K={OPENAI}".encode() + b"\n\t" * 40
    assert blocked(run(oracle(), anthropic({"type": "text", "text": b64(data)}))) == blocks.ENCODED_SECRET


def test_fr_gate_11_text_filter_boundary() -> None:
    from Harpocrates.gate import encoded
    assert encoded._is_text("a" * 91 + "\x01" * 9) and not encoded._is_text("a" * 85 + "\x01" * 15)
    assert not encoded._is_text("")
    short = b"\x01" * 4 + b"SECRETVALUE1" + b"\x01" * 4  # 60% text, no 32-character stretch: not scanned
    assert encoded.decoded_texts(b64(short)) == []


@pytest.mark.parametrize("route_name", ["anthropic_text_block", "responses_reasoning_summary_class_s"])
def test_fr_gate_11_detector_error_on_a_decoding_blocks(route_name) -> None:
    def detect(text: str) -> list:
        if "PORT=8080" in text:
            raise RuntimeError("detector crashed")
        return []

    name, route, body = next(f for f in text_fields(f"file: {b64(CLEAN_FILE)}") if f[0] == route_name)
    scan = RequestScanner(Finder(detector=detect))
    assert run_scan(scan, json.dumps(body).encode(), route, HEAD, 5.0) == Block("HARPO-BLOCK-SCAN-ERROR")


def test_fr_gate_11_too_much_base64_blocks_quickly() -> None:
    import time

    from Harpocrates.gate import encoded
    rng = random.Random(4)
    one_run = b64(rng.randbytes(encoded.MAX_RUN_CHARS // 4 * 3 + 30))
    many_runs = " ! ".join(b64(rng.randbytes(24)) for _ in range(encoded.MAX_RUN_CHARS // 32 + 10))
    for huge in (one_run, many_runs):
        started = time.monotonic()
        result = run_scan(oracle(), json.dumps(anthropic({"type": "text", "text": huge})).encode(), MESSAGES, HEAD, 5.0)
        assert result == Block(blocks.ENCODED_TOO_LARGE)  # the specific code, not a scan timeout
        assert time.monotonic() - started < 5
    assert is_block_code(blocks.ENCODED_TOO_LARGE) and "base64" in block_message(blocks.ENCODED_TOO_LARGE)


def test_fr_gate_11_media_fields_are_not_scanned_by_this_rule() -> None:
    """Class M payloads belong to FR-GATE-07: a PNG passes with its media note, and a text file with a secret keeps
    the media code."""
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png())}}
    data = json.dumps(anthropic(image)).encode()
    result = run(oracle(), data)
    assert forwarded(result) == data and result.media == (("image/png", len(png())),)
    text_file = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(SECRET_FILE)}}
    assert blocked(run(oracle(), anthropic(text_file))) == blocks.MEDIA_SECRET
