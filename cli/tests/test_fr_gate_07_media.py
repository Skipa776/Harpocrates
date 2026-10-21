"""FR-GATE-07 (G5 spec, part 1): images, PDFs and audio pass unchanged with a warning; a text file sent as base64 is
decoded and scanned; a string is media only in a field the parser tables mark M, and only when it is strict base64.

Every secret is a generated canary (gate_canaries.py). The real detector is used unless a test says otherwise.
"""
from __future__ import annotations

import base64
import hashlib
import json

import pytest
from gate_canaries import PLACEHOLDER, known_format
from gate_rig import ANTHROPIC, JSON, running_rig
from gate_scan_support import MESSAGES, RESPONSES, blocked, forwarded, run, scanner

from Harpocrates.gate import blocks
from Harpocrates.gate.errors import block_message, is_block_code
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.request_log import format_entry
from Harpocrates.gate.request_scan import RequestScanner

OPENAI = known_format()["openai_key"]
REAL = Finder()

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 3
JPEG = b"\xff\xd8\xff\xe0" + bytes(range(255, -1, -1)) * 2
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + bytes(range(256)) * 4
WAV = b"RIFF\x00\x01\x00\x00WAVEfmt " + bytes(range(256))
SECRET_FILE = f"# local settings\nOPENAI_API_KEY={OPENAI}\nDEBUG=1\n".encode()
CLEAN_FILE = b"# local settings\nDEBUG=1\nPORT=8080\n"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def data_url(media_type: str, data: bytes) -> str:
    return f"data:{media_type};base64,{b64(data)}"


def real() -> RequestScanner:
    return scanner(finder=REAL)


# --- request builders: one per media position in the parser tables ---

def anthropic(*blocks_: dict) -> dict:
    return {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": list(blocks_)}]}


def a_image(payload: str, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": payload}}


def a_document(payload: str, media_type: str = "application/pdf") -> dict:
    return {"type": "document", "source": {"type": "base64", "media_type": media_type, "data": payload}}


def a_tool_result_image(payload: str) -> dict:
    return {"type": "tool_result", "tool_use_id": "toolu_01", "content": [a_image(payload)]}


def responses(*items: dict) -> dict:
    return {"model": "m", "input": list(items), "store": False}


def r_message(*parts: dict) -> dict:
    return {"type": "message", "role": "user", "content": list(parts)}


def r_image(url: str) -> dict:
    return r_message({"type": "input_image", "image_url": url, "detail": "auto"})


def r_file(file_data: str, filename: str = "report.pdf") -> dict:
    return r_message({"type": "input_file", "filename": filename, "file_data": file_data})


def r_audio(payload: str) -> dict:
    return r_message({"type": "input_audio", "data": payload, "format": "wav"})


def r_screenshot(url: str) -> dict:
    return {"type": "computer_call_output", "call_id": "call_01",
            "output": {"type": "computer_screenshot", "image_url": url}}


def r_generated(payload: str) -> dict:
    return {"type": "image_generation_call", "id": "ig_01", "status": "completed", "result": payload}


# (name, route, body builder from a payload string, the payload for raw bytes, expected kind)
POSITIONS = [
    ("anthropic_image", MESSAGES, lambda p: anthropic(a_image(p)), b64, "image/png"),
    ("anthropic_tool_result_image", MESSAGES, lambda p: anthropic(a_tool_result_image(p)), b64, "image/png"),
    ("anthropic_document_pdf", MESSAGES, lambda p: anthropic(a_document(p)), b64, "application/pdf"),
    ("responses_input_image", RESPONSES, lambda p: responses(r_image(p)), lambda d: data_url("image/png", d), "image/png"),
    ("responses_input_file_pdf", RESPONSES, lambda p: responses(r_file(p)), lambda d: data_url("application/pdf", d),
     "application/pdf"),
    ("responses_input_audio", RESPONSES, lambda p: responses(r_audio(p)), b64, "audio"),
    ("responses_computer_screenshot", RESPONSES, lambda p: responses(r_screenshot(p)),
     lambda d: data_url("image/png", d), "image/png"),
    ("responses_image_generation_result", RESPONSES, lambda p: responses(r_generated(p)), b64, "image/png"),
]
RAW_FOR = {"application/pdf": PDF, "audio": WAV}


@pytest.mark.parametrize("name,route,build,encode,kind", POSITIONS, ids=[p[0] for p in POSITIONS])
def test_fr_gate_07_image_and_pdf_pass_with_warning(name, route, build, encode, kind) -> None:
    """Binary media is forwarded byte-identical, not decoded or scanned, and noted as (kind, decoded size) only."""
    raw = RAW_FOR.get(kind, PNG)
    body = json.dumps(build(encode(raw))).encode()
    result = run(real(), body, route)
    assert forwarded(result) == body
    assert result.media == ((kind, len(raw)),)


def test_fr_gate_07_several_media_items_are_noted_in_document_order() -> None:
    body = anthropic(a_image(b64(JPEG), "image/jpeg"), {"type": "text", "text": "compare"}, a_document(b64(PDF)))
    result = run(real(), body)
    assert result.media == (("image/jpeg", len(JPEG)), ("application/pdf", len(PDF)))


def test_fr_gate_07_a_request_without_media_has_no_note() -> None:
    assert run(real(), anthropic({"type": "text", "text": "hello"})).media == ()


PREFIXED = f"data:image/png;base64,iVBORw0KGgo=\nOPENAI_API_KEY={OPENAI}\n"
TEXT_FIELDS = [
    ("anthropic_text_block", MESSAGES, anthropic({"type": "text", "text": PREFIXED})),
    ("anthropic_tool_result_string", MESSAGES,
     anthropic({"type": "tool_result", "tool_use_id": "toolu_01", "content": PREFIXED})),
    ("anthropic_system", MESSAGES, {**anthropic({"type": "text", "text": "hi"}), "system": PREFIXED}),
    ("responses_input_text", RESPONSES, responses(r_message({"type": "input_text", "text": PREFIXED}))),
    ("responses_function_call_output", RESPONSES,
     responses({"type": "function_call_output", "call_id": "call_01", "output": PREFIXED})),
]


@pytest.mark.parametrize("name,route,body", TEXT_FIELDS, ids=[t[0] for t in TEXT_FIELDS])
def test_fr_gate_07_data_url_in_text_field_is_scanned(name, route, body) -> None:
    """A string that starts like a data URL is still text outside a media field: the secret after it is redacted."""
    out = forwarded(run(real(), body, route)).decode()
    assert OPENAI not in out and PLACEHOLDER.search(out)


MALFORMED = {
    "line_break": lambda d: b64(d)[:40] + "\n" + b64(d)[40:],
    "readable_tail": lambda d: b64(d) + f" OPENAI_API_KEY={OPENAI}",
    "bad_character": lambda d: b64(d)[:40] + "!" + b64(d)[41:],
    "padding_in_the_middle": lambda d: "AAAA=AAA" + b64(d),
    "length_not_a_multiple_of_four": lambda d: b64(d) + "A",
}


@pytest.mark.parametrize("variant", sorted(MALFORMED))
@pytest.mark.parametrize("name,route,build,encode,kind", POSITIONS, ids=[p[0] for p in POSITIONS])
def test_fr_gate_07_malformed_base64_media_blocks(variant, name, route, build, encode, kind) -> None:
    """A media field must be strict base64 to its end; anything else is not forwarded."""
    payload = MALFORMED[variant](PNG)
    if encode is not b64:
        payload = f"data:{'application/pdf' if kind == 'application/pdf' else 'image/png'};base64,{payload}"
    assert blocked(run(real(), build(payload), route)) == blocks.MEDIA_MALFORMED


TEXT_FILES = [
    ("anthropic_document_text_plain", MESSAGES, lambda d: anthropic(a_document(b64(d), "text/plain"))),
    ("anthropic_document_labelled_pdf", MESSAGES, lambda d: anthropic(a_document(b64(d)))),
    ("anthropic_image_labelled_png", MESSAGES, lambda d: anthropic(a_image(b64(d)))),
    ("responses_input_file_env", RESPONSES, lambda d: responses(r_file(data_url("text/plain", d), ".env"))),
    ("responses_input_file_octet_stream", RESPONSES,
     lambda d: responses(r_file(data_url("application/octet-stream", d), "settings"))),
    ("responses_input_file_raw_base64", RESPONSES, lambda d: responses(r_file(b64(d), "config.json"))),
]


@pytest.mark.parametrize("name,route,build", TEXT_FILES, ids=[t[0] for t in TEXT_FILES])
def test_fr_gate_07_text_file_with_secret_blocks(name, route, build) -> None:
    """A payload that decodes to text is scanned whatever it is labelled; a secret in it blocks (base64 is not
    rewritten in v1)."""
    assert blocked(run(real(), build(SECRET_FILE), route)) == blocks.MEDIA_SECRET


@pytest.mark.parametrize("name,route,build", TEXT_FILES, ids=[t[0] for t in TEXT_FILES])
def test_fr_gate_07_clean_text_file_passes(name, route, build) -> None:
    """A text file without a secret is forwarded byte-identical. It was scanned, so it carries no media warning."""
    body = json.dumps(build(CLEAN_FILE)).encode()
    result = run(real(), body, route)
    assert forwarded(result) == body and result.media == ()


def test_fr_gate_07_media_secret_has_advice_and_no_request_text() -> None:
    assert is_block_code(blocks.MEDIA_SECRET) and is_block_code(blocks.MEDIA_MALFORMED)
    message = block_message(blocks.MEDIA_SECRET)
    assert "as text" in message and OPENAI not in message


@pytest.mark.parametrize("media_type", ["image/svg+xml", f"image/png;x={OPENAI}", "IMAGE/PNG", "", "text/plain"],
                         ids=["svg", "parameter_with_canary", "upper_case", "empty", "text_plain"])
def test_fr_gate_07_media_kind_is_from_fixed_list(media_type) -> None:
    """The kind comes from the payload's first bytes, never from the label: no string from the request reaches the note."""
    result = run(real(), anthropic(a_image(b64(PNG), media_type)))
    forwarded(result)
    assert result.media == (("image/png", len(PNG)),)
    assert OPENAI not in repr(result.media)


GIF = b"GIF89a" + bytes(range(256))
WEBP = b"RIFF\x00\x02\x00\x00WEBPVP8 " + bytes(range(256))
MP3 = b"ID3\x04\x00" + bytes(range(256))


@pytest.mark.parametrize("raw,kind", [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp"),
                                      (PDF, "application/pdf"), (WAV, "audio"), (MP3, "audio")],
                         ids=["png", "jpeg", "gif", "webp", "pdf", "wav", "mp3"])
def test_fr_gate_07_media_kind_is_read_from_the_payload_signature(raw, kind) -> None:
    result = run(real(), responses(r_file(data_url("application/octet-stream", raw), "blob")), RESPONSES)
    assert result.media == ((kind, len(raw)),)


TEXT_DISGUISES = {
    "utf_16_le": lambda d: "\ufeff".encode("utf-16-le") + d.decode().encode("utf-16-le"),
    "utf_16_be": lambda d: d.decode().encode("utf-16-be"),
    "nul_then_secret": lambda d: b"\x00" + d,
    "one_invalid_utf8_byte": lambda d: d + b"\xff",
    "unknown_binary_header": lambda d: b"PK\x03\x04\x00\x00" + d,
    "secret_past_8_kib": lambda d: b"#" * 9000 + b"\n" + d,
}


@pytest.mark.parametrize("disguise", sorted(TEXT_DISGUISES))
@pytest.mark.parametrize("label", ["image/png", "application/javascript", "text/plain"])
def test_fr_gate_07_payload_without_a_binary_signature_is_scanned(disguise, label) -> None:
    """Only a known image, PDF or audio signature passes unscanned: text in any other wrapping is read and scanned."""
    payload = TEXT_DISGUISES[disguise](SECRET_FILE)
    assert blocked(run(real(), anthropic(a_image(b64(payload), label)))) == blocks.MEDIA_SECRET
    assert blocked(run(real(), responses(r_file(data_url(label, payload), "x")), RESPONSES)) == blocks.MEDIA_SECRET


@pytest.mark.parametrize("payload", ["", "data:image/png;base64,"], ids=["empty", "empty_data_url"])
def test_fr_gate_07_empty_media_payload_passes_without_a_note(payload) -> None:
    """An empty payload is empty text: nothing to scan, nothing unscanned to warn about."""
    body = json.dumps(responses(r_file(payload))).encode()
    result = run(real(), body, RESPONSES)
    assert forwarded(result) == body and result.media == ()


URL_FIELDS = [
    ("responses_input_image", lambda u: responses(r_image(u))),
    ("responses_computer_screenshot", lambda u: responses(r_screenshot(u))),
    ("anthropic_url_source", lambda u: anthropic({"type": "image", "source": {"type": "url", "url": u}})),
]


@pytest.mark.parametrize("url", [f"https://img.example/x.png?key={OPENAI}", f"data:text/plain,OPENAI_API_KEY={OPENAI}"],
                         ids=["https_url", "data_url_without_base64"])
@pytest.mark.parametrize("name,build", URL_FIELDS, ids=[u[0] for u in URL_FIELDS])
def test_fr_gate_07_image_url_that_is_not_base64_is_scanned(name, build, url) -> None:
    route = MESSAGES if name.startswith("anthropic") else RESPONSES
    out = forwarded(run(real(), build(url), route)).decode()
    assert OPENAI not in out and PLACEHOLDER.search(out)


def test_fr_gate_07_base64_data_url_in_a_url_source_is_media() -> None:
    body = json.dumps(anthropic({"type": "image", "source": {"type": "url", "url": data_url("image/png", PNG)}})).encode()
    result = run(real(), body)
    assert forwarded(result) == body and result.media == (("image/png", len(PNG)),)


def test_fr_gate_07_base64_marker_past_200_characters_is_still_checked() -> None:
    url = "data:image/png;" + "x" * 300 + ";base64," + b64(PNG) + f" key={OPENAI}"
    assert blocked(run(real(), responses(r_image(url)), RESPONSES)) == blocks.MEDIA_MALFORMED


NON_STRINGS = [
    ("anthropic_source_data", MESSAGES, lambda v: anthropic(a_image(v))),
    ("responses_file_data", RESPONSES, lambda v: responses(r_file(v))),
    ("responses_audio_data", RESPONSES, lambda v: responses(r_audio(v))),
    ("responses_generation_result", RESPONSES, lambda v: responses(r_generated(v))),
    ("responses_image_url", RESPONSES, lambda v: responses(r_image(v))),
]


@pytest.mark.parametrize("wrap", [lambda c: {"text": c}, lambda c: [c]], ids=["object", "array"])
@pytest.mark.parametrize("name,route,build", NON_STRINGS, ids=[n[0] for n in NON_STRINGS])
def test_fr_gate_07_non_string_media_value_is_scanned(name, route, build, wrap) -> None:
    out = forwarded(run(real(), build(wrap(f"OPENAI_API_KEY={OPENAI}")), route)).decode()
    assert OPENAI not in out and PLACEHOLDER.search(out)


def test_fr_gate_07_other_members_of_a_base64_source_are_scanned() -> None:
    block = a_image(b64(PNG))
    block["source"]["note"] = f"OPENAI_API_KEY={OPENAI}"
    out = forwarded(run(real(), anthropic(block))).decode()
    assert OPENAI not in out and PLACEHOLDER.search(out) and b64(PNG) in out


def test_fr_gate_07_anthropic_source_without_media_type_is_still_checked() -> None:
    source = {"type": "image", "source": {"type": "base64", "data": b64(SECRET_FILE)}}
    assert blocked(run(real(), anthropic(source))) == blocks.MEDIA_SECRET
    source["source"]["media_type"] = 7
    assert blocked(run(real(), anthropic(source))) == blocks.MEDIA_SECRET


def test_fr_gate_07_image_in_a_function_call_output_passes_with_warning() -> None:
    part = {"type": "input_image", "image_url": data_url("image/png", PNG)}
    body = json.dumps(responses({"type": "function_call_output", "call_id": "call_01", "output": [part]})).encode()
    result = run(real(), body, RESPONSES)
    assert forwarded(result) == body and result.media == (("image/png", len(PNG)),)


def test_fr_gate_07_log_keeps_only_known_kinds_and_sizes() -> None:
    entry = {"media": [{"kind": "image/png", "bytes": 3}, {"kind": "image/svg+xml", "bytes": 5}, {"kind": "other", "bytes": 1},
                       {"kind": "image/png", "bytes": -1}, {"kind": "image/png", "bytes": True},
                       {"kind": "application/pdf", "bytes": 7, "name": OPENAI}, "junk", ("audio", 2)]}
    line = json.loads(format_entry(entry))
    assert line["media"] == [{"kind": "image/png", "bytes": 3}, {"kind": "application/pdf", "bytes": 7}]
    assert "media" not in json.loads(format_entry({"media": []}))
    assert "media" not in json.loads(format_entry({"media": "image/png"}))


def test_fr_gate_07_media_warning_reaches_the_request_log_through_the_gate() -> None:
    """Through the real gate and the mock provider: the image arrives byte-identical and the log line says so."""
    body = json.dumps(anthropic({"type": "text", "text": "what is this?"}, a_image(b64(PNG)))).encode()
    with running_rig(scan_step=RequestScanner()) as rig:
        reply = rig.call("POST", "/v1/messages", body, {**JSON, **ANTHROPIC})
        lines = rig.wait_log(1)
        records = rig.mock.records()
    assert reply.status == 200
    assert records[0]["body_sha256"] == hashlib.sha256(body).hexdigest()
    entry = json.loads(lines[0])
    assert entry["decision"] == "forwarded" and entry["media"] == [{"kind": "image/png", "bytes": len(PNG)}]
    assert b64(PNG)[:40] not in lines[0]


def test_fr_gate_07_text_file_with_secret_is_blocked_through_the_gate() -> None:
    """Through the real gate: nothing reaches the provider, the agent is told to send the file as text, and neither the
    answer nor the log holds the secret or the payload."""
    payload = b64(SECRET_FILE)
    body = json.dumps(anthropic(a_document(payload, "text/plain"))).encode()
    with running_rig(scan_step=RequestScanner()) as rig:
        reply = rig.call("POST", "/v1/messages", body, {**JSON, **ANTHROPIC})
        lines = rig.wait_log(1)
        records = rig.mock.records()
    assert reply.status == 400 and records == []
    message = json.loads(reply.body)["error"]["message"]
    assert blocks.MEDIA_SECRET in message and "as text" in message
    for text in (reply.body.decode(), lines[0]):
        assert OPENAI not in text and payload[:24] not in text
    assert json.loads(lines[0])["reason"] == blocks.MEDIA_SECRET
