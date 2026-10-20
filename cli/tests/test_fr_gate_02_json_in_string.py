"""FR-GATE-02 decision 2: JSON inside a string (tool-call arguments). Parse it, redact the string leaves, splice them back
and encode the whole once; scan the keys; scan what does not parse as plain text. Unit tests of
Harpocrates.gate.jsonstring and the block codes of Harpocrates.gate.blocks."""
from __future__ import annotations

import json

import pytest
from gate_canaries import PLACEHOLDER, known_format

from Harpocrates.gate import jsonstring
from Harpocrates.gate.blocks import (
    HEADER_SECRET,
    HISTORY,
    JSON_ARGS,
    KEY_SECRET,
    PARSE,
    PROVIDER_TEXT,
    UNKNOWN_TYPE,
    RefusalError,
)
from Harpocrates.gate.errors import is_block_code
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.jsonstring import redact_json_string

CANARY = known_format()["github_token"]


def redacted(text: str):
    return redact_json_string(text, Finder())


def test_fr_gate_02_every_scan_step_block_code_is_a_safe_code_for_the_user() -> None:
    codes = [PARSE, UNKNOWN_TYPE, PROVIDER_TEXT, KEY_SECRET, HEADER_SECRET, HISTORY, JSON_ARGS]
    assert all(is_block_code(code) for code in codes) and len(set(codes)) == 7
    error = RefusalError(KEY_SECRET)
    assert error.code == KEY_SECRET and CANARY not in str(error)


def test_fr_gate_02_clean_arguments_are_left_alone() -> None:
    assert redacted('{"cmd": "ls -la", "workdir": "/repo"}') is None
    assert redacted("[1, 2, {\"a\": \"b\"}]") is None
    assert redacted("not json at all") is None


def test_fr_gate_02_a_secret_in_a_leaf_is_replaced_and_the_arguments_still_parse() -> None:
    raw = json.dumps({"cmd": f"echo {CANARY}", "workdir": "/repo", "n": 3})
    out = redacted(raw)
    assert out is not None and CANARY not in out
    parsed = json.loads(out)
    assert PLACEHOLDER.sub("@", parsed["cmd"]) == "echo @" and parsed["workdir"] == "/repo" and parsed["n"] == 3
    assert out == raw.replace(CANARY, PLACEHOLDER.search(out).group())  # nothing else moved


def test_fr_gate_02_escapes_next_to_a_secret_do_not_break_the_arguments() -> None:
    """A span cut out of the raw string could end between a backslash and its letter. Redacting the decoded leaf cannot."""
    raw = json.dumps({"cmd": f'printf "%s\\n" "{CANARY}"\nline two\ttab', "x": "é "})
    out = redacted(raw)
    assert out is not None
    parsed = json.loads(out)
    assert PLACEHOLDER.sub("@", parsed["cmd"]) == 'printf "%s\\n" "@"\nline two\ttab' and parsed["x"] == "é "


def test_fr_gate_02_json_nested_in_json_strings_is_followed_three_levels_deep() -> None:
    level3 = json.dumps({"deep": f"k {CANARY}"})
    level2 = json.dumps({"inner": level3})
    level1 = json.dumps({"payload": level2})
    out = redacted(level1)
    assert out is not None and CANARY not in out
    assert PLACEHOLDER.search(json.loads(json.loads(json.loads(out)["payload"])["inner"])["deep"])
    level4 = json.dumps({"payload": json.dumps({"a": json.dumps({"b": json.dumps({"c": f"k {CANARY}"})})})})
    out4 = redacted(level4)  # deeper than three levels is scanned as plain text: still redacted, the nesting kept valid
    assert out4 is not None and CANARY not in out4 and json.loads(out4)


def test_fr_gate_02_arguments_that_do_not_parse_are_scanned_as_plain_text() -> None:
    out = redacted('{"cmd": "echo ' + CANARY + '", "tail": ')
    assert out is not None and CANARY not in out
    assert PLACEHOLDER.sub("@", out) == '{"cmd": "echo @", "tail": '
    dup = '{"a": "x", "a": "echo ' + CANARY + '"}'  # a repeated key is not strict JSON either
    assert PLACEHOLDER.sub("@", redacted(dup)) == '{"a": "x", "a": "echo @"}'


def test_fr_gate_02_json_that_is_a_scalar_is_plain_text() -> None:
    assert redacted(json.dumps(CANARY)) is not None  # the string "<canary>" with its quotes
    assert redacted("12345") is None


def test_fr_gate_02_secret_in_json_key_blocks() -> None:
    for raw in (json.dumps({CANARY: 1}), json.dumps({"a": {"b": [{CANARY: True}]}}), json.dumps({"p": json.dumps({CANARY: 1})})):
        with pytest.raises(RefusalError) as caught:
            redacted(raw)
        assert caught.value.code == KEY_SECRET


def test_fr_gate_02_arguments_must_still_parse_after_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsonstring, "encode_literal", lambda text: text)  # a broken encoder: unquoted, unescaped
    with pytest.raises(RefusalError) as caught:
        redacted(json.dumps({"cmd": f"echo {CANARY}"}))
    assert caught.value.code == JSON_ARGS


def test_fr_gate_02_the_detector_is_called_once_per_distinct_leaf_even_when_nested() -> None:
    calls: list[str] = []

    def detect(text: str) -> list:
        calls.append(text)
        return []

    raw = json.dumps({"a": "same", "b": "same", "c": json.dumps({"d": "same"})})
    assert redact_json_string(raw, Finder(detector=detect)) is None
    assert calls.count("same") == 1
