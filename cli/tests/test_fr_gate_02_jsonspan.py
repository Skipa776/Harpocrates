"""FR-GATE-02 (decision 15) and FR-GATE-05: strict JSON parsing, literal encoding and byte-exact splicing.

Nothing here holds a secret. Unit tests of Harpocrates.gate.jsontext's parsing and splicing.
"""
from __future__ import annotations

import json

import pytest

from Harpocrates.gate.jsontext import (
    MAX_DEPTH,
    BodyParseError,
    apply_edits,
    encode_literal,
    nesting_depth,
    parse_strict,
    parse_text_strict,
)


def nested(depth: int) -> bytes:
    return b"[" * depth + b"]" * depth


def test_fr_gate_02_parse_strict_returns_the_text_and_the_tree() -> None:
    text, tree = parse_strict(b'{"a": [1, 2.50, "x\\u00e9"], "b": {"c": null}}')
    assert text == '{"a": [1, 2.50, "x\\u00e9"], "b": {"c": null}}'
    assert tree == {"a": [1, 2.5, "xé"], "b": {"c": None}}


def test_fr_gate_02_malformed_blocked_parse_strict_refuses_what_decision_15_lists() -> None:
    for bad in (b"\xff\xfe{}", b'{"a": "\xc3("}', b"\xef\xbb\xbf{}", b"", b"{", b'{"a": 1,}', b"{'a': 1}", b'{"a": 1} x',
                b'{"a": NaN}', b'{"a": Infinity}', b'{"a": -Infinity}', b'{"a": "\x01"}', b'{"a": 1e999999999999999999999}x'):
        with pytest.raises(BodyParseError):
            parse_strict(bad)


def test_fr_gate_02_duplicate_key_blocked_at_any_depth() -> None:
    for bad in (b'{"a": 1, "a": 2}', b'{"x": {"a": 1, "b": 2, "a": 3}}', b'[{"k": 1}, {"k": 1, "k": 1}]',
                b'{"a\\u0062": 1, "ab": 2}'):  # the same key spelled two ways is still the same key
        with pytest.raises(BodyParseError):
            parse_strict(bad)
    parse_strict(b'{"a": {"a": 1}, "b": [{"a": 1}]}')  # the same name in different objects is fine


def test_fr_gate_02_nesting_deeper_than_128_blocked() -> None:
    assert MAX_DEPTH == 128
    parse_strict(nested(128))
    parse_strict(b'{"a":' * 127 + b"[]" + b"}" * 127)  # 128 deep
    for bad in (nested(129), b'{"a":' * 128 + b"[]" + b"}" * 128, nested(5000), b"[" * 1_000_000):
        with pytest.raises(BodyParseError):
            parse_strict(bad)


def test_fr_gate_02_nesting_depth_counts_brackets_outside_strings_only() -> None:
    assert nesting_depth('{"a": "[[[[{{{{"}') == 1
    assert nesting_depth('{"a": ["\\"[[", [1, {"b": []}]]}') == 5
    assert nesting_depth("7") == 0 and nesting_depth("[]") == 1 and nesting_depth("[[]][[]]") == 2


def test_fr_gate_02_parse_text_strict_is_the_same_rules_for_json_inside_a_string() -> None:
    assert parse_text_strict('{"cmd": "ls"}') == {"cmd": "ls"}
    for bad in ('{"cmd": ', '{"a": 1, "a": 2}', "NaN", "[" * 200 + "]" * 200):
        with pytest.raises(BodyParseError):
            parse_text_strict(bad)


@pytest.mark.parametrize("text", ["plain", 'quote " and \\ backslash', "line\nbreak\ttab\r", "\x00\x01\x1f", "é漢字😀",
                                  "  \x7f", "\ud800", "a\ud83d", ""])
def test_fr_gate_02_encode_literal_makes_a_valid_json_string_for_any_text(text: str) -> None:
    literal = encode_literal(text)
    assert literal.startswith('"') and literal.endswith('"')
    assert literal.isascii() or all(ch not in literal[1:-1] for ch in '"\\\n\r\t\x00')
    literal.encode("utf-8")  # no lone surrogate may be left raw: the body must stay valid UTF-8
    assert json.loads(literal) == text


def test_fr_gate_02_encode_literal_escapes_quotes_backslashes_and_control_characters_only() -> None:
    assert encode_literal("é漢<>&'/") == '"é漢<>&\'/"'
    assert encode_literal('a"b\\c\nd') == '"a\\"b\\\\c\\nd"'
    assert encode_literal("\x01") == '"\\u0001"'


def test_fr_gate_05_apply_edits_replaces_only_the_given_spans() -> None:
    text = '{"a": "one", "b": [1, 2], "c": "three"}'
    one, three = text.index('"one"'), text.index('"three"')
    out = apply_edits(text, [(three, three + 7, '"3"'), (one, one + 5, '"1"')])  # given out of order
    assert out == '{"a": "1", "b": [1, 2], "c": "3"}'
    assert apply_edits(text, []) is text  # nothing to do: the same object, not a copy


def test_fr_gate_05_apply_edits_refuses_overlapping_or_out_of_range_spans() -> None:
    for bad in ([(0, 5, "x"), (3, 8, "y")], [(4, 2, "x")], [(0, 999, "x")], [(-1, 2, "x")]):
        with pytest.raises(ValueError):
            apply_edits("0123456789", bad)
    assert apply_edits("0123456789", [(0, 0, "ab"), (0, 3, "X")]) == "abX3456789"  # an insertion may touch a replacement
