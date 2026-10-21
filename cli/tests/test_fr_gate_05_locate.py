"""FR-GATE-05: finding the character span of a value, or the span to cut out of an array, by its path in the tree.

Unit tests of Harpocrates.gate.jsontext.locate. Nothing here holds a secret.
"""
from __future__ import annotations

import itertools
import json
import random
import time

import pytest

from Harpocrates.gate.jsontext import apply_edits, locate

DOC = ('  {"a": "one", "b" : [ 1 ,\n "two\\"]}", {"c": [ ] , "d":"é漢\\u00e9"} ] ,\t"e":{"f":{"g":"deep"}},"h":[[],[ "x" ]] }\n')


def spans(text: str, *paths: tuple, drop: tuple = ()) -> tuple[dict, list]:
    return locate(text, list(paths), list(drop))


def test_fr_gate_05_locate_finds_the_exact_literal_of_a_value_at_any_depth() -> None:
    tree = json.loads(DOC)
    paths = [("a",), ("b", 1), ("b", 2, "d"), ("e", "f", "g"), ("h", 1, 0), ("b", 0), ("e",), ("b", 2, "c")]
    found, _ = spans(DOC, *paths)
    for path in paths:
        start, end = found[path]
        value = tree
        for step in path:
            value = value[step]
        assert json.loads(DOC[start:end]) == value, path
    assert DOC[slice(*found[("a",)])] == '"one"' and DOC[slice(*found[("b", 1)])] == '"two\\"]}"'


def test_fr_gate_05_locate_counts_characters_not_bytes() -> None:
    text = '{"k": "日本語 😀", "v": "target"}'
    start, end = spans(text, ("v",))[0][("v",)]
    assert text[start:end] == '"target"'


def test_fr_gate_05_replacing_located_literals_changes_nothing_else() -> None:
    found, _ = spans(DOC, ("a",), ("e", "f", "g"), ("h", 1, 0))
    edits = [(s, e, '"X"') for s, e in found.values()]
    out = apply_edits(DOC, edits)
    expected = json.loads(DOC)
    expected["a"] = "X"
    expected["e"]["f"]["g"] = "X"
    expected["h"][1][0] = "X"
    assert json.loads(out) == expected
    assert out.replace('"X"', "") == DOC.replace('"one"', "").replace('"deep"', "").replace('"x"', "")  # all else identical


def test_fr_gate_05_a_path_that_is_not_in_the_document_is_an_error() -> None:
    for path in (("zzz",), ("b", 9), ("a", "x"), ("b", "x"), (0,)):
        with pytest.raises(LookupError):
            spans(DOC, path)


def cut(text: str, indexes: list[int], at: tuple = ()) -> str:
    _, removals = locate(text, [], [at + (i,) for i in indexes])
    return apply_edits(text, [(s, e, "") for s, e in removals])


@pytest.mark.parametrize("layout", ["[1,2,3,4,5]", "[ 1 , 2 ,\n3,\t4 ,5 ]", "[\n  1,\n  2,\n  3,\n  4,\n  5\n]"])
def test_fr_gate_05_dropping_array_elements_leaves_valid_json_with_the_others_untouched(layout: str) -> None:
    items = [1, 2, 3, 4, 5]
    for size in range(0, 6):
        for chosen in itertools.combinations(range(5), size):
            out = cut(layout, list(chosen))
            assert json.loads(out) == [v for i, v in enumerate(items) if i not in chosen], (layout, chosen)


def test_fr_gate_05_dropping_every_element_leaves_an_empty_array_in_its_place() -> None:
    assert json.loads(cut('{"x": [ "a" , "b" ], "y": 1}', [0, 1], ("x",))) == {"x": [], "y": 1}
    assert json.loads(cut('{"x": ["a"]}', [0], ("x",))) == {"x": []}


def test_fr_gate_05_dropping_elements_in_two_arrays_and_replacing_a_value_at_once() -> None:
    text = '{"m": [{"k": [1, 2, 3]}, {"k": [4, 5]}, "tail"], "z": "val"}'
    found, removals = locate(text, [("z",)], [("m", 0, "k", 1), ("m", 1), ("m", 2)])
    out = apply_edits(text, [(s, e, "") for s, e in removals] + [(*found[("z",)], '"new"')])
    assert json.loads(out) == {"m": [{"k": [1, 3]}], "z": "new"}


def test_fr_gate_05_random_documents_agree_with_the_parser() -> None:
    rng = random.Random(5)
    for _ in range(200):
        items = [{"t": "x" * rng.randint(0, 5), "n": rng.random()} for _ in range(rng.randint(1, 8))]
        text = json.dumps({"items": items}, indent=rng.choice([None, 1, 3]), separators=rng.choice([None, (",", ":")]))
        drop = [i for i in range(len(items)) if rng.random() < 0.5]
        out = cut(text, drop, ("items",))
        assert json.loads(out) == {"items": [v for i, v in enumerate(items) if i not in drop]}


def test_fr_gate_05_locate_a_value_near_the_end_of_a_megabyte_is_quick() -> None:
    filler = [{"role": "user", "content": [{"type": "text", "text": "word " * 200}]} for _ in range(900)]
    text = json.dumps({"messages": filler + [{"role": "user", "content": "last"}]})
    assert len(text) > 900_000
    started = time.perf_counter()
    found, _ = locate(text, [("messages", 900, "content")], [])
    elapsed = time.perf_counter() - started
    assert text[slice(*found[("messages", 900, "content")])] == '"last"' and elapsed < 0.25
