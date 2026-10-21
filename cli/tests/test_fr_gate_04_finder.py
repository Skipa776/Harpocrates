"""FR-GATE-04, FR-GATE-05, SEC-02: the finder turns text into spans (cached) and placeholders (session-keyed).

Every secret is a generated canary (gate_canaries.py). The detector is the real one unless a test hands in its own.
"""
from __future__ import annotations

import pytest
from gate_canaries import PLACEHOLDER, canaries, known_format

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.gate import finder as finder_module
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.spancache import SpanCache, cache_key
from Harpocrates.read import Redactor

KNOWN = known_format()


def fake_finding(token: str, category: str = "generic_secret") -> Finding:
    return Finding(type="X", file=None, line=1, snippet="", entropy=0.0, evidence=EvidenceType.REGEX, severity=None,
                   confidence=1.0, token=token, category=category)


def counting(values: list[str]):
    """A detector that finds the given values and counts its calls."""
    calls: list[str] = []

    def detect(text: str) -> list:
        calls.append(text)
        return [fake_finding(v) for v in values if v in text]

    return detect, calls


@pytest.mark.parametrize("kind", sorted(KNOWN))
def test_fr_gate_04_the_real_detector_finds_every_known_format_canary_in_context(kind: str) -> None:
    value = KNOWN[kind]
    text = f"my deployment key is {value} and that is all\nnext line"
    out = Finder().redact(text, "R")
    assert out is not None and value not in out and PLACEHOLDER.search(out)
    assert out.startswith("my deployment key is ") and out.endswith(" and that is all\nnext line")


def test_fr_gate_04_clean_text_has_no_spans_and_redact_says_none() -> None:
    finder = Finder()
    assert finder.redact("just some ordinary words about a function", "R") is None
    assert finder.spans("just some ordinary words about a function", "R") == ()
    assert not finder.has_secret("hello", "S")


def test_fr_gate_04_every_occurrence_is_replaced_and_nothing_else_changes() -> None:
    detect, _ = counting(["SECRETVALUE123"])
    finder = Finder(detector=detect)
    text = "a SECRETVALUE123 b\nSECRETVALUE123,SECRETVALUE123 c"
    out = finder.redact(text, "R")
    assert out is not None and "SECRETVALUE123" not in out
    assert PLACEHOLDER.sub("@", out) == "a @ b\n@,@ c"
    assert len({m.group() for m in PLACEHOLDER.finditer(out)}) == 1  # one value, one placeholder


def test_fr_gate_04_a_longer_value_swallows_a_shorter_one_inside_it() -> None:
    detect, _ = counting(["ABCDEFGHIJ", "CDEF"])
    out = Finder(detector=detect).redact("x ABCDEFGHIJ y CDEF z", "R")
    assert out is not None and PLACEHOLDER.sub("@", out) == "x @ y @ z" and "CDEF" not in out.replace("<<HARPO", "")


def test_fr_gate_04_a_private_key_block_is_one_value() -> None:
    pem = canaries()["private_key"]
    out = Finder().redact(f"before\n{pem}\nafter", "R")
    assert out is not None and PLACEHOLDER.sub("@", out) == "before\n@\nafter"


def test_fr_gate_05_same_secret_same_placeholder_across_texts_and_turns() -> None:
    value = KNOWN["github_token"]
    finder = Finder()
    first = finder.redact(f"turn one: {value}", "R")
    second = finder.redact(f"a completely different sentence mentioning {value} later", "R")
    assert first is not None and second is not None
    assert PLACEHOLDER.search(first).group() == PLACEHOLDER.search(second).group()


def test_sec_02_new_gate_new_key() -> None:
    values = list(KNOWN.values())
    mine = [Finder().redact(f"x {v}", "R") for v in values]
    theirs = [Finder().redact(f"x {v}", "R") for v in values]
    again_same_finder = Finder()
    assert [again_same_finder.redact(f"x {v}", "R") for v in values] == [again_same_finder.redact(f"x {v}", "R") for v in values]
    assert mine != theirs  # eleven 16-bit hashes cannot all agree under two random keys, short of a 1-in-2^176 chance
    assert all(PLACEHOLDER.search(m) for m in mine)


def test_sec_02_cache_holds_no_values() -> None:
    values = list(KNOWN.values())
    cache = SpanCache()
    finder = Finder(cache=cache)
    texts = [f"prefix {v} suffix" for v in values]
    for text in texts:
        assert finder.redact(text, "R") is not None
    held = repr(cache._entries) + repr(list(cache._entries)) + repr(vars(cache))
    for v in values:
        for part in v.splitlines():
            if len(part) >= 8 and "-----" not in part:
                assert part not in held
    assert all(isinstance(k, bytes) and len(k) == 32 for k in cache._entries) and len(cache) == len(texts)
    assert all(all(isinstance(x, (int, str)) for span in spans for x in span) for spans in cache._entries.values())
    assert "prefix" not in held and "suffix" not in held


def test_fr_gate_05_the_second_time_a_leaf_is_seen_the_detector_is_not_called() -> None:
    detect, calls = counting(["SECRETVALUE123"])
    finder = Finder(detector=detect)
    first = finder.redact("abc SECRETVALUE123", "R")
    clean_one = finder.redact("nothing here", "R")
    again = finder.redact("abc SECRETVALUE123", "R")
    clean_again = finder.redact("nothing here", "R")
    assert first == again and clean_one is None and clean_again is None
    assert calls == ["abc SECRETVALUE123", "nothing here"]  # both the hit and the clean leaf were remembered


def test_fr_gate_05_the_field_class_is_part_of_the_cache_key() -> None:
    detect, calls = counting(["SECRETVALUE123"])
    finder = Finder(detector=detect)
    finder.spans("x SECRETVALUE123", "R")
    finder.spans("x SECRETVALUE123", "S")
    assert len(calls) == 2
    assert cache_key("R", "x") != cache_key("S", "x")


def test_fr_gate_05_a_cache_hit_gives_the_same_bytes_as_a_fresh_scan_with_the_same_key() -> None:
    shared = Redactor()
    warm, cold = Finder(redactor=shared), Finder(redactor=shared)
    text = f"export KEY={KNOWN['openai_key']} # and more"
    first = warm.redact(text, "R")
    assert first is not None and warm.redact(text, "R") == first == cold.redact(text, "R")
    assert len(warm.cache) == len(cold.cache) == 1


def test_fr_gate_03_detector_error_blocks_it_propagates_to_the_caller() -> None:
    def broken(text: str) -> list:
        raise RuntimeError("detector crashed")

    with pytest.raises(RuntimeError):
        Finder(detector=broken).redact("anything", "R")


def test_fr_gate_03_a_verifier_error_is_an_error_not_a_redact_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """The read tool redacts every candidate when the model cannot run; the gate fails the request instead (G4 spec:
    a scanner failure blocks)."""
    def no_model():
        raise OSError("model cannot load")

    monkeypatch.setattr(finder_module, "_gate_verifier", no_model)
    with pytest.raises(OSError):
        finder_module.strict_findings("password = 'x7Kq9LmT2vBn4Zw8Yc1R'")
