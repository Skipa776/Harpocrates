"""FR-GATE-02, FR-GATE-10: one test per case of the G2 corpus (bench/gate_corpus*.py: 94 cases; the spec said 93), named as the parser spec names
it (test_fr_gate_02_<format>_<type>).

Each case is run twice through the request scan step:
  * with an oracle detector that finds exactly the planted canary, so the test proves the gate reaches the field, replaces
    the canary with a well-formed placeholder, leaves every other byte alone, and blocks what must block;
  * with the real detector, which must do the same for every known-format canary. An ML-only canary the detector does not
    recognise in that context (a bare password in a sentence) must leave the request exactly as it was; the number of such
    cases is bounded at the end of the file and reported, not hidden.
"""
from __future__ import annotations

import json
import re

from gate_canaries import PLACEHOLDER
from gate_scan_support import blocked, finding, forwarded, run, scanner

from bench import canary_match, gate_corpus
from Harpocrates.gate import blocks
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.routes import classify
from Harpocrates.gate.scan import Forward, RequestHead
from Harpocrates.read import Redactor

CASES = gate_corpus.build_corpus(0)
assert len(CASES) == 94
# What G4 does where the G2 corpus still holds the pre-decision expectation (docs/gate-scan-spec.md):
#  * thinking and redacted_thinking blocks are dropped, not blocked or passed (decision 5, FR-GATE-10);
#  * previous_response_id blocks whether or not the gate has seen the id: v1 has no relay tee (decision 1).
EXPECT = {"anthropic_thinking": "drop", "anthropic_redacted_thinking": "drop",
          "responses_previous_response_id_seen_passes": "block"}
CODES = {
    "anthropic_compaction": blocks.PROVIDER_TEXT, "anthropic_web_search_tool_result": blocks.PROVIDER_TEXT,
    "anthropic_mcp_tool_listing": blocks.PROVIDER_TEXT, "responses_reasoning": blocks.PROVIDER_TEXT,
    "anthropic_unknown_block_blocked": blocks.UNKNOWN_TYPE, "anthropic_tools_unknown_type_blocked": blocks.UNKNOWN_TYPE,
    "responses_unknown_item_blocked": blocks.UNKNOWN_TYPE, "responses_tools_unknown_type_blocked": blocks.UNKNOWN_TYPE,
    "responses_function_call_arguments_key_blocked": blocks.KEY_SECRET,
    "responses_previous_response_id_seen_passes": blocks.HISTORY, "responses_previous_response_id_unseen_blocked": blocks.HISTORY,
    "responses_conversation_blocked": blocks.HISTORY,
}
FINDING_DEPENDENT = {blocks.PROVIDER_TEXT, blocks.KEY_SECRET}  # blocks that need the detector to find something
REAL = Finder()  # one Finder (one model load) for the whole module
ML_ONLY_UNDETECTED: list[str] = []
THINKING = {"thinking", "redacted_thinking"}


def head_for(case) -> RequestHead:
    headers = (("content-type", "application/json"),) + (
        (("anthropic-version", "2023-06-01"),) if case.fmt == "anthropic" else ())
    return RequestHead(headers, case.route, "")


def present(text: str, canary: str) -> bool:
    """Whether the canary, or one long line of it, is in a text (also when JSON escaped, once or nested)."""
    forms = [canary]
    for _ in range(3):
        forms.append(json.dumps(forms[-1])[1:-1])
    return any(needle in text for form in forms for kind, needle in canary_match.needles(form) if kind in ("plain", "line"))


def leaves(node: object, path: tuple = ()) -> dict:
    """Every scalar of a JSON tree, by path."""
    if isinstance(node, dict):
        return {k: v for key, value in node.items() for k, v in leaves(value, path + (key,)).items()}
    if isinstance(node, list):
        return {k: v for i, value in enumerate(node) for k, v in leaves(value, path + (i,)).items()}
    return {path: node}


def drop_thinking(body: dict) -> dict:
    """The body as G4 should leave it: no thinking blocks, and no message left empty."""
    out = json.loads(json.dumps(body))
    kept = []
    for message in out.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            remaining = [b for b in content if not (isinstance(b, dict) and b.get("type") in THINKING)]
            if content and not remaining:
                continue
            message["content"] = remaining
        kept.append(message)
    if "messages" in out:
        out["messages"] = kept
    return out


def oracle_values(case) -> list[str]:
    """The strings the oracle treats as secret for a case: what the Redactor would replace for the canary's finding."""
    return list(Redactor().secret_values(case.canary_value, findings=[finding(case.canary_value)]))


def check_redacted(case, original: bytes, out: bytes) -> None:
    """The canary is gone, placeholders are well formed, and the only leaves that changed are the ones that held it."""
    text = out.decode("utf-8")
    assert not present(text, case.canary_value)
    assert PLACEHOLDER.search(text)
    before, after = leaves(json.loads(original)), leaves(json.loads(text))
    assert before.keys() == after.keys()  # same shape: nothing added, removed or moved
    changed = [k for k in before if before[k] != after[k]]
    assert changed
    for key in changed:
        assert PLACEHOLDER.search(after[key]), key  # a leaf that changed now holds a placeholder


def check_bytes_outside_spans(case, original: bytes, out: bytes) -> None:
    """With the oracle, everything outside the replaced spans is byte for byte what the client sent."""
    forms = []
    for value in oracle_values(case):
        escaped = value
        for _ in range(4):
            forms.append(escaped)
            escaped = json.dumps(escaped)[1:-1]
    pattern = "|".join(re.escape(f) for f in sorted(set(forms), key=len, reverse=True))
    assert re.split(pattern, original.decode()) == PLACEHOLDER.split(out.decode())


def check_case(case, scan, with_oracle: bool) -> None:
    body = json.dumps(case.body).encode()
    result = run(scan, body, classify("POST", case.route, dict(head_for(case).headers)), head_for(case))
    kind = EXPECT.get(case.case_id, case.expect)
    canary = case.canary_value
    if kind == "block":
        if CODES[case.case_id] in FINDING_DEPENDENT and isinstance(result, Forward) and not with_oracle:
            assert result.body == body and case.canary_class == "ml-only", case.case_id  # the detector missed the canary
            if case.case_id not in ML_ONLY_UNDETECTED:
                ML_ONLY_UNDETECTED.append(case.case_id)
            return
        assert blocked(result) == CODES[case.case_id]
    elif kind == "pass":
        assert forwarded(result) == body  # unchanged, down to the byte (a credential for a connector stays as sent)
    elif kind == "drop":
        out = forwarded(result)
        assert not canary or not present(out.decode(), canary)
        assert json.loads(out) == drop_thinking(case.body)
    else:
        out = forwarded(result)
        if present(out.decode(), canary):  # only the real detector may miss, and only an ML-only canary in bare context
            assert not with_oracle and case.canary_class == "ml-only" and out == body, case.case_id
            if case.case_id not in ML_ONLY_UNDETECTED:
                ML_ONLY_UNDETECTED.append(case.case_id)
            return
        check_redacted(case, body, out)
        if with_oracle:
            check_bytes_outside_spans(case, body, out)


def make_test(case):
    def test() -> None:
        check_case(case, scanner([case.canary_value] if case.canary_value else []), with_oracle=True)
        check_case(case, scanner(finder=REAL), with_oracle=False)

    test.__name__ = case.test_name or gate_corpus.TEST_PREFIX + case.case_id
    test.__doc__ = f"{case.case_id}: {case.fmt} {case.section}/{case.block_type}, canary {case.canary_kind}, expect {case.expect}"
    return test


for _case in CASES:
    globals()[_case.test_name or gate_corpus.TEST_PREFIX + _case.case_id] = make_test(_case)


def test_fr_gate_02_the_corpus_has_94_cases_and_the_g4_expectations_name_real_cases() -> None:
    ids = {c.case_id for c in CASES}
    assert len(CASES) == 94 and set(EXPECT) <= ids and set(CODES) <= ids
    assert {c.case_id for c in CASES if EXPECT.get(c.case_id, c.expect) == "block"} == set(CODES)


def test_fr_gate_02_ml_only_corpus_misses_are_a_bounded_detector_limit() -> None:
    """An ML-only canary in bare context (a password inside a sentence) is a detector recall question, FR-CORE-03, not a
    gate one: the gate forwards such a request untouched. The count is bounded so a regression is seen, never hidden."""
    for case in CASES:  # make sure every case ran in this process, whatever the test order
        if case.canary_class == "ml-only" and case.expect == "redact" and case.case_id not in ML_ONLY_UNDETECTED:
            check_case(case, scanner(finder=REAL), with_oracle=False)
    assert len(ML_ONLY_UNDETECTED) <= 15, sorted(ML_ONLY_UNDETECTED)
    print(f"ml-only corpus canaries the detector did not recognise in context: {len(ML_ONLY_UNDETECTED)} of "
          f"{sum(c.canary_class == 'ml-only' and c.expect == 'redact' for c in CASES)}")
