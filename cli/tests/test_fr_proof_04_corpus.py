"""FR-PROOF-04 and FR-GATE-02: the synthetic request corpus (bench/gate_corpus.py).

One request per block, item and tool type covered by FR-GATE-02 (docs/design.md), in both provider formats, each carrying a
generated canary. The corpus is test data for the future test_fr_gate_02_* parser tests; the canary harness sends it
to the mock provider in direct mode. Every canary is generated; nothing here is a real credential.
"""
from __future__ import annotations

import json
import re

import pytest

from bench import canary_match as match
from bench import gate_corpus as corpus

CASES = corpus.build_corpus(0)
EXPECTS = {"redact", "block", "pass"}

# Frozen FR-GATE-02 coverage contract (docs/design.md), independent of the corpus builders.
PARSER_CASE_NAMES = frozenset({
    "test_fr_gate_02_anthropic_code_execution_tool_result",
    "test_fr_gate_02_anthropic_compaction",
    "test_fr_gate_02_anthropic_container_upload",
    "test_fr_gate_02_anthropic_document_content",
    "test_fr_gate_02_anthropic_document_pdf",
    "test_fr_gate_02_anthropic_document_text",
    "test_fr_gate_02_anthropic_document_url",
    "test_fr_gate_02_anthropic_image",
    "test_fr_gate_02_anthropic_mcp_tool_listing",
    "test_fr_gate_02_anthropic_mcp_tool_result",
    "test_fr_gate_02_anthropic_mcp_tool_use",
    "test_fr_gate_02_anthropic_redacted_thinking",
    "test_fr_gate_02_anthropic_search_result",
    "test_fr_gate_02_anthropic_server_tool_use",
    "test_fr_gate_02_anthropic_system_role_text",
    "test_fr_gate_02_anthropic_text",
    "test_fr_gate_02_anthropic_text_editor_tool_result",
    "test_fr_gate_02_anthropic_thinking",
    "test_fr_gate_02_anthropic_tool_addition",
    "test_fr_gate_02_anthropic_tool_removal",
    "test_fr_gate_02_anthropic_tool_result_blocks",
    "test_fr_gate_02_anthropic_tool_result_string",
    "test_fr_gate_02_anthropic_tool_search_tool_result",
    "test_fr_gate_02_anthropic_tool_use",
    "test_fr_gate_02_anthropic_tools_custom",
    "test_fr_gate_02_anthropic_tools_mcp_toolset",
    "test_fr_gate_02_anthropic_tools_server",
    "test_fr_gate_02_anthropic_tools_unknown_type_blocked",
    "test_fr_gate_02_anthropic_unknown_block_blocked",
    "test_fr_gate_02_anthropic_web_fetch_tool_result",
    "test_fr_gate_02_anthropic_web_search_tool_result",
    "test_fr_gate_02_responses_additional_tools",
    "test_fr_gate_02_responses_apply_patch_call",
    "test_fr_gate_02_responses_apply_patch_call_output",
    "test_fr_gate_02_responses_code_interpreter_call",
    "test_fr_gate_02_responses_compaction",
    "test_fr_gate_02_responses_computer_call",
    "test_fr_gate_02_responses_computer_call_output",
    "test_fr_gate_02_responses_conversation_blocked",
    "test_fr_gate_02_responses_custom_tool_call",
    "test_fr_gate_02_responses_custom_tool_call_output",
    "test_fr_gate_02_responses_file_search_call",
    "test_fr_gate_02_responses_function_call",
    "test_fr_gate_02_responses_function_call_output",
    "test_fr_gate_02_responses_image_generation_call",
    "test_fr_gate_02_responses_input_file",
    "test_fr_gate_02_responses_input_image",
    "test_fr_gate_02_responses_input_text",
    "test_fr_gate_02_responses_item_reference",
    "test_fr_gate_02_responses_local_shell_call",
    "test_fr_gate_02_responses_local_shell_call_output",
    "test_fr_gate_02_responses_mcp_approval_request",
    "test_fr_gate_02_responses_mcp_approval_response",
    "test_fr_gate_02_responses_mcp_call",
    "test_fr_gate_02_responses_mcp_list_tools",
    "test_fr_gate_02_responses_message_string",
    "test_fr_gate_02_responses_output_text",
    "test_fr_gate_02_responses_previous_response_id_seen_passes",
    "test_fr_gate_02_responses_previous_response_id_unseen_blocked",
    "test_fr_gate_02_responses_reasoning",
    "test_fr_gate_02_responses_refusal",
    "test_fr_gate_02_responses_shell_call",
    "test_fr_gate_02_responses_shell_call_output",
    "test_fr_gate_02_responses_tool_search",
    "test_fr_gate_02_responses_tools_builtin",
    "test_fr_gate_02_responses_tools_custom",
    "test_fr_gate_02_responses_tools_function",
    "test_fr_gate_02_responses_tools_mcp",
    "test_fr_gate_02_responses_tools_namespace",
    "test_fr_gate_02_responses_tools_unknown_type_blocked",
    "test_fr_gate_02_responses_unknown_item_blocked",
    "test_fr_gate_02_responses_web_search_call",
})


def test_fr_proof_04_corpus_covers_every_parser_spec_row() -> None:
    have = {c.test_name for c in CASES if c.test_name}
    assert PARSER_CASE_NAMES <= have, sorted(PARSER_CASE_NAMES - have)


def test_fr_proof_04_corpus_case_ids_and_test_names_are_unique() -> None:
    ids = [c.case_id for c in CASES]
    names = [c.test_name for c in CASES if c.test_name]
    assert len(ids) == len(set(ids)) and len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-z0-9_]+", i) for i in ids)


def test_fr_proof_04_corpus_is_the_same_for_the_same_seed_and_differs_for_another() -> None:
    again = corpus.build_corpus(0)
    assert [(c.case_id, c.canary_value, json.dumps(c.body)) for c in CASES] == [
        (c.case_id, c.canary_value, json.dumps(c.body)) for c in again]
    other = corpus.build_corpus(1)
    assert [c.case_id for c in other] == [c.case_id for c in CASES]
    assert [c.canary_value for c in other if c.canary_value] != [c.canary_value for c in CASES if c.canary_value]


def test_fr_proof_04_corpus_every_planted_canary_is_in_its_body_at_the_named_field() -> None:
    for case in CASES:
        text = json.dumps(case.body)
        if case.canary_value is None:
            assert case.canary_kind is None and case.field_path is None, case.case_id
            assert case.expect == "pass" or (case.expect == "block" and case.section == "policy"), case.case_id
            continue
        hits = match.locate(text, case.canary_value)
        assert hits, case.case_id
        assert case.field_path in {h.path for h in hits}, (case.case_id, case.field_path)
        others = [c.canary_value for c in CASES if c.canary_value and c.canary_value != case.canary_value]
        assert not any(match.locate(text, other) for other in others), case.case_id  # one canary per request


def test_fr_proof_04_corpus_canaries_are_generated_and_both_classes_appear() -> None:
    kinds = {c.canary_kind for c in CASES if c.canary_kind}
    assert kinds <= match.KNOWN_FORMAT_KINDS | match.ML_ONLY_KINDS
    classes = {match.kind_class(k) for k in kinds}
    assert classes == {"known-format", "ml-only"} and len(kinds) >= 10
    assert all(c.canary_class == match.kind_class(c.canary_kind) for c in CASES if c.canary_kind)


def test_fr_proof_04_corpus_expectations_follow_the_spec_classes() -> None:
    by_id = {c.case_id: c for c in CASES}
    assert {c.expect for c in CASES} == EXPECTS
    blocked = {"anthropic_thinking", "anthropic_compaction", "anthropic_web_search_tool_result", "anthropic_mcp_tool_listing",
               "anthropic_unknown_block_blocked", "anthropic_tools_unknown_type_blocked", "responses_reasoning",
               "responses_unknown_item_blocked", "responses_tools_unknown_type_blocked", "responses_conversation_blocked"}
    assert blocked <= {c.case_id for c in CASES if c.expect == "block"}
    assert all(by_id[i].canary_value is not None for i in blocked - {"responses_conversation_blocked"})
    passed = {"anthropic_redacted_thinking", "anthropic_container_upload", "anthropic_tool_removal", "anthropic_tools_server",
              "anthropic_tools_mcp_toolset", "responses_item_reference", "responses_compaction",
              "responses_computer_call_output", "anthropic_tool_search_tool_result"}
    assert passed <= {c.case_id for c in CASES if c.expect == "pass"}
    redacted = {"anthropic_text", "anthropic_tool_use", "anthropic_tool_result_string", "responses_function_call",
                "responses_function_call_output", "responses_local_shell_call", "responses_tools_function"}
    assert redacted <= {c.case_id for c in CASES if c.expect == "redact"}


def test_fr_proof_04_corpus_routes_and_envelopes_match_the_format() -> None:
    for case in CASES:
        assert case.fmt in ("anthropic", "responses")
        if case.fmt == "anthropic":
            assert case.route == "/v1/messages" and {"model", "max_tokens", "messages"} <= set(case.body), case.case_id
        else:
            assert case.route == "/v1/responses" and {"model", "input"} <= set(case.body), case.case_id
        assert json.loads(json.dumps(case.body)) == case.body  # plain JSON all the way down
        assert case.source in ("spec", "captured")


def test_fr_proof_04_corpus_repr_never_holds_a_value() -> None:
    secrets = [c.canary_value for c in CASES if c.canary_value]
    text = repr(CASES) + str([c.summary() for c in CASES])
    assert not any(s in text for s in secrets)
    assert "canary_value" not in repr(CASES[0]) and set(CASES[0].summary()) >= {"case_id", "expect", "canary_kind"}


@pytest.mark.parametrize("case_id", ["responses_function_call_arguments_bad_json", "responses_function_call_arguments_nested_json",
                                     "responses_function_call_arguments_key_blocked", "anthropic_text_cited_text",
                                     "anthropic_system_string", "anthropic_system_blocks", "responses_instructions"])
def test_fr_proof_04_corpus_has_the_extra_cases_the_rewrite_rules_need(case_id: str) -> None:
    case = next(c for c in CASES if c.case_id == case_id)
    assert case.canary_value is not None and match.locate(json.dumps(case.body), case.canary_value)


def test_fr_proof_04_corpus_json_in_string_cases_hold_the_canary_only_inside_the_string() -> None:
    case = next(c for c in CASES if c.case_id == "responses_function_call")
    item = next(i for i in case.body["input"] if i.get("type") == "function_call")
    assert isinstance(item["arguments"], str) and json.loads(item["arguments"])  # a JSON string, parseable
    assert "{json}" in case.field_path
    key_case = next(c for c in CASES if c.case_id == "responses_function_call_arguments_key_blocked")
    assert key_case.expect == "block" and key_case.field_path.endswith("<key>")


def test_fr_proof_04_corpus_marks_the_cases_a_recorded_agent_session_confirmed() -> None:
    ids = {c.case_id for c in CASES}
    assert corpus.CAPTURED <= ids and len(corpus.CAPTURED) >= 15
    by_id = {c.case_id: c for c in CASES}
    for case_id in corpus.CAPTURED:
        assert by_id[case_id].source == "captured", case_id
    assert {c.source for c in CASES} == {"spec", "captured"}
    assert by_id["anthropic_system_role_string"].source == "captured"  # sent by Claude Code, absent from the spec table
    assert by_id["anthropic_thinking"].source == "captured" and by_id["anthropic_compaction"].source == "spec"
    assert corpus.CAPTURED_RUN.endswith("-g2")


def test_fr_proof_04_corpus_builds_for_many_seeds_including_those_with_short_passwords() -> None:
    for seed in range(2, 14):
        built = corpus.build_corpus(seed)
        assert all(match.locate(json.dumps(c.body), c.canary_value) for c in built if c.canary_value), seed
