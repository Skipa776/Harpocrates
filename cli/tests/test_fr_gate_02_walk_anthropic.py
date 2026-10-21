"""FR-GATE-02, FR-GATE-10: which strings of an Anthropic Messages request are scanned, and as what (parser spec, "Anthropic
Messages"). Unit tests of Harpocrates.gate.walk_anthropic: it yields operations and decides nothing about secrets.

Marker strings (MARK-...) stand in for text; none is a credential."""
from __future__ import annotations

import pytest

from Harpocrates.gate.blocks import PARSE, UNKNOWN_TYPE, RefusalError
from Harpocrates.gate.walk_anthropic import walk
from Harpocrates.gate.walk_common import Drop, KeyText, Leaf


def ops(body: dict) -> list:
    return list(walk(body))


def texts(body: dict, cls: str = "R") -> list[str]:
    return [op.text for op in ops(body) if isinstance(op, Leaf) and op.cls == cls]


def req(*messages: dict, **extra: object) -> dict:
    return {"model": "m", "max_tokens": 1, "messages": list(messages), **extra}


ASK = {"role": "user", "content": "x"}


def user(*blocks: object) -> dict:
    return {"role": "user", "content": list(blocks)}


def test_fr_gate_02_plain_strings_are_leaves_and_identifiers_are_not() -> None:
    body = req({"role": "user", "content": "MARK-user"}, system="MARK-system", metadata={"user_id": "MARK-uid"},
               stop_sequences=["MARK-stop"], tool_choice={"type": "tool", "name": "NOT-SCANNED"}, thinking={"type": "enabled"})
    assert sorted(texts(body)) == ["MARK-stop", "MARK-system", "MARK-uid", "MARK-user"]


def test_fr_gate_02_leaves_know_their_path_and_empty_strings_are_not_leaves_but_data_urls_in_text_are() -> None:
    """Media is decided by the field, not the content (FR-GATE-07): a data URL in message text is scanned."""
    body = req(user({"type": "text", "text": "MARK-a"}, {"type": "text", "text": ""}),
               {"role": "user", "content": "data:image/png;base64,AAAA"})
    assert [(op.path, op.text) for op in ops(body)] == [(("messages", 0, "content", 0, "text"), "MARK-a"),
                                                        (("messages", 1, "content"), "data:image/png;base64,AAAA")]


def test_fr_gate_02_system_blocks_are_text_blocks_and_any_other_type_blocks() -> None:
    body = req(ASK, system=[{"type": "text", "text": "MARK-1", "cache_control": {"type": "ephemeral"}},
                                  {"type": "text", "text": "MARK-2"}])
    assert sorted(texts(body)) == ["MARK-1", "MARK-2", "x"]
    with pytest.raises(RefusalError) as caught:
        ops(req(ASK, system=[{"type": "image"}]))
    assert caught.value.code == UNKNOWN_TYPE


def test_fr_gate_02_text_block_scans_cited_text_and_other_string_fields() -> None:
    body = req(user({"type": "text", "text": "MARK-t", "citations": [{"type": "char_location", "cited_text": "MARK-cite",
                                                                      "document_title": "MARK-title", "start_char_index": 0}]}))
    assert sorted(texts(body)) == ["MARK-cite", "MARK-t", "MARK-title"]


def test_fr_gate_10_thinking_and_redacted_thinking_blocks_are_dropped_not_scanned() -> None:
    body = req({"role": "user", "content": "go"},
               {"role": "assistant", "content": [{"type": "thinking", "thinking": "MARK-secret-thought", "signature": "SIG"},
                                                 {"type": "redacted_thinking", "data": "DATA"},
                                                 {"type": "text", "text": "MARK-answer"}]})
    found = ops(body)
    assert [op.path for op in found if isinstance(op, Drop)] == [("messages", 1, "content", 0), ("messages", 1, "content", 1)]
    assert [op.text for op in found if isinstance(op, Leaf)] == ["go", "MARK-answer"]


def test_fr_gate_10_a_message_that_is_only_thinking_is_dropped_whole() -> None:
    body = req({"role": "user", "content": "go"}, {"role": "assistant", "content": [{"type": "thinking", "thinking": "t",
                                                                                      "signature": "s"}]},
               {"role": "user", "content": "next"})
    found = ops(body)
    assert [op.path for op in found if isinstance(op, Drop)] == [("messages", 1)]
    assert [op.text for op in found if isinstance(op, Leaf)] == ["go", "next"]


def test_fr_gate_02_unknown_block_types_block_at_every_level() -> None:
    for body in (req(user({"type": "future_block", "text": "x"})),
                 req(user({"type": "tool_result", "tool_use_id": "t", "content": [{"type": "future_block"}]})),
                 req(user({"type": "text", "text": "x"}), tools=[{"type": "future_tool_20990101", "name": "x"}]),
                 req(user({"type": "image", "source": {"type": "future_source", "data": "x"}})),
                 req(user({"type": "document", "source": {"type": "future_source"}})),
                 req({"role": "user", "content": [{"text": "no type at all"}]}),
                 req({"role": "user", "content": [{"type": 7}]})):
        with pytest.raises(RefusalError) as caught:
            ops(body)
        assert caught.value.code == UNKNOWN_TYPE, body


def test_fr_gate_02_a_request_that_is_not_shaped_like_messages_blocks() -> None:
    for body in ({"messages": {"role": "user"}}, {"messages": ["text"]}, {"messages": [{"role": "user", "content": 5}]},
                 {"messages": [], "tools": "nope"}, {"messages": [], "system": 3},
                 {"messages": [{"role": "user", "content": ["MARK-bare-string"]}]}):
        with pytest.raises(RefusalError) as caught:
            ops(body)
        assert caught.value.code == PARSE, body


def test_fr_gate_02_identifiers_signatures_ciphertext_and_media_are_never_leaves() -> None:
    body = req(
        user({"type": "tool_result", "tool_use_id": "toolu_ID", "content": "MARK-result"},
             {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BASE64"}},
             {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF64"},
              "title": "MARK-doc-title"}),
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_ID", "name": "Bash", "input": {"command": "MARK-cmd"}},
                                          {"type": "server_tool_use", "id": "srv_ID", "name": "web_search",
                                           "input": {"query": "MARK-q"}},
                                          {"type": "container_upload", "file_id": "file_ID"},
                                          {"type": "tool_search_tool_result", "tool_use_id": "t", "content": {
                                              "type": "tool_search_tool_search_result",
                                              "tool_references": [{"type": "tool_reference", "tool_name": "NOT-SCANNED"}]}}]})
    assert sorted(texts(body)) == ["MARK-cmd", "MARK-doc-title", "MARK-q", "MARK-result"]


def test_fr_gate_02_tool_input_values_are_leaves_and_its_keys_are_scanned_too() -> None:
    body = req({"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Edit",
                                                  "input": {"path": "MARK-p", "opts": {"MARK-key": ["MARK-item", 3, None]}}}]})
    found = ops(body)
    assert sorted(op.text for op in found if isinstance(op, Leaf)) == ["MARK-item", "MARK-p"]
    assert sorted(op.text for op in found if isinstance(op, KeyText)) == ["MARK-key", "opts", "path"]


def test_fr_gate_10_text_the_provider_signed_is_scan_only_class_s() -> None:
    body = req({"role": "assistant", "content": [
        {"type": "compaction", "content": "MARK-summary", "signature": "SIG"},
        {"type": "web_search_tool_result", "tool_use_id": "s", "content": [
            {"type": "web_search_result", "title": "MARK-title", "url": "https://x.test/u", "encrypted_content": "ENC",
             "page_age": "1 day ago"}]},
        {"type": "mcp_tool_listing", "server_name": "NOT-SCANNED", "tools": [
            {"name": "NOT-SCANNED-NAME", "description": "MARK-desc", "input_schema": {"type": "object", "properties": {}}}]}]})
    s_texts = texts(body, "S")
    assert {"MARK-summary", "MARK-title", "https://x.test/u", "MARK-desc"} <= set(s_texts)
    assert not {"SIG", "ENC", "NOT-SCANNED", "NOT-SCANNED-NAME", "s"} & set(s_texts)
    assert texts(body, "R") == []


def test_fr_gate_02_connector_credentials_are_class_k_and_not_scanned() -> None:
    body = req(ASK, mcp_servers=[{"type": "url", "url": "https://mcp.test/sse", "name": "MARK-name",
                                        "authorization_token": "NOT-SCANNED-TOKEN"}])
    assert sorted(texts(body)) == ["MARK-name", "https://mcp.test/sse", "x"]


def test_fr_gate_02_mid_conversation_system_messages_string_and_blocks() -> None:
    body = req({"role": "user", "content": "hi"}, {"role": "system", "content": "MARK-env-string"},
               {"role": "system", "content": [{"type": "text", "text": "MARK-note"},
                                              {"type": "tool_addition", "definition": {
                                                  "name": "NOT-SCANNED", "description": "MARK-tool-desc",
                                                  "input_schema": {"type": "object", "properties": {
                                                      "q": {"type": "string", "description": "MARK-prop"}}}}},
                                              {"type": "tool_removal", "tool_reference": {"type": "tool_reference",
                                                                                         "tool_name": "NOT-SCANNED"}}]})
    assert {"MARK-env-string", "MARK-note", "MARK-tool-desc", "MARK-prop"} <= set(texts(body))
    assert "NOT-SCANNED" not in texts(body)


def test_fr_gate_02_tools_custom_server_and_mcp_toolset() -> None:
    body = req(ASK, tools=[
        {"name": "NOT-SCANNED", "description": "MARK-d", "input_schema": {"type": "object", "properties": {
            "p": {"type": "string", "description": "MARK-pd", "enum": ["MARK-e"]}}}, "input_examples": [{"p": "MARK-ex"}],
         "cache_control": {"type": "ephemeral"}},
        {"type": "bash_20250124", "name": "bash"}, {"type": "code_execution_20250522", "name": "code_execution"},
        {"type": "mcp_toolset", "mcp_server_name": "MARK-srv"}])
    assert {"MARK-d", "MARK-pd", "MARK-e", "MARK-ex", "MARK-srv"} <= set(texts(body))
    assert "NOT-SCANNED" not in texts(body) and "ephemeral" not in texts(body)


def test_fr_gate_02_documents_follow_their_source_type() -> None:
    body = req(user(
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "MARK-plain"}, "title": "MARK-t",
         "context": "MARK-c"},
        {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "MARK-inner"}]}},
        {"type": "document", "source": {"type": "url", "url": "https://f.test/x.pdf?k=MARK-url"}},
        {"type": "document", "source": {"type": "file", "file_id": "file_ID"}}))
    assert {"MARK-plain", "MARK-t", "MARK-c", "MARK-inner", "https://f.test/x.pdf?k=MARK-url"} <= set(texts(body))
    assert "file_ID" not in texts(body)
