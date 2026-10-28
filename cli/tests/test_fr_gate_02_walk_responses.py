"""FR-GATE-02: which strings of an OpenAI Responses request are scanned, and as what (parser spec, "OpenAI Responses").
Unit tests of Harpocrates.gate.walk_responses: it yields operations and decides nothing about secrets.

Marker strings (MARK-...) stand in for text; none is a credential."""
from __future__ import annotations

import json

import pytest

from Harpocrates.gate.blocks import HISTORY, PARSE, UNKNOWN_TYPE, RefusalError
from Harpocrates.gate.walk_common import Leaf
from Harpocrates.gate.walk_responses import walk


def ops(body: dict) -> list:
    return list(walk(body))


def texts(body: dict, cls: str = "R") -> list[str]:
    return [op.text for op in ops(body) if isinstance(op, Leaf) and op.cls == cls]


def req(*items: dict, **extra: object) -> dict:
    return {"model": "m", "input": list(items), "store": False, **extra}


def test_fr_gate_02_top_level_strings_are_leaves_and_identifiers_are_not() -> None:
    body = {"model": "NOT-SCANNED", "input": "MARK-input", "instructions": "MARK-instr", "include": ["NOT-SCANNED"],
            "reasoning": {"effort": "NOT-SCANNED"}, "tool_choice": {"type": "function", "name": "NOT-SCANNED"},
            "metadata": {"k": "MARK-meta"}, "client_metadata": {"x": "MARK-cm"}, "user": "MARK-user",
            "prompt_cache_key": "MARK-pck", "prompt": {"id": "NOT-SCANNED", "variables": {"name": "MARK-var"}},
            "text": {"format": {"type": "json_schema", "name": "out", "description": "MARK-fmt"}}}
    assert {"MARK-input", "MARK-instr", "MARK-meta", "MARK-cm", "MARK-user", "MARK-pck", "MARK-var", "MARK-fmt"} <= set(texts(body))
    assert not {"NOT-SCANNED"} & set(texts(body))


def test_fr_gate_02_a_message_without_a_type_is_a_message_and_roles_are_not_scanned() -> None:
    body = req({"role": "user", "content": "MARK-short"}, {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
                                                           "content": [{"type": "output_text", "text": "MARK-out",
                                                                        "annotations": [{"type": "url_citation", "title": "MARK-t",
                                                                                         "url": "https://x.test", "start_index": 0}]}]})
    got = texts(body)
    assert {"MARK-short", "MARK-out", "MARK-t", "https://x.test"} <= set(got)
    assert not {"user", "assistant", "msg_1", "completed", "message", "output_text", "url_citation"} & set(got)


def test_fr_gate_02_message_parts_and_media() -> None:
    body = req({"role": "user", "content": [
        {"type": "input_text", "text": "MARK-in"}, {"type": "refusal", "refusal": "MARK-ref"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto", "file_id": "NOT-SCANNED"},
        {"type": "input_image", "image_url": "https://img.test/a.png?k=MARK-img"},
        {"type": "input_file", "filename": "MARK-name.txt", "file_data": "BASE64DATA", "file_id": "NOT-SCANNED"}]})
    got = texts(body)
    assert {"MARK-in", "MARK-ref", "https://img.test/a.png?k=MARK-img", "MARK-name.txt"} <= set(got)
    assert not {"data:image/png;base64,AAAA", "BASE64DATA", "NOT-SCANNED"} & set(got)


def test_fr_gate_02_function_call_arguments_are_json_in_a_string_and_ids_are_not_scanned() -> None:
    body = req({"type": "function_call", "call_id": "call_1", "name": "exec_command", "namespace": "NOT-SCANNED",
                "arguments": json.dumps({"cmd": "MARK-cmd"})},
               {"type": "function_call_output", "call_id": "call_1", "output": "MARK-out"},
               {"type": "mcp_call", "id": "m", "server_label": "NOT-SCANNED", "name": "NOT-SCANNED", "arguments": '{"q": 1}',
                "output": "MARK-mcp-out", "error": None})
    found = ops(body)
    assert [(op.path[-1], op.as_json) for op in found if isinstance(op, Leaf) and op.as_json] == [("arguments", True), ("arguments", True)]
    assert {"MARK-out", "MARK-mcp-out"} <= {op.text for op in found if isinstance(op, Leaf) and not op.as_json}
    assert not {"call_1", "exec_command", "NOT-SCANNED", "m"} & {op.text for op in found if isinstance(op, Leaf)}


def test_fr_gate_10_reasoning_text_is_class_s_and_its_ciphertext_is_not_scanned() -> None:
    body = req({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "MARK-sum"}],
                "content": [{"type": "reasoning_text", "text": "MARK-content"}], "encrypted_content": "CIPHERTEXT"})
    s_texts = texts(body, "S")
    assert {"MARK-sum", "MARK-content"} <= set(s_texts) and not {"CIPHERTEXT", "rs_1"} & set(s_texts)
    assert texts(body, "R") == []


def test_fr_gate_02_shell_patch_search_and_code_items() -> None:
    body = req(
        {"type": "local_shell_call", "id": "l", "call_id": "c", "status": "completed",
         "action": {"type": "exec", "command": ["bash", "-lc", "MARK-sh"], "env": {"K": "MARK-env"}, "working_directory": "MARK-wd"}},
        {"type": "local_shell_call_output", "id": "l", "output": json.dumps({"output": "MARK-lso"})},
        {"type": "shell_call", "call_id": "c2", "action": {"commands": ["MARK-cmds"]}},
        {"type": "shell_call_output", "call_id": "c2", "output": [{"stdout": "MARK-so", "stderr": "MARK-se",
                                                                  "outcome": {"type": "exit", "exit_code": 0}}]},
        {"type": "apply_patch_call", "call_id": "c3", "operation": {"type": "update_file", "path": "MARK-path", "diff": "MARK-diff"}},
        {"type": "apply_patch_call_output", "call_id": "c3", "output": "MARK-apo"},
        {"type": "web_search_call", "id": "w", "action": {"type": "search", "query": "MARK-query"}},
        {"type": "file_search_call", "id": "f", "queries": ["MARK-fq"], "results": [{"file_id": "NOT-SCANNED", "filename": "MARK-fn",
                                                                                    "text": "MARK-ft"}]},
        {"type": "code_interpreter_call", "id": "ci", "container_id": "NOT-SCANNED", "code": "MARK-code",
         "outputs": [{"type": "logs", "logs": "MARK-logs"}]},
        {"type": "image_generation_call", "id": "ig", "revised_prompt": "MARK-rp", "result": "IMAGEBASE64"},
        {"type": "computer_call", "call_id": "cc", "action": {"type": "type", "text": "MARK-typed"}},
        {"type": "computer_call_output", "call_id": "cc", "output": {"type": "computer_screenshot", "image_url": "https://s.test/x?k=MARK-shot"}},
        {"type": "custom_tool_call", "call_id": "ct", "name": "apply_patch", "input": "MARK-ct"},
        {"type": "custom_tool_call_output", "call_id": "ct", "output": "MARK-cto"})
    got = set(texts(body))
    want = {"MARK-sh", "MARK-env", "MARK-wd", "MARK-cmds", "MARK-so", "MARK-se", "MARK-path", "MARK-diff", "MARK-apo",
            "MARK-query", "MARK-fq", "MARK-fn", "MARK-ft", "MARK-code", "MARK-logs", "MARK-rp", "MARK-typed",
            "https://s.test/x?k=MARK-shot", "MARK-ct", "MARK-cto"}
    assert want <= got
    assert not {"IMAGEBASE64", "NOT-SCANNED", "exec", "update_file"} & got


def test_fr_gate_02_mcp_tool_search_and_additional_tools_items() -> None:
    tool = {"type": "function", "name": "NOT-SCANNED", "description": "MARK-desc", "parameters": {"type": "object", "properties": {
        "q": {"type": "string", "description": "MARK-pdesc"}}}}
    body = req({"type": "mcp_list_tools", "id": "l", "server_label": "NOT-SCANNED", "tools": [
        {"name": "NOT-SCANNED", "description": "MARK-ld", "input_schema": {"type": "object"}}]},
               {"type": "mcp_approval_request", "id": "a", "name": "NOT-SCANNED", "server_label": "x", "arguments": "{}"},
               {"type": "mcp_approval_response", "approval_request_id": "a", "approve": True, "reason": "MARK-why"},
               {"type": "tool_search_call", "call_id": "t", "arguments": json.dumps({"query": "MARK-tsq"})},
               {"type": "tool_search_output", "call_id": "t", "tools": [tool]}, {"type": "additional_tools", "tools": [tool]},
               {"type": "item_reference", "id": "NOT-SCANNED"}, {"type": "compaction", "id": "cmp", "encrypted_content": "CIPHER"})
    got = set(texts(body))
    assert {"MARK-ld", "MARK-why", "MARK-desc", "MARK-pdesc"} <= got
    assert not {"NOT-SCANNED", "CIPHER", "cmp"} & got
    assert sum(1 for op in ops(body) if isinstance(op, Leaf) and op.as_json) == 2  # the approval request's arguments and the tool search call's


def test_fr_gate_02_tools_function_custom_mcp_namespace_builtin() -> None:
    body = req({"role": "user", "content": "x"}, tools=[
        {"type": "function", "name": "NOT-SCANNED", "description": "MARK-fd", "parameters": {"type": "object", "properties": {
            "p": {"type": "string", "description": "MARK-fp"}}}, "strict": False},
        {"type": "custom", "name": "NOT-SCANNED", "description": "MARK-cd", "format": {"type": "grammar", "syntax": "lark",
                                                                                          "definition": "MARK-grammar"}},
        {"type": "mcp", "server_label": "MARK-label", "server_url": "https://m.test", "server_description": "MARK-sd",
         "authorization": "NOT-SCANNED", "headers": {"Authorization": "NOT-SCANNED"}, "require_approval": "never"},
        {"type": "namespace", "name": "NOT-SCANNED", "description": "MARK-nd",
         "tools": [{"type": "function", "name": "NOT-SCANNED", "description": "MARK-nested", "parameters": {}}]},
        {"type": "web_search", "user_location": {"type": "approximate", "city": "MARK-city"}}, {"type": "local_shell"},
        {"type": "apply_patch"}, {"type": "tool_search"}])
    got = set(texts(body))
    assert {"MARK-fd", "MARK-fp", "MARK-cd", "MARK-grammar", "MARK-label", "https://m.test", "MARK-sd", "MARK-nd", "MARK-nested",
            "MARK-city"} <= got
    assert "NOT-SCANNED" not in got


def test_fr_gate_02_unknown_types_block_at_every_level() -> None:
    bad = (req({"type": "future_item", "text": "x"}), req({"role": "user", "content": [{"type": "future_part"}]}),
           req({"text": "no type or role"}), {"input": "x", "tools": [{"type": "future_tool"}]},
           {"input": "x", "tools": [{"type": "namespace", "tools": [{"type": "future_tool"}]}]})
    for body in bad:
        with pytest.raises(RefusalError) as caught:
            ops(body)
        assert caught.value.code == UNKNOWN_TYPE, body


def test_fr_gate_02_server_side_history_blocks_in_v1() -> None:
    for extra in ({"previous_response_id": "resp_1"}, {"conversation": "conv_1"}, {"conversation": {"id": "conv_1"}}):
        with pytest.raises(RefusalError) as caught:
            ops({"input": "x", **extra})
        assert caught.value.code == HISTORY
    assert ops({"input": "x", "previous_response_id": None, "conversation": None}) == ops({"input": "x"})


def test_fr_gate_02_a_request_that_is_not_shaped_like_responses_blocks() -> None:
    for body in ({"input": 5}, {"input": ["bare string"]}, {"input": [5]}, {"input": "x", "tools": "nope"},
                 {"input": [{"type": "message", "role": "user", "content": 5}]}, []):
        with pytest.raises(RefusalError) as caught:
            ops(body)
        assert caught.value.code == PARSE, body
