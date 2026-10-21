"""OpenAI Responses rows of the gate corpus (FR-PROOF-04, FR-GATE-02): one request per input item and tool type in
FR-GATE-02 (docs/design.md). Shapes follow the OpenAI Responses API documentation.
"""
from __future__ import annotations

import json

from bench.gate_corpus import Row

PNG_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
CIPHER = "gAAAAABoXjQ2k3u9b0vE0pP5dX8y7mQnR1sT4wZlC6hJ2aKfG9iBqNe3VtYUo5LrMxHcDzSgWbAyEiOjFkPl"
USER = {"type": "message", "role": "user", "content": "go"}


def _req(items: object, **extra: object) -> dict:
    """
    Description: Wrap input items in a Responses API request.
    Parameters:
        items: the input: a string or a list of items.
        **extra: further top-level fields (instructions, tools, metadata).
    Output: A request body with model, input and store false.
    """
    return {"model": "gpt-mock", "input": items, "store": False, **extra}


def _after_user(*items: dict) -> dict:
    """
    Description: A request whose input is a user message followed by the items.
    Parameters:
        *items: input items.
    Output: A request body.
    """
    return _req([USER, *items])


def _call(call_id: str, name: str = "exec_command") -> dict:
    """
    Description: A function_call item with harmless arguments, to pair with an output item.
    Parameters:
        call_id: the call id.
        name: the function name.
    Output: A function_call item.
    """
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps({"cmd": "ls"})}


def _tools(*tools: dict) -> dict:
    """
    Description: A request that offers the tools.
    Parameters:
        *tools: tool definitions.
    Output: A request body.
    """
    return _req("hi", tools=list(tools))


def _params(description: str) -> dict:
    """
    Description: A small JSON schema whose property description is the given text.
    Parameters:
        description: the text.
    Output: An object schema dict.
    """
    return {"type": "object", "properties": {"q": {"type": "string", "description": description}}}


def rows() -> list[Row]:
    """
    Description: List the OpenAI Responses corpus rows.
    Parameters: None
    Output: Rows in the order of the parser spec: input items, tools, then top-level fields, the previous_response_id and
        conversation policy cases, and extra cases for the JSON-in-string rule.
    """
    return [
        # input items
        Row("responses_message_string", "input-item", "message", "redact", lambda c: _req(
            [{"type": "message", "role": "user", "content": f"key {c}"}])),
        Row("responses_input_text", "input-item", "input_text", "redact", lambda c: _req(
            [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"key {c}"}]}])),
        Row("responses_output_text", "input-item", "output_text", "redact", lambda c: _after_user(
            {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed", "content": [
                {"type": "output_text", "text": f"I saw {c}", "annotations": [
                    {"type": "url_citation", "title": "t", "url": "https://example.test", "start_index": 0, "end_index": 1}]}]})),
        Row("responses_refusal", "input-item", "refusal", "redact", lambda c: _after_user(
            {"type": "message", "role": "assistant", "id": "msg_2", "status": "completed",
             "content": [{"type": "refusal", "refusal": f"cannot {c}"}]})),
        Row("responses_input_image", "input-item", "input_image", "redact", lambda c: _req([{
            "type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": f"https://img.example.test/a.png?sig={c}", "detail": "auto"}]}])),
        Row("responses_input_image_data_url", "input-item", "input_image", "pass", lambda c: _req([{
            "type": "message", "role": "user", "content": [{"type": "input_image", "image_url": PNG_URL}]}]),
            plant=False, spec=False),
        Row("responses_input_file", "input-item", "input_file", "redact", lambda c: _req([{
            "type": "message", "role": "user", "content": [
                {"type": "input_file", "filename": f"{c}.txt", "file_data": "data:text/plain;base64,aGVsbG8="}]}])),
        Row("responses_function_call", "input-item", "function_call", "redact", lambda c: _after_user({
            "type": "function_call", "call_id": "call_1", "name": "exec_command",
            "arguments": json.dumps({"cmd": f"echo {c}", "workdir": "/repo"})})),
        Row("responses_function_call_output", "input-item", "function_call_output", "redact", lambda c: _after_user(
            _call("call_1"), {"type": "function_call_output", "call_id": "call_1", "output": f"cat deploy/id_rsa\n{c}"}),
            kind="private_key"),
        Row("responses_custom_tool_call", "input-item", "custom_tool_call", "redact", lambda c: _after_user({
            "type": "custom_tool_call", "call_id": "call_2", "name": "apply_patch",
            "input": f"*** Begin Patch\n+K={c}\n*** End Patch"})),
        Row("responses_custom_tool_call_output", "input-item", "custom_tool_call_output", "redact", lambda c: _after_user(
            {"type": "custom_tool_call", "call_id": "call_2", "name": "apply_patch", "input": "x"},
            {"type": "custom_tool_call_output", "call_id": "call_2", "output": f"done {c}"})),
        Row("responses_reasoning", "input-item", "reasoning", "block", lambda c: _after_user({
            "type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": f"thinking {c}"}],
            "encrypted_content": CIPHER})),
        Row("responses_local_shell_call", "input-item", "local_shell_call", "redact", lambda c: _after_user({
            "type": "local_shell_call", "id": "lsh_1", "call_id": "call_3", "status": "completed", "action": {
                "type": "exec", "command": ["bash", "-lc", f"echo {c}"], "env": {"K": "v"},
                "working_directory": "/repo", "timeout_ms": 1000, "user": None}})),
        Row("responses_local_shell_call_output", "input-item", "local_shell_call_output", "redact", lambda c: _after_user({
            "type": "local_shell_call_output", "id": "lsh_1",
            "output": json.dumps({"output": f"{c}\n", "metadata": {"exit_code": 0, "duration_seconds": 0.1}})})),
        Row("responses_shell_call", "input-item", "shell_call", "redact", lambda c: _after_user({
            "type": "shell_call", "id": "sh_1", "call_id": "call_4", "status": "completed",
            "action": {"commands": [f"echo {c}"], "timeout_ms": 1000, "max_output_length": 4096}})),
        Row("responses_shell_call_output", "input-item", "shell_call_output", "redact", lambda c: _after_user({
            "type": "shell_call_output", "id": "sho_1", "call_id": "call_4", "output": [
                {"stdout": f"{c}\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]})),
        Row("responses_apply_patch_call", "input-item", "apply_patch_call", "redact", lambda c: _after_user({
            "type": "apply_patch_call", "id": "ap_1", "call_id": "call_5", "status": "completed", "operation": {
                "type": "update_file", "path": "app/config.py", "diff": f"@@\n-K=1\n+K={c}\n"}})),
        Row("responses_apply_patch_call_output", "input-item", "apply_patch_call_output", "redact", lambda c: _after_user({
            "type": "apply_patch_call_output", "call_id": "call_5", "status": "completed", "output": f"Updated {c}"})),
        Row("responses_computer_call", "input-item", "computer_call", "redact", lambda c: _after_user({
            "type": "computer_call", "id": "cu_1", "call_id": "call_6", "status": "completed",
            "action": {"type": "type", "text": c}, "pending_safety_checks": []})),
        Row("responses_computer_call_output", "input-item", "computer_call_output", "pass", lambda c: _after_user({
            "type": "computer_call_output", "call_id": "call_6",
            "output": {"type": "computer_screenshot", "image_url": PNG_URL}}), plant=False),
        Row("responses_web_search_call", "input-item", "web_search_call", "redact", lambda c: _after_user({
            "type": "web_search_call", "id": "ws_1", "status": "completed",
            "action": {"type": "search", "query": f"leak {c}"}})),
        Row("responses_file_search_call", "input-item", "file_search_call", "redact", lambda c: _after_user({
            "type": "file_search_call", "id": "fs_1", "status": "completed", "queries": [f"find {c}"],
            "results": [{"file_id": "file-1", "filename": "a.txt", "text": "snippet", "score": 0.5}]})),
        Row("responses_code_interpreter_call", "input-item", "code_interpreter_call", "redact", lambda c: _after_user({
            "type": "code_interpreter_call", "id": "ci_1", "status": "completed", "container_id": "cntr_1",
            "code": f"print('{c}')", "outputs": [{"type": "logs", "logs": "ok"}]})),
        Row("responses_image_generation_call", "input-item", "image_generation_call", "redact", lambda c: _after_user({
            "type": "image_generation_call", "id": "ig_1", "status": "completed",
            "revised_prompt": f"a cat holding {c}", "result": "iVBORw0KGgo="})),
        Row("responses_mcp_call", "input-item", "mcp_call", "redact", lambda c: _after_user({
            "type": "mcp_call", "id": "mcp_1", "server_label": "srv", "name": "search",
            "arguments": json.dumps({"q": c}), "output": None, "error": None})),
        Row("responses_mcp_list_tools", "input-item", "mcp_list_tools", "redact", lambda c: _after_user({
            "type": "mcp_list_tools", "id": "mcpl_1", "server_label": "srv", "tools": [
                {"name": "search", "description": f"search {c}", "input_schema": {"type": "object", "properties": {}}}]})),
        Row("responses_mcp_approval_request", "input-item", "mcp_approval_request", "redact", lambda c: _after_user({
            "type": "mcp_approval_request", "id": "mcpr_1", "server_label": "srv", "name": "search",
            "arguments": json.dumps({"q": c})})),
        Row("responses_mcp_approval_response", "input-item", "mcp_approval_response", "redact", lambda c: _after_user({
            "type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True, "reason": f"ok {c}"})),
        Row("responses_tool_search", "input-item", "tool_search_call", "redact", lambda c: _after_user({
            "type": "tool_search_call", "id": "ts_1", "call_id": "call_7", "arguments": json.dumps({"query": c})})),
        Row("responses_additional_tools", "input-item", "additional_tools", "redact", lambda c: _after_user({
            "type": "additional_tools", "tools": [{"type": "function", "name": "f", "description": f"does {c}",
                                                   "parameters": _params("q")}]})),
        Row("responses_item_reference", "input-item", "item_reference", "pass", lambda c: _after_user(
            {"type": "item_reference", "id": "msg_prev_1"}), plant=False),
        Row("responses_compaction", "input-item", "compaction", "pass", lambda c: _after_user(
            {"type": "compaction", "id": "cmp_1", "encrypted_content": CIPHER}), plant=False),
        Row("responses_unknown_item_blocked", "input-item", "unknown", "block", lambda c: _after_user(
            {"type": "future_item_xyz", "text": f"{c}"})),
        # JSON inside a string (rewrite rule 3)
        Row("responses_function_call_arguments_bad_json", "input-item", "function_call", "redact", lambda c: _after_user({
            "type": "function_call", "call_id": "call_8", "name": "exec_command",
            "arguments": '{"cmd": "echo ' + c + '", "tail": '}), spec=False),
        Row("responses_function_call_arguments_nested_json", "input-item", "function_call", "redact", lambda c: _after_user({
            "type": "function_call", "call_id": "call_9", "name": "exec_command",
            "arguments": json.dumps({"payload": json.dumps({"inner": c})})}), spec=False),
        Row("responses_function_call_arguments_key_blocked", "input-item", "function_call", "block", lambda c: _after_user({
            "type": "function_call", "call_id": "call_10", "name": "exec_command",
            "arguments": json.dumps({c: 1})}), spec=False),
        # tools
        Row("responses_tools_function", "tools", "function", "redact", lambda c: _tools({
            "type": "function", "name": "lookup", "description": f"look up {c}", "parameters": _params("query"),
            "strict": False})),
        Row("responses_tools_custom", "tools", "custom", "redact", lambda c: _tools({
            "type": "custom", "name": "apply_patch", "description": "d",
            "format": {"type": "grammar", "syntax": "lark", "definition": f'start: "{c}"'}})),
        Row("responses_tools_mcp", "tools", "mcp", "redact", lambda c: _tools({
            "type": "mcp", "server_label": "srv", "server_url": "https://mcp.example.test",
            "server_description": f"srv {c}", "require_approval": "never"})),
        Row("responses_tools_mcp_authorization", "tools", "mcp", "pass", lambda c: _tools({
            "type": "mcp", "server_label": "srv", "server_url": "https://mcp.example.test", "authorization": c,
            "require_approval": "never"}), spec=False),
        Row("responses_tools_namespace", "tools", "namespace", "redact", lambda c: _tools({
            "type": "namespace", "name": "ns", "description": f"ns {c}",
            "tools": [{"type": "function", "name": "f", "description": "d", "parameters": _params("q")}]})),
        Row("responses_tools_builtin", "tools", "web_search", "redact", lambda c: _tools({
            "type": "web_search", "user_location": {"type": "approximate", "city": c}})),
        Row("responses_tools_unknown_type_blocked", "tools", "unknown", "block", lambda c: _tools({
            "type": "future_tool_xyz", "description": f"{c}"})),
        # top-level fields and policy
        Row("responses_instructions", "top-level", "instructions", "redact", lambda c: _req(
            "hi", instructions=f"Be brief. {c}"), spec=False),
        Row("responses_input_string", "top-level", "input", "redact", lambda c: _req(f"my key {c}"), spec=False),
        Row("responses_prompt_variables", "top-level", "prompt", "redact", lambda c: _req(
            "hi", prompt={"id": "pmpt_1", "variables": {"name": c}}), spec=False),
        Row("responses_metadata", "top-level", "metadata", "redact", lambda c: _req("hi", metadata={"k": c}), spec=False),
        Row("responses_client_metadata", "top-level", "client_metadata", "redact", lambda c: _req(
            "hi", client_metadata={"x-codex-turn-metadata": json.dumps({"remote": f"https://user:{c}@example.test/r.git"})}),
            spec=False),
        Row("responses_text_format_schema", "top-level", "text.format", "redact", lambda c: _req(
            "hi", text={"format": {"type": "json_schema", "name": "out", "description": f"use {c}",
                                   "schema": _params("q")}}), spec=False),
        Row("responses_previous_response_id_seen_blocked", "policy", "previous_response_id", "block", lambda c: _req(
            [{"type": "message", "role": "user", "content": "next"}], previous_response_id="resp_seen_001"), plant=False),
        Row("responses_previous_response_id_unseen_blocked", "policy", "previous_response_id", "block", lambda c: _req(
            [{"type": "message", "role": "user", "content": "next"}], previous_response_id="resp_unseen_999"), plant=False),
        Row("responses_conversation_blocked", "policy", "conversation", "block", lambda c: _req(
            [{"type": "message", "role": "user", "content": "next"}], conversation="conv_123"), plant=False),
    ]
