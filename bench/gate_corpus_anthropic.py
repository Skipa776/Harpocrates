"""Anthropic Messages rows of the gate corpus (FR-PROOF-04, FR-GATE-02): one request per block and tool type in
FR-GATE-02 (docs/design.md). Shapes follow the Anthropic Messages API documentation.
"""
from __future__ import annotations

from bench.gate_corpus import Row

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
SIG = "EqQBCkYIBhgCKkDq1x7Yw0b8mQ3sSsq8bC1n9m2l5Qx4a7pYtqV0wKj3e6RrZt1u8vNf2h5LdXo9cGiAyBkMzEpTnFqHsUw4DaJvOgCbRlYe"


def _msgs(*turns: tuple[str, object]) -> list[dict]:
    """
    Description: Build a message list from (role, content) pairs.
    Parameters:
        *turns: (role, content) pairs, content a string or a list of blocks.
    Output: A list of {"role", "content"} dicts.
    """
    return [{"role": role, "content": content} for role, content in turns]


def _req(messages: list[dict], **extra: object) -> dict:
    """
    Description: Wrap messages in a Messages API request.
    Parameters:
        messages: the message list.
        **extra: further top-level fields (system, tools, metadata).
    Output: A request body with model, max_tokens and messages.
    """
    return {"model": "claude-mock", "max_tokens": 64, "messages": messages, **extra}


def _user(*blocks: dict) -> dict:
    """
    Description: A request whose last user message holds the blocks.
    Parameters:
        *blocks: content blocks.
    Output: A request body.
    """
    return _req(_msgs(("user", list(blocks))))


def _assistant(*blocks: dict) -> dict:
    """
    Description: A request whose conversation has an assistant turn holding the blocks.
    Parameters:
        *blocks: content blocks the assistant produced.
    Output: A request body: user, assistant, user.
    """
    return _req(_msgs(("user", "go"), ("assistant", list(blocks)), ("user", "continue")))


def _tool_pair(result: object, tool: str = "Bash") -> dict:
    """
    Description: A request with a tool call from the assistant and a tool_result from the user.
    Parameters:
        result: the tool_result content: a string or a list of blocks.
        tool: the tool name.
    Output: A request body: user, assistant (tool_use), user (tool_result).
    """
    use = {"type": "tool_use", "id": "toolu_01", "name": tool, "input": {"command": "ls"}}
    return _req(_msgs(("user", "go"), ("assistant", [use]),
                      ("user", [{"type": "tool_result", "tool_use_id": "toolu_01", "content": result}])))


def _schema(description: str) -> dict:
    """
    Description: A small JSON schema whose property description is the given text.
    Parameters:
        description: the text.
    Output: An object schema dict.
    """
    return {"type": "object", "properties": {"q": {"type": "string", "description": description}}, "required": ["q"]}


def rows() -> list[Row]:
    """
    Description: List the Anthropic Messages corpus rows.
    Parameters: None
    Output: Rows in the order of the parser spec: content blocks, mid-conversation system messages, tools, then top-level
        fields and extra cases the rewrite rules need (secondary fields, key names, credentials for hosted connectors).
    """
    doc_text = {"type": "text", "media_type": "text/plain"}
    return [
        # content blocks
        Row("anthropic_text", "content-block", "text", "redact", lambda c: _user({"type": "text", "text": f"My key is {c}"})),
        Row("anthropic_text_cited_text", "content-block", "text.citations", "redact", lambda c: _assistant(
            {"type": "text", "text": "answer", "citations": [{"type": "char_location", "cited_text": f"line {c}",
                                                              "document_index": 0, "document_title": "d",
                                                              "start_char_index": 0, "end_char_index": 9}]}), spec=False),
        Row("anthropic_image", "content-block", "image", "redact", lambda c: _user(
            {"type": "image", "source": {"type": "url", "url": f"https://img.example.test/a.png?sig={c}"}})),
        Row("anthropic_image_base64", "content-block", "image", "pass", lambda c: _user(
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}), plant=False, spec=False),
        Row("anthropic_document_text", "content-block", "document", "redact", lambda c: _user(
            {"type": "document", "source": {**doc_text, "data": f"notes {c}"}, "title": "notes", "context": "ctx"}),
            kind="private_key"),
        Row("anthropic_document_pdf", "content-block", "document", "redact", lambda c: _user(
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"},
             "title": f"report {c}"})),
        Row("anthropic_document_content", "content-block", "document", "redact", lambda c: _user(
            {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": f"inner {c}"}]}})),
        Row("anthropic_document_url", "content-block", "document", "redact", lambda c: _user(
            {"type": "document", "source": {"type": "url", "url": f"https://files.example.test/x.pdf?sig={c}"}})),
        Row("anthropic_search_result", "content-block", "search_result", "redact", lambda c: _user(
            {"type": "search_result", "source": "https://docs.example.test/p", "title": "t",
             "content": [{"type": "text", "text": f"snippet {c}"}]})),
        Row("anthropic_tool_use", "content-block", "tool_use", "redact", lambda c: _req(_msgs(
            ("user", "go"), ("assistant", [{"type": "tool_use", "id": "toolu_01", "name": "Bash",
                                            "input": {"command": f"echo {c}"}}]),
            ("user", [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}])))),
        Row("anthropic_tool_result_string", "content-block", "tool_result", "redact", lambda c: _tool_pair(f"cat deploy/id_rsa\n{c}\n"),
            kind="private_key"),
        Row("anthropic_tool_result_blocks", "content-block", "tool_result", "redact", lambda c: _tool_pair(
            [{"type": "text", "text": f"out {c}"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                                "data": PNG}}])),
        Row("anthropic_thinking", "content-block", "thinking", "drop", lambda c: _assistant(
            {"type": "thinking", "thinking": f"I should use {c}", "signature": SIG})),
        Row("anthropic_redacted_thinking", "content-block", "redacted_thinking", "drop", lambda c: _assistant(
            {"type": "redacted_thinking", "data": SIG}), plant=False),
        Row("anthropic_compaction", "content-block", "compaction", "block", lambda c: _assistant(
            {"type": "compaction", "content": f"summary {c}", "signature": SIG})),
        Row("anthropic_server_tool_use", "content-block", "server_tool_use", "redact", lambda c: _assistant(
            {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": f"how to {c}"}})),
        Row("anthropic_web_search_tool_result", "content-block", "web_search_tool_result", "block", lambda c: _assistant(
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01", "content": [
                {"type": "web_search_result", "title": f"page {c}", "url": "https://example.test/p",
                 "encrypted_content": SIG, "page_age": "1 day ago"}]})),
        Row("anthropic_web_fetch_tool_result", "content-block", "web_fetch_tool_result", "redact", lambda c: _assistant(
            {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_02", "content": {
                "type": "web_fetch_result", "url": "https://example.test/doc", "retrieved_at": "2026-10-19T00:00:00Z",
                "content": {"type": "document", "source": {**doc_text, "data": f"fetched {c}"}, "title": "doc"}}})),
        Row("anthropic_code_execution_tool_result", "content-block", "code_execution_tool_result", "redact", lambda c: _assistant(
            {"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_03", "content": {
                "type": "code_execution_result", "stdout": f"{c}\n", "stderr": "", "return_code": 0, "content": []}})),
        Row("anthropic_bash_code_execution_tool_result", "content-block", "bash_code_execution_tool_result", "redact",
            lambda c: _assistant({"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_04", "content": {
                "type": "bash_code_execution_result", "stdout": f"{c}\n", "stderr": "", "return_code": 0, "content": []}}),
            spec=False),
        Row("anthropic_text_editor_tool_result", "content-block", "text_editor_code_execution_tool_result", "redact",
            lambda c: _assistant({"type": "text_editor_code_execution_tool_result", "tool_use_id": "srvtoolu_05", "content": {
                "type": "text_editor_code_execution_view_result", "content": f"1: KEY={c}", "file_type": "text",
                "num_lines": 1, "start_line": 1, "total_lines": 1}})),
        Row("anthropic_tool_search_tool_result", "content-block", "tool_search_tool_result", "pass", lambda c: _assistant(
            {"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_06", "content": {
                "type": "tool_search_tool_search_result", "tool_references": [{"type": "tool_reference",
                                                                               "tool_name": "get_weather"}]}}), plant=False),
        Row("anthropic_mcp_tool_use", "content-block", "mcp_tool_use", "redact", lambda c: _assistant(
            {"type": "mcp_tool_use", "id": "mcptoolu_01", "name": "search", "server_name": "srv", "input": {"q": c}})),
        Row("anthropic_mcp_tool_result", "content-block", "mcp_tool_result", "redact", lambda c: _assistant(
            {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_01", "is_error": False,
             "content": [{"type": "text", "text": f"found {c}"}]})),
        Row("anthropic_mcp_tool_listing", "content-block", "mcp_tool_listing", "block", lambda c: _assistant(
            {"type": "mcp_tool_listing", "server_name": "srv", "tools": [
                {"name": "search", "description": f"search {c}", "input_schema": {"type": "object", "properties": {}}}]})),
        Row("anthropic_container_upload", "content-block", "container_upload", "pass", lambda c: _user(
            {"type": "container_upload", "file_id": "file_011CNha8iCJcU1wXNR6q4V8w"}), plant=False),
        Row("anthropic_unknown_block_blocked", "content-block", "unknown", "block", lambda c: _user(
            {"type": "future_block_xyz", "text": f"x {c}"})),
        # mid-conversation system messages
        Row("anthropic_system_role_text", "system-role", "text", "redact", lambda c: _req(_msgs(
            ("user", "hi"), ("system", [{"type": "text", "text": f"note {c}"}])))),
        Row("anthropic_system_role_string", "system-role", "text", "redact", lambda c: _req(_msgs(
            ("user", "hi"), ("system", f"# Environment\nYou have been invoked in the following environment: key {c}"))),
            spec=False),
        Row("anthropic_tool_addition", "system-role", "tool_addition", "redact", lambda c: _req(_msgs(
            ("user", "hi"), ("system", [{"type": "tool_addition", "definition": {
                "name": "lookup", "description": f"look up {c}", "input_schema": _schema("query")}}])))),
        Row("anthropic_tool_removal", "system-role", "tool_removal", "pass", lambda c: _req(_msgs(
            ("user", "hi"), ("system", [{"type": "tool_removal", "tool_reference": {"type": "tool_reference",
                                                                                     "tool_name": "lookup"}}]))), plant=False),
        # tools
        Row("anthropic_tools_custom", "tools", "custom", "redact", lambda c: _req(_msgs(("user", "hi")), tools=[{
            "name": "lookup", "description": f"look up {c}", "input_schema": _schema("query"),
            "input_examples": [{"q": "example"}]}])),
        Row("anthropic_tools_server", "tools", "server", "pass", lambda c: _req(_msgs(("user", "hi")), tools=[
            {"type": "bash_20250124", "name": "bash"}, {"type": "code_execution_20250522", "name": "code_execution"}]),
            plant=False),
        Row("anthropic_tools_mcp_toolset", "tools", "mcp_toolset", "pass", lambda c: _req(_msgs(("user", "hi")), tools=[
            {"type": "mcp_toolset", "mcp_server_name": "srv"}]), plant=False),
        Row("anthropic_tools_unknown_type_blocked", "tools", "unknown", "block", lambda c: _req(_msgs(("user", "hi")), tools=[
            {"type": "future_tool_20990101", "name": "x", "description": f"{c}"}])),
        Row("anthropic_tools_custom_schema_description", "tools", "custom", "redact", lambda c: _req(_msgs(("user", "hi")), tools=[
            {"name": "lookup", "description": "d", "input_schema": _schema(f"the {c} to use")}]), spec=False),
        # top-level fields
        Row("anthropic_system_string", "top-level", "system", "redact", lambda c: _req(
            _msgs(("user", "hi")), system=f"You are helpful. Key {c}"), spec=False),
        Row("anthropic_system_blocks", "top-level", "system", "redact", lambda c: _req(_msgs(("user", "hi")), system=[
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=0.0.0; cc_entrypoint=sdk-cli;"},
            {"type": "text", "text": f"rules {c}", "cache_control": {"type": "ephemeral"}}]), spec=False),
        Row("anthropic_metadata_user_id", "top-level", "metadata", "redact", lambda c: _req(
            _msgs(("user", "hi")), metadata={"user_id": f"user-{c}"}), spec=False),
        Row("anthropic_stop_sequences", "top-level", "stop_sequences", "redact", lambda c: _req(
            _msgs(("user", "hi")), stop_sequences=[c]), spec=False),
        Row("anthropic_output_config_schema", "top-level", "output_config", "redact", lambda c: _req(
            _msgs(("user", "hi")), output_config={"format": {"type": "json_schema", "schema": _schema(f"use {c}")}}),
            spec=False),
        Row("anthropic_mcp_servers_authorization_token", "top-level", "mcp_servers", "pass", lambda c: _req(
            _msgs(("user", "hi")), mcp_servers=[{"type": "url", "url": "https://mcp.example.test/sse", "name": "srv",
                                                  "authorization_token": c}]), spec=False),
    ]
