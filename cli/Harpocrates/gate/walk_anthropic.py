"""Walk an Anthropic Messages request (POST /v1/messages and /count_tokens): parser spec, "Anthropic Messages".

The block types, tool types and source types are closed lists: an unknown one raises RefusalError(UNKNOWN_TYPE). The fields
inside a known block are an open list: a string the tables do not name is scanned like its neighbours. Thinking and
redacted-thinking blocks are dropped (decision 5, FR-GATE-10) before anything in them could be scanned.
"""
from __future__ import annotations

from typing import Callable, Iterator

from Harpocrates.gate.walk_common import (
    NO_SKIP,
    OPAQUE,
    Drop,
    Op,
    fields,
    leaf,
    refuse_shape,
    refuse_type,
    strings,
)

THINKING = frozenset({"thinking", "redacted_thinking"})
# Class I at the top of the body: nothing in them is text the user wrote.
TOP_IDENTIFIERS = frozenset({"model", "container", "thinking", "tool_choice", "context_management", "service_tier", "speed",
                             "inference_geo", "cache_control"})
IDS = frozenset({"type", "id", "tool_use_id", "name", "server_name", "caller", "is_error"})
# Typed server tools hold settings, not free text; custom tools have no type or "custom".
SERVER_TOOLS = ("bash_", "code_execution_", "text_editor_", "computer_", "web_search_", "web_fetch_", "memory_",
                "tool_search_tool_", "browser_toolset", "mcp_toolset")

def _content(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk content that is a string, a block, or a list of blocks (tool_result, mcp_tool_result, web fetch).
    Parameters:
        value: the content.
        path: its path.
    Output: Operations. A string is one Leaf; blocks follow the block table. Anything else raises RefusalError(PARSE).
    """
    if isinstance(value, str):
        yield from leaf(path, value)
    elif isinstance(value, list):
        for position, block in enumerate(value):
            yield from _block(block, path + (position,))
    elif isinstance(value, dict):
        yield from _block(value, path)
    elif value is not None:
        raise refuse_shape()


def _block(block: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk one content block by its type.
    Parameters:
        block: the block.
        path: its path.
    Output: Operations. Raises RefusalError(PARSE) when the block is not an object and RefusalError(UNKNOWN_TYPE) when its
        type is missing or not in the table.
    """
    if not isinstance(block, dict):
        raise refuse_shape()
    kind = block.get("type")
    handler = _BLOCKS.get(kind) if isinstance(kind, str) else None
    if handler is None:
        raise refuse_type()
    yield from handler(block, path)


def _source(url_key: str, kinds: dict) -> Callable[[object, tuple], Iterator[Op]]:
    """
    Description: Build the rule for an image or document `source` member.
    Parameters:
        url_key: the member that holds a URL ("url").
        kinds: source type -> "skip" (media or a provider file id), "url" (scan the URL), "text" (scan the text fields) or
            "blocks" (nested content blocks).
    Output: A function(value, path) for fields(special=...). An unlisted source type raises RefusalError(UNKNOWN_TYPE).
    """
    def walk_source(value: object, path: tuple) -> Iterator[Op]:
        """
        Description: Walk one source.
        Parameters:
            value: the source object.
            path: its path.
        Output: Operations by the source type.
        """
        if not isinstance(value, dict):
            raise refuse_shape()
        how = kinds.get(value.get("type")) if isinstance(value.get("type"), str) else None
        if how is None:
            raise refuse_type()
        if how == "url" and isinstance(value.get(url_key), str):
            yield from leaf(path + (url_key,), value[url_key])
        elif how == "text":
            yield from fields(value, path, skip=frozenset({"type", "media_type"}))
        elif how == "blocks":
            yield from _content(value.get("content"), path + ("content",))
    return walk_source


def _text(block: dict, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a text block: its text and its citations (cited_text and the other strings, not their types).
    Parameters:
        block: the block.
        path: its path.
    Output: Operations.
    """
    yield from fields(block, path, skip=frozenset({"type"}),
                      special={"citations": lambda v, p: strings(v, p, skip=frozenset({"type"}), deep=OPAQUE)})


def _tool_call(block: dict, path: tuple) -> Iterator[Op]:
    """
    Description: Walk tool_use, server_tool_use and mcp_tool_use: the input object's values and its free-form keys.
    Parameters:
        block: the block.
        path: its path.
    Output: Operations; input is walked with keys=True and without the signature skip (a tool may take such a parameter).
    """
    yield from fields(block, path, skip=IDS, special={"input": lambda v, p: strings(v, p, keys=True, deep=NO_SKIP)})


def _provider_signed(block: dict, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a block of readable text the provider signs or pairs with ciphertext (compaction, mcp_tool_listing):
        class S, scan only.
    Parameters:
        block: the block.
        path: its path.
    Output: Leaf operations of class S for every string except identifiers, tool names and ciphertext.
    """
    yield from fields(block, path, skip=IDS, cls="S",
                      special={"tools": lambda v, p: strings(v, p, "S", skip=frozenset({"name"}), deep=NO_SKIP)})


def _web_search_result(block: dict, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a web_search_tool_result: titles and URLs are provider-signed readable text (class S).
    Parameters:
        block: the block.
        path: its path.
    Output: Leaf operations of class S; ciphertext, page age and error codes are not scanned.
    """
    skip = frozenset({"type", "page_age", "error_code"})
    yield from fields(block, path, skip=IDS, cls="S", special={"content": lambda v, p: strings(v, p, "S", skip=skip, deep=OPAQUE)})


def _tool_output(block: dict, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a code execution, bash or text editor tool result: stdout, stderr, file text and lines are class R.
    Parameters:
        block: the block.
        path: its path.
    Output: Operations; file ids, types and return codes are not scanned.
    """
    skip = frozenset({"type", "file_id", "file_type", "return_code", "encrypted_stdout"})
    yield from fields(block, path, skip=IDS, special={"content": lambda v, p: strings(v, p, skip=skip, deep=OPAQUE)})


def _nothing(block: dict, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a block that holds no text (tool_search_tool_result, container_upload, tool_reference, tool_removal).
    Parameters:
        block: the block.
        path: its path.
    Output: No operations.
    """
    return iter(())


def _tool_definition(tool: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a custom tool definition (a tools[] entry or the definition of a tool_addition).
    Parameters:
        tool: the definition.
        path: its path.
    Output: Operations: the description, and every string of input_schema and input_examples; the name is an identifier.
    """
    if not isinstance(tool, dict):
        raise refuse_shape()
    free = lambda v, p: strings(v, p, deep=NO_SKIP)  # noqa: E731 - a schema may name a property "signature"
    yield from fields(tool, path, skip=frozenset({"type", "name"}), special={"input_schema": free, "input_examples": free})


_IMAGE = _source("url", {"base64": "skip", "file": "skip", "url": "url"})
_DOCUMENT = _source("url", {"base64": "skip", "file": "skip", "url": "url", "text": "text", "content": "blocks"})
_BLOCKS: dict[str, Callable[[dict, tuple], Iterator[Op]]] = {
    "text": _text,
    "image": lambda b, p: fields(b, p, skip=frozenset({"type"}), special={"source": _IMAGE}),
    "document": lambda b, p: fields(b, p, skip=frozenset({"type"}), special={"source": _DOCUMENT}),
    "search_result": lambda b, p: fields(b, p, skip=frozenset({"type"}), special={"content": _content}),
    "tool_use": _tool_call, "server_tool_use": _tool_call, "mcp_tool_use": _tool_call,
    "tool_result": lambda b, p: fields(b, p, skip=IDS, special={"content": _content}),
    "mcp_tool_result": lambda b, p: fields(b, p, skip=IDS, special={"content": _content}),
    "web_fetch_tool_result": lambda b, p: fields(b, p, skip=IDS, special={"content": lambda v, q: fields(
        v, q, skip=frozenset({"type", "retrieved_at"}), special={"content": _content}) if isinstance(v, dict) else strings(v, q)}),
    "compaction": _provider_signed, "mcp_tool_listing": _provider_signed, "web_search_tool_result": _web_search_result,
    "code_execution_tool_result": _tool_output, "bash_code_execution_tool_result": _tool_output,
    "text_editor_code_execution_tool_result": _tool_output,
    "browser_state": lambda b, p: strings(b, p, skip=frozenset({"type"})),
    "tool_addition": lambda b, p: fields(b, p, skip=frozenset({"type"}), special={"definition": _tool_definition}),
    "tool_search_tool_result": _nothing, "container_upload": _nothing, "tool_reference": _nothing, "tool_removal": _nothing,
}


def _system(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk the top-level system prompt: a string, or a list of text blocks (their order and cache_control stay).
    Parameters:
        value: the system member.
        path: its path.
    Output: Operations. A block that is not a text block raises RefusalError(UNKNOWN_TYPE).
    """
    if isinstance(value, str):
        yield from leaf(path, value)
    elif isinstance(value, list):
        for position, block in enumerate(value):
            if not isinstance(block, dict):
                raise refuse_shape()
            if block.get("type") != "text":
                raise refuse_type()
            yield from _text(block, path + (position,))
    elif value is not None:
        raise refuse_shape()


def _messages(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk the conversation, dropping thinking blocks (FR-GATE-10).
    Parameters:
        value: the messages member.
        path: its path.
    Output: Operations. A thinking or redacted-thinking block becomes a Drop; a message that would be left with no block
        becomes one Drop of the whole message (the API accepts consecutive messages of one role, not an empty content).
        A string content is one Leaf (and stays a string). Mid-conversation system messages follow the same table.
    """
    if not isinstance(value, list):
        raise refuse_shape()
    for position, message in enumerate(value):
        if not isinstance(message, dict):
            raise refuse_shape()
        where, content = path + (position,), message.get("content")
        for name, other in message.items():
            if name not in ("role", "content"):
                yield from strings(other, where + (name,))
        if isinstance(content, str):
            yield from leaf(where + ("content",), content)
        elif isinstance(content, list):
            dropped = [i for i, b in enumerate(content) if isinstance(b, dict) and b.get("type") in THINKING]
            if dropped and len(dropped) == len(content):
                yield Drop(where)
                continue
            for i in dropped:
                yield Drop(where + ("content", i))
            for i, block in enumerate(content):
                if i not in dropped:
                    yield from _block(block, where + ("content", i))
        elif content is not None:
            raise refuse_shape()


def _tools(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk the tool list: custom tools, typed server tools and mcp_toolset.
    Parameters:
        value: the tools member.
        path: its path.
    Output: Operations. A tool type that is not custom and not a known server tool raises RefusalError(UNKNOWN_TYPE).
    """
    if value is None:
        return
    if not isinstance(value, list):
        raise refuse_shape()
    for position, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise refuse_shape()
        kind = tool.get("type", "custom")
        if kind == "custom":
            yield from _tool_definition(tool, path + (position,))
        elif isinstance(kind, str) and kind.startswith(SERVER_TOOLS):
            yield from fields(tool, path + (position,), skip=frozenset({"type", "name"}))
        else:
            raise refuse_type()


def walk(body: object) -> Iterator[Op]:
    """
    Description: Walk a whole Messages request.
    Parameters:
        body: the parsed request (must be an object).
    Output: Operations in document order. Identifiers (model, tool_choice, thinking settings, container, ...) and connector
        credentials (mcp_servers[].authorization_token) are not yielded; every other string member of the body is.
        Raises RefusalError(PARSE) for a shape the format does not have and RefusalError(UNKNOWN_TYPE) for an unlisted
        block, tool or source type.
    """
    if not isinstance(body, dict):
        raise refuse_shape()
    for name, value in body.items():
        path = (name,)
        if name == "system":
            yield from _system(value, path)
        elif name == "messages":
            yield from _messages(value, path)
        elif name == "tools":
            yield from _tools(value, path)
        elif name == "mcp_servers":
            yield from strings(value, path, skip=frozenset({"type", "authorization_token"}))
        elif name not in TOP_IDENTIFIERS:
            yield from strings(value, path)
