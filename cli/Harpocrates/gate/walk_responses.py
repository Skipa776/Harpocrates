"""Walk an OpenAI Responses request (POST /v1/responses): parser spec, "OpenAI Responses".

Item types, message part types and tool types are closed lists: an unknown one raises RefusalError(UNKNOWN_TYPE). The fields
inside a known item are an open list. History the gate cannot see (previous_response_id, conversation) blocks in v1
(decision 1), and so does anything that is not shaped as the format says.
"""
from __future__ import annotations

from typing import Callable, Iterator

from Harpocrates.gate.blocks import HISTORY, RefusalError
from Harpocrates.gate.walk_common import (
    NO_SKIP,
    Op,
    fields,
    leaf,
    refuse_shape,
    refuse_type,
    strings,
)

# Class I at the top of the body.
TOP_IDENTIFIERS = frozenset({"model", "include", "reasoning", "store", "stream", "service_tier", "tool_choice",
                             "parallel_tool_calls", "truncation", "background", "stream_options"})
IDS = frozenset({"type", "id", "call_id", "status", "name", "namespace", "server_label", "approval_request_id"})
BUILTIN_TOOLS = ("web_search", "file_search", "code_interpreter", "computer", "image_generation", "shell", "local_shell",
                 "apply_patch", "tool_search")
ARGUMENTS = frozenset({"arguments"})
TYPE = frozenset({"type"})


def _free(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk free-form JSON (a schema, a tool's action) where a member may be named like anything.
    Parameters:
        value: the value.
        path: its path.
    Output: Every string as a Leaf; nothing is skipped by name.
    """
    return strings(value, path, deep=NO_SKIP)


def _arguments(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk tool-search arguments, which arrive as a JSON string or as an object.
    Parameters:
        value: the arguments.
        path: their path.
    Output: A string is one Leaf with as_json=True; an object is walked with its free-form keys.
    """
    if isinstance(value, str):
        yield from leaf(path, value, as_json=True)
    else:
        yield from strings(value, path, keys=True, deep=NO_SKIP)


def _output(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk the output of a tool call: a string, or a list of input parts.
    Parameters:
        value: the output.
        path: its path.
    Output: Operations. Anything else raises RefusalError(PARSE).
    """
    if isinstance(value, str):
        yield from leaf(path, value)
    elif isinstance(value, list):
        for position, part in enumerate(value):
            yield from _part(part, path + (position,))
    elif value is not None:
        raise refuse_shape()


_PARTS: dict[str, Callable[[dict, tuple], Iterator[Op]]] = {
    "input_text": lambda p, w: fields(p, w, skip=TYPE),
    "output_text": lambda p, w: fields(p, w, skip=TYPE, special={
        "annotations": lambda v, q: strings(v, q, skip=TYPE), "logprobs": lambda v, q: strings(v, q, deep=NO_SKIP)}),
    "refusal": lambda p, w: fields(p, w, skip=TYPE),
    "input_image": lambda p, w: fields(p, w, skip=frozenset({"type", "file_id", "detail"})),
    "input_file": lambda p, w: fields(p, w, skip=frozenset({"type", "file_id", "file_data"})),
    "input_audio": lambda p, w: fields(p, w, skip=frozenset({"type", "data", "format"})),
}


def _part(part: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk one message or output content part by its type.
    Parameters:
        part: the part.
        path: its path.
    Output: Operations. A part that is not an object raises RefusalError(PARSE); an unlisted type, RefusalError(UNKNOWN_TYPE).
    """
    if not isinstance(part, dict):
        raise refuse_shape()
    handler = _PARTS.get(part.get("type")) if isinstance(part.get("type"), str) else None
    if handler is None:
        raise refuse_type()
    yield from handler(part, path)


def _message(item: dict, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a message item: a content string, or a list of parts.
    Parameters:
        item: the item (with or without type "message").
        path: its path.
    Output: Operations; role, id, status and phase are identifiers.
    """
    yield from fields(item, path, skip=frozenset({"type", "id", "role", "status", "phase"}), special={"content": _output})


def _tools(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk a tool list: function, custom, mcp, namespace and the built-in tools.
    Parameters:
        value: the list.
        path: its path.
    Output: Operations. mcp credentials (authorization, headers) are class K and are not scanned. An unlisted tool type
        raises RefusalError(UNKNOWN_TYPE).
    """
    if value is None:
        return
    if not isinstance(value, list):
        raise refuse_shape()
    for position, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise refuse_shape()
        where, kind = path + (position,), tool.get("type")
        if kind == "function":
            yield from fields(tool, where, skip=frozenset({"type", "name"}), special={"parameters": _free, "output_schema": _free})
        elif kind == "custom":
            yield from fields(tool, where, skip=frozenset({"type", "name"}), special={
                "format": lambda v, q: strings(v, q, skip=frozenset({"type", "syntax"}))})
        elif kind == "mcp":
            yield from fields(tool, where, skip=frozenset({"type", "authorization", "headers"}))
        elif kind == "namespace":
            yield from fields(tool, where, skip=frozenset({"type", "name"}), special={"tools": _tools})
        elif isinstance(kind, str) and kind.startswith(BUILTIN_TOOLS):
            yield from fields(tool, where, skip=frozenset({"type", "name"}), special={"user_location": _free})
        else:
            raise refuse_type()


def _simple(skip: frozenset = IDS, **special: Callable[[object, tuple], Iterator[Op]]) -> Callable[[dict, tuple], Iterator[Op]]:
    """
    Description: Build the rule for an item whose members are scanned generically, with a few given their own rule.
    Parameters:
        skip: member names that are identifiers.
        **special: member name -> function(value, path).
    Output: A function(item, path) for the item table.
    """
    return lambda item, path: fields(item, path, skip=skip, special=special or None)


_ITEMS: dict[str, Callable[[dict, tuple], Iterator[Op]]] = {
    "message": _message,
    "function_call": lambda i, p: fields(i, p, skip=IDS, json_keys=ARGUMENTS),
    "function_call_output": _simple(output=_output), "custom_tool_call_output": _simple(output=_output),
    "custom_tool_call": _simple(),
    "reasoning": lambda i, p: fields(i, p, skip=frozenset({"type", "id", "status"}), cls="S"),
    "local_shell_call": _simple(action=lambda v, q: strings(v, q, skip=TYPE, deep=NO_SKIP)),
    "local_shell_call_output": lambda i, p: fields(i, p, skip=IDS, json_keys=frozenset({"output"})),
    "shell_call": _simple(action=lambda v, q: strings(v, q, skip=TYPE)),
    "shell_call_output": _simple(),
    "apply_patch_call": _simple(operation=lambda v, q: strings(v, q, skip=TYPE)), "apply_patch_call_output": _simple(),
    "computer_call": _simple(action=lambda v, q: strings(v, q, skip=TYPE)),
    "computer_call_output": _simple(output=lambda v, q: strings(v, q, skip=frozenset({"type", "file_id"}))),
    "web_search_call": _simple(action=lambda v, q: strings(v, q, skip=TYPE)),
    "file_search_call": _simple(results=lambda v, q: strings(v, q, skip=frozenset({"file_id"}))),
    "code_interpreter_call": _simple(skip=IDS | {"container_id"}, outputs=lambda v, q: strings(v, q, skip=TYPE)),
    "image_generation_call": _simple(skip=IDS | {"result"}),
    "mcp_call": lambda i, p: fields(i, p, skip=IDS, json_keys=ARGUMENTS),
    "mcp_list_tools": _simple(tools=lambda v, q: strings(v, q, skip=frozenset({"name"}), deep=NO_SKIP)),
    "mcp_approval_request": lambda i, p: fields(i, p, skip=IDS, json_keys=ARGUMENTS),
    "mcp_approval_response": _simple(),
    "tool_search_call": _simple(arguments=_arguments),
    "tool_search_output": _simple(tools=_tools), "additional_tools": _simple(tools=_tools),
    "item_reference": lambda i, p: iter(()), "compaction": lambda i, p: iter(()),
}


def _input(value: object, path: tuple) -> Iterator[Op]:
    """
    Description: Walk the input: a string, or a list of items.
    Parameters:
        value: the input member.
        path: its path.
    Output: Operations. An item without a type but with a role is a message. A non-object item raises RefusalError(PARSE);
        an unlisted item type, RefusalError(UNKNOWN_TYPE).
    """
    if isinstance(value, str):
        yield from leaf(path, value)
    elif isinstance(value, list):
        for position, item in enumerate(value):
            if not isinstance(item, dict):
                raise refuse_shape()
            kind = item.get("type", "message" if "role" in item else None)
            handler = _ITEMS.get(kind) if isinstance(kind, str) else None
            if handler is None:
                raise refuse_type()
            yield from handler(item, path + (position,))
    elif value is not None:
        raise refuse_shape()


def walk(body: object) -> Iterator[Op]:
    """
    Description: Walk a whole Responses request.
    Parameters:
        body: the parsed request (must be an object).
    Output: Operations in document order. Identifiers, ciphertext (reasoning and compaction items), base64 media and mcp
        credentials are not yielded; every other string member of the body is. Raises RefusalError(HISTORY) for a
        previous_response_id or conversation, RefusalError(PARSE) for a wrong shape and RefusalError(UNKNOWN_TYPE) for
        an unlisted item, part or tool type.
    """
    if not isinstance(body, dict):
        raise refuse_shape()
    if body.get("previous_response_id") is not None or body.get("conversation") is not None:
        raise RefusalError(HISTORY)
    for name, value in body.items():
        path = (name,)
        if name == "input":
            yield from _input(value, path)
        elif name == "tools":
            yield from _tools(value, path)
        elif name == "prompt":
            yield from strings(value, path, skip=frozenset({"id", "version"}))
        elif name not in TOP_IDENTIFIERS and name not in ("previous_response_id", "conversation"):
            yield from strings(value, path)
