#!/usr/bin/env python3
"""A tiny MCP server for the g5 gate test (FR-GATE-07): one tool that returns a local file as an image or as a PDF
resource, without checking that the bytes are either, as a careless MCP server might. Standard library only; JSON-RPC
over stdio, one message per line.

    python -I bench/mcp_image_server.py ROOT

Tool `show_file(path, as)` answers with one content block holding the file's bytes in base64: an image labelled image/png
(as "image", the default) or an embedded resource labelled application/pdf (as "pdf"). Used only by bench/gate_e2e.py
in a throwaway home. It reads a file only when its resolved path is inside ROOT (the throwaway repo).
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Optional

PROTOCOL = "2025-06-18"
TOOL = {"name": "show_file", "description": "Show a local file as an image.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "as": {"type": "string", "enum": ["image", "pdf"]}},
                        "required": ["path"]}}


def answer(request: dict, root: Path) -> Optional[dict]:
    """
    Description: Answer one JSON-RPC message.
    Parameters:
        request: the parsed message.
        root: the only folder files may be read from.
    Output: The response object, or None for a notification or a response (no id, or no method). An unknown method gets
        error -32601; a call to another tool, or on a file outside root or that cannot be read, gets an MCP tool error
        (isError true) with no file content.
    """
    if "id" not in request or not isinstance(request.get("method"), str):
        return None
    method, params, reply = request.get("method"), request.get("params") or {}, {"jsonrpc": "2.0", "id": request["id"]}
    if method == "initialize":
        return {**reply, "result": {"protocolVersion": params.get("protocolVersion", PROTOCOL),
                                    "capabilities": {"tools": {}}, "serverInfo": {"name": "media", "version": "1"}}}
    if method == "tools/list":
        return {**reply, "result": {"tools": [TOOL]}}
    if method == "tools/call":
        arguments = params.get("arguments") or {}
        refused = {**reply, "result": {"content": [{"type": "text", "text": "cannot read the file"}], "isError": True}}
        if params.get("name") != "show_file":
            return refused
        try:
            target = Path(str(arguments.get("path", ""))).resolve()
            if not target.is_relative_to(root):
                return refused
            data = base64.b64encode(target.read_bytes()).decode("ascii")
        except (OSError, RuntimeError):
            return refused
        if arguments.get("as") == "pdf":
            block = {"type": "resource", "resource": {"uri": "file:///attachment.pdf", "mimeType": "application/pdf",
                                                      "blob": data}}
        else:
            block = {"type": "image", "data": data, "mimeType": "image/png"}
        return {**reply, "result": {"content": [block]}}
    if method == "ping":
        return {**reply, "result": {}}
    return {**reply, "error": {"code": -32601, "message": "method not found"}}


def main(argv: list[str]) -> int:
    """
    Description: Serve requests from stdin until it closes.
    Parameters:
        argv: the command line; argv[1] is the root folder.
    Output: Exit status 0; 2 without a root. A line that is not JSON, or a batch, is skipped.
    """
    if len(argv) != 2:
        sys.stderr.write("usage: mcp_image_server.py ROOT\n")
        return 2
    root = Path(argv[1]).resolve()
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        response = answer(request, root) if isinstance(request, dict) else None
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
