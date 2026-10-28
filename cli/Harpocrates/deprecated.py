"""The old command names (FR-NAME-02): `harpocrates` and `harpocrates-mcp` still work, for two more minor releases.

The commands are `harpo` and `harpo-mcp`. Each old name is a console script that points here: it prints one notice line
and then runs the same main as the new name, so the behavior and the exit code are the new command's. The notice goes to
stderr, never stdout: stdout carries the JSON of `--json` and the MCP stdio protocol, and one extra line on it would
break a script or a client.
"""
from __future__ import annotations

import sys

CLI_NOTICE = "harpocrates is now harpo; this name stops working after two minor releases."
MCP_NOTICE = "harpocrates-mcp is now harpo-mcp; this name stops working after two minor releases."


def _warn(notice: str) -> None:
    """
    Description: Write one notice line to stderr, and carry on if stderr cannot take it.
    Parameters:
        notice: the line to print, without a newline.
    Output: None. Writes notice and a newline to sys.stderr, then flushes. A closed, missing or broken stderr is ignored:
        a command that works must not stop because it could not say it is deprecated.
    """
    try:
        sys.stderr.write(notice + "\n")
        sys.stderr.flush()
    except (AttributeError, OSError, ValueError):
        pass


def harpocrates() -> None:
    """
    Description: Entry point of the old `harpocrates` console script: print the notice, then run the CLI.
    Parameters: None
    Output: None. Writes CLI_NOTICE to stderr, then calls Harpocrates.cli.main, so its output is unchanged and its
        SystemExit (the exit code) propagates as it does from `harpo`.
    """
    _warn(CLI_NOTICE)
    from Harpocrates.cli import main

    main()


def harpocrates_mcp() -> None:
    """
    Description: Entry point of the old `harpocrates-mcp` console script: print the notice, then run the MCP server.
    Parameters: None
    Output: None. Writes MCP_NOTICE to stderr, then calls Harpocrates.mcp.server.main (the stdio server). Nothing is
        written to stdout, which belongs to the MCP protocol.
    """
    _warn(MCP_NOTICE)
    from Harpocrates.mcp.server import main

    main()
