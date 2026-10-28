"""FR-NAME-02: run the CLI tests under every command name.

`harpo` is the entry point `Harpocrates.cli.main`. `harpocrates` is the deprecation wrapper
`Harpocrates.deprecated.harpocrates`: one notice line on stderr, then the same main. Both are driven the way a console
script drives them (sys.argv set, entry function called) inside click's CliRunner, so a test written as
`runner.invoke(app, [...])` checks the real entry for the name it runs under.

The notice is captured instead of written (deprecated._warn is replaced) because click 8.0's CliRunner mixes stderr
into `result.stdout`, which would break every JSON assertion. That the notice really goes to stderr, and only there,
is checked on its own in test_fr_name_02_commands.py.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any
from unittest import mock

import click.testing
import pytest

from Harpocrates.cli import app, main

COMMAND_NAMES = ("harpo", "harpocrates")
BIN = "<bin>"  # in a shell command under test: the program word, run once under each command name


def with_each_name(cmds: list[str]) -> list[str]:
    """Each command holding the BIN placeholder, once per command name (in order); the others once."""
    return [out for c in cmds for out in ([c.replace(BIN, n) for n in COMMAND_NAMES] if BIN in c else [c])]


class _Entry:
    """What CliRunner.invoke needs of a command: a name and a main() that runs the console-script function."""

    def __init__(self, name: str, run: Callable[[], None]) -> None:
        self.name = name
        self._run = run

    def main(self, args: Any = None, prog_name: Any = None, **extra: Any) -> None:
        with mock.patch.object(sys, "argv", [self.name, *(args or ())]):
            self._run()


class CommandNameRunner:
    """A stand-in for `CliRunner()` whose `invoke(app, args)` runs under the command name chosen with `use`."""

    def __init__(self) -> None:
        self.name = "harpo"
        self.notices: list[str] = []
        self._runner = click.testing.CliRunner()

    def use(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run the next invokes under `name`; for the old name, capture the deprecation notice line."""
        self.name = name
        if name == "harpocrates":
            from Harpocrates import deprecated

            monkeypatch.setattr(deprecated, "_warn", self.notices.append)

    def invoke(self, cli: Any, args: Any = None, **kwargs: Any) -> click.testing.Result:
        """Same as CliRunner.invoke(app, args, ...), but through the entry function of the command name in use."""
        assert cli is app, "the command-name runner only drives the real CLI"
        if self.name == "harpocrates":
            from Harpocrates import deprecated

            entry, expected = _Entry(self.name, deprecated.harpocrates), [deprecated.CLI_NOTICE]
        else:
            entry, expected = _Entry(self.name, main), []
        self.notices.clear()
        result = click.testing.CliRunner.invoke(self._runner, entry, args, **kwargs)  # type: ignore[arg-type]
        assert self.notices == expected, f"`{self.name}` must print exactly {len(expected)} notice line(s)"
        return result
