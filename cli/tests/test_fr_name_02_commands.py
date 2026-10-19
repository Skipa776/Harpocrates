"""FR-NAME-02: the commands are `harpo` and `harpo-mcp`. `harpocrates` and `harpocrates-mcp` keep working for two
minor releases and print one deprecation line, on stderr only (stdout carries JSON and the MCP stdio protocol).

The CLI tests in test_cli.py and test_cli_version.py also run once under each name (cli_entry.py).
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[2]
CLI_NOTICE = "harpocrates is now harpo; this name stops working after two minor releases."
MCP_NOTICE = "harpocrates-mcp is now harpo-mcp; this name stops working after two minor releases."
SCRIPTS = {
    "harpo": "Harpocrates.cli:main",
    "harpo-mcp": "Harpocrates.mcp.server:main",
    "harpocrates": "Harpocrates.deprecated:harpocrates",
    "harpocrates-mcp": "Harpocrates.deprecated:harpocrates_mcp",
}


def _run(entry: Callable[[], object], argv: list[str], monkeypatch: pytest.MonkeyPatch,
         capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    """Run a console-script function as the shell would: argv set, SystemExit read as the exit code."""
    monkeypatch.setattr(sys, "argv", argv)
    capsys.readouterr()
    try:
        entry()
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    out, err = capsys.readouterr()
    return code, out, err


def _same_but_timing(text: str) -> str:
    """The scan summary says how long it took (`in 5ms`); that is the one thing two runs do not repeat."""
    return re.sub(r"\d+(?:\.\d+)?ms", "<ms>", text)


def _scripts() -> dict[str, str]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]


def test_fr_name_02_console_scripts_are_declared() -> None:
    assert _scripts() == SCRIPTS
    for target in SCRIPTS.values():
        module, _, attr = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attr)), target


def test_fr_name_02_notice_text_is_exact() -> None:
    from Harpocrates import deprecated

    assert deprecated.CLI_NOTICE == CLI_NOTICE
    assert deprecated.MCP_NOTICE == MCP_NOTICE


@pytest.fixture
def cases(tmp_path: Path) -> list[list[str]]:
    clean = tmp_path / "config.txt"
    clean.write_text("APP_NAME=Test\n", encoding="utf-8")
    secret = tmp_path / "secrets.env"
    secret.write_text("AWS_KEY=AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")
    return [
        ["--version"],
        ["version"],
        ["scan", str(clean)],
        ["scan", str(secret), "--fail-on", "none"],
        ["scan", str(secret)],
        ["scan", str(clean), "--fail-on", "bogus"],  # a usage error: exit 2, text on stderr
        ["no-such-command"],
    ]


def test_fr_name_02_old_commands_warn_and_work(cases: list[list[str]], monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture) -> None:
    from Harpocrates import deprecated
    from Harpocrates.cli import main

    for args in cases:
        new = _run(main, ["harpo", *args], monkeypatch, capsys)
        old = _run(deprecated.harpocrates, ["harpocrates", *args], monkeypatch, capsys)
        assert old[0] == new[0], args  # the same exit code
        assert _same_but_timing(old[1]) == _same_but_timing(new[1]), args  # the same stdout
        notice, _, rest = old[2].partition("\n")
        assert notice == CLI_NOTICE, args  # one line first, on stderr
        # what follows is the new command's stderr; a usage error names the command typed, the one thing that differs
        assert _same_but_timing(rest).replace("harpocrates", "harpo") == _same_but_timing(new[2]).replace(
            "harpocrates", "harpo"), args
        assert "now harpo" not in new[1] + new[2], args


def test_fr_name_02_new_commands_no_notice(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from Harpocrates import __version__
    from Harpocrates.cli import main
    from Harpocrates.mcp import server

    code, out, err = _run(main, ["harpo", "--version"], monkeypatch, capsys)
    assert (code, out, err) == (0, f"harpocrates {__version__}\n", "")

    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: None)
    assert _run(server.main, ["harpo-mcp"], monkeypatch, capsys) == (0, "", "")


def test_fr_name_02_mcp_notice_on_stderr_only(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from Harpocrates import deprecated
    from Harpocrates.mcp import server

    calls: list[dict] = []
    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: calls.append(kwargs))
    code, out, err = _run(deprecated.harpocrates_mcp, ["harpocrates-mcp"], monkeypatch, capsys)
    assert (code, out, err) == (0, "", MCP_NOTICE + "\n")
    assert calls == [{"transport": "stdio"}]  # the same server, started the same way as harpo-mcp


def test_fr_name_02_mcp_process_keeps_stdout_clean(tmp_path: Path) -> None:
    """The real thing: a stdio server told nothing and closed at once writes the notice to stderr and nothing to stdout."""
    code = "from Harpocrates.deprecated import harpocrates_mcp; harpocrates_mcp()"
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(ROOT / "cli")}
    done = subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=60, check=False, env=env, cwd=tmp_path)
    assert done.stdout == "", done.stdout
    assert done.stderr.splitlines()[:1] == [MCP_NOTICE], done.stderr
    assert done.stderr.count("is now harpo") == 1


@pytest.mark.parametrize("wrapper, target", [("harpocrates", "Harpocrates.cli"), ("harpocrates_mcp", "Harpocrates.mcp.server")])
def test_fr_name_02_exit_code_is_the_main_s(wrapper: str, target: str, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture) -> None:
    from Harpocrates import deprecated

    module = importlib.import_module(target)
    entry = getattr(deprecated, wrapper)
    monkeypatch.setattr(module, "main", lambda: (_ for _ in ()).throw(SystemExit(7)))
    assert _run(entry, ["x"], monkeypatch, capsys)[0] == 7
    monkeypatch.setattr(module, "main", lambda: None)
    assert _run(entry, ["x"], monkeypatch, capsys)[0] == 0


@pytest.mark.parametrize("wrapper, target", [("harpocrates", "Harpocrates.cli"), ("harpocrates_mcp", "Harpocrates.mcp.server")])
def test_fr_name_02_a_closed_stderr_does_not_stop_the_command(wrapper: str, target: str,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from Harpocrates import deprecated

    ran: list[str] = []
    monkeypatch.setattr(importlib.import_module(target), "main", lambda: ran.append("main"))
    for stream in (None, io.StringIO()):
        if stream is not None:
            stream.close()
        monkeypatch.setattr(sys, "stderr", stream)
        getattr(deprecated, wrapper)()
    assert ran == ["main", "main"]
