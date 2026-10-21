"""Throwaway homes and a read-only guard for running real agents against the mock provider (FR-PROOF-04, G2).

Agent mode runs `claude` and `codex` for real, so nothing it does may reach the user's own files. Three layers:

1. Each agent gets a throwaway HOME (and CODEX_HOME, TMPDIR, repo) under the system temp folder. Its environment is built
   from scratch: PATH and a few harmless variables, a fake API key, a loopback base URL. No token, proxy or config
   variable of the calling process is passed on.
2. Harpo is installed into that home through a test-side bootstrap that points Harpocrates.paths.home at it, so the real
   ~/.harpo is never touched (SEC-08: paths.home() reads the password database, not $HOME, which is exactly why a plain
   HOME change is not enough). The hook and MCP commands the installer writes start `python-throwaway`, a wrapper that runs
   the same interpreter with the same bootstrap, so the agent's own hook processes also see the throwaway home.
3. The real ~/.harpo, ~/.harpocrates, ~/.claude/settings.json, ~/.codex/config.toml and ~/.codex/auth.json are snapshotted
   (read-only; auth.json by existence, mode, size and mtime only, never its content) before and after, and the difference
   is reported.
"""
from __future__ import annotations

import importlib.util
import ipaddress
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
FAKE_API_KEY = "sk-ant-api03-FAKE-NOT-A-REAL-KEY-000000000000000000000000"
FAKE_OPENAI_KEY = "sk-FAKE-NOT-A-REAL-KEY-0000000000000000"
PREFIX = "harpo-canary-"
REFUSED_EXIT = 70
# A proxy that nothing listens on (the discard port): a client that honors proxy variables and tries to reach anything but
# the loopback mock (NO_PROXY) fails to connect instead of reaching the internet. Defense in depth, not the only guard.
DEAD_PROXY = "http://127.0.0.1:9"
INSTALL_TIMEOUT = 180

# Runs in the child interpreter (python -I -c BOOT HOME RECORD_PYTHON MODULE ARGS...). No single quote may appear in it:
# the wrapper script embeds it in single quotes.
BOOT = """import os, pathlib, pwd, runpy, sys
home = sys.argv.pop(1)
record_python = sys.argv.pop(1)
target = pathlib.Path(home)
real = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
installing = record_python != "-"
if not target.is_absolute() or target.resolve() == real or real in target.resolve().parents or (installing and os.environ.get("HOME") != home):
    sys.stderr.write("bootstrap: refusing a home that is, or is inside, the real home, or (installing) that HOME does not name\\n")
    sys.exit(70)
import Harpocrates.paths as paths
paths.home = lambda: target
if installing:
    sys.executable = record_python
upstream = os.environ.get("HARPO_E2E_UPSTREAM")
if upstream and sys.argv[1] == "Harpocrates.gate":
    import dataclasses
    import Harpocrates.gate.config as config
    names = ("anthropic_upstream", "openai_upstream", "chatgpt_upstream")
    config.GateConfig = dataclasses.make_dataclass("GateConfig", [(n, str, dataclasses.field(default=upstream)) for n in names],
                                                   bases=(config.GateConfig,), frozen=True)
runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)
"""
# HARPO_E2E_UPSTREAM (bench/guard_e2e.py only): a gate that `harpo guard up` starts inside the throwaway home forwards to the
# mock provider. It exists only in this test bootstrap; the product has no setting that moves the gate's upstream.


class IsolationError(Exception):
    """
    Description: A step would have reached outside the throwaway home, or used a non-loopback address.
    Parameters:
        None: the message says which path or address.
    Output: Raised before anything is written or started.
    """


@dataclass(frozen=True)
class Throwaway:
    """
    Description: The folders of one throwaway agent session.
    Parameters:
        root: the session's folder, named harpo-canary-XXXX under the system temp folder.
        home: HOME for the agent (and for Harpo's installer).
        repo: the seeded canary repo the agent works in.
    Output: An immutable record; tmp and bin are folders under root.
    """

    root: Path
    home: Path
    repo: Path

    @property
    def tmp(self) -> Path:
        """
        Description: The agent's TMPDIR.
        Parameters: None
        Output: root/tmp.
        """
        return self.root / "tmp"

    @property
    def wrapper(self) -> Path:
        """
        Description: Where the python-throwaway wrapper lives.
        Parameters: None
        Output: root/bin/python-throwaway.
        """
        return self.root / "bin" / "python-throwaway"


def _real_home_module():
    """
    Description: Load the session guard's snapshot module from cli/tests/_real_home.py.
    Parameters: None
    Output: The module (snapshot, changed, real_home), loaded by file path so that bench/ needs no test path.
    """
    spec = importlib.util.spec_from_file_location("_real_home_for_bench", ROOT / "cli" / "tests" / "_real_home.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_outside_real_home(path: Path) -> None:
    """
    Description: Refuse a path that is, or is inside, the real home folder (the password database's, never $HOME).
    Parameters:
        path: the path to check.
    Output: None. Raises IsolationError when path resolves to the real home or below it.
    """
    real = _real_home_module().real_home().resolve()
    resolved = Path(path).resolve()
    if resolved == real or real in resolved.parents:
        raise IsolationError(f"{path} is inside the real home folder")


def make_throwaway(parent: Optional[Path] = None) -> Throwaway:
    """
    Description: Create the folders of a throwaway agent session.
    Parameters:
        parent: the folder to create it in; None means the system temp folder.
    Output: A Throwaway with root, home, repo, tmp and bin created. Raises IsolationError when the root would be in the
        real home.
    """
    root = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent)).resolve()
    try:
        assert_outside_real_home(root)
    except IsolationError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    for name in ("home", "repo", "tmp", "bin"):
        (root / name).mkdir()
    return Throwaway(root, root / "home", root / "repo")


def discard(tw: Throwaway) -> None:
    """
    Description: Delete a throwaway session folder made by make_throwaway.
    Parameters:
        tw: the session.
    Output: None. Raises IsolationError, deleting nothing, unless the folder is named harpo-canary-* and is outside the
        real home.
    """
    if not tw.root.name.startswith(PREFIX):
        raise IsolationError(f"refusing to delete {tw.root}: not a folder this harness made")
    assert_outside_real_home(tw.root)
    shutil.rmtree(tw.root)


def real_home_snapshot(home: Optional[Path] = None) -> dict:
    """
    Description: Take a read-only snapshot of the real home's Harpo state and agent config.
    Parameters:
        home: the home to describe, defaulting to the real one.
    Output: A dict from relative path to (kind, mode, size, mtime_ns, sha256) for ~/.harpo and ~/.harpocrates (every entry
        under them), ~/.claude/settings.json and ~/.codex/config.toml; for ~/.codex/auth.json the same row with sha256
        None: its content is never opened. None marks a path that is not there. Nothing is written.
    """
    module = _real_home_module()
    base = home if home is not None else module.real_home()
    rows = module.snapshot(base)
    try:
        info = os.lstat(base / ".codex" / "auth.json")
        kind = "f" if stat.S_ISREG(info.st_mode) else "o"
        rows[".codex/auth.json"] = (kind, stat.S_IMODE(info.st_mode), info.st_size, info.st_mtime_ns, None)
    except OSError:
        rows[".codex/auth.json"] = None
    return rows


def snapshot_diff(before: dict, after: dict) -> list[str]:
    """
    Description: List the paths whose snapshot row changed.
    Parameters:
        before: the earlier snapshot.
        after: the later one.
    Output: Sorted relative paths that were added, removed or changed; never a content.
    """
    return _real_home_module().changed(before, after)


def write_wrapper(tw: Throwaway, python: Optional[str] = None) -> Path:
    """
    Description: Write python-throwaway, an interpreter stand-in that runs the bootstrap before the requested module.
    Parameters:
        tw: the session; the script goes to tw.wrapper (mode 0755).
        python: the real interpreter to run, defaulting to this one.
    Output: The script's path. Invoked as `python-throwaway [flags] -m MODULE ARGS`, it execs
        `PYTHON flags -c BOOT HOME - MODULE ARGS`, so MODULE sees paths.home() pinned to tw.home. Without -m it exits 2.
    """
    real = python or sys.executable
    script = "\n".join([
        "#!/bin/sh",
        "# Throwaway-home interpreter for the canary harness (FR-PROOF-04): same Python, paths.home() pinned to one folder.",
        'flags=""',
        'while [ "$#" -gt 0 ] && [ "$1" != "-m" ]; do flags="$flags $1"; shift; done',
        'if [ "$1" != "-m" ] || [ -z "$2" ]; then echo "python-throwaway: expected flags, then -m MODULE" >&2; exit 2; fi',
        'module="$2"; shift 2',
        f'exec {shlex.quote(real)} $flags -c {shlex.quote(BOOT)} {shlex.quote(str(tw.home))} - "$module" "$@"',
        "",
    ])
    tw.wrapper.parent.mkdir(parents=True, exist_ok=True)
    tw.wrapper.write_text(script, encoding="utf-8")
    tw.wrapper.chmod(0o755)
    return tw.wrapper


def assert_loopback_url(url: str) -> None:
    """
    Description: Refuse a mock URL that is not on the loopback interface.
    Parameters:
        url: the URL an agent would be pointed at.
    Output: None. Raises IsolationError for any host but an IPv4 or IPv6 loopback literal or localhost.
    """
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        ok = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = False
    if not ok:
        raise IsolationError(f"the mock URL {url!r} is not on the loopback interface")


def clean_env(tw: Throwaway, mock_url: str, extra: dict, path: Optional[str] = None) -> dict:
    """
    Description: Build the environment of an agent or installer process from scratch.
    Parameters:
        tw: the session.
        mock_url: the mock provider's URL; must be loopback.
        extra: variables to add, such as the fake API key and the base URL.
        path: PATH to use instead of the caller's.
    Output: A dict with PATH, HOME, TMPDIR, USER/LOGNAME/SHELL when set, locale, a dead proxy for everything but loopback,
        and git isolation, plus extra. No other variable of the calling process is included. Raises IsolationError for a
        non-loopback mock URL.
    """
    assert_loopback_url(mock_url)
    env = {"PATH": path or os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tw.home), "TMPDIR": str(tw.tmp),
           "TERM": "dumb", "LANG": "en_US.UTF-8", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
           **{name: DEAD_PROXY for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")},
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0"}
    for name in ("USER", "LOGNAME", "SHELL"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    env.update(extra)
    return env


def install_harpo(tw: Throwaway, target: str, path: Optional[str] = None) -> subprocess.CompletedProcess:
    """
    Description: Run `harpo install --TARGET` into the throwaway home through the bootstrap.
    Parameters:
        tw: the session.
        target: claude-code or codex.
        path: PATH for the installer, or None for the caller's (the installer calls `claude mcp add` when it finds claude).
    Output: The CompletedProcess (stdout and stderr captured as text). The installer sees tw.home as both HOME and
        paths.home(), records the wrapper as the interpreter in every hook and MCP command, and never gets the real home.
        Raises IsolationError, running nothing, if tw.home is inside the real home.
    """
    assert_outside_real_home(tw.home)
    wrapper = write_wrapper(tw)
    env = clean_env(tw, "http://127.0.0.1:1", {}, path)
    argv = [sys.executable, "-I", "-c", BOOT, str(tw.home), str(wrapper), "Harpocrates.cli", "install", f"--{target}"]
    return subprocess.run(argv, cwd=tw.home, env=env, capture_output=True, text=True, timeout=INSTALL_TIMEOUT,
                          stdin=subprocess.DEVNULL, check=False)
