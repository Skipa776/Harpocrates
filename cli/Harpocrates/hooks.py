"""Claude Code and Codex PreToolUse hook (FR-READ-03): block built-in reads of any file that holds a secret.

Path deny rules only cover files named like secrets (.env, *.pem). A token in settings.py or app.yaml
would still reach the agent through Read, Grep, or `cat`. This hook scans the target first, with the
same detector and gate threshold as safe_read, and denies the call if anything would be redacted, telling
the agent to use safe_read or safe_grep instead. It prints only paths, counts, types and variable names,
never a value.

Fail closed: both harnesses treat a hook crash or timeout as "allow", so every error, an unparseable
command, a search pattern it can't evaluate safely, and running out of its time budget all exit 2, which
blocks the call. Run as `python -m Harpocrates.hooks` with the hook event JSON on stdin.

SEC-10: before any of that, when the install record says the gate is on, the hook asks Harpocrates/gatecheck.py
whether the Claude Code settings in force send requests anywhere but the gate (base URL, Bedrock, Vertex, custom headers
without the gate token). If so, every tool call of a Claude Code session is denied, in every mode and with or without
--ask: the deny names the setting file, nothing prompts and nothing waits for an answer. Codex events (they carry
`turn_id` and `model`) are skipped, since Codex does not read Claude Code settings. No gate installed: unchanged.

FR-READ-05: paths on the user's allowlist (Harpocrates/allowlist.py) pass, each use logged.
FR-READ-06: reads never prompt. A Read, Grep or Bash call that would expose a secret-bearing file is refused in every
mode. With `--ask` (Claude Code only) an edit of such a file (Edit, MultiEdit, Write, NotebookEdit) asks the user
instead, in the modes where a human answers; any other mode gets the refusal. Yes lets Claude read and edit that one
path with its built-in tools for the rest of the session, never through Bash (Harpocrates/grants.py). The grant is
recorded by `python -m Harpocrates.hooks --post` (PostToolUse, PostToolUseFailure, SessionEnd): those fire only for a
call that ran, so a No never reaches them. Trade-off: an approved edit sends those values to the provider; that is
the user's explicit call. An edit that would leave a Harpo placeholder in the file that it did not hold before
is denied in every mode, granted or not: it would swap a real secret for a fake value. Codex runs without --ask: its
handling of "ask" isn't verified.

Bash coverage is best effort. It follows `sh -c` bodies, `$(...)`, backticks, `cd`, `git -C`,
globs, `~` and `$VAR` paths, `git show rev:path`, and the environment. A script that opens files itself
(`python x.py`, `node -e`, `python - <<EOF`), `find -exec`, `xargs` fed by a pipe, and `git diff` / `git log -p` output get
through. The egress gate (M2) covers what reaches the provider whatever the tool.
A recursive search inside a git work tree is scoped to the files git lists plus git-ignored files named like secret
stores; a secret in an ignored file with an ordinary name can pass `grep -r` (see _scope). Only the gate stops it.
"""
from __future__ import annotations

import errno
import fnmatch
import glob
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from Harpocrates import gatecheck
from Harpocrates.harness import SENSITIVE_FILES

_PLACEHOLDER_TYPE = re.compile(r"<<HARPO:(\w+):[0-9a-f]{4}>>")
_READ_COMMANDS = {"cat", "head", "tail", "less", "more", "bat", "nl", "sed", "awk", "strings", "xxd", "od",
                  "hexdump", "base64", "tac", "cut", "sort", "uniq", "diff", "jq", "yq", "type", "Get-Content",
                  "cp", "dd", "paste", "column", "fold", "rev", "tee", "xargs", "zcat", "zless", "pr", "fmt",
                  "expand", "join", "comm", "source", ".", "blame", "annotate",
                  "mv", "ln", "rsync", "install", "tar", "zip", "ditto"}  # mv/ln: moving a file into an allowlisted folder
# Commands that read every file under a directory argument (cp -r DIR dst, tar -cf a.tar DIR). _TARGET_LAST: the
# last word is where they write, which they don't read (unless -t/--target-directory moves it elsewhere).
_COPY_COMMANDS = {"cp", "mv", "rsync", "install", "ln", "tar", "zip", "ditto"}
_TARGET_LAST = {"cp", "mv", "rsync", "install", "ln", "ditto"}
_SEARCH_COMMANDS = {"grep", "egrep", "fgrep", "rg", "ag", "ack"}
_NAMES_ONLY = {"-l", "--files-with-matches", "-L", "--files-without-match", "-c", "--count", "--files",
               "-q", "--quiet", "--count-matches"}
_SHELLS = {"sh", "bash", "zsh", "dash", "fish", "ksh"}
# An environment dump: `env`, `printenv`, `compgen` anywhere as a word (`echo env | xargs`, `$SHELL -c env`, `\\env`,
# `/usr/bin/env` and `alias e=env` all run it), `export` bare or with -p/-x, `declare`/`typeset`/`readonly`/`local`
# with -p/-x, a bare `set`, `ps e...` (BSD option e prints each process's environment), /proc/*/environ. Not counted,
# as code and not a command: `env` followed by `=`, `:`, `,` or `]` (`env=os.environ`, a YAML or JSON key, a list),
# or preceded by a word character, dot or dash (`my_env`, `.env`, `--env`), or the argument of a call (`dict(env)`;
# a bare `(env)` or `$(env)` still counts). Trade-off: `f(a, env)` and a path such as `proj/env` count, and a
# file or key named env in prose does too, which refuses only while the environment holds a flagged variable.
_ENV_NAME = r"(?:env|printenv|compgen)(?![\w./-])(?![\"']?\s*[:=,\]])"
_ENV_OTHER = (r"\b(?:export|declare|typeset|readonly|local)\s+-[a-zA-Z]*[px]"
              r"|(?:^|[;&|(`\n]\s*)(?:set|export|declare|typeset|readonly)[ \t]*(?:$|[;&|)`\n])"
              r"|/proc/[^\s]*environ|(?<![\w./-])ps\s+[a-zA-Z]*e[a-zA-Z]*(?![\w.=/-])")
_ENV_DUMP = re.compile(r"(?<![\w.(-])" + _ENV_NAME + r"|(?:^|[^\w.)\]])\(\s*" + _ENV_NAME + "|" + _ENV_OTHER)
_SEPARATORS = set(";&|()")  # a token made only of these ends a command (shlex groups them: `&&`, `|&`, `;;`)
_PIPES = {"|", "|&"}
_GREPS = {"grep", "egrep", "fgrep"}
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*\+?=")
# Words that run what follows, with the short options that take a separate value. A search that is the command
# word after them may be reading stdin (_search_command); behind any other command it is scoped without that.
_PREFIXES = {"sudo": "ugphCDrtTUR", "doas": "uC", "env": "uCSP", "command": "", "nice": "n", "nohup": "", "time": "fo",
             "timeout": "sk", "xargs": "adEIJLnPs", "stdbuf": "ioe", "ionice": "cnpPu", "setsid": "", "exec": "",
             "builtin": "", "if": "", "then": "", "elif": "", "else": "", "do": "", "while": "", "until": "", "!": "",
             "{": ""}
# Commands that only name programs, never run them (`command -v` is handled by _command_word): a search name
# among their words is not a search (`which rg`, `echo grep`) unless the text is run (_feeds_a_shell).
_NAMING = {"which", "type", "whereis", "hash", "man", "info", "help", "tldr", "echo", "printf"}
_LONG_VALUE = {"--user", "--group", "--host", "--prompt", "--chdir", "--role", "--type", "--unset", "--split-string",
               "--signal", "--kill-after", "--arg-file", "--delimiter", "--eof", "--max-args", "--max-lines",
               "--max-procs", "--max-chars", "--adjustment"}
_DURATION = re.compile(r"\d+(\.\d+)?[smhd]?$")
_UNRESOLVED = re.compile(r"[*?\[{`$]")  # a word the hook can't turn into paths: glob, brace, backtick, any $
_SINKS = _SHELLS | {"source", ".", "eval", "xargs", "parallel"}  # a pipe into one of these runs its input as commands
_PROSE_SPLIT = re.compile(r"(\|&|\|\||&&|[|&;()`])")  # what ends a command in a heredoc body
_HEREDOC = re.compile(
    r"""'[^']*'|"(?:\\.|[^"\\])*"|\\.|(?<!\S)#[^\n]*"""  # quotes, escapes and comments hide what they hold
    r"""|(?<!<)<<(?!<)(?P<tabs>-?)[ \t]*"""
    r"""(?:(?P<q>['"])(?P<qword>[^\n]+?)(?P=q)|(?P<esc>\\?)(?P<word>[^\s;&|()<>'"\\]+))"""
    r"|\n", re.S)
# A reader's operand that is a variable or a substitution (`cat "$f"`, `cat $(ls)`, backticks): the file names come from
# elsewhere, so every word in the command counts as read. A `$` inside an awk or sed program (it has spaces) is no operand.
_DYNAMIC = re.compile(r"""[^\s'"]*(?:\$[\w@*#?!]|\$\{|\$$|`)[^\s'"]*""")
_CLI_MODULE = re.compile(r"Harpocrates(\.cli(\.__main__)?)?$")
_CLI_NAMES = ("harpo", "harpocrates")  # the command and its old name (FR-NAME-02): both print values when they scan
def _tool_ref(tool: str) -> str:
    """
    Description: Name an MCP tool of the Harpo server the way an agent calls it, and the name an install from before
        the rename gave it, so a refusal still helps a user who has not re-run `harpo install`.
    Parameters:
        tool: The tool's name, such as safe_read.
    Output: A phrase with both qualified names; pure string building.
    """
    return f"mcp__harpo__{tool} (mcp__harpocrates__{tool} until you re-run `harpo install`)"


SCAN_REFUSAL = (f"harpo scan prints secret values. Use {_tool_ref('scan_file')} or scan_text, which return "
                "findings without values.")
SCAN_UNCHECKABLE = SCAN_REFUSAL + " (git history and --show-secrets can't be checked)"


class _ScanUncheckableError(Exception):
    """
    Description: Marks an agent-run scan that uses --history (reads commits) or --show-secrets (prints the values). No
        work-tree check covers those modes.
    Parameters:
        *args: Exception arguments forwarded to Exception.
    Output: Exception instance that causes an outright refusal of the --history or --show-secrets scan.
    """
_SHELL_C = re.compile(r"^-[a-zA-Z]*c[a-zA-Z]*$")
_SUBSHELL = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_ENV_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")
_VAR = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
_ASSIGN_ANY = re.compile(r"(?<![\w$])([A-Za-z_]\w*)(\+?)=(\"[^\"]*\"|'[^']*'|[^\s;&|()`]*)")
_SENSITIVE = re.compile("|".join(map(fnmatch.translate, SENSITIVE_FILES)))  # file names, matched with .match
MAX_SCOPE_FILES = 2000  # a search wider than this is denied rather than scanned: safe_grep handles it
MAX_WORDS = 64  # spellings one shell word may take once variables are filled in; more is denied
MATCH_TIMEOUT = 5.0  # seconds for one pattern match over a tree (runs in a child process)
BUDGET = 25.0  # seconds for the whole decision; Codex's documented example hook timeout is 30
_deadline = float("inf")
_EDIT_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit"}
STATE_REFUSAL = ("Only the user can change the Harpo allowlist. Ask them to run "
                 "`harpo allow <path>` in their own terminal.")
_PROMPTING_MODES = {"default", "acceptEdits", "plan", "auto"}  # auto: a hook's ask forces a real prompt
_can_ask = False
_session: object = None  # the event's session_id, for the findings log

# Child-process matcher for when ripgrep isn't installed: Python re, killed after MATCH_TIMEOUT, so an
# agent-supplied pattern with catastrophic backtracking can't stall the hook into a timeout (= allow).
_PY_MATCHER = r"""
import json, os, re, sys
a = json.loads(sys.argv[1])
rx = re.compile(a["pattern"], a["flags"])
for path in a["files"]:
    try:
        if os.path.getsize(path) > a["max_bytes"] or rx.search(open(path, encoding="utf-8", errors="replace").read()):
            sys.stdout.write(path + "\0")
    except OSError:
        pass
"""


_GIT_ENV = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_PAGER": "cat"}


class _OutOfTimeError(Exception):
    """
    Description: Represent exhausted hook time or scope limits that require refusing the call.
    Parameters:
        *args: Exception arguments forwarded to Exception.
    Output: Exception instances caught by decide to emit a refusal.
    """
    pass


def _tick() -> None:
    """
    Description: Enforce the current decision deadline before further work.
    Parameters: None
    Output: None; raises _OutOfTimeError when monotonic time exceeds the deadline.
    """
    if time.monotonic() > _deadline:
        raise _OutOfTimeError


def _left() -> float:
    """
    Description: Bounds one subprocess timeout by the hook's remaining budget. This avoids timeout-as-allow: a hook
        that times out counts as allow, and chained full timeouts could outlast the budget.
    Parameters: None
    Output: Seconds: MATCH_TIMEOUT, or what is left of the budget if that is less. Never less than 0.1 seconds.
    """
    return min(MATCH_TIMEOUT, max(_deadline - time.monotonic(), 0.1))


_ABSENT = {errno.ENOENT, errno.ENOTDIR, errno.ENAMETOOLONG}  # a name that can't be on disk (ENAMETOOLONG: a 300-char word)


def _mode(path: Path) -> Optional[int]:
    """
    Description: Reads a path's file mode and tells an absent path apart from a failure. A failure must fail closed,
        because the path may be the secret file.
    Parameters:
        path: Filesystem path to stat.
    Output: st_mode of path, or None when it names nothing (ENOENT, ENOTDIR, ENAMETOOLONG). Any other OSError (EACCES
        behind a mode-000 directory, ELOOP, EIO) is raised, and the caller refuses the file as "unreadable" (_types).
    """
    try:
        return os.stat(path).st_mode
    except OSError as exc:
        if exc.errno in _ABSENT:
            return None
        raise


def _isfile(path: Path) -> bool:
    """
    Description: Says whether a path is a file to check. An uncertain stat failure counts as a file, so an unreadable
        path is still refused later.
    Parameters:
        path: Filesystem path to probe.
    Output: True for a regular file or for a stat error that is not an absence (_ABSENT), otherwise False. Only
        existence probes use this. A read that fails on a file that is there still raises, and decide() blocks on it.
    """
    try:
        mode = _mode(path)
    except OSError:
        return True
    return mode is not None and stat.S_ISREG(mode)


def _isdir(path: Path) -> bool:
    """
    Description: Identify directories without claiming paths that _isfile must treat as unreadable
        files.
    Parameters:
        path: Filesystem path to probe.
    Output: True for a confirmed directory, otherwise False, including stat errors.
    """
    try:
        mode = _mode(path)
    except OSError:
        return False  # _isfile claims it (as a file to refuse)
    return mode is not None and stat.S_ISDIR(mode)


def _exists(path: Path) -> bool:
    """
    Description: Treat uncertain stat failures as existing paths so scope checks do not silently
        omit them.
    Parameters:
        path: Filesystem path to probe.
    Output: False only for recognized absence; True for successful stat or other OSError failures.
    """
    try:
        return _mode(path) is not None
    except OSError:
        return True


def _types(path: Path, redactor, grep_args: Optional[dict] = None) -> Optional[list[str]]:
    """
    Description: Finds the secret types a file exposes, keeping risk markers for unreadable or unscanned content.
        Files over the Read tool's whole-file limit are checked in chunks within the hook's time budget (FR-CORE-05).
    Parameters:
        path: File path to inspect.
        redactor: Redactor used to detect replaced content.
        grep_args: Optional structured Grep arguments for judging oversize matching output.
    Output: Secret-type list, [] for no findings, or None for a missing, nonregular, or ordinary binary file.
        Unreadable stat results and risky unscanned content return markers. Deadline, read, stat and detector errors
        propagate for refusal. For an oversize file checked through the Grep tool (grep_args), only the matching lines
        it would print are judged.
    """
    from Harpocrates.read import MAX_BYTES, SNIFF_BYTES, _read_text, text_encoding

    _tick()
    try:
        mode = _mode(path)
    except OSError:
        return ["unreadable"]  # can't even stat it: it may be the secret file, so refuse rather than pass
    if mode is None or not stat.S_ISREG(mode):
        return None
    # Binary (a NUL in the first 8 KiB, as _read_text sniffs) passes whatever its size, before the size check:
    # a big executable named in a command (`.venv/bin/tool --version`) is no secret store. Trade-off: a binary
    # file is never scanned; secrets inside binaries were already not scanned when under the limit. Two
    # exceptions: UTF-16/32 text (text_encoding) goes on to be decoded and scanned, and a file named like a secret
    # store (SENSITIVE_FILES) never passes as binary, since a NUL in front of the text would hide it: refused.
    try:
        with path.open("rb") as f:
            head = f.read(SNIFF_BYTES)
        if b"\x00" in head and text_encoding(head) is None:
            return ["unscanned"] if _SENSITIVE.match(path.name) else None
    except OSError:
        pass  # unreadable: fall through to the size check and read below, which behave as before
    if path.stat().st_size > MAX_BYTES:
        return _printed_types(path, grep_args, redactor) if grep_args else _chunked_types(path, redactor)
    try:
        text = _read_text(path)
    except ValueError:  # binary: images and PDFs pass, as FR-GATE-07 sets for the gate
        return None
    return _found(redactor, text)


def _chunked_types(path: Path, redactor) -> Optional[list[str]]:
    """
    Description: The secret types in a file over the read tool's whole-file limit, chunk by chunk, each chunk seen with
        the lines before it (FR-CORE-05). Every chunk counts against the hook's deadline, so a file too big to check
        in time is refused by the caller (_OutOfTimeError).
    Parameters:
        path: the file.
        redactor: the detector.
    Output: Sorted types, [] for none; ["unscanned"] when a line or key block is too long to check within the cap.
        Every other error propagates, so the hook blocks (a binary file never gets here: _types passes it first).
    """
    from Harpocrates.read import CONTEXT_LINES, TooLargeToScanError, file_chunks

    found: set = set()
    tail = ""
    try:
        for _, text in file_chunks(path):
            _tick()
            found.update(_found(redactor, tail + text))
            tail = "".join(text.splitlines(keepends=True)[-CONTEXT_LINES:])
    except TooLargeToScanError:
        return ["unscanned"]
    return sorted(found)


def _found(redactor, text: str) -> list[str]:
    """
    Description: Finds the secret types that redaction would replace in text. A placeholder already in the text (in
        docs or plans, for example) is not a finding; only placeholders that redaction adds count.
    Parameters:
        redactor: Object whose redact method transforms source text.
        text: Text to scan without printing its values.
    Output: Sorted list of the added placeholder types; "generic" if the text changed without a new type; [] if
        unchanged. Detector errors propagate.
    """
    redacted = redactor.redact(text)
    if redacted == text:
        return []
    grown = Counter(_PLACEHOLDER_TYPE.findall(redacted)) - Counter(_PLACEHOLDER_TYPE.findall(text))
    return sorted(grown) or ["generic"]


def _printed_types(path: Path, args: dict, redactor) -> list[str]:
    """
    Description: Judges the bounded output that an oversize Grep call could print, using that call's flags and widest
        context. The rest of the file is not judged, so a secret the detector only recognises from other lines (a PEM
        body below its header) can pass; the Grep tool prints no more than that.
    Parameters:
        path: Oversize file to search.
        args: Structured Grep arguments including pattern and context.
        redactor: Redactor used to inspect matched output.
    Output: Secret-type list, or ["unscanned"] for anything short of a complete scan: rg unavailable, invalid pattern
        type, incomplete search, timeout, or output over MAX_BYTES (such as a pattern matching most lines). Runs one
        bounded rg over the file with the Grep tool's flags and context lines. Invalid context values, OS, deadline,
        or detector errors propagate for refusal.
    """
    from Harpocrates.read import MAX_BYTES

    rg, pattern = shutil.which("rg"), args.get("pattern")
    if not rg or not isinstance(pattern, str):
        return ["unscanned"]
    _tick()
    # rg lets -A/-B override -C, so scan the widest context any flag could print. A non-number raises: blocked.
    wide = max(int(args.get("-C") or 0), int(args.get("context") or 0))
    before, after = max(int(args.get("-B") or 0), wide), max(int(args.get("-A") or 0), wide)
    flags = _grep_flags(args)[0] + [x for f, n in (("-B", before), ("-A", after)) if n > 0 for x in (f, str(n))]
    proc = subprocess.Popen([rg, "-n", "--no-messages", *flags, "-e", pattern, "--", str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    timer = threading.Timer(_left(), proc.kill)
    timer.start()
    try:
        out = proc.stdout.read(MAX_BYTES + 1)  # bounded: a pattern matching everything can't fill memory
        if len(out) > MAX_BYTES:
            proc.kill()
        proc.wait()
    finally:
        timer.cancel()
        proc.kill()  # no-op once it has exited
        proc.stdout.close()
    _tick()
    if len(out) > MAX_BYTES or proc.returncode not in (0, 1):
        return ["unscanned"]
    return _found(redactor, out.decode("utf-8", errors="replace"))


def _deny(reason: str) -> int:
    """
    Description: Emit a structured PreToolUse deny decision understood by the harness.
    Parameters:
        reason: Refusal explanation that must not contain secret values.
    Output: Integer zero; prints deny JSON on stdout, with output errors propagated.
    """
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    return 0


def _ask(reason: str) -> int:
    """
    Description: Emit a structured PreToolUse ask decision for a human approval prompt.
    Parameters:
        reason: Approval explanation that must not contain secret values.
    Output: Integer zero; prints ask JSON on stdout, with output errors propagated.
    """
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                             "permissionDecisionReason": reason}}))
    return 0


def _first_flagged(paths: Iterable[Path], cwd: Path, grep_args: Optional[dict] = None,
                   session_id: object = None) -> tuple[Optional[tuple[str, str]], list[Path]]:
    """
    Description: Finds the first file that holds a secret, skipping allowlisted paths and paths the user granted in
        this session (FR-READ-06). A grant covers one real path only; a search that reaches other files still checks
        them.
    Parameters:
        paths: Paths to deduplicate and check in order.
        cwd: Working directory used for relative display names.
        grep_args: Optional structured Grep arguments for oversize output checks.
        session_id: Session identifier for approved real paths, or None to use no session grants.
    Output: Tuple of the first hit, as (display name, secret types, their list), or None, and the allowlisted paths
        that passed before it. Does not log uses. Allowlist resolution errors disable that exemption; other scan and
        resolution errors propagate.
    """
    from Harpocrates import allowlist, grants
    from Harpocrates.read import Redactor

    redactor = Redactor()
    approved = grants.granted(session_id)
    passed: list[Path] = []
    for path in dict.fromkeys(paths):
        try:
            listed = allowlist.allowed(path)
        except (OSError, RuntimeError):  # RuntimeError: resolve() on a symlink loop
            listed = False  # can't stat it: not allowlisted; _types then refuses it as unreadable
        if listed:
            passed.append(path)
            continue
        if approved and str(path.resolve()) in approved:
            continue
        found = _types(path, redactor, grep_args)
        if found:
            shown = allowlist.printable(str(path.relative_to(cwd) if path.is_relative_to(cwd) else path))
            kinds = {"unscanned": "unscanned (too large to check)", "unreadable": "unreadable (can't be checked)"}
            return (shown, kinds.get(found[0], found[0]) if len(found) == 1 else ", ".join(found), found), passed
    return None, passed


def _check_files(paths: Iterable[Path], tool: str, cwd: Path, event_tool: str = "Read",
                 grep_args: Optional[dict] = None, session_id: object = None) -> int:
    """
    Description: Refuses a read of a file that holds a secret, never with a prompt (FR-READ-06). Allowlist uses are
        logged only for calls that are allowed.
    Parameters:
        paths: Paths the call could expose.
        tool: Safe MCP tool suffix recommended on refusal.
        cwd: Working directory for path display.
        event_tool: Harness tool name for allowlist audit records.
        grep_args: Optional structured Grep arguments for oversize output checks.
        session_id: Session identifier for built-in-tool grants; Bash passes none.
    Output: Integer 0 after an optional deny JSON response; allowed calls append audit records. Scan, resolution,
        logging, and output errors propagate for fail-closed handling. A grant covers only the built-in tools, so Bash
        passes no session_id.
    """
    from Harpocrates import allowlist

    hit, passed = _first_flagged(paths, cwd, grep_args, session_id)
    if hit is not None:
        _log_refusal(hit[0], hit[2], event_tool)
        return _deny(f"{hit[0]} ({hit[1]}) contains secrets. Use {_tool_ref(tool)} instead: it returns the "
                     "same content with each secret replaced by a <<HARPO:type:hash>> placeholder.")
    for path in passed:  # the call may run: log the allowlist uses (not on a refusal)
        allowlist.log_use(path, event_tool)
    return 0


def _log_refusal(shown: str, found: list[str], tool: str) -> None:
    """
    Description: Log a refused read to the findings log, one line per secret type (FR-LOG-01).
    Parameters:
        shown: the file as the refusal names it.
        found: its secret types, or "unscanned" or "unreadable" for a file that could not be checked.
        tool: the agent's tool.
    Output: None. The hook holds no session key, so a line has no hash. A failed write changes nothing.
    """
    from Harpocrates.audit import FindingsLog

    log = FindingsLog("hook", str(_session) if _session is not None else "")
    for kind in found:
        unchecked = kind in ("unscanned", "unreadable")
        log.record("read_refused", None if unchecked else kind, None, f"{shown} ({tool})", "refuse",
                   kind if unchecked else None)


def _placeholders(text: str) -> Counter:
    """
    Description: Count exact Harpo placeholders to detect newly introduced copies.
    Parameters:
        text: Text in which to count placeholders.
    Output: Counter mapping complete matched placeholder strings to occurrence counts.
    """
    return Counter(m.group(0) for m in _PLACEHOLDER_TYPE.finditer(text))


def _current(path: Path) -> Optional[str]:
    """
    Description: Reads the text an edit starts from, for comparison. The text is never printed.
    Parameters:
        path: Target file path to read.
    Output: The text, keeping its newline spelling; "" for a file that does not exist yet; None if it cannot be
        compared (over the size limit, a directory, or unreadable).
    """
    from Harpocrates.read import MAX_BYTES

    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        with open(path, encoding="utf-8", errors="replace", newline="") as f:  # newline="": as the tool matches it
            return f.read()
    except FileNotFoundError:
        return ""
    except OSError:
        return None


def _adds_placeholder(tool: str, args: dict, path: Path) -> bool:
    """
    Description: Simulates the result of an edit, so a real secret is not swapped for a new placeholder. The result is
        judged, not the new text alone, because one placeholder can be built from two edits or two MultiEdit steps.
    Parameters:
        tool: Edit, MultiEdit, Write, or NotebookEdit tool name.
        args: Tool input including replacement text or edit steps.
        path: Target path whose current text is compared.
    Output: True if the edit would leave a Harpo placeholder that the file did not hold before, otherwise False. A
        placeholder the file already holds may stay or move (docs that quote one), but an extra copy of one, or a
        different one, counts as new (compared as a multiset). Edit and MultiEdit are applied as the tool applies them
        (replace_all honored, each step on the last step's result). Write is compared with the file it replaces;
        NotebookEdit checks the new source directly. A step that cannot be applied here (old_string absent, or the
        file not comparable) falls back to its new text alone, which errs toward deny. Current content is read, never
        printed.
    """
    if tool == "NotebookEdit":
        source = args.get("new_source")
        source = "".join(part for part in source if isinstance(part, str)) if isinstance(source, list) else source
        return isinstance(source, str) and bool(_PLACEHOLDER_TYPE.search(source))
    before = _current(path)
    if tool == "Write":
        content = args.get("content")
        if not isinstance(content, str):
            return False
        return bool(_PLACEHOLDER_TYPE.search(content)) if before is None else bool(
            _placeholders(content) - _placeholders(before))
    after = before
    for step in (args.get("edits") or ()) if tool == "MultiEdit" else (args,):
        if not isinstance(step, dict) or not isinstance(step.get("new_string"), str):
            continue
        old, new = step.get("old_string"), step["new_string"]
        if after is not None and isinstance(old, str) and old in after:
            after = after.replace(old, new, -1 if step.get("replace_all") else 1)
        elif _PLACEHOLDER_TYPE.search(new):
            return True
    return before is not None and bool(_placeholders(after) - _placeholders(before))


_PATH_KEY = re.compile(r"path|file|dir", re.I)  # file_path, notebook_path, path, filePath, target_file, directory ...


def _path_values(node: object, in_path_key: bool = False, depth: int = 0) -> list[str]:
    """
    Description: Collect every string held under a key whose name says it is a path, anywhere in a tool call's input
        (SEC-09): the edit tools take file_path or notebook_path, but a call may carry others and the hook must not
        guess which one the tool will write.
    Parameters:
        node: the tool input, or a part of it.
        in_path_key: whether the enclosing key names a path (strings and lists of strings under it count).
        depth: how deep the walk is; nothing below 6 levels is looked at.
    Output: The distinct non-empty path strings in order of appearance. Keys such as old_string, new_string, content and
        new_source never match, so edit content is not taken for a path. No side effects.
    """
    if depth > 6:
        return []
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += _path_values(value, in_path_key or bool(_PATH_KEY.search(str(key))), depth + 1)
    elif isinstance(node, list):
        for item in node:
            found += _path_values(item, in_path_key, depth + 1)
    elif isinstance(node, str) and in_path_key and node:
        found.append(node)
    return list(dict.fromkeys(found))


def _edit(tool: str, args: dict, cwd: Path, event: dict) -> int:
    """
    Description: Decides an edit in three checks: state-directory protection, then placeholder introduction, then
        secret exposure. The state directories (~/.harpo, ~/.harpocrates) are protected in every mode, because only
        the user may change the allowlist and an agent-made ~/.harpo would replace the migrated one (SEC-09).
    Parameters:
        tool: Edit, MultiEdit, Write, or NotebookEdit tool name.
        args: Tool input containing a file or notebook path and edit content.
        cwd: Working directory for resolving the target.
        event: Hook event containing session and tool-use identifiers.
    Output: Integer 0 after allowing, denying, or asking; may log an allowlist use or save a pending grant. A target
        inside a state directory, or a hard link to a file in one, is denied in any mode, as is a target that cannot
        be placed. Every path-valued key of the call is checked, not only the tool's own; a call naming two different
        files is refused, since the hook cannot tell which one the tool writes (SEC-09). An edit that would add a
        placeholder is denied in any mode, granted or not. A secret-bearing file asks the user only when a human can
        answer, and the call is first recorded as pending so `--post` can turn Yes into a grant. With no session_id or
        tool_use_id to record it under, or no human to ask, it is refused (fail closed). Other errors propagate for
        fail-closed handling.
    """
    from Harpocrates import allowlist, grants, paths

    own, other = ("notebook_path", "file_path") if tool == "NotebookEdit" else ("file_path", "notebook_path")
    raw = args.get(own) or args[other]  # the key the tool itself uses first; KeyError (blocked) when neither is there
    path = cwd / os.path.expanduser(raw)
    named = _path_values(args)  # every path-valued key, whatever its name (SEC-09): not only the two the tools use
    try:
        in_state_dir = False
        for value in named:
            spellings = [cwd / os.path.expanduser(value)]
            if value == "~" or value.startswith("~/"):  # the tool may expand ~ from the password database (SEC-08)
                spellings.append(paths.home().joinpath(value[2:]))
            in_state_dir = in_state_dir or any(allowlist.inside_state_dir(target) or allowlist.links_to_state_file(target)
                                               for target in spellings)
        several = len({str((cwd / os.path.expanduser(value)).resolve()) for value in named}) > 1
    except (OSError, RuntimeError, ValueError):  # the target cannot be placed: fail closed
        in_state_dir, several = True, False
    if in_state_dir:
        return _deny(STATE_REFUSAL)
    if several:  # which of the files the tool writes is not for the hook to guess
        return _deny("That edit names more than one file. Ask the user to make it, or edit one file at a time.")
    if _adds_placeholder(tool, args, path):
        return _deny("That edit would write a Harpo placeholder into the file, replacing a real secret with a "
                     "fake value. Edit around the secret, or ask the user to change it.")
    session_id = event.get("session_id")
    hit, passed = _first_flagged([path], cwd, None, session_id)
    if hit is None:
        for allowed in passed:
            allowlist.log_use(allowed, tool)
        return 0
    shown, kinds, _ = hit
    if _can_ask and grants.add_pending(session_id, event.get("tool_use_id"), path, tool):
        return _ask(f"Harpo: Claude wants to edit {shown} without secret protection. It contains secrets "
                    f"({kinds}). Allowing lets Claude read and edit this file with its built-in tools for the rest "
                    "of this session; these values go to the model provider as-is. Choose No to keep them "
                    "protected: Claude can still read it with mcp__harpo__safe_read.")
    return _deny(f"{shown} ({kinds}) contains secrets, and a built-in edit shows its content to the model. "
                 "Ask the user to make this change.")


# ---------------------------------------------------------------------------------------- searching


def _git(root: Path, *args: str, ok: tuple[int, ...] = (0,)) -> Optional[bytes]:
    """
    Description: Runs a read-only git query in a bounded time, with core.excludesFile (from repo config or the XDG
        default) switched off. Ignore rules then come only from .gitignore files and .git/info/exclude, which the
        scope checks vet.
    Parameters:
        root: Working directory for the Git command.
        *args: Read-only Git arguments.
        ok: Accepted exit statuses, defaulting to zero.
    Output: Captured stdout as bytes, or None if git exits with a status outside ok, stalls, or is not installed
        (OSError). Invokes git and checks the deadline in every case, which can raise _OutOfTimeError.
    """
    try:
        proc = subprocess.run(["git", "--no-pager", "-c", "core.fsmonitor=false", "-c", f"core.excludesFile={os.devnull}",
                               "-C", str(root), *args], capture_output=True, timeout=_left(), env=_GIT_ENV)
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        _tick()
    return proc.stdout if proc.returncode in ok else None


def _in_git(root: Path) -> bool:
    """
    Description: Check whether a directory or a file parent lies inside a Git work tree.
    Parameters:
        root: Directory or file path to test.
    Output: True only for Git output true followed by a newline; Git failures yield False and
        deadline errors propagate.
    """
    return _git(root if root.is_dir() else root.parent, "rev-parse", "--is-inside-work-tree") == b"true\n"


def _git_internals(gitdir: Path) -> list[Path]:
    """
    Description: Collects the files a search of .git reads, except the compressed objects/ directory, whose size would
        exceed MAX_SCOPE_FILES in any real history. The LFS store (lfs/, including its objects/) stays in, since it
        holds the raw content of large files.
    Parameters:
        gitdir: Git metadata directory to walk.
    Output: List of files: config, refs, logs, hooks, and LFS content. Raises _OutOfTimeError on deadline or scope-cap
        exhaustion. Trade-off: a secret committed into an object is not checked; the config (for example a token in a
        remote URL), refs, logs, and hooks are.
    """
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(gitdir):
        _tick()
        rel = os.path.relpath(dirpath, gitdir)
        if rel != "lfs" and not rel.startswith("lfs" + os.sep):
            dirnames[:] = [d for d in dirnames if d != "objects"]
        files += [Path(dirpath) / f for f in filenames]
        if len(files) > MAX_SCOPE_FILES:
            raise _OutOfTimeError  # too wide to vouch for: deny
    return files


def _ident(path: str | Path) -> tuple[int, int]:
    """
    Description: Identifies a filesystem object by its device and inode, whatever link leads to it. This is cheaper
        than resolving the path.
    Parameters:
        path: Path to stat.
    Output: (device, inode) tuple. Raises OSError if the stat fails.
    """
    st = os.stat(path)
    return st.st_dev, st.st_ino


def _walk(root: Path, secret_names: bool = False, git_dir: bool = False,
          seen: Optional[set[tuple[int, int]]] = None) -> list[Path]:
    """
    Description: Walks directory contents conservatively, with symlink-loop detection and a bounded file count.
        Directory links are followed, as grep -R and rg -L do, so the scope may read more than the tool does, never
        less.
    Parameters:
        root: Directory to walk.
        secret_names: Whether to include only names matching secret stores.
        git_dir: Whether to include Git internals when not restricted to secret names.
        seen: Shared set of visited directory identities, or None to create one.
    Output: List of files under root, excluding .git unless git_dir is set; with secret_names, only files named like a
        secret store. Each real directory is walked once, through seen, so a link loop ends and a link to an already
        covered place adds nothing. Mutates seen when supplied, skips directories whose identity probe fails, and the
        file cap still applies. Raises _OutOfTimeError on deadline or file-cap exhaustion.
    """
    files: list[Path] = []
    seen = set() if seen is None else seen
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        _tick()
        try:
            here = _ident(dirpath)
        except OSError:  # gone since it was listed
            continue
        if here in seen:
            dirnames.clear()
            continue
        seen.add(here)
        if git_dir and not secret_names and ".git" in dirnames:
            files += _git_internals(Path(dirpath) / ".git")
        dirnames[:] = [d for d in dirnames if d != ".git"]
        files += [Path(dirpath) / f for f in filenames if not secret_names or _SENSITIVE.match(f)]
        if len(files) > MAX_SCOPE_FILES:
            raise _OutOfTimeError  # too wide to vouch for: deny
    return files


def _unvetted_ignore_rules(root: Path) -> bool:
    """
    Description: Detects ignore rules an agent could have written since the last commit, so the scope can fall back to
        a full walk.
    Parameters:
        root: Git work-tree location to inspect.
    Output: True if a .gitignore anywhere in the work tree is untracked or has uncommitted changes, if
        .git/info/exclude has any rule, or if the Git or read results are uncertain. False when none of these apply.
        Deadline errors propagate.
    """
    dirty = _git(root, "--no-optional-locks", "status", "--porcelain", "-z", "--untracked-files=all",
                 "--ignored=no", "--", ":(top,glob)**/.gitignore")
    exclude = _git(root, "rev-parse", "--git-path", "info/exclude")
    if dirty != b"" or exclude is None:  # None: git couldn't say
        return True
    try:
        lines = (root / os.fsdecode(exclude).strip()).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return any(line.strip() and not line.lstrip().startswith("#") for line in lines)


def _hides_shown_files(visible: bytes, ignored: bytes) -> bool:
    """
    Description: Detects an untracked .gitignore that ignores itself (with `*`) in a directory that still shows files.
        Such a file hides the untracked ones in that directory.
    Parameters:
        visible: NUL-separated visible Git file paths.
        ignored: NUL-separated ignored Git file paths.
    Output: True when a self-ignoring .gitignore shares an ancestor directory with visible files, otherwise False.
        Empty cache-only directories are not flagged, since tool caches (.ruff_cache, .venv) write the same file into
        them.
    """
    dirs = {os.fsdecode(e.rpartition(b"/")[0]) for e in ignored.split(b"\0") if e.rpartition(b"/")[2] == b".gitignore"}
    if not dirs:
        return False
    for entry in filter(None, visible.split(b"\0")):
        parts = os.fsdecode(entry).split("/")[:-1]
        if any("/".join(parts[:i]) in dirs for i in range(len(parts) + 1)):
            return True
    return False


def _scope(root: Path, full: bool = False, git_dir: bool = False) -> list[Path]:
    """
    Description: Computes the bounded set of files that a recursive read of root can reach, excluding .git. In a git
        work tree this is git's listing (tracked files, and untracked files that are not ignored), plus git-ignored
        files named like secret stores, wherever they sit.
    Parameters:
        root: File or directory being read.
        full: Whether to force a full walk.
        git_dir: Whether to include searchable Git internals.
    Output: List of candidate files, including files reached through followed directory links. Everything is walked
        instead when full is set, when root is itself ignored (naming a directory is explicit), when ignore rules were
        written since the last commit (_unvetted_ignore_rules, _hides_shown_files), or when git cannot list the tree
        (not a work tree, missing, failing, or stalling). Errors never shrink the fallback scope. Raises
        _OutOfTimeError on budget or cap exhaustion; filesystem identity errors may propagate. git_dir adds the files
        of .git (_git_internals). Trade-off: a secret in a git-ignored file with an ordinary name (notes.local) can
        pass a recursive shell grep, which keeps data/, .venv and target/ from making the scope too wide to vouch for;
        a direct read of such a file is still checked. Residual gaps: a rule the agent commits, `git update-index
        --assume-unchanged` on an edited .gitignore, or a self-ignoring .gitignore in a directory with no tracked
        files still hides files.
    """
    if _isfile(root):
        return [root]
    ls = ["ls-files", "-z", "--others", "--exclude-standard"]
    ignored_root = _git(root, "check-ignore", ".", ok=(0, 1)) != b""  # prints "." if ignored; None on error
    visible = None if full or ignored_root or _unvetted_ignore_rules(root) else _git(root, *ls, "--cached")
    ignored = None if visible is None else _git(root, *ls, "--ignored")
    if ignored is None or _hides_shown_files(visible, ignored):
        return _walk(root, git_dir=git_dir)
    seen = {_ident(root)}  # a link back into the tree adds nothing: git listed it
    files: list[Path] = []
    for listing, secret_names in ((visible, False), (ignored, True)):
        for rel in filter(None, listing.split(b"\0")):
            _tick()
            name = os.fsdecode(rel)
            if secret_names and not name.endswith("/") and not _SENSITIVE.match(name.rsplit("/", 1)[-1]):
                continue
            path = root / name
            if _isdir(path):  # a nested repo or submodule (git lists it as one entry), or a link to a directory
                files += _walk(path, secret_names, git_dir, seen)
            elif _isfile(path):
                files.append(path)
            if len(files) > MAX_SCOPE_FILES:
                raise _OutOfTimeError  # too wide to vouch for: deny
    if git_dir and (root / ".git").is_dir():
        files += _git_internals(root / ".git")
        if len(files) > MAX_SCOPE_FILES:
            raise _OutOfTimeError
    return files


def _matching(pattern: Optional[str], roots: list[Path], rg_flags: list[str], py_flags: int) -> list[Path]:
    """
    Description: Finds the files under roots whose content a search for pattern would print. When matching cannot be
        judged exactly, every file in scope counts as a match.
    Parameters:
        pattern: Search expression, or None when exact semantics cannot be evaluated.
        roots: Search root paths.
        rg_flags: Ripgrep flags controlling matching and scope.
        py_flags: Equivalent Python regular-expression flags for fallback matching.
    Output: Candidate file paths, from a bounded rg run or isolated Python subprocesses. A None pattern, a matcher
        error, or a timeout falls back to the whole scope. Deadline and unhandled OS errors propagate for refusal.
    """
    from Harpocrates.read import MAX_BYTES

    roots = [r for r in roots if _exists(r)]
    full = "-g" in rg_flags  # a Grep glob can name files .gitignore hides: the fallback scope is the whole tree
    rg = shutil.which("rg")
    run = {"capture_output": True, "encoding": "utf-8", "errors": "surrogateescape"}  # a name may not be UTF-8
    if pattern is not None and rg:  # exact semantics; rg also lists oversize files, which _types flags
        base = [rg, "-l", "-0", "--hidden", "--no-messages", *rg_flags]
        tail = ["-e", pattern, "--", *map(str, roots)]
        if roots and all(_in_git(r) for r in roots):
            # rg honours .gitignore here, as the Grep tool does; a second pass without it re-searches the
            # secret-store names (SENSITIVE_FILES) that would hide. Trade-off: see _scope.
            globs = [a for g in SENSITIVE_FILES for a in ("-g", g)]
            passes = [base + tail, base + ["--no-ignore", *globs] + tail]
        else:
            passes = [base + ["--no-ignore"] + tail]
        try:
            outputs = []
            for argv in passes:
                proc = subprocess.run(argv, timeout=_left(), **run)
                _tick()
                if proc.returncode not in (0, 1):
                    break
                outputs.append(proc.stdout)
            else:
                return _paths("".join(outputs), roots, full)
        except subprocess.TimeoutExpired:
            _tick()
        return [f for r in roots for f in _scope(r, full)]
    scope = [f for r in roots for f in _scope(r, full)]
    if pattern is None:
        return scope
    arg = json.dumps({"pattern": pattern, "flags": py_flags | re.M, "files": list(map(str, scope)),
                      "max_bytes": MAX_BYTES})
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", _PY_MATCHER, arg], timeout=_left(), **run)
    except subprocess.TimeoutExpired:
        _tick()
        return scope
    _tick()
    if proc.returncode != 0:  # a pattern Python rejects: assume the real tool matches
        return scope
    return _paths(proc.stdout, roots, full)


def _paths(output: str, roots: list[Path], full: bool = False) -> list[Path]:
    """
    Description: Validates NUL-separated matcher output before it is trusted as a file scope. Output that names
        anything other than a file cannot be trusted.
    Parameters:
        output: Matcher stdout with NUL-separated file names.
        roots: Original roots used for fallback scope.
        full: Whether fallback must walk the full tree.
    Output: Parsed paths if all of them are file candidates, otherwise the whole scope. File names may contain
        newlines. Scope and deadline errors propagate.
    """
    paths = [Path(p) for p in output.split("\0") if p]
    return paths if all(_isfile(p) for p in paths) else [f for r in roots for f in _scope(r, full)]


def _grep_flags(args: dict) -> tuple[list[str], int]:
    """
    Description: Translates the Grep tool's case and multiline settings into matcher flags. These flags decide what a
        Grep call matches.
    Parameters:
        args: Grep input arguments.
    Output: Tuple of the rg argument list and the matching Python re flag bits. Launches no subprocess.
    """
    flags: list[str] = []
    py = 0
    if args.get("-i"):
        flags.append("-i")
        py |= re.I
    if args.get("multiline"):
        flags += ["-U", "--multiline-dotall"]
        py |= re.S
    return flags, py


def _grep_tool(args: dict, cwd: Path) -> list[Path]:
    """
    Description: Resolve a structured Grep call into candidate files using its pattern, globs, and
        type filters.
    Parameters:
        args: Grep arguments with optional path, pattern, glob, type, and matching flags.
        cwd: Working directory used as the default search root.
    Output: Matching file paths; uses full-scope matching for POSIX classes without rg. Matcher,
        input-shape, filesystem, and deadline errors may propagate for refusal.
    """
    flags, py = _grep_flags(args)
    for piece in (args.get("glob") or "").split():  # as the tool does: split on spaces, then commas unless braces
        for g in [piece] if "{" in piece or "}" in piece else piece.split(","):
            if g:
                flags += ["-g", g]
    if args.get("type"):
        flags += ["-t", args["type"]]
    pattern = args.get("pattern", "")
    if "[[:" in pattern and not shutil.which("rg"):
        pattern = None  # POSIX classes mean something else to Python re
    return _matching(pattern, [cwd / args.get("path", ".")], flags, py)


def _recursive(args: list[str]) -> bool:
    """
    Description: Recognize grep recursion options conservatively, so an ambiguous argument word cannot shrink the
        scope.
    Parameters:
        args: Grep argument words to inspect.
    Output: True if the arguments make grep descend into directories: -r or -R (alone or in a cluster), --recursive or
        an abbreviation of it, or -d recurse. A word that merely looks like one of these counts, so the check
        over-reads; otherwise False.
    """
    for a in args:
        if a.startswith("--"):
            name = a.partition("=")[0]
            if "recurs" in name or (len(name) > 3 and "--recursive".startswith(name)):
                return True
        elif a.startswith("-") and set(a) & set("rR"):
            return True
    return any(a == "recurse" or a.endswith("=recurse") or a.endswith("drecurse") for a in args)


def _names_pattern(flag: str) -> bool:
    """
    Description: Recognize flags that may supply a search pattern, so a file operand is not mistaken for the pattern.
    Parameters:
        flag: Short cluster or long option to inspect.
    Output: True if the flag may carry the pattern (-e, -f, --regexp or --file, in a cluster or abbreviated); then no
        positional word is taken as the pattern. The check over-reads, so a real file operand is never mistaken for
        the pattern. Otherwise False.
    """
    if flag.startswith("--"):
        name = flag.partition("=")[0]
        return len(name) > 3 and ("--regexp".startswith(name) or "--file".startswith(name))
    return bool(set(flag) & set("ef"))


def _reads_git(name: str, flags: list[str], recursive: bool) -> bool:
    """
    Description: Estimate whether search flags make the search read inside .git, using a conservative policy for each
        tool.
    Parameters:
        name: Search executable name.
        flags: Search option words.
        recursive: Whether grep recursion is enabled.
    Output: Boolean estimate, True if the search run with these flags reads inside .git. It does not run the search.
        Checked on this machine: BSD grep -r/-R read .git; rg does only with --hidden, -. or -uu; git grep never. ag
        (with --hidden or -u) and ack (with --noignore-dir) are assumed, since neither is installed here; ack skips
        .git by default. An over-estimate is safe.
    """
    short = "".join(a[1:] for a in flags if not a.startswith("--"))
    if name in _GREPS:
        return recursive
    if name == "rg":
        return "--hidden" in flags or "." in short or short.count("u") + flags.count("--unrestricted") >= 2
    if name == "ag":
        return "--hidden" in flags or "u" in short or "--unrestricted" in flags
    return any(a.startswith("--noignore-dir") for a in flags)


def _search_command(argv: list[str], here: Path, piped: bool = False, tree: bool = False,
                    git: bool = False, env_opts: bool = False) -> list[Path]:
    """
    Description: Scope shell search commands without evaluating their patterns, because flags can change which lines
        print.
    Parameters:
        argv: Search command name and arguments.
        here: Effective working directory.
        piped: Whether the command receives a pipe.
        tree: Whether file-name input or command semantics force tree search.
        git: Whether this is git grep.
        env_opts: Whether GREP_OPTIONS may enable grep recursion.
    Output: Files the search could print from. An empty list when it prints only names or counts, or reads only stdin.
        Attached or clustered flags (-eX, --regexp=X, -nA 3), -v, --passthru and basic-regex syntax change which lines
        print, so every file in scope counts. Without a file operand, grep reads stdin unless it recurses, and rg, ag
        or ack read stdin after a pipe. A word that could name a file the hook cannot see (glob, $VAR, $(...)) counts
        as an operand. The scope includes .git where _reads_git says the search reads it. Scope, filesystem and
        deadline errors may propagate and refuse the call. The Grep tool has its own exact check (_grep_tool).
    """
    name, opts = Path(argv[0]).name, argv[1:]
    if any(a in (_NAMES_ONLY - {"-L"} if name == "rg" else _NAMES_ONLY) for a in opts):  # rg -L is --follow
        return []  # file names or counts only
    head = opts[:opts.index("--")] if "--" in opts else opts
    flags = [a for a in head if a.startswith("-")]
    words = [a for a in head if not a.startswith("-")] + opts[len(head) + 1:]
    roots = [r for r in (_resolve(a, here) for a in words) if _exists(r)]
    recursive = name in _GREPS and (env_opts or _recursive(head))
    operands = words if any(_names_pattern(a) for a in flags) else words[1:]  # else the first word is the pattern
    if not roots and not (tree or git or recursive or any(_UNRESOLVED.search(a) for a in operands)
                          or (name not in _GREPS and not piped)):
        return []  # reads stdin
    descends = not git and _reads_git(name, flags, recursive)
    return [f for r in (roots or [here]) if _exists(r) for f in _scope(r, git_dir=descends)]


# ---------------------------------------------------------------------------------------- shell


def _assignments(command: str, inherited: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Description: Collect every shell assignment alternative without modeling control flow, so the check
        over-approximates the possible paths.
    Parameters:
        command: Shell text containing assignments in any branch or subshell.
        inherited: Previously known variable values.
    Output: New mapping from each variable name to its ordered, unique possible values, including appended assignments
        (D+=x). Every value a variable is given anywhere in the command counts, so a later $D is checked against all
        of them. The check can see more paths than the shell reads, never fewer.
    """
    values = dict(inherited)
    for name, append, raw in _ASSIGN_ANY.findall(command):
        value = raw.strip("'\"")
        given = [old + value for old in values.get(name, [""])] if append else [value]
        values = {**values, name: list(dict.fromkeys(values.get(name, []) + given))}
    return values


def _words(word: str, values: dict[str, list[str]], depth: int = 0) -> list[str]:
    """
    Description: Expand command-assigned variables into the possible shell-word spellings, for conservative path
        inspection.
    Parameters:
        word: Shell word to expand.
        values: Mapping of variable names to possible assigned values.
        depth: Current recursive expansion depth, initially zero.
    Output: List of unsplit spellings and whitespace-split spellings. Each $NAME or ${NAME} is replaced by each value
        the command gave it, so F='a b'; cat $F reads two files. A leading ~ counts as an assigned $HOME. Limits
        recursive self-reference, and raises _OutOfTimeError above MAX_WORDS.
    """
    if depth == 0 and word.startswith("~") and "HOME" in values:
        return [word] + _words("$HOME" + word[1:], values, 1)
    m = next((m for m in _VAR.finditer(word) if (m.group(1) or m.group(2)) in values), None)
    if m is None or depth > 8:  # depth: a value that names itself (`D=$D/x`)
        return [word] + (word.split() if depth and len(word.split()) > 1 else [])
    out = [w for v in values[m.group(1) or m.group(2)] for w in _words(word[:m.start()] + v + word[m.end():],
                                                                       values, depth + 1)]
    if len(out) > MAX_WORDS:
        raise _OutOfTimeError  # too many spellings to vouch for: deny
    return out


def _resolve(word: str, here: Path) -> Path:
    """
    Description: Expand home and environment references and anchor a shell word to the effective
        directory.
    Parameters:
        word: Shell word interpreted as a path.
        here: Effective working directory.
    Output: Path after expanduser and expandvars; does not resolve symlinks or verify existence.
    """
    return here / os.path.expandvars(os.path.expanduser(word))


def _expand(word: str, here: Path) -> list[Path]:
    """
    Description: Expand a shell word or assignment-style value into candidate paths, including globs.
    Parameters:
        word: Shell word to inspect, after stripping surrounding quote and backtick characters.
        here: Effective working directory.
    Output: List of paths the word may name: the word itself, if=FILE-style values, globs, and ~ or $VAR expansions.
        Glob enumeration checks the deadline and raises _OutOfTimeError at the file cap.
    """
    words = [word.strip("`'\"")]
    if "=" in words[0]:
        words.append(words[0].split("=", 1)[1])
    out: list[Path] = []
    for w in words:
        path = _resolve(w, here)
        if any(c in w for c in "*?["):
            for n, m in enumerate(glob.iglob(str(path), recursive=True)):
                _tick()
                if n >= MAX_SCOPE_FILES:
                    raise _OutOfTimeError  # too wide to vouch for: deny
                out.append(Path(m))
        else:
            out.append(path)
    return out


def _cut_body(text: str, start: int, word: str, strip: bool) -> tuple[str, int]:
    """
    Description: Extract a heredoc body up to its delimiter, or to the end of the command.
    Parameters:
        text: Complete shell text.
        start: Offset where the body starts.
        word: Terminating delimiter.
        strip: Whether to ignore leading tabs when matching the delimiter.
    Output: Tuple of (body text, offset just after the terminating line). A body that is never closed runs to the end
        of the text, and its offset is the end of the text.
    """
    lines: list[str] = []
    while True:
        stop = text.find("\n", start)
        line = text[start:] if stop < 0 else text[start:stop]
        if (line.lstrip("\t") if strip else line) == word:
            return "\n".join(lines), len(text) if stop < 0 else stop + 1
        lines.append(line)
        if stop < 0:
            return "\n".join(lines), len(text)
        start = stop + 1


def _heredocs(command: str) -> tuple[list[str], str, list[str]]:
    """
    Description: Separate logical shell lines, expansion-search text, and heredoc bodies without discarding any
        content.
    Parameters:
        command: Shell command text to split.
    Output: Tuple of (logical lines, text searched for substitutions, bodies). Lines are split at newlines outside
        quotes and keep their <<WORD operators. An unquoted delimiter's body expands before the command runs, so it is
        searched for $(...) and backticks; a quoted delimiter's body is read as lines instead. Bodies are kept, not
        dropped, so a << that is no heredoc (e.g. $((1<<X))) hides nothing. _bash_files reads bodies as prose
        (_prose), because an apostrophe would break the tokenizer.
    """
    lines: list[str] = []
    scan: list[str] = []
    bodies: list[str] = []
    pending: list[tuple[str, bool, bool]] = []  # (delimiter, strip tabs, quoted) of operators awaiting a body
    start = pos = kept = 0  # start: text not yet in a line; kept: text not yet in scan
    while (m := _HEREDOC.search(command, pos)) is not None:
        pos = m.end()
        if m.group() == "\n":
            lines.append(command[start:m.start()])
            start = pos
            if pending:  # the bodies follow, one after the other
                scan.append(command[kept:pos])
                for word, strip, quoted in pending:
                    body, end = _cut_body(command, pos, word, strip)
                    bodies.append(body)
                    if not quoted:
                        scan.append(command[pos:end])
                    pos = end
                pending, start, kept = [], pos, pos
        elif m.group().startswith("<<"):
            pending.append((m["qword"] or m["word"], bool(m["tabs"]), bool(m["q"] or m["esc"])))
    lines.append(command[start:])
    scan.append(command[kept:])
    return lines, "".join(scan), bodies


def _prose(body: str) -> list[tuple[list[str], bool, bool]]:
    """
    Description: Read heredoc bodies as conservative command segments, whatever reads them, without rejecting stray
        prose quotes.
    Parameters:
        body: Heredoc body to split into lines and words.
    Output: List of (words, pipe-fed, True) tuples. Splits at ; & | ( ) and backticks, and strips word quotes with
        str.split, never shlex, which would raise on prose. The True marks a body line: no cd failure, and no eval or
        -c recursion on it. Over-reads (a Python comment naming grep counts as a search), but never under-reads a
        script.
    """
    out: list[tuple[list[str], bool, bool]] = []
    piped = False  # the separator before the words being read was a pipe
    for line in body.splitlines():
        for part in _PROSE_SPLIT.split(line):
            if _PROSE_SPLIT.fullmatch(part):
                piped = piped or part in _PIPES
                continue
            words = [w for w in (x.strip("'\"") for x in part.split()) if w]
            if words:
                out.append((words, piped, True))
                piped = False
    return out


def _feeds_a_shell(segs: list[tuple[list[str], bool, bool]]) -> bool:
    """
    Description: Detect input execution, so names inside echo or printf arguments get no naming-only exemption.
    Parameters:
        segs: Command segments containing words, pipe-input flags, and prose markers.
    Output: True if a command evals its input, or a pipe feeds text to a shell, source, ., eval or xargs, including
        xargs-fed commands. Arguments of echo or printf may then run, so _NAMING excuses nothing. Otherwise False.
    """
    for argv, piped, _prose_line in segs:
        i, fed = _command_word(argv)
        name = (Path(argv[i]).name or argv[i]) if i < len(argv) else ""
        if name == "eval" or (piped and (fed or name in _SINKS)):
            return True
    return False


def _command_word(argv: list[str]) -> tuple[int, bool]:
    """
    Description: Locate the executable command after assignments, shell keywords, and supported wrappers.
    Parameters:
        argv: Segment words to inspect.
    Output: Tuple of (index of the command, whether xargs feeds it). Skips VAR=x words, shell keywords (do, then) and
        the wrappers in _PREFIXES with their options. Returns len(argv) for a naming-only command -v or -V, which only
        looks a name up, and for a segment that runs no command. Wrapper options can consume missing values, so the
        index may reach or exceed the end.
    """
    i, fed = 0, False
    while i < len(argv):
        name = Path(argv[i]).name
        if _ASSIGNMENT.match(argv[i]):
            i += 1
            continue
        if name not in _PREFIXES:
            break
        fed, i = fed or name == "xargs", i + 1
        while i < len(argv) and argv[i].startswith("-") and argv[i] != "-":
            opt, i = argv[i], i + 1
            if opt == "--":
                break
            if opt.startswith("--"):
                i += opt in _LONG_VALUE
                continue
            if name == "command" and set(opt) & set("vV"):
                return len(argv), fed
            for k, c in enumerate(opt[1:], 1):
                if c in _PREFIXES[name]:
                    i += k == len(opt) - 1  # the value is the next word unless attached
                    break
        if name == "timeout" and i < len(argv) and _DURATION.match(argv[i]):
            i += 1
    return i, fed


def _find_paths(args: list[str]) -> list[str]:
    """
    Description: Extract the find starting paths, before the first search expression.
    Parameters:
        args: Words following find.
    Output: List of path words. Skips -H, -L and -P, and stops at any other option, !, or an opening parenthesis.
    """
    paths: list[str] = []
    for a in args:
        if a in ("-H", "-L", "-P"):
            continue
        if a.startswith("-") or a in ("!", "("):
            break
        paths.append(a)
    return paths


def _search_word(argv: list[str], start: int) -> Optional[tuple[int, bool]]:
    """
    Description: Identify direct search commands, and git grep after Git options.
    Parameters:
        argv: Segment words.
        start: Index of the actual command word.
    Output: Tuple of (index of the search command in argv, whether it is git grep), or None if the command at start is
        not a search. Searches behind other wrappers are handled by _bash_files.
    """
    if start >= len(argv):
        return None
    name = Path(argv[start]).name
    if name in _SEARCH_COMMANDS:
        return start, False
    if name == "git":
        i = start + 1
        while i < len(argv) and argv[i].startswith("-"):  # git's own options come before the subcommand
            i += 2 if argv[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") else 1
        if i < len(argv) and argv[i] == "grep":
            return i, True
    return None


def _segments(lines: list[str]) -> list[tuple[list[str], bool]]:
    """
    Description: Tokenize logical shell lines into command segments, keeping pipe-input information.
    Parameters:
        lines: Logical lines produced by heredoc separation.
    Output: List of (words, pipe-fed) tuples, one per command, split at ; & | ( ). Raises ValueError on unbalanced
        quotes, so decide blocks the call.
    """
    out: list[tuple[list[str], bool]] = []
    piped = False  # the separator before the command being read was a pipe
    for line in lines:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        seg: list[str] = []
        for tok in lexer:  # unbalanced quotes raise ValueError -> decide() blocks the call
            if set(tok) <= _SEPARATORS:
                if seg:
                    out.append((seg, piped))
                    seg, piped = [], False
                piped = piped or tok in _PIPES
            else:
                seg.append(tok)
        if seg:
            out.append((seg, piped))
            piped = False
    return out


def _git_blobs(argv: list[str], here: Path) -> Optional[str]:
    """
    Description: Fetch the content named by git show revision:path operands, using read-only Git commands, so it can
        be scanned.
    Parameters:
        argv: Git command argument words to inspect, from index two.
        here: Effective Git working directory.
    Output: Joined, decoded blob text, or None if no content was read. Git runs without text conversion or external
        diffs. Subprocess, OS and deadline errors propagate and refuse the call.
    """
    texts = []
    for tok in argv[2:]:
        if ":" in tok and not tok.startswith("-") and "://" not in tok:
            _tick()
            proc = subprocess.run(["git", "--no-pager", "-c", "core.fsmonitor=false", "-C", str(here), "show",
                                   "--no-textconv", "--no-ext-diff", tok],
                                  capture_output=True, timeout=_left(), env=_GIT_ENV)
            _tick()
            if proc.returncode == 0:
                texts.append(proc.stdout.decode("utf-8", errors="replace"))
    return "\n".join(texts) or None


def _copy_dirs(name: str, args: list[str], here: Path) -> list[Path]:
    """
    Description: Scope the source directories of copy, move, link and archive commands to prevent allowlist
        laundering: a secret copied into an allowlisted folder would otherwise go unscanned.
    Parameters:
        name: Copy or archive executable name.
        args: Arguments after the command name.
        here: Effective working directory.
    Output: List of files under those directories, judged like a recursive search of each (same scope, cap and
        fail-closed rules). Every non-option word counts, option values included, so it over-reads (tar -C DIR scans
        DIR). A copy's last word is its target and is left out, unless a -t or --target-directory flag (or any short
        cluster holding t) may have moved it. Trade-off: cp x existing-dir does not scan secrets already in
        existing-dir. Expansion and scope errors propagate.
    """
    words = [a for a in args if not a.startswith("-")]
    targeted = any(a.startswith("--target") or (a[:1] == "-" and a[1:2] != "-" and "t" in a[1:]) for a in args)
    sources = words if name not in _TARGET_LAST or targeted else words[:-1]
    return [f for w in sources for p in _expand(w, here) if _isdir(p) for f in _scope(p)]


def _scan_operands(argv: list[str]) -> Optional[list[str]]:
    """
    Description: Identify the operands of an agent-run Harpo scan, and refuse modes that a work-tree check cannot
        cover.
    Parameters:
        argv: Command words, including possible wrappers or a Python module invocation.
    Output: Non-option words after scan, when the command runs harpo scan or harpocrates scan (in any letter case,
        since macOS folds case), or python -m Harpocrates[.cli] behind any wrapper. None if no scan runs or only
        --help is asked for. Raises _ScanUncheckableError for --history or --show-secrets.
    """
    for i, tok in enumerate(argv):
        if Path(tok).name.casefold() in _CLI_NAMES:
            rest = argv[i + 1:]
        elif tok == "-m" and i + 1 < len(argv) and _CLI_MODULE.match(argv[i + 1]):
            rest = argv[i + 2:]
        else:
            continue
        if "scan" in rest and not {"--help", "-h"} & set(rest):
            after = rest[rest.index("scan") + 1:]
            if any(a == "--history" or a.startswith("--show-secrets") for a in after):
                raise _ScanUncheckableError
            return [a for a in after if not a.startswith("-")]
    return None


def _bash_files(command: str, cwd: Path, depth: int = 0, inherited: Optional[dict[str, list[str]]] = None,
                named_ok: bool = True, scans: Optional[list[Path]] = None) -> tuple[list[Path], list[str]]:
    """
    Description: Collect the files and extra text a shell command could expose, using conservative parsing and bounded
        recursion.
    Parameters:
        command: Shell command text to inspect.
        cwd: Initial working directory.
        depth: Shell recursion depth, initially zero.
        inherited: Inherited variable alternatives, or None.
        named_ok: Whether naming-only commands can be exempted. False for executed substitution output, which runs as
            a command, so _NAMING excuses nothing.
        scans: Optional list that receives the files an agent-run harpocrates scan could print values from.
            That scan reads its operands recursively (default .), so it is scoped and capped like a recursive search.
    Output: Tuple of (candidate files, extra text such as Git blobs). May read Git blobs and add to scans; _bash
        refuses the files in scans with its own message. Raises ValueError for excessive nesting, an ambiguous cd, or
        invalid quoting, and _ScanUncheckableError for unsafe scan modes. Scope, deadline and filesystem errors
        propagate and refuse the call.
    """
    files: list[Path] = []
    texts: list[str] = []
    if depth > 3:
        raise ValueError("command nested too deep to check")
    lines, scan, bodies = _heredocs(command)  # bodies are cut out of what is tokenised and read apart (_prose)
    values = _assignments(command, inherited or {})  # nested bodies inherit even unexported names: more paths
    for m in _SUBSHELL.finditer(scan):  # $(...) and `...` run first (not in a quoted heredoc): check their bodies too
        sub_files, sub_texts = _bash_files(m.group(1) or m.group(2) or "", cwd, depth + 1, values, False, scans)
        files, texts = files + sub_files, texts + sub_texts
    segs = [(argv, piped, False) for argv, piped in _segments(lines)] + [s for body in bodies for s in _prose(body)]
    # `D=dir; cat $D/f`, `R=HEAD:.env; git show $R`
    segs = [([w for t in argv for w in _words(t, values)], piped, prose) for argv, piped, prose in segs]
    sink = _feeds_a_shell(segs)
    if sink:  # `echo 'grep -r x .' | sh`: its quoted words are commands
        segs = [([w for t in argv for w in t.split() or [t]], piped, prose) for argv, piped, prose in segs]
    exempt = named_ok and not sink
    env_opts = any("GREP_OPTIONS" in where for where in (command, values, os.environ))  # may add -r to a grep
    here = cwd
    # Words count as read for the commands that read files themselves (a reader of stdin only, `| tail -3`, promotes
    # nothing of another command's: `pytest tests/x.py | tail -1`). Everything named counts (reads_all) when a
    # reader takes a variable or substitution (`for f in a b; do cat "$f"; done`) or xargs feeds it names.
    # A pipe into an interpreter (sink: a shell, source, eval, xargs, parallel) runs whatever it is fed, so there the
    # old command-wide rule applies: any reader makes every named word count (`echo f | sed 's/^/cat /' | sh`).
    reading: list[Path] = []
    named: list[Path] = []
    reads_all = any_reader = False
    for argv, piped, prose in segs:
        names = [Path(t).name for t in argv]
        start, fed = _command_word(argv)
        if names[0] == "cd":
            if len(argv) > 2 and not prose:
                raise ValueError("cd to a directory that depends on a branch")  # can't tell which: block
            if len(argv) <= 2:
                here = _resolve(argv[1], here) if len(argv) > 1 else Path.home()
            continue
        if not prose and names[0] in ("eval", "exec", "watch") and len(argv) > 1:
            sub_files, sub_texts = _bash_files(" ".join(argv[1:]), here, depth + 1, values, False, scans)
            files, texts = files + sub_files, texts + sub_texts
            continue
        body = None  # sudo / env ... bash -x -c BODY
        if not prose:
            body = next((argv[j + 1] for i, n in enumerate(names) if n in _SHELLS
                         for j in range(i + 1, len(argv) - 1) if _SHELL_C.match(argv[j])), None)
        if body is not None:
            sub_files, sub_texts = _bash_files(body, here, depth + 1, values, True, scans)
            files, texts = files + sub_files, texts + sub_texts
            continue
        base = here
        if names[0] == "git":
            if "-C" in argv[:-1]:
                base = _resolve(argv[argv.index("-C") + 1], here)
            blob = _git_blobs(argv, base)
            if blob:
                texts.append(blob)
            files += [base / t.split(":", 1)[1] for t in argv[2:]  # rev:path -> also the working-tree file
                      if ":" in t and not t.startswith("-") and "://" not in t]
        found = _search_word(argv, start)
        if found is not None:
            files += _search_command(argv[found[0]:], base, piped, fed, found[1], env_opts)
        elif start < len(argv) and not (exempt and names[start] in _NAMING):
            i = next((i for i, n in enumerate(names) if n in _SEARCH_COMMANDS), None)
            if i is not None:  # behind a command we don't know (flock, parallel, find -exec): never stdin
                scoped = ["grep", "-r", *_find_paths(argv[start + 1:])] if names[start] == "find" else argv[i:]
                files += _search_command(scoped, base, piped, tree=True, env_opts=env_opts)
        if scans is not None and start < len(argv) and not (exempt and names[start] in _NAMING):
            operands = _scan_operands(argv)
            if operands is not None:
                roots = [p for w in operands for p in _expand(w, base) if _exists(p)]
                scans += [f for r in (roots or [base]) for f in _scope(r)]  # a flag's value is no path: default `.`
        words = [p for w in argv for p in _expand(w, base)]
        named += words
        if any(n in _READ_COMMANDS for n in names) or "<" in argv:
            any_reader = True
            reading += words
            reads_all = reads_all or any(_DYNAMIC.fullmatch(w) for w in argv) or (
                fed and start < len(argv) and names[start] in _READ_COMMANDS)
        if start < len(argv) and names[start] in _COPY_COMMANDS:
            files += _copy_dirs(names[start], argv[start + 1:], base)
    files += named if reads_all or (sink and any_reader) else reading
    return [f for f in files if _isfile(f)], texts


def _env_secrets() -> list[str]:
    """
    Description: Identify the environment variables whose values the detector would redact, without returning any
        values.
    Parameters: None
    Output: List of flagged variable names, in sorted environment order. Runs the redaction detector, and detector
        errors propagate.
    """
    from Harpocrates.read import Redactor

    items = sorted(os.environ.items())
    lines = [f"{k}={v}" for k, v in items]
    redacted = Redactor().redact("\n".join(lines)).split("\n")
    return [k for (k, _), before, after in zip(items, lines, redacted) if before != after]


def _check_env(command: str) -> Optional[str]:
    """
    Description: Check environment dumps and variable references for exposure of flagged environment
        values.
    Parameters:
        command: Raw shell text, including nested command bodies.
    Output: Refusal explanation naming at most five variables, or None; never includes values and
        propagates detector errors.
    """
    dumps = bool(_ENV_DUMP.search(command))  # raw text, so nested `bash -c` and $(...) bodies count too
    refs = set(_ENV_REF.findall(command))
    if not dumps and not refs:
        return None
    flagged = _env_secrets()
    hit = flagged if dumps else sorted(refs & set(flagged))
    if hit:
        return (f"This command would print environment variables that hold secrets ({', '.join(hit[:5])}). "
                "Reference the variable by name instead of printing its value.")
    return None


def _bash(command: str, cwd: Path) -> int:
    """
    Description: Check state mutation, environment exposure, shell scope, Git blobs, scan output,
        then ordinary file reads in order.
    Parameters:
        command: Shell command to inspect.
        cwd: Effective working directory.
    Output: Integer zero after optional deny JSON; Bash never uses session grants. May scan files,
        read Git blobs, and log allowed overrides; uncheckable scans are denied and other errors
        propagate for fail-closed handling.
    """
    from Harpocrates import allowlist
    from Harpocrates.read import Redactor

    if allowlist.touches(command, cwd):
        return _deny(STATE_REFUSAL)
    env_reason = _check_env(command)
    if env_reason:
        return _deny(env_reason)
    scans: list[Path] = []
    try:
        files, texts = _bash_files(command, cwd, scans=scans)
    except _ScanUncheckableError:
        return _deny(SCAN_UNCHECKABLE)
    for text in texts:
        if Redactor().redact(text) != text:
            return _deny("That git object contains secrets. Read the working-tree file with "
                         f"{_tool_ref('safe_read')} instead.")
    if _first_flagged([f for f in scans if _isfile(f)], cwd)[0] is not None:
        return _deny(SCAN_REFUSAL)
    return _check_files(files, "safe_read", cwd, "Bash")


# ---------------------------------------------------------------------------------------- entry


def _from_codex(event: dict) -> bool:
    """
    Description: Whether a PreToolUse event comes from Codex. Codex documents `model` and `turn_id` as its own fields
        on top of the common ones (https://learn.chatgpt.com/docs/hooks, checked 2026-10-19); Claude Code sends neither.
    Parameters:
        event: The parsed hook event.
    Output: True when the event has both fields; False otherwise. A Codex event that lacked them would be judged as a
        Claude Code one, which can only refuse more, never less.
    """
    return "turn_id" in event and "model" in event


def decide(raw: str, ask: bool = False) -> int:
    """
    Description: Hook entry point: dispatch a PreToolUse event under a fixed budget, and fail closed on errors.
    Parameters:
        raw: JSON hook event text.
        ask: Whether supported human-answerable modes may prompt for edits.
    Output: Zero after a deny or ask decision, or after an allowed call; 2 (block) on any other error. Sets the
        deadline and ask mode, may read files, and may update audit or grant state. Prints only safe decisions or
        exception types. Budget exhaustion emits a deny decision. SEC-10: when the gate is on and a Claude Code
        setting sends requests around it (gatecheck.refusal), every tool call is denied before anything else is
        checked, whatever ask and the permission mode are. This never prompts.
    """
    global _deadline, _can_ask, _session
    _deadline = time.monotonic() + BUDGET
    try:
        event = json.loads(raw)
        _session = event.get("session_id")
        tool, args = event.get("tool_name"), event.get("tool_input") or {}
        cwd = Path(event.get("cwd") or ".")
        if not _from_codex(event):
            bypass = gatecheck.refusal(cwd)
            if bypass:
                return _deny(bypass)
        _can_ask = ask and event.get("permission_mode") in _PROMPTING_MODES
        if tool == "Read":
            return _check_files([cwd / args["file_path"]], "safe_read", cwd, session_id=event.get("session_id"))
        if tool == "Grep" and args.get("output_mode") == "content":
            return _check_files(_grep_tool(args, cwd), "safe_grep", cwd, "Grep", grep_args=args,
                                session_id=event.get("session_id"))
        if tool == "Bash":
            return _bash(args.get("command", ""), cwd)
        if tool in _EDIT_TOOLS:
            return _edit(tool, args, cwd, event)
        return 0
    except _OutOfTimeError:
        return _deny(f"Too many files to check for secrets in time. Use {_tool_ref('safe_grep')} or "
                     f"{_tool_ref('safe_read')} on a narrower path instead.")
    except Exception as exc:  # fail closed; the type only, never a message that could quote file content
        print(f"harpo hook error ({type(exc).__name__}); blocked. Use {_tool_ref('safe_read')}.",
              file=sys.stderr)
        return 2


def post(raw: str) -> int:
    """
    Description: Hook entry point for --post: record completed, approved edits as grants, or clear a session on
        SessionEnd, without blocking the harness.
    Parameters:
        raw: JSON PostToolUse, PostToolUseFailure, or SessionEnd event text.
    Output: Integer zero. Turns the pending edit with this tool_use_id into a grant, or drops the session's grants on
        SessionEnd. Only the event name decides: a stray PreToolUse event never grants, and the event must name the
        edit tool the hook asked about (grants.promote). A failed tool still ran, so its edit is granted; an Edit that
        fails with "not read yet" grants, which lets the Read through. An interrupt never grants. Prints nothing and
        suppresses ordinary exceptions, so a crash here cannot reach the harness.
    """
    try:
        from Harpocrates import grants

        event = json.loads(raw)
        name, session_id = event.get("hook_event_name"), event.get("session_id")
        if name == "SessionEnd":
            grants.clear(session_id)
        elif (name in ("PostToolUse", "PostToolUseFailure") and not event.get("is_interrupt")
              and event.get("tool_name") in _EDIT_TOOLS):
            grants.promote(session_id, event.get("tool_use_id"), event.get("tool_name"))
    except Exception:  # noqa: BLE001
        pass
    return 0


def main() -> None:
    """
    Description: Read a hook event from stdin and dispatch pre-tool enforcement or nonblocking
        post-event recording.
    Parameters: None
    Output: No return; raises SystemExit with the decision code, using 2 for any escaping pre-hook
        BaseException and always 0 for post mode. May print safe decisions or error types and read
        or update hook state.
    """
    if "--post" in sys.argv[1:]:
        try:
            post(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
        except BaseException:  # noqa: BLE001 - recording only: no exit code but 0 for the harness
            pass
        sys.exit(0)
    try:
        code = decide(sys.stdin.buffer.read().decode("utf-8", errors="replace"), ask="--ask" in sys.argv[1:])
    except BaseException:  # noqa: BLE001 - anything that escapes would exit 1, which harnesses treat as allow
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
