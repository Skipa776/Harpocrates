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
    Description: Represent an agent-run scan whose history or secret-display mode cannot be covered
        by a work-tree check.
    Parameters:
        *args: Exception arguments forwarded to Exception.
    Output: Exception instances causing outright refusal of --history or --show-secrets scans.
    Notes: An agent-run scan with --history (reads commits) or --show-secrets (prints the values):
        no work-tree check covers it, so the call is refused outright.
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
    Description: Bound each subprocess timeout by the remaining hook budget to avoid
        timeout-as-allow behavior.
    Parameters: None
    Output: Seconds bounded above by MATCH_TIMEOUT and with a remaining-budget floor of 0.1 seconds.
    Notes: Timeout for one subprocess: MATCH_TIMEOUT, or what is left of BUDGET if that is less.
        Chained full timeouts could outlast the budget, and a hook that times out counts as allow.
    """
    return min(MATCH_TIMEOUT, max(_deadline - time.monotonic(), 0.1))


_ABSENT = {errno.ENOENT, errno.ENOTDIR, errno.ENAMETOOLONG}  # a name that can't be on disk (ENAMETOOLONG: a 300-char word)


def _mode(path: Path) -> Optional[int]:
    """
    Description: Read file mode while distinguishing absent paths from failures that must fail
        closed.
    Parameters:
        path: Filesystem path to stat.
    Output: st_mode or None for ENOENT, ENOTDIR, or ENAMETOOLONG; raises other OSError failures for
        caller refusal.
    Notes: st_mode of path, None if it names nothing (_ABSENT). Any other failure (EACCES behind a
        mode-000 directory, ELOOP, EIO) raises: the path may be the secret file, so the caller fails
        closed (_types: "unreadable").
    """
    try:
        return os.stat(path).st_mode
    except OSError as exc:
        if exc.errno in _ABSENT:
            return None
        raise


def _isfile(path: Path) -> bool:
    """
    Description: Treat regular files and uncertain stat failures as file candidates so unreadable
        paths are refused.
    Parameters:
        path: Filesystem path to probe.
    Output: True for a regular file or nonabsence stat error, otherwise False; uncertain files
        proceed to _types.
    Notes: True if path is a regular file, or if the hook can't tell (a stat error that is not
        _ABSENT): such a path goes on to _types, which refuses it as unreadable. Only the existence
        probes use this; a read that fails on a file that is there still raises, and decide() blocks
        on it.
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
    Description: Determine which secret types a file exposes, retaining unreadable and unscanned
        risk markers.
    Parameters:
        path: File path to inspect.
        redactor: Redactor used to detect replaced content.
        grep_args: Optional structured Grep arguments for judging oversize matching output.
    Output: Secret-type list, empty list for no findings, or None for an absent, nonregular, or
        ordinary binary file; unreadable stat results and risky unscanned content return markers.
        Deadline, read, stat, and detector errors may propagate for refusal.
    Notes: Secret types in a file, [] if none, None if it isn't a scannable text file (missing, a
        directory, binary). A file over the read tool's whole-file limit is checked in chunks within
        the hook's time budget (FR-CORE-05; _chunked_types); for the Grep tool (grep_args), which
        prints only matching lines, those are judged instead (_printed_types).
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
    Description: Find secret types introduced by redaction without counting placeholders already
        present in text.
    Parameters:
        redactor: Object whose redact method transforms source text.
        text: Text to scan without printing its values.
    Output: Sorted added placeholder types, generic if changed without a new type, or an empty list
        if unchanged; detector errors propagate.
    Notes: Types of the secrets redaction replaces in text, [] if it changes nothing. A
        placeholder-shaped string already in text (docs, plans) is not a finding: only the
        placeholders redaction added count.
    """
    redacted = redactor.redact(text)
    if redacted == text:
        return []
    grown = Counter(_PLACEHOLDER_TYPE.findall(redacted)) - Counter(_PLACEHOLDER_TYPE.findall(text))
    return sorted(grown) or ["generic"]


def _printed_types(path: Path, args: dict, redactor) -> list[str]:
    """
    Description: Scan the bounded output an oversize Grep call could print using its flags and
        widest context.
    Parameters:
        path: Oversize file to search.
        args: Structured Grep arguments including pattern and context.
        redactor: Redactor used to inspect matched output.
    Output: Secret-type list or unscanned marker for unavailable rg, invalid pattern type,
        incomplete search, timeout, or excess output; launches and bounds an rg subprocess. Invalid
        context values, OS, deadline, or detector errors propagate for refusal.
    Notes: Secret types in what the Grep tool would print from an oversize file: one rg run over
        that file with the tool's flags and context lines. Anything short of a complete scan (no rg,
        an rg error, a timeout, more output than MAX_BYTES such as a pattern matching most lines)
        stays ["unscanned"]. Trade-off: the matched lines are judged without the rest of the file,
        so a secret the detector only recognises from other lines (a PEM body below its header) can
        pass; the Grep tool prints no more than that.
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
    Description: Find the first risky file while honoring user allowlist entries and
        session-specific built-in-tool grants.
    Parameters:
        paths: Paths to deduplicate and check in order.
        cwd: Working directory used for relative display names.
        grep_args: Optional structured Grep arguments for oversize output checks.
        session_id: Session identifier for approved real paths, or None to use no session grants.
    Output: Tuple of the first (display name, secret types, their list) hit or None and preceding
        allowlisted paths; does not log uses. Allowlist resolution errors disable that exemption; other scan and
        resolution errors propagate.
    Notes: ((name to show, secret types) of the first file that holds a secret, or None), and the
        allowlisted paths that passed on the way. Not scanned: a path on the user's allowlist, and
        one the user granted in this session (session_id, FR-READ-06). A grant names one real path;
        a search that reaches other files still checks them.
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
    Description: Refuse secret-bearing reads without prompting under FR-READ-06 and log allowlist
        uses only on allowed calls.
    Parameters:
        paths: Paths the call could expose.
        tool: Safe MCP tool suffix recommended on refusal.
        cwd: Working directory for path display.
        event_tool: Harness tool name for allowlist audit records.
        grep_args: Optional structured Grep arguments for oversize output checks.
        session_id: Session identifier for built-in-tool grants; Bash passes none.
    Output: Integer zero after an optional deny JSON response; allowed calls append audit records.
        Scan, resolution, logging, and output errors propagate for fail-closed handling.
    Notes: Refuse a read of a file that holds a secret (FR-READ-06: never a prompt). Bash passes no
        session_id: a grant covers the built-in tools only.
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
    Description: Read the text an edit starts from for comparison without printing it.
    Parameters:
        path: Target file path to read.
    Output: Text preserving newline spelling, an empty string for a missing file, or None for
        oversized or unreadable targets, including directories.
    Notes: The text an edit starts from: "" for a file that doesn't exist yet, None if it can't be
        compared (over the size limit, a directory, unreadable). Read here to judge the result;
        never printed.
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
    Description: Simulate edit results to prevent replacing real secrets with newly introduced
        placeholder values.
    Parameters:
        tool: Edit, MultiEdit, Write, or NotebookEdit tool name.
        args: Tool input including replacement text or edit steps.
        path: Target path whose current text is compared.
    Output: True when the edit introduces a placeholder under the comparison or conservative
        fallback rules, otherwise False; reads current content without printing it. NotebookEdit
        checks new source directly.
    Notes: True if the edit would leave a Harpo placeholder in the file that it did not hold
        before: writing one swaps a real secret for a fake value. The result is judged, not the new
        text alone, because a placeholder can be built from two edits (`<<HAR` now, the rest beside
        it later) or from two MultiEdit steps. Edit and MultiEdit are applied as the tool does
        (replace_all honoured, each step on the last one's result); Write is compared with the file
        it replaces. A placeholder the file already holds may stay or move (docs that quote one); a
        copy of it, or another, may not (compared as a multiset). A step that can't be applied here
        (old_string absent, or the file not comparable) falls back to its new text alone, which errs
        towards deny.
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
    Description: Check state-directory protection, placeholder introduction, then secret exposure
        before deciding an edit.
    Parameters:
        tool: Edit tool name.
        args: Tool input containing a file or notebook path and edit content.
        cwd: Working directory for resolving the target.
        event: Hook event containing session and tool-use identifiers.
    Output: Integer zero after allowing, denying, or asking; may log an allowlist use or save a
        pending grant. Unplaceable state targets are refused; asking requires a human-capable mode
        and successfully recorded identifiers. Other errors propagate for fail-closed handling.
    Notes: Edit, MultiEdit, Write or NotebookEdit (the tool result echoes the edited file back to
        the model). First of all, deny a target inside ~/.harpo or ~/.harpocrates
        (allowlist.inside_state_dir; any mode, with or without --ask, granted or not; a target that
        cannot be placed is refused too), or a hard link to a file in them
        (allowlist.links_to_state_file, SEC-09): only the user changes the allowlist, and a ~/.harpo the
        agent made would replace the migrated one. Every path-valued key of the call is checked, not only
        the tool's own (_path_values; SEC-09), and a call that names two different files is refused because the
        hook cannot know which one the tool writes. Then deny an edit that would add a placeholder
        (_adds_placeholder; any mode, granted or not). Then a secret-bearing file asks the user when
        a human can answer, after recording the call as pending so `--post` can turn Yes into a
        grant; with no session_id or tool_use_id to record it under, or no human, it is refused
        (fail closed).
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
    Description: Run a bounded read-only Git query with external ignore rules disabled for vetted
        scope decisions.
    Parameters:
        root: Working directory for the Git command.
        *args: Read-only Git arguments.
        ok: Accepted exit statuses, defaulting to zero.
    Output: Captured stdout bytes or None on an unaccepted status, timeout, or OSError; invokes Git
        and checks the deadline in all cases, which can raise _OutOfTimeError.
    Notes: stdout of a read-only git command run in root; None if git exits outside ok, stalls or
        isn't installed. core.excludesFile (repo config, or the XDG default) is switched off: ignore
        rules come from .gitignore files and .git/info/exclude only, which _scope vets.
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
    Description: Collect searchable Git metadata while excluding compressed objects except raw LFS
        objects.
    Parameters:
        gitdir: Git metadata directory to walk.
    Output: List of files including config, refs, logs, hooks, and LFS content; raises
        _OutOfTimeError on deadline or scope-cap exhaustion.
    Notes: Files a search that descends into .git reads, except objects/ (compressed; a real history
        would trip MAX_SCOPE_FILES on every `grep -r .`). lfs/ stays in, its objects/ too: that is
        the raw content of large files. Trade-off: a secret committed into an object is not checked;
        the config (a token in a remote URL), refs, logs and hooks, where one usually sits, are.
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
    Description: Identify a filesystem object by device and inode regardless of the symlink used to
        reach it.
    Parameters:
        path: Path to stat.
    Output: (device, inode) tuple; raises OSError on stat failure.
    Notes: Which directory this is, whatever link led to it (cheaper than resolving the path).
    """
    st = os.stat(path)
    return st.st_dev, st.st_ino


def _walk(root: Path, secret_names: bool = False, git_dir: bool = False,
          seen: Optional[set[tuple[int, int]]] = None) -> list[Path]:
    """
    Description: Walk directory contents conservatively with symlink-loop detection and a bounded
        file count.
    Parameters:
        root: Directory to walk.
        secret_names: Whether to include only names matching secret stores.
        git_dir: Whether to include Git internals when not restricted to secret names.
        seen: Shared set of visited directory identities, or None to create one.
    Output: List of files; follows directory links, mutates seen when supplied, and skips failed
        directory identity probes. Raises _OutOfTimeError on deadline or file-cap exhaustion.
    Notes: Every file under root, .git excluded (git_dir: included, see _git_internals); with
        secret_names only those named like a secret store. Links to directories are followed, as
        `grep -R` and `rg -L` do (the scope may read more than the tool, never less): each real
        directory is walked once (seen), so a link loop ends and a link to a place already covered
        adds nothing. The file cap still counts.
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
    Description: Detect ignore rules an agent could have changed since the last commit so scope can
        fall back to walking.
    Parameters:
        root: Git work-tree location to inspect.
    Output: True for dirty or untracked .gitignore files, effective info/exclude rules, or uncertain
        Git/read results; False when no such rules are found. Deadline errors propagate.
    Notes: True if an ignore rule an agent could have written since the last commit may be in
        effect: a .gitignore anywhere in the work tree that is untracked or has uncommitted changes,
        or any rule in .git/info/exclude.
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
    Description: Detect self-ignoring .gitignore files beneath directories that still have visible
        files.
    Parameters:
        visible: NUL-separated visible Git file paths.
        ignored: NUL-separated ignored Git file paths.
    Output: True when a self-ignored .gitignore shares an ancestor directory with visible files,
        otherwise False; empty cache-only directories are not flagged.
    Notes: True if an untracked .gitignore that ignores itself (`*`) sits in a directory that still
        shows files, as `*` in config/.gitignore does to a repo with tracked files in config/: it
        hides the untracked ones. Tool caches (.ruff_cache, .venv) write the same file into a
        directory with nothing else shown, which is fine.
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
    Description: Compute a bounded recursive read scope using vetted Git listings or a conservative
        full walk.
    Parameters:
        root: File or directory being read.
        full: Whether to force a full walk.
        git_dir: Whether to include searchable Git internals.
    Output: List of candidate files including ignored secret-store names and followed directory
        links; errors do not shrink the fallback scope. Raises _OutOfTimeError on budget or cap
        exhaustion; filesystem identity errors may propagate.
    Notes: Every file a recursive search of root reads, .git excluded. In a git work tree that is
        what git lists (tracked, and untracked but not ignored) plus git-ignored files named like
        secret stores (SENSITIVE_FILES), wherever they sit. Trade-off: a secret in a git-ignored
        file with an ordinary name (notes.local) can now pass a recursive shell grep; that is what
        stops data/, .venv and target/ from making every `grep -r .` too wide to vouch for. Reading
        such a file directly is still checked. Everything is walked instead when: full; root is
        itself ignored (naming a directory is explicit, unlike meeting it inside `grep -r .`);
        ignore rules were written since the last commit (_unvetted_ignore_rules,
        _hides_shown_files); or git can't list the tree (not a work tree, missing, failing or
        stalling): an error never shrinks the scope. RESIDUAL: a rule the agent commits still hides
        files (same class as the script gap in the module docstring), as does `git update-index
        --assume-unchanged` on an edited .gitignore, or a self-ignoring .gitignore in a directory
        with no tracked files. Links to directories are followed (_walk). git_dir adds the files of
        .git (_git_internals), for the shell searches that descend into it.
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
    Description: Find files a structured search could print, falling back to the whole scope when
        matching is uncertain.
    Parameters:
        pattern: Search expression, or None when exact semantics cannot be evaluated.
        roots: Search root paths.
        rg_flags: Ripgrep flags controlling matching and scope.
        py_flags: Equivalent Python regular-expression flags for fallback matching.
    Output: Candidate file paths; runs bounded rg or isolated Python subprocesses. Matcher errors or
        timeouts fall back to scope, while deadline and unhandled OS errors propagate for refusal.
    Notes: Files under roots whose content a search for pattern would print. pattern None means the
        hook can't evaluate it the way the real tool would: every file in scope counts as a match.
        So do a timeout and a matcher error.
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
    Description: Validate NUL-separated matcher output before trusting it as a file scope.
    Parameters:
        output: Matcher stdout with NUL-separated file names.
        roots: Original roots used for fallback scope.
        full: Whether fallback must walk the full tree.
    Output: Parsed paths if all are file candidates, otherwise the whole scope; filenames may
        contain newlines. Scope and deadline errors propagate.
    Notes: NUL-separated matcher output (a file name may contain a newline). Anything that isn't a
        file means the output can't be trusted: fall back to the whole scope.
    """
    paths = [Path(p) for p in output.split("\0") if p]
    return paths if all(_isfile(p) for p in paths) else [f for r in roots for f in _scope(r, full)]


def _grep_flags(args: dict) -> tuple[list[str], int]:
    """
    Description: Translate structured Grep case and multiline settings into matcher flags.
    Parameters:
        args: Grep input arguments.
    Output: Tuple of rg argument list and Python re flag bits; no subprocess is launched.
    Notes: rg flags, and the Python re flags that match them, which change what a Grep tool call
        matches.
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
    Description: Recognize grep recursion options conservatively so ambiguous argument words cannot
        shrink scope.
    Parameters:
        args: Grep argument words to inspect.
    Output: True for -r/-R clusters, recursive long options or abbreviations, and recurse
        directory-mode forms; otherwise False.
    Notes: True if these grep arguments make it descend into directories: -r or -R (alone or in a
        cluster), --recursive and its abbreviations, -d recurse. Over-reads: a word that looks like
        one counts.
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
    Description: Recognize flags that may supply a search pattern so file operands are not mistaken
        for it.
    Parameters:
        flag: Short cluster or long option to inspect.
    Output: True for possible -e, -f, --regexp, or --file forms, including abbreviations; otherwise
        False.
    Notes: True if a flag may carry the pattern (-e, -f, --regexp, --file, in a cluster or
        abbreviated): then no positional word is the pattern. Over-reads, so a real file operand is
        never taken for the pattern.
    """
    if flag.startswith("--"):
        name = flag.partition("=")[0]
        return len(name) > 3 and ("--regexp".startswith(name) or "--file".startswith(name))
    return bool(set(flag) & set("ef"))


def _reads_git(name: str, flags: list[str], recursive: bool) -> bool:
    """
    Description: Estimate whether search flags cause traversal into .git using a conservative
        tool-specific policy.
    Parameters:
        name: Search executable name.
        flags: Search option words.
        recursive: Whether grep recursion is enabled.
    Output: Boolean estimate for grep, rg, ag, or ack behavior; does not execute the search.
    Notes: True if the search, run with these flags, reads inside .git. Checked on this machine: BSD
        grep -r/-R do; rg only with --hidden, -. or -uu; git grep never. ag and ack aren't installed
        here, so assumed: ag with --hidden or -u, ack with --noignore-dir (it skips .git by
        default). Over-approximation is safe.
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
    Description: Scope shell search commands without evaluating their patterns because flags can
        change printed content.
    Parameters:
        argv: Search command name and arguments.
        here: Effective working directory.
        piped: Whether the command receives a pipe.
        tree: Whether file-name input or command semantics force tree search.
        git: Whether this is git grep.
        env_opts: Whether GREP_OPTIONS may enable grep recursion.
    Output: Files the search could print, or an empty list for names/counts-only or stdin-only
        cases; scope, filesystem, and deadline errors may propagate for refusal.
    Notes: Files a grep/rg-style shell command could print from. Its pattern is never evaluated:
        attached and clustered flags (`-eX`, `--regexp=X`, `-nA 3`), inversion (`-v`), `--passthru`
        and basic-regex syntax all change which lines print, so every file in scope counts unless
        only names are printed. The Grep tool, with its structured input, gets the exact check
        (_grep_tool). With no file operand it reads stdin, not the tree: grep unless it recurses,
        rg/ag/ack after a pipe. A word that could name files the hook can't see (glob, $VAR, $(...))
        counts as an operand. tree: it searches the work tree whatever its stdin (`git grep`) or
        xargs feeds it file names. The scope includes .git (_scope) for the searches that descend
        into it (_reads_git). env_opts: GREP_OPTIONS may make any grep recursive.
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
    Description: Collect all shell assignment alternatives without modeling control flow to
        over-approximate possible paths.
    Parameters:
        command: Shell text containing assignments in any branch or subshell.
        inherited: Previously known variable values.
    Output: New mapping from variable names to ordered unique possible values, including appended
        assignments.
    Notes: Every value each variable is given anywhere in the command (`D=x`, `export D=x`, `D+=x`,
        in any branch or subshell). Control flow isn't modelled: a later `$D` is checked against all
        of them, so the check can see more paths than the shell reads, never fewer.
    """
    values = dict(inherited)
    for name, append, raw in _ASSIGN_ANY.findall(command):
        value = raw.strip("'\"")
        given = [old + value for old in values.get(name, [""])] if append else [value]
        values = {**values, name: list(dict.fromkeys(values.get(name, []) + given))}
    return values


def _words(word: str, values: dict[str, list[str]], depth: int = 0) -> list[str]:
    """
    Description: Expand command-assigned variables into possible shell-word spellings for
        conservative path inspection.
    Parameters:
        word: Shell word to expand.
        values: Mapping of variable names to possible assigned values.
        depth: Current recursive expansion depth, initially zero.
    Output: List of unsplit and whitespace-split spellings including assigned HOME expansion; limits
        recursive self-reference and raises _OutOfTimeError above MAX_WORDS.
    Notes: Every spelling `word` may take: each `$NAME` / `${NAME}` the command set replaced by each
        value it was given, unsplit and split on whitespace (`F='a b'; cat $F` reads two files). A
        leading `~` also counts as a `$HOME` the command set.
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
    Description: Expand a shell word and assignment-style value into candidate paths, including
        globs.
    Parameters:
        word: Shell word to inspect after stripping surrounding quote and backtick characters.
        here: Effective working directory.
    Output: List of paths from literal, if=FILE-style, glob, home, or environment expansion; glob
        enumeration checks the deadline and raises _OutOfTimeError at the file cap.
    Notes: Paths a shell word may name: itself, `if=FILE`-style values, globs, `~` and `$VAR`.
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
    Description: Extract a heredoc body up to its delimiter or the end of the command.
    Parameters:
        text: Complete shell text.
        start: Offset where the body starts.
        word: Terminating delimiter.
        strip: Whether to ignore leading tabs when matching the delimiter.
    Output: Tuple of body text and offset after the terminating line, or end-of-text offset for an
        unclosed body.
    Notes: (heredoc body, index after its terminating line) for a body starting at start. A body
        that is never closed runs to the end of the text.
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
    Description: Separate logical shell lines, expansion-search text, and heredoc bodies without
        discarding content.
    Parameters:
        command: Shell command text to split.
    Output: Tuple of logical lines retaining heredoc operators, text searched for substitutions, and
        separate bodies; quoted bodies are excluded from expansion search and unquoted bodies remain
        in it.
    Notes: Split command into logical lines (at newlines outside quotes) and cut every heredoc body
        out of them. Returns the lines (the `<<WORD` operators stay), the text `$(...)` and
        backticks are searched in (an unquoted delimiter's body expands them before the command
        runs; a quoted one's is read as lines instead), and the bodies, which _bash_files reads
        apart (_prose): an apostrophe in a body would break the tokeniser. Nothing is ever dropped,
        so a `<<` that is no heredoc (`$((1<<X))`) hides nothing.
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
    Description: Read heredoc bodies as conservative command segments without rejecting stray prose
        quotes.
    Parameters:
        body: Heredoc body to split into lines and words.
    Output: List of (words, pipe-fed, True) tuples; splits at shell separators and strips word
        quotes without shlex. The True marker suppresses cd failure and eval/-c recursion on body
        lines.
    Notes: A heredoc body as command segments, whatever reads it (a shell down a pipe, `source
        /dev/stdin`, xargs, or plain cat): split at ; & | ( ) and backticks, words by str.split with
        their quotes stripped. Never shlex: prose has stray quotes and must not raise. It over-reads
        (a python comment naming grep is a search) and never under-reads a script. The True marks a
        body line: no cd failure, no eval or -c recursion on it.
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
    Description: Detect input execution so names inside echo or printf arguments receive no
        naming-only exemption.
    Parameters:
        segs: Command segments containing words, pipe-input flags, and prose markers.
    Output: True for eval or a pipe feeding an execution sink, including xargs-fed commands;
        otherwise False.
    Notes: True if a command evals its input or a pipe feeds text to a shell, `source`, `.`, `eval`
        or xargs: what looks like an argument of echo or printf may then run, so _NAMING excuses
        nothing.
    """
    for argv, piped, _prose_line in segs:
        i, fed = _command_word(argv)
        name = (Path(argv[i]).name or argv[i]) if i < len(argv) else ""
        if name == "eval" or (piped and (fed or name in _SINKS)):
            return True
    return False


def _command_word(argv: list[str]) -> tuple[int, bool]:
    """
    Description: Locate the executable command after assignments, shell keywords, and supported
        wrappers.
    Parameters:
        argv: Segment words to inspect.
    Output: Tuple of command index and whether xargs feeds it; returns len(argv) for naming-only
        command -v/-V, and may reach or exceed the end when wrapper options consume missing values.
    Notes: (index of the command a segment runs, whether xargs feeds it): skips `VAR=x` words, shell
        keywords (`do`, `then`) and the wrappers in _PREFIXES with their options. len(argv) if it
        runs none (`command -v rg` only looks a name up).
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
    Description: Extract find starting paths before the first search expression.
    Parameters:
        args: Words following find.
    Output: List of path words, skipping -H/-L/-P and stopping at another option, !, or an opening
        parenthesis.
    Notes: The start paths of a `find` command line (args: the words after `find`), up to its first
        expression.
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
    Description: Identify direct search commands and git grep after Git options.
    Parameters:
        argv: Segment words.
        start: Index of the actual command word.
    Output: Tuple of search-command index and git-grep boolean, or None; searches behind unknown
        wrappers are handled by _bash_files.
    Notes: (index of the search command in argv, whether it is `git grep`) for a segment whose
        command word is at start; None if that command is not one. A search behind another command
        is _bash_files' business.
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
    Description: Tokenize logical shell lines into command segments while retaining pipe-input
        information.
    Parameters:
        lines: Logical lines produced by heredoc separation.
    Output: List of (words, pipe-fed) tuples split at ; & | ( ); raises ValueError on unbalanced
        quotes so decide blocks the call.
    Notes: (words, whether a pipe feeds it) for each command in the logical lines (_heredocs), split
        at ; & | ( ).
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
    Description: Fetch content named by git show revision:path operands through read-only Git
        commands for scanning.
    Parameters:
        argv: Git command argument words to inspect from index two.
        here: Effective Git working directory.
    Output: Joined decoded blob text or None for no successful content; invokes Git without text
        conversion or external diffs. Subprocess, OS, and deadline errors propagate for refusal.
    Notes: Content `git show rev:path` would print, fetched read-only so it can be scanned.
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
    Description: Scope directory sources of copy, move, link, and archive commands to prevent
        allowlist laundering.
    Parameters:
        name: Copy or archive executable name.
        args: Arguments after the command name.
        here: Effective working directory.
    Output: List of source-directory files using recursive scope and the same cap and fail-closed
        behavior; generally omits the last destination word unless target-directory flags change its
        meaning. Expansion and scope errors propagate.
    Notes: Files under the directories a copy, move, link or archive command takes in, judged like a
        recursive search of each (same _scope, same cap, same fail-closed): `cp -r DIR
        graphify-out/y` would otherwise launder a secret into an allowlisted folder, since only
        files named as arguments were read. Every non-option word counts, option values included
        (over-reads: `tar -C DIR` scans DIR). A copy's last word is its target, left out unless a
        -t/--target-directory (or any short cluster holding t) may have moved it; trade-off: `cp x
        existing-dir` is not scanned for the secrets already in existing-dir, which this command
        doesn't read.
    """
    words = [a for a in args if not a.startswith("-")]
    targeted = any(a.startswith("--target") or (a[:1] == "-" and a[1:2] != "-" and "t" in a[1:]) for a in args)
    sources = words if name not in _TARGET_LAST or targeted else words[:-1]
    return [f for w in sources for p in _expand(w, here) if _isdir(p) for f in _scope(p)]


def _scan_operands(argv: list[str]) -> Optional[list[str]]:
    """
    Description: Identify agent-run Harpo scan operands while refusing modes a work-tree check
        cannot cover.
    Parameters:
        argv: Command words including possible wrappers or Python module invocation.
    Output: Non-option words after scan, or None when no scan runs or help is requested; raises
        _ScanUncheckableError for --history or --show-secrets.
    Notes: The non-option words after `scan` if this command runs `harpo scan` or `harpocrates scan` (the
        script, in any letter case since the default macOS file system folds it, or `python -m
        Harpocrates[.cli]`, behind any wrapper); None if it doesn't, or only asks for --help.
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
    Description: Collect files and extra text a shell command could expose using conservative
        parsing and bounded recursion.
    Parameters:
        command: Shell command text to inspect.
        cwd: Initial working directory.
        depth: Shell recursion depth, initially zero.
        inherited: Inherited variable alternatives, or None.
        named_ok: Whether naming-only commands can be exempted; False for executed substitution
            output.
        scans: Optional list to receive files whose values an agent-run Harpo scan could
            print.
    Output: Tuple of candidate files and extra text such as Git blobs; may read Git blobs and mutate
        scans. Raises ValueError for excessive nesting, ambiguous cd, or invalid quoting,
        _ScanUncheckableError for unsafe scan modes, and scope, deadline, or filesystem errors for
        refusal.
    Notes: (files the command could print, extra text it could print such as git blobs). named_ok
        False: the text is run as a command (a substitution's output), so _NAMING excuses nothing.
        scans: collects the files an agent-run `harpocrates scan` would print values from (a
        recursive reader of its operands, default `.`: scoped like a recursive search, same cap);
        _bash refuses those with its own message.
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
    Description: Identify environment variable names whose assignments the detector would redact
        without returning values.
    Parameters: None
    Output: List of flagged variable names in sorted environment order; invokes redaction and
        propagates detector errors.
    Notes: Names of environment variables whose value the detector would redact (never the values).
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
    Description: Dispatch a PreToolUse event under a fixed budget and fail closed on errors.
    Parameters:
        raw: JSON hook event text.
        ask: Whether supported human-answerable modes may prompt for edits.
    Output: Zero after a deny/ask decision or an allowed call, or 2 on other errors; sets deadline
        and ask mode, may read files or update audit/grant state, and prints only safe decisions or
        exception types. Budget exhaustion emits a deny decision. SEC-10: when the gate is on and a Claude Code
        setting sends requests around it (gatecheck.refusal), every tool call gets a deny, before anything else is
        looked at and whatever `ask` and the permission mode are; this never prompts.
    Notes: Hook entry point: print a deny or ask decision, or nothing; 2 (block) on any error.
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
    Description: Record completed approved edits as grants or clear a session on SessionEnd without
        blocking the harness.
    Parameters:
        raw: JSON PostToolUse, PostToolUseFailure, or SessionEnd event text.
    Output: Integer zero; may promote a matching pending edit or delete session grants. Prints
        nothing and suppresses ordinary exceptions; interrupts and nonedit or stray PreToolUse
        events never grant.
    Notes: `--post` entry point (PostToolUse, PostToolUseFailure, SessionEnd): turn the pending edit
        of this tool_use_id into a grant, or drop the session's grants. A failed tool still ran, so
        it grants (an Edit that fails on "not read yet" is what lets the Read through); an interrupt
        does not. Only the event name decides (a stray PreToolUse event never grants), and the event
        must name an edit tool, the one the hook asked about (grants.promote). It only records: it
        prints nothing and never raises (a crash here must not reach the harness).
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
