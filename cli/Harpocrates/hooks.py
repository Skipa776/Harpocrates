"""Claude Code and Codex PreToolUse hook (FR-READ-03): block built-in reads of any file that holds a secret.

Path deny rules only cover files named like secrets (.env, *.pem). A token in settings.py or app.yaml
would still reach the agent through Read, Grep, or `cat`. This hook scans the target first, with the
same detector and gate threshold as safe_read, and denies the call if anything would be redacted, telling
the agent to use safe_read or safe_grep instead. It prints only paths, counts, types and variable names,
never a value.

Fail closed: both harnesses treat a hook crash or timeout as "allow", so every error, an unparseable
command, a search pattern it can't evaluate safely, and running out of its time budget all exit 2, which
blocks the call. Run as `python -m Harpocrates.hooks` with the hook event JSON on stdin.

FR-READ-05: paths on the user's allowlist (Harpocrates/allowlist.py) pass, each use logged.
FR-READ-06: reads never prompt. A Read, Grep or Bash call that would expose a secret-bearing file is refused in every
mode. With `--ask` (Claude Code only) an edit of such a file (Edit, MultiEdit, Write, NotebookEdit) asks the user
instead, in the modes where a human answers; any other mode gets the refusal. Yes lets Claude read and edit that one
path with its built-in tools for the rest of the session, never through Bash (Harpocrates/grants.py). The grant is
recorded by `python -m Harpocrates.hooks --post` (PostToolUse, PostToolUseFailure, SessionEnd): those fire only for a
call that ran, so a No never reaches them. Trade-off: an approved edit sends those values to the provider; that is
the user's explicit call. An edit that would leave a Harpocrates placeholder in the file that it did not hold before
is denied in every mode, granted or not: it would swap a real secret for a fake value. Codex runs without --ask: its
handling of "ask" isn't verified.

ponytail: Bash coverage is best effort. It follows `sh -c` bodies, `$(...)`, backticks, `cd`, `git -C`,
globs, `~` and `$VAR` paths, `git show rev:path`, and the environment. A script that opens files itself
(`python x.py`, `node -e`, `python - <<EOF`), `find -exec`, `xargs` fed by a pipe, and `git diff` / `git log -p` output get
through. The egress gate (M2) covers what reaches the provider whatever the tool.
A recursive search inside a git work tree is scoped to the files git lists plus git-ignored files named like secret
stores; a secret in an ignored file with an ordinary name can pass `grep -r` (see _scope). Only the gate stops it.
"""
from __future__ import annotations

import fnmatch
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

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
_ENV_DUMP = re.compile(r"(?<![\w./-])(?:env|printenv|compgen)(?![\w./-])|\b(?:export|declare|typeset)\s+-p\b"
                       r"|(?:^|[;&|(`]\s*)set\s*(?:$|[;&|)`])|/proc/[^\s]*environ")
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
_SINKS = _SHELLS | {"source", ".", "eval", "xargs"}  # a pipe into one of these runs its input as commands
_PROSE_SPLIT = re.compile(r"(\|&|\|\||&&|[|&;()`])")  # what ends a command in a heredoc body
_HEREDOC = re.compile(
    r"""'[^']*'|"(?:\\.|[^"\\])*"|\\.|(?<!\S)#[^\n]*"""  # quotes, escapes and comments hide what they hold
    r"""|(?<!<)<<(?!<)(?P<tabs>-?)[ \t]*"""
    r"""(?:(?P<q>['"])(?P<qword>[^\n]+?)(?P=q)|(?P<esc>\\?)(?P<word>[^\s;&|()<>'"\\]+))"""
    r"|\n", re.S)
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
_PROMPTING_MODES = {"default", "acceptEdits", "plan", "auto"}  # auto: a hook's ask forces a real prompt
_can_ask = False

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
    pass


def _tick() -> None:
    if time.monotonic() > _deadline:
        raise _OutOfTimeError


def _left() -> float:
    """Timeout for one subprocess: MATCH_TIMEOUT, or what is left of BUDGET if that is less. Chained full
    timeouts could outlast the budget, and a hook that times out counts as allow."""
    return min(MATCH_TIMEOUT, max(_deadline - time.monotonic(), 0.1))


def _types(path: Path, redactor, grep_args: Optional[dict] = None) -> Optional[list[str]]:
    """Secret types in a file, [] if none, None if it isn't a scannable text file (missing, a directory,
    binary). An oversize file counts as unscannable-but-risky and is reported as ["unscanned"], except for
    the Grep tool (grep_args), which prints only matching lines: those are judged instead (_printed_types)."""
    from Harpocrates.read import MAX_BYTES, SNIFF_BYTES, _read_text, text_encoding

    _tick()
    if not path.is_file():
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
        return _printed_types(path, grep_args, redactor) if grep_args else ["unscanned"]
    try:
        text = _read_text(path)
    except ValueError:  # binary: images and PDFs pass, as FR-GATE-07 sets for the gate
        return None
    return _found(redactor, text)


def _found(redactor, text: str) -> list[str]:
    """Types of the secrets redaction replaces in text, [] if it changes nothing. A placeholder-shaped string
    already in text (docs, plans) is not a finding: only the placeholders redaction added count."""
    redacted = redactor.redact(text)
    if redacted == text:
        return []
    grown = Counter(_PLACEHOLDER_TYPE.findall(redacted)) - Counter(_PLACEHOLDER_TYPE.findall(text))
    return sorted(grown) or ["generic"]


def _printed_types(path: Path, args: dict, redactor) -> list[str]:
    """Secret types in what the Grep tool would print from an oversize file: one rg run over that file with the
    tool's flags and context lines. Anything short of a complete scan (no rg, an rg error, a timeout, more
    output than MAX_BYTES such as a pattern matching most lines) stays ["unscanned"].
    Trade-off: the matched lines are judged without the rest of the file, so a secret the detector only
    recognises from other lines (a PEM body below its header) can pass; the Grep tool prints no more than that."""
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
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    return 0


def _ask(reason: str) -> int:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                             "permissionDecisionReason": reason}}))
    return 0


def _first_flagged(paths: Iterable[Path], cwd: Path, grep_args: Optional[dict] = None,
                   session_id: object = None) -> tuple[Optional[tuple[str, str]], list[Path]]:
    """((name to show, secret types) of the first file that holds a secret, or None), and the allowlisted paths
    that passed on the way. Not scanned: a path on the user's allowlist, and one the user granted in this session
    (session_id, FR-READ-06). A grant names one real path; a search that reaches other files still checks them."""
    from Harpocrates import allowlist, grants
    from Harpocrates.read import Redactor

    redactor = Redactor()
    approved = grants.granted(session_id)
    passed: list[Path] = []
    for path in dict.fromkeys(paths):
        if allowlist.allowed(path):
            passed.append(path)
            continue
        if approved and str(path.resolve()) in approved:
            continue
        found = _types(path, redactor, grep_args)
        if found:
            shown = allowlist.printable(str(path.relative_to(cwd) if path.is_relative_to(cwd) else path))
            return (shown, "unscanned (over the size limit)" if found == ["unscanned"] else ", ".join(found)), passed
    return None, passed


def _check_files(paths: Iterable[Path], tool: str, cwd: Path, event_tool: str = "Read",
                 grep_args: Optional[dict] = None, session_id: object = None) -> int:
    """Refuse a read of a file that holds a secret (FR-READ-06: never a prompt). Bash passes no session_id: a
    grant covers the built-in tools only."""
    from Harpocrates import allowlist

    hit, passed = _first_flagged(paths, cwd, grep_args, session_id)
    if hit is not None:
        return _deny(f"{hit[0]} ({hit[1]}) contains secrets. Use mcp__harpocrates__{tool} instead: it returns the "
                     "same content with each secret replaced by a <<HARPO:type:hash>> placeholder.")
    for path in passed:  # the call may run: log the allowlist uses (not on a refusal)
        allowlist.log_use(path, event_tool)
    return 0


def _placeholders(text: str) -> Counter:
    return Counter(m.group(0) for m in _PLACEHOLDER_TYPE.finditer(text))


def _current(path: Path) -> Optional[str]:
    """The text an edit starts from: "" for a file that doesn't exist yet, None if it can't be compared (over the
    size limit, a directory, unreadable). Read here to judge the result; never printed."""
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
    """True if the edit would leave a Harpocrates placeholder in the file that it did not hold before: writing one
    swaps a real secret for a fake value. The result is judged, not the new text alone, because a placeholder can
    be built from two edits (`<<HAR` now, the rest beside it later) or from two MultiEdit steps. Edit and MultiEdit
    are applied as the tool does (replace_all honoured, each step on the last one's result); Write is compared with
    the file it replaces. A placeholder the file already holds may stay or move (docs that quote one); a copy of it,
    or another, may not (compared as a multiset). A step that can't be applied here (old_string absent, or the file
    not comparable) falls back to its new text alone, which errs towards deny."""
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


def _edit(tool: str, args: dict, cwd: Path, event: dict) -> int:
    """Edit, MultiEdit, Write or NotebookEdit (the tool result echoes the edited file back to the model).
    Deny an edit that would add a placeholder first (_adds_placeholder; any mode, granted or not). Then a
    secret-bearing file asks the user
    when a human can answer, after recording the call as pending so `--post` can turn Yes into a grant; with no
    session_id or tool_use_id to record it under, or no human, it is refused (fail closed)."""
    from Harpocrates import allowlist, grants

    path = cwd / os.path.expanduser(args.get("file_path") or args["notebook_path"])
    if _adds_placeholder(tool, args, path):
        return _deny("That edit would write a Harpocrates placeholder into the file, replacing a real secret with a "
                     "fake value. Edit around the secret, or ask the user to change it.")
    session_id = event.get("session_id")
    hit, passed = _first_flagged([path], cwd, None, session_id)
    if hit is None:
        for allowed in passed:
            allowlist.log_use(allowed, tool)
        return 0
    shown, kinds = hit
    if _can_ask and grants.add_pending(session_id, event.get("tool_use_id"), path, tool):
        return _ask(f"Harpocrates: Claude wants to edit {shown} without secret protection. It contains secrets "
                    f"({kinds}). Allowing lets Claude read and edit this file with its built-in tools for the rest "
                    "of this session; these values go to the model provider as-is. Choose No to keep them "
                    "protected: Claude can still read it with mcp__harpocrates__safe_read.")
    return _deny(f"{shown} ({kinds}) contains secrets, and a built-in edit shows its content to the model. "
                 "Ask the user to make this change.")


# ---------------------------------------------------------------------------------------- searching


def _git(root: Path, *args: str, ok: tuple[int, ...] = (0,)) -> Optional[bytes]:
    """stdout of a read-only git command run in root; None if git exits outside ok, stalls or isn't installed.
    core.excludesFile (repo config, or the XDG default) is switched off: ignore rules come from .gitignore files
    and .git/info/exclude only, which _scope vets."""
    try:
        proc = subprocess.run(["git", "--no-pager", "-c", "core.fsmonitor=false", "-c", f"core.excludesFile={os.devnull}",
                               "-C", str(root), *args], capture_output=True, timeout=_left(), env=_GIT_ENV)
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        _tick()
    return proc.stdout if proc.returncode in ok else None


def _in_git(root: Path) -> bool:
    return _git(root if root.is_dir() else root.parent, "rev-parse", "--is-inside-work-tree") == b"true\n"


def _git_internals(gitdir: Path) -> list[Path]:
    """Files a search that descends into .git reads, except objects/ (compressed; a real history would trip
    MAX_SCOPE_FILES on every `grep -r .`). lfs/ stays in, its objects/ too: that is the raw content of large files.
    Trade-off: a secret committed into an object is not checked; the config (a token in a remote URL), refs, logs
    and hooks, where one usually sits, are."""
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
    """Which directory this is, whatever link led to it (cheaper than resolving the path)."""
    st = os.stat(path)
    return st.st_dev, st.st_ino


def _walk(root: Path, secret_names: bool = False, git_dir: bool = False,
          seen: Optional[set[tuple[int, int]]] = None) -> list[Path]:
    """Every file under root, .git excluded (git_dir: included, see _git_internals); with secret_names only those
    named like a secret store. Links to directories are followed, as `grep -R` and `rg -L` do (the scope may read
    more than the tool, never less): each real directory is walked once (seen), so a link loop ends and a link to a
    place already covered adds nothing. The file cap still counts."""
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
    """True if an ignore rule an agent could have written since the last commit may be in effect: a .gitignore
    anywhere in the work tree that is untracked or has uncommitted changes, or any rule in .git/info/exclude."""
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
    """True if an untracked .gitignore that ignores itself (`*`) sits in a directory that still shows files, as
    `*` in config/.gitignore does to a repo with tracked files in config/: it hides the untracked ones. Tool
    caches (.ruff_cache, .venv) write the same file into a directory with nothing else shown, which is fine."""
    dirs = {os.fsdecode(e.rpartition(b"/")[0]) for e in ignored.split(b"\0") if e.rpartition(b"/")[2] == b".gitignore"}
    if not dirs:
        return False
    for entry in filter(None, visible.split(b"\0")):
        parts = os.fsdecode(entry).split("/")[:-1]
        if any("/".join(parts[:i]) in dirs for i in range(len(parts) + 1)):
            return True
    return False


def _scope(root: Path, full: bool = False, git_dir: bool = False) -> list[Path]:
    """Every file a recursive search of root reads, .git excluded. In a git work tree that is what git lists
    (tracked, and untracked but not ignored) plus git-ignored files named like secret stores (SENSITIVE_FILES),
    wherever they sit. Trade-off: a secret in a git-ignored file with an ordinary name (notes.local) can now
    pass a recursive shell grep; that is what stops data/, .venv and target/ from making every `grep -r .` too
    wide to vouch for. Reading such a file directly is still checked. Everything is walked instead when: full;
    root is itself ignored (naming a directory is explicit, unlike meeting it inside `grep -r .`); ignore rules
    were written since the last commit (_unvetted_ignore_rules, _hides_shown_files); or git can't list the tree
    (not a work tree, missing, failing or stalling): an error never shrinks the scope.
    RESIDUAL: a rule the agent commits still hides files (same class as the script gap in the module docstring),
    as does `git update-index --assume-unchanged` on an edited .gitignore, or a self-ignoring .gitignore in a
    directory with no tracked files.
    Links to directories are followed (_walk). git_dir adds the files of .git (_git_internals), for the shell
    searches that descend into it."""
    if root.is_file():
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
            if path.is_dir():  # a nested repo or submodule (git lists it as one entry), or a link to a directory
                files += _walk(path, secret_names, git_dir, seen)
            elif path.is_file():
                files.append(path)
            if len(files) > MAX_SCOPE_FILES:
                raise _OutOfTimeError  # too wide to vouch for: deny
    if git_dir and (root / ".git").is_dir():
        files += _git_internals(root / ".git")
        if len(files) > MAX_SCOPE_FILES:
            raise _OutOfTimeError
    return files


def _matching(pattern: Optional[str], roots: list[Path], rg_flags: list[str], py_flags: int) -> list[Path]:
    """Files under roots whose content a search for pattern would print. pattern None means the hook
    can't evaluate it the way the real tool would: every file in scope counts as a match. So do a
    timeout and a matcher error."""
    from Harpocrates.read import MAX_BYTES

    roots = [r for r in roots if r.exists()]
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
    """NUL-separated matcher output (a file name may contain a newline). Anything that isn't a file
    means the output can't be trusted: fall back to the whole scope."""
    paths = [Path(p) for p in output.split("\0") if p]
    return paths if all(p.is_file() for p in paths) else [f for r in roots for f in _scope(r, full)]


def _grep_flags(args: dict) -> tuple[list[str], int]:
    """rg flags, and the Python re flags that match them, which change what a Grep tool call matches."""
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
    """True if these grep arguments make it descend into directories: -r or -R (alone or in a cluster), --recursive
    and its abbreviations, -d recurse. Over-reads: a word that looks like one counts."""
    for a in args:
        if a.startswith("--"):
            name = a.partition("=")[0]
            if "recurs" in name or (len(name) > 3 and "--recursive".startswith(name)):
                return True
        elif a.startswith("-") and set(a) & set("rR"):
            return True
    return any(a == "recurse" or a.endswith("=recurse") or a.endswith("drecurse") for a in args)


def _names_pattern(flag: str) -> bool:
    """True if a flag may carry the pattern (-e, -f, --regexp, --file, in a cluster or abbreviated): then no
    positional word is the pattern. Over-reads, so a real file operand is never taken for the pattern."""
    if flag.startswith("--"):
        name = flag.partition("=")[0]
        return len(name) > 3 and ("--regexp".startswith(name) or "--file".startswith(name))
    return bool(set(flag) & set("ef"))


def _reads_git(name: str, flags: list[str], recursive: bool) -> bool:
    """True if the search, run with these flags, reads inside .git. Checked on this machine: BSD grep -r/-R do; rg
    only with --hidden, -. or -uu; git grep never. ag and ack aren't installed here, so assumed: ag with --hidden or
    -u, ack with --noignore-dir (it skips .git by default). Over-approximation is safe."""
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
    """Files a grep/rg-style shell command could print from. Its pattern is never evaluated: attached
    and clustered flags (`-eX`, `--regexp=X`, `-nA 3`), inversion (`-v`), `--passthru` and basic-regex
    syntax all change which lines print, so every file in scope counts unless only names are printed.
    The Grep tool, with its structured input, gets the exact check (_grep_tool).
    With no file operand it reads stdin, not the tree: grep unless it recurses, rg/ag/ack after a pipe. A word
    that could name files the hook can't see (glob, $VAR, $(...)) counts as an operand. tree: it searches the work
    tree whatever its stdin (`git grep`) or xargs feeds it file names. The scope includes .git (_scope) for the
    searches that descend into it (_reads_git). env_opts: GREP_OPTIONS may make any grep recursive."""
    name, opts = Path(argv[0]).name, argv[1:]
    if any(a in (_NAMES_ONLY - {"-L"} if name == "rg" else _NAMES_ONLY) for a in opts):  # rg -L is --follow
        return []  # file names or counts only
    head = opts[:opts.index("--")] if "--" in opts else opts
    flags = [a for a in head if a.startswith("-")]
    words = [a for a in head if not a.startswith("-")] + opts[len(head) + 1:]
    roots = [r for r in (_resolve(a, here) for a in words) if r.exists()]
    recursive = name in _GREPS and (env_opts or _recursive(head))
    operands = words if any(_names_pattern(a) for a in flags) else words[1:]  # else the first word is the pattern
    if not roots and not (tree or git or recursive or any(_UNRESOLVED.search(a) for a in operands)
                          or (name not in _GREPS and not piped)):
        return []  # reads stdin
    descends = not git and _reads_git(name, flags, recursive)
    return [f for r in (roots or [here]) if r.exists() for f in _scope(r, git_dir=descends)]


# ---------------------------------------------------------------------------------------- shell


def _assignments(command: str, inherited: dict[str, list[str]]) -> dict[str, list[str]]:
    """Every value each variable is given anywhere in the command (`D=x`, `export D=x`, `D+=x`, in any
    branch or subshell). Control flow isn't modelled: a later `$D` is checked against all of them, so the
    check can see more paths than the shell reads, never fewer."""
    values = dict(inherited)
    for name, append, raw in _ASSIGN_ANY.findall(command):
        value = raw.strip("'\"")
        given = [old + value for old in values.get(name, [""])] if append else [value]
        values = {**values, name: list(dict.fromkeys(values.get(name, []) + given))}
    return values


def _words(word: str, values: dict[str, list[str]], depth: int = 0) -> list[str]:
    """Every spelling `word` may take: each `$NAME` / `${NAME}` the command set replaced by each value it
    was given, unsplit and split on whitespace (`F='a b'; cat $F` reads two files). A leading `~` also
    counts as a `$HOME` the command set."""
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
    return here / os.path.expandvars(os.path.expanduser(word))


def _expand(word: str, here: Path) -> list[Path]:
    """Paths a shell word may name: itself, `if=FILE`-style values, globs, `~` and `$VAR`."""
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
    """(heredoc body, index after its terminating line) for a body starting at start. A body that is never closed
    runs to the end of the text."""
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
    """Split command into logical lines (at newlines outside quotes) and cut every heredoc body out of them.
    Returns the lines (the `<<WORD` operators stay), the text `$(...)` and backticks are searched in (an unquoted
    delimiter's body expands them before the command runs; a quoted one's is read as lines instead), and the
    bodies, which _bash_files reads apart (_prose): an apostrophe in a body would break the tokeniser. Nothing is
    ever dropped, so a `<<` that is no heredoc (`$((1<<X))`) hides nothing."""
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
    """A heredoc body as command segments, whatever reads it (a shell down a pipe, `source /dev/stdin`, xargs, or
    plain cat): split at ; & | ( ) and backticks, words by str.split with their quotes stripped. Never shlex:
    prose has stray quotes and must not raise. It over-reads (a python comment naming grep is a search) and never
    under-reads a script. The True marks a body line: no cd failure, no eval or -c recursion on it."""
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
    """True if a command evals its input or a pipe feeds text to a shell, `source`, `.`, `eval` or xargs: what
    looks like an argument of echo or printf may then run, so _NAMING excuses nothing."""
    for argv, piped, _prose_line in segs:
        i, fed = _command_word(argv)
        name = (Path(argv[i]).name or argv[i]) if i < len(argv) else ""
        if name == "eval" or (piped and (fed or name in _SINKS)):
            return True
    return False


def _command_word(argv: list[str]) -> tuple[int, bool]:
    """(index of the command a segment runs, whether xargs feeds it): skips `VAR=x` words, shell keywords (`do`,
    `then`) and the wrappers in _PREFIXES with their options. len(argv) if it runs none (`command -v rg` only
    looks a name up)."""
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
    """The start paths of a `find` command line (args: the words after `find`), up to its first expression."""
    paths: list[str] = []
    for a in args:
        if a in ("-H", "-L", "-P"):
            continue
        if a.startswith("-") or a in ("!", "("):
            break
        paths.append(a)
    return paths


def _search_word(argv: list[str], start: int) -> Optional[tuple[int, bool]]:
    """(index of the search command in argv, whether it is `git grep`) for a segment whose command word is at
    start; None if that command is not one. A search behind another command is _bash_files' business."""
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
    """(words, whether a pipe feeds it) for each command in the logical lines (_heredocs), split at ; & | ( )."""
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
    """Content `git show rev:path` would print, fetched read-only so it can be scanned."""
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
    """Files under the directories a copy, move, link or archive command takes in, judged like a recursive search
    of each (same _scope, same cap, same fail-closed): `cp -r DIR graphify-out/y` would otherwise launder a secret
    into an allowlisted folder, since only files named as arguments were read. Every non-option word counts, option
    values included (over-reads: `tar -C DIR` scans DIR). A copy's last word is its target, left out unless a
    -t/--target-directory (or any short cluster holding t) may have moved it; trade-off: `cp x existing-dir` is not
    scanned for the secrets already in existing-dir, which this command doesn't read."""
    words = [a for a in args if not a.startswith("-")]
    targeted = any(a.startswith("--target") or (a[:1] == "-" and a[1:2] != "-" and "t" in a[1:]) for a in args)
    sources = words if name not in _TARGET_LAST or targeted else words[:-1]
    return [f for w in sources for p in _expand(w, here) if p.is_dir() for f in _scope(p)]


def _bash_files(command: str, cwd: Path, depth: int = 0, inherited: Optional[dict[str, list[str]]] = None,
                named_ok: bool = True) -> tuple[list[Path], list[str]]:
    """(files the command could print, extra text it could print such as git blobs). named_ok False: the text is
    run as a command (a substitution's output), so _NAMING excuses nothing."""
    files: list[Path] = []
    texts: list[str] = []
    if depth > 3:
        raise ValueError("command nested too deep to check")
    lines, scan, bodies = _heredocs(command)  # bodies are cut out of what is tokenised and read apart (_prose)
    values = _assignments(command, inherited or {})  # nested bodies inherit even unexported names: more paths
    for m in _SUBSHELL.finditer(scan):  # $(...) and `...` run first (not in a quoted heredoc): check their bodies too
        sub_files, sub_texts = _bash_files(m.group(1) or m.group(2) or "", cwd, depth + 1, values, False)
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
    reads, named = False, []  # a reader anywhere makes every named file count (`for f in a b; do cat "$f"`)
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
            sub_files, sub_texts = _bash_files(" ".join(argv[1:]), here, depth + 1, values, False)
            files, texts = files + sub_files, texts + sub_texts
            continue
        body = None  # sudo / env ... bash -x -c BODY
        if not prose:
            body = next((argv[j + 1] for i, n in enumerate(names) if n in _SHELLS
                         for j in range(i + 1, len(argv) - 1) if _SHELL_C.match(argv[j])), None)
        if body is not None:
            sub_files, sub_texts = _bash_files(body, here, depth + 1, values)
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
        reads = reads or any(n in _READ_COMMANDS for n in names) or "<" in argv
        named += [p for w in argv for p in _expand(w, base)]
        if start < len(argv) and names[start] in _COPY_COMMANDS:
            files += _copy_dirs(names[start], argv[start + 1:], base)
    if reads:
        files += named
    return [f for f in files if f.is_file()], texts


def _env_secrets() -> list[str]:
    """Names of environment variables whose value the detector would redact (never the values)."""
    from Harpocrates.read import Redactor

    items = sorted(os.environ.items())
    lines = [f"{k}={v}" for k, v in items]
    redacted = Redactor().redact("\n".join(lines)).split("\n")
    return [k for (k, _), before, after in zip(items, lines, redacted) if before != after]


def _check_env(command: str) -> Optional[str]:
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
    from Harpocrates import allowlist
    from Harpocrates.read import Redactor

    if allowlist.touches(command, cwd):
        return _deny("Only the user can change the Harpocrates allowlist. Ask them to run "
                     "`harpocrates allow <path>` in their own terminal.")
    env_reason = _check_env(command)
    if env_reason:
        return _deny(env_reason)
    files, texts = _bash_files(command, cwd)
    for text in texts:
        if Redactor().redact(text) != text:
            return _deny("That git object contains secrets. Read the working-tree file with "
                         "mcp__harpocrates__safe_read instead.")
    return _check_files(files, "safe_read", cwd, "Bash")


# ---------------------------------------------------------------------------------------- entry


def decide(raw: str, ask: bool = False) -> int:
    """Hook entry point: print a deny or ask decision, or nothing; 2 (block) on any error."""
    global _deadline, _can_ask
    _deadline = time.monotonic() + BUDGET
    try:
        event = json.loads(raw)
        tool, args = event.get("tool_name"), event.get("tool_input") or {}
        cwd = Path(event.get("cwd") or ".")
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
        return _deny("Too many files to check for secrets in time. Use mcp__harpocrates__safe_grep or "
                     "mcp__harpocrates__safe_read on a narrower path instead.")
    except Exception as exc:  # fail closed; the type only, never a message that could quote file content
        print(f"harpocrates hook error ({type(exc).__name__}); blocked. Use mcp__harpocrates__safe_read.",
              file=sys.stderr)
        return 2


def post(raw: str) -> int:
    """`--post` entry point (PostToolUse, PostToolUseFailure, SessionEnd): turn the pending edit of this tool_use_id
    into a grant, or drop the session's grants. A failed tool still ran, so it grants (an Edit that fails on "not
    read yet" is what lets the Read through); an interrupt does not. Only the event name decides (a stray PreToolUse
    event never grants), and the event must name an edit tool, the one the hook asked about (grants.promote). It only records: it prints nothing and never raises (a crash here must not reach the harness)."""
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
