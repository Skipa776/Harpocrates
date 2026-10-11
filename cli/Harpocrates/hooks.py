"""Claude Code and Codex PreToolUse hook (FR-READ-03): block built-in reads of any file that holds a secret.

Path deny rules only cover files named like secrets (.env, *.pem). A token in settings.py or app.yaml
would still reach the agent through Read, Grep, or `cat`. This hook scans the target first, with the
same detector and gate threshold as safe_read, and denies the call if anything would be redacted, telling
the agent to use safe_read or safe_grep instead. It prints only paths, counts, types and variable names,
never a value.

Fail closed: both harnesses treat a hook crash or timeout as "allow", so every error, an unparseable
command, a search pattern it can't evaluate safely, and running out of its time budget all exit 2, which
blocks the call. Run as `python -m Harpocrates.hooks` with the hook event JSON on stdin.

FR-READ-05: paths on the user's allowlist (Harpocrates/allowlist.py) pass, each use logged. With `--ask`
(Claude Code only) a secret-bearing read or edit prompts the user instead of being refused, naming every flagged
file. Trade-off: an approved prompt sends those values to the provider; that is the user's explicit call.
The prompt only happens in modes where a human answers it; any other mode (bypassPermissions, dontAsk, an
unknown one) gets the refusal. Codex runs without --ask: its handling of "ask" isn't verified.

ponytail: Bash coverage is best effort. It follows `sh -c` bodies, `$(...)`, backticks, `cd`, `git -C`,
globs, `~` and `$VAR` paths, `git show rev:path`, and the environment. A script that opens files itself
(`python x.py`, `node -e`), `find -exec`, `xargs` fed by a pipe, and `git diff` / `git log -p` output get
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
                  "mv", "ln", "rsync", "install"}  # mv/ln: moving a file into an allowlisted folder
_SEARCH_COMMANDS = {"grep", "egrep", "fgrep", "rg", "ag", "ack"}
_NAMES_ONLY = {"-l", "--files-with-matches", "-L", "--files-without-match", "-c", "--count", "--files",
               "-q", "--quiet", "--count-matches"}
_SHELLS = {"sh", "bash", "zsh", "dash", "fish", "ksh"}
_ENV_DUMP = re.compile(r"(?<![\w./-])(?:env|printenv|compgen)(?![\w./-])|\b(?:export|declare|typeset)\s+-p\b"
                       r"|(?:^|[;&|(`]\s*)set\s*(?:$|[;&|)`])|/proc/[^\s]*environ")
_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "\n"}
_SHELL_C = re.compile(r"^-[a-zA-Z]*c[a-zA-Z]*$")
_SUBSHELL = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_ENV_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")
_VAR = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
_ASSIGN_ANY = re.compile(r"(?<![\w$])([A-Za-z_]\w*)(\+?)=(\"[^\"]*\"|'[^']*'|[^\s;&|()`]*)")
_SENSITIVE = re.compile("|".join(map(fnmatch.translate, SENSITIVE_FILES)))  # file names, matched with .match
MAX_SCOPE_FILES = 2000  # a search wider than this is denied rather than scanned: safe_grep handles it
MAX_WORDS = 64  # spellings one shell word may take once variables are filled in; more is denied
MAX_ASK_FILES = 20  # more secret files than this in one call is refused, not put to the user unread
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
    from Harpocrates.read import MAX_BYTES, _read_text

    _tick()
    if not path.is_file():
        return None
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


def _check_files(paths: Iterable[Path], tool: str, cwd: Path, event_tool: str = "Read",
                 action: str = "read", grep_args: Optional[dict] = None) -> int:
    from Harpocrates import allowlist
    from Harpocrates.read import Redactor

    redactor = Redactor()
    flagged: list[str] = []
    passed: list[Path] = []
    for path in dict.fromkeys(paths):
        if allowlist.allowed(path):
            passed.append(path)
            continue
        found = _types(path, redactor, grep_args)
        if found:
            shown = allowlist.printable(str(path.relative_to(cwd) if path.is_relative_to(cwd) else path))
            kinds = "unscanned (over the size limit)" if found == ["unscanned"] else ", ".join(found)
            flagged.append(f"{shown} ({kinds})")
            if not _can_ask:
                break  # refusing anyway: the first file is enough
    ask = _can_ask and len(flagged) <= MAX_ASK_FILES
    if not flagged or ask:  # the call may run: log the allowlist uses (not on a refusal)
        for path in passed:
            allowlist.log_use(path, event_tool)
    if not flagged:
        return 0
    if ask:
        listed = "\n".join(f"  {f}" for f in flagged)
        caveat = (" Shell commands are checked on a best effort basis: this one may read files not listed."
                  if event_tool == "Bash" else "")
        return _ask(f"Harpocrates: Claude wants to {action} files that contain secrets:\n{listed}\n"
                    f"Allowing sends these values to the model provider as-is.{caveat} Choose No to keep them out; "
                    f"Claude can use mcp__harpocrates__{tool}, which redacts them. To always allow a path, "
                    "run `harpocrates allow <path>` in your own terminal.")
    if action != "read":
        return _deny(f"{flagged[0]} contains secrets, and a built-in edit shows its content to the model. "
                     "Ask the user to make this change.")
    return _deny(f"{flagged[0]} contains secrets. Use mcp__harpocrates__{tool} instead: it returns the same "
                 "content with each secret replaced by a <<HARPO:type:hash>> placeholder.")


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


def _walk(root: Path, secret_names: bool = False) -> list[Path]:
    """Every file under root, .git excluded; with secret_names only those named like a secret store."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        _tick()
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


def _scope(root: Path, full: bool = False) -> list[Path]:
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
    directory with no tracked files."""
    if root.is_file():
        return [root]
    ls = ["ls-files", "-z", "--others", "--exclude-standard"]
    ignored_root = _git(root, "check-ignore", ".", ok=(0, 1)) != b""  # prints "." if ignored; None on error
    visible = None if full or ignored_root or _unvetted_ignore_rules(root) else _git(root, *ls, "--cached")
    ignored = None if visible is None else _git(root, *ls, "--ignored")
    if ignored is None or _hides_shown_files(visible, ignored):
        return _walk(root)
    files: list[Path] = []
    for listing, secret_names in ((visible, False), (ignored, True)):
        for rel in filter(None, listing.split(b"\0")):
            _tick()
            name = os.fsdecode(rel)
            if secret_names and not name.endswith("/") and not _SENSITIVE.match(name.rsplit("/", 1)[-1]):
                continue
            path = root / name
            if path.is_dir() and not path.is_symlink():  # a nested repo or submodule: git lists it as one entry
                files += _walk(path, secret_names)
            elif path.is_file():
                files.append(path)
            if len(files) > MAX_SCOPE_FILES:
                raise _OutOfTimeError  # too wide to vouch for: deny
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
        proc = subprocess.run([sys.executable, "-c", _PY_MATCHER, arg], timeout=_left(), **run)
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


def _search_command(argv: list[str], here: Path) -> list[Path]:
    """Files a grep/rg-style shell command could print from. Its pattern is never evaluated: attached
    and clustered flags (`-eX`, `--regexp=X`, `-nA 3`), inversion (`-v`), `--passthru` and basic-regex
    syntax all change which lines print, so every file in scope counts unless only names are printed.
    The Grep tool, with its structured input, gets the exact check (_grep_tool)."""
    if any(a in _NAMES_ONLY for a in argv[1:]):
        return []  # file names or counts only
    roots = [_resolve(a, here) for a in argv[1:] if not a.startswith("-") and _resolve(a, here).exists()]
    return _matching(None, roots or [here], [], 0)


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


def _segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for tok in lexer:  # unbalanced quotes raise ValueError -> decide() blocks the call
        if tok in _SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(tok)
    return [seg for seg in segments if seg]


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


def _bash_files(command: str, cwd: Path, depth: int = 0,
                inherited: Optional[dict[str, list[str]]] = None) -> tuple[list[Path], list[str]]:
    """(files the command could print, extra text it could print such as git blobs)."""
    files: list[Path] = []
    texts: list[str] = []
    if depth > 3:
        raise ValueError("command nested too deep to check")
    values = _assignments(command, inherited or {})  # nested bodies inherit even unexported names: more paths
    for m in _SUBSHELL.finditer(command):  # $(...) and `...` run first: check their bodies too
        sub_files, sub_texts = _bash_files(m.group(1) or m.group(2) or "", cwd, depth + 1, values)
        files, texts = files + sub_files, texts + sub_texts
    here = cwd
    reads, named = False, []  # a reader anywhere makes every named file count (`for f in a b; do cat "$f"`)
    for argv in _segments(command):
        argv = [w for t in argv for w in _words(t, values)]  # `D=dir; cat $D/f`, `R=HEAD:.env; git show $R`
        names = [Path(t).name for t in argv]
        if names[0] == "cd":
            if len(argv) > 2:
                raise ValueError("cd to a directory that depends on a branch")  # can't tell which: block
            here = _resolve(argv[1], here) if len(argv) > 1 else Path.home()
            continue
        if names[0] in ("eval", "exec", "watch") and len(argv) > 1:
            sub_files, sub_texts = _bash_files(" ".join(argv[1:]), here, depth + 1, values)
            files, texts = files + sub_files, texts + sub_texts
            continue
        body = next((argv[j + 1] for i, n in enumerate(names) if n in _SHELLS  # sudo / env ... bash -x -c BODY
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
        for i, n in enumerate(names):
            if n in _SEARCH_COMMANDS:
                files += _search_command(argv[i:], base)
                break
        reads = reads or any(n in _READ_COMMANDS for n in names) or "<" in argv
        named += [p for w in argv for p in _expand(w, base)]
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

    if allowlist.touches(command):
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
            return _check_files([cwd / args["file_path"]], "safe_read", cwd)
        if tool == "Grep" and args.get("output_mode") == "content":
            return _check_files(_grep_tool(args, cwd), "safe_grep", cwd, "Grep", grep_args=args)
        if tool == "Bash":
            return _bash(args.get("command", ""), cwd)
        if tool in _EDIT_TOOLS:  # the tool result echoes the edited file back to the model
            path = os.path.expanduser(args.get("file_path") or args["notebook_path"])
            return _check_files([cwd / path], "safe_read", cwd, tool, action="read and change")
        return 0
    except _OutOfTimeError:
        return _deny("Too many files to check for secrets in time. Use mcp__harpocrates__safe_grep or "
                     "mcp__harpocrates__safe_read on a narrower path instead.")
    except Exception as exc:  # fail closed; the type only, never a message that could quote file content
        print(f"harpocrates hook error ({type(exc).__name__}); blocked. Use mcp__harpocrates__safe_read.",
              file=sys.stderr)
        return 2


def main() -> None:
    try:
        code = decide(sys.stdin.buffer.read().decode("utf-8", errors="replace"), ask="--ask" in sys.argv[1:])
    except BaseException:  # noqa: BLE001 - anything that escapes would exit 1, which harnesses treat as allow
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
