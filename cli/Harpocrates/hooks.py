"""Claude Code and Codex PreToolUse hook (FR-READ-03): block built-in reads of any file that holds a secret.

Path deny rules only cover files named like secrets (.env, *.pem). A token in settings.py or app.yaml
would still reach the agent through Read, Grep, or `cat`. This hook scans the target first, with the
same detector and gate threshold as safe_read, and denies the call if anything would be redacted, telling
the agent to use safe_read or safe_grep instead. It prints only paths, counts, types and variable names,
never a value.

Fail closed: both harnesses treat a hook crash or timeout as "allow", so every error, an unparseable
command, a search pattern it can't evaluate safely, and running out of its time budget all exit 2, which
blocks the call. Run as `python -m Harpocrates.hooks` with the hook event JSON on stdin.

ponytail: Bash coverage is best effort. It follows `sh -c` bodies, `$(...)`, backticks, `cd`, `git -C`,
globs, `~` and `$VAR` paths, `git show rev:path`, and the environment. A script that opens files itself
(`python x.py`, `node -e`), `find -exec`, `xargs` fed by a pipe, and `git diff` / `git log -p` output get
through. The egress gate (M2) covers what reaches the provider whatever the tool.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

_PLACEHOLDER_TYPE = re.compile(r"<<HARPO:(\w+):[0-9a-f]{4}>>")
_READ_COMMANDS = {"cat", "head", "tail", "less", "more", "bat", "nl", "sed", "awk", "strings", "xxd", "od",
                  "hexdump", "base64", "tac", "cut", "sort", "uniq", "diff", "jq", "yq", "type", "Get-Content",
                  "cp", "dd", "paste", "column", "fold", "rev", "tee", "xargs", "zcat", "zless", "pr", "fmt",
                  "expand", "join", "comm", "source", ".", "blame", "annotate"}
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
MAX_SCOPE_FILES = 2000  # a search wider than this is denied rather than scanned: safe_grep handles it
MATCH_TIMEOUT = 5.0  # seconds for one pattern match over a tree (runs in a child process)
BUDGET = 25.0  # seconds for the whole decision; Codex's documented example hook timeout is 30
_deadline = float("inf")

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


def _types(path: Path, redactor) -> Optional[list[str]]:
    """Secret types in a file, [] if none, None if it isn't a scannable text file (missing, a directory,
    binary). An oversize file counts as unscannable-but-risky and is reported as ["unscanned"]."""
    from Harpocrates.read import MAX_BYTES, _read_text

    _tick()
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_BYTES:
        return ["unscanned"]
    try:
        text = _read_text(path)
    except ValueError:  # binary: images and PDFs pass, as FR-GATE-07 sets for the gate
        return None
    return sorted(set(_PLACEHOLDER_TYPE.findall(redactor.redact(text))))


def _deny(reason: str) -> int:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    return 0


def _check_files(paths: Iterable[Path], tool: str, cwd: Path) -> int:
    from Harpocrates.read import Redactor

    redactor = Redactor()
    for path in paths:
        found = _types(path, redactor)
        if found:
            shown = path.relative_to(cwd) if path.is_relative_to(cwd) else path
            kinds = "unscanned (over the size limit)" if found == ["unscanned"] else ", ".join(found)
            return _deny(f"{shown} contains secrets ({kinds}). Use mcp__harpocrates__{tool} instead: it "
                         "returns the same content with each secret replaced by a <<HARPO:type:hash>> placeholder.")
    return 0


# ---------------------------------------------------------------------------------------- searching


def _scope(root: Path) -> list[Path]:
    """Every file a recursive search of root reads (rg --hidden --no-ignore), .git excluded."""
    if root.is_file():
        return [root]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        _tick()
        dirnames[:] = [d for d in dirnames if d != ".git"]
        files += [Path(dirpath) / f for f in filenames]
        if len(files) > MAX_SCOPE_FILES:
            raise _OutOfTimeError  # too wide to vouch for: deny
    return files


def _matching(pattern: Optional[str], roots: list[Path], rg_flags: list[str], py_flags: int) -> list[Path]:
    """Files under roots whose content a search for pattern would print. pattern None means the hook
    can't evaluate it the way the real tool would: every file in scope counts as a match. So do a
    timeout and a matcher error."""
    from Harpocrates.read import MAX_BYTES

    roots = [r for r in roots if r.exists()]
    rg = shutil.which("rg")
    if pattern is not None and rg:  # exact semantics; rg also lists oversize files, which _types flags
        argv = [rg, "-l", "-0", "--hidden", "--no-ignore", "--no-messages", *rg_flags, "-e", pattern, "--",
                *map(str, roots)]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=MATCH_TIMEOUT)
            if proc.returncode in (0, 1):
                return _paths(proc.stdout, roots)
        except subprocess.TimeoutExpired:
            pass
        return [f for r in roots for f in _scope(r)]
    scope = [f for r in roots for f in _scope(r)]
    if pattern is None:
        return scope
    arg = json.dumps({"pattern": pattern, "flags": py_flags | re.M, "files": list(map(str, scope)),
                      "max_bytes": MAX_BYTES})
    try:
        proc = subprocess.run([sys.executable, "-c", _PY_MATCHER, arg], capture_output=True, text=True,
                              timeout=MATCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return scope
    if proc.returncode != 0:  # a pattern Python rejects: assume the real tool matches
        return scope
    return _paths(proc.stdout, roots)


def _paths(output: str, roots: list[Path]) -> list[Path]:
    """NUL-separated matcher output (a file name may contain a newline). Anything that isn't a file
    means the output can't be trusted: fall back to the whole scope."""
    paths = [Path(p) for p in output.split("\0") if p]
    return paths if all(p.is_file() for p in paths) else [f for r in roots for f in _scope(r)]


def _grep_tool(args: dict, cwd: Path) -> list[Path]:
    flags: list[str] = []
    py = 0
    if args.get("-i"):
        flags.append("-i")
        py |= re.I
    if args.get("multiline"):
        flags += ["-U", "--multiline-dotall"]
        py |= re.S
    if args.get("glob"):
        flags += ["-g", args["glob"]]
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
                                  capture_output=True, timeout=MATCH_TIMEOUT, env=_GIT_ENV)
            if proc.returncode == 0:
                texts.append(proc.stdout.decode("utf-8", errors="replace"))
    return "\n".join(texts) or None


def _bash_files(command: str, cwd: Path, depth: int = 0) -> tuple[list[Path], list[str]]:
    """(files the command could print, extra text it could print such as git blobs)."""
    files: list[Path] = []
    texts: list[str] = []
    if depth > 3:
        raise ValueError("command nested too deep to check")
    for m in _SUBSHELL.finditer(command):  # $(...) and `...` run first: check their bodies too
        sub_files, sub_texts = _bash_files(m.group(1) or m.group(2) or "", cwd, depth + 1)
        files, texts = files + sub_files, texts + sub_texts
    here = cwd
    reads, named = False, []  # a reader anywhere makes every named file count (`for f in a b; do cat "$f"`)
    for argv in _segments(command):
        names = [Path(t).name for t in argv]
        if names[0] == "cd":
            here = _resolve(argv[1], here) if len(argv) > 1 else Path.home()
            continue
        if names[0] in ("eval", "exec", "watch") and len(argv) > 1:
            sub_files, sub_texts = _bash_files(" ".join(argv[1:]), here, depth + 1)
            files, texts = files + sub_files, texts + sub_texts
            continue
        body = next((argv[j + 1] for i, n in enumerate(names) if n in _SHELLS  # sudo / env ... bash -x -c BODY
                     for j in range(i + 1, len(argv) - 1) if _SHELL_C.match(argv[j])), None)
        if body is not None:
            sub_files, sub_texts = _bash_files(body, here, depth + 1)
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
    from Harpocrates.read import Redactor

    env_reason = _check_env(command)
    if env_reason:
        return _deny(env_reason)
    files, texts = _bash_files(command, cwd)
    for text in texts:
        if Redactor().redact(text) != text:
            return _deny("That git object contains secrets. Read the working-tree file with "
                         "mcp__harpocrates__safe_read instead.")
    return _check_files(files, "safe_read", cwd)


# ---------------------------------------------------------------------------------------- entry


def decide(raw: str) -> int:
    """Hook entry point: print a deny decision or nothing; 2 (block) on any error."""
    global _deadline
    _deadline = time.monotonic() + BUDGET
    try:
        event = json.loads(raw)
        tool, args = event.get("tool_name"), event.get("tool_input") or {}
        cwd = Path(event.get("cwd") or ".")
        if tool == "Read":
            return _check_files([cwd / args["file_path"]], "safe_read", cwd)
        if tool == "Grep" and args.get("output_mode") == "content":
            return _check_files(_grep_tool(args, cwd), "safe_grep", cwd)
        if tool == "Bash":
            return _bash(args.get("command", ""), cwd)
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
        code = decide(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
    except BaseException:  # noqa: BLE001 - anything that escapes would exit 1, which harnesses treat as allow
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
