"""User-approved paths the read hook lets through even when they hold secrets (FR-READ-05).

One absolute path per line in ~/.harpocrates/allow; a directory covers everything under it. Paths are
compared after resolving symlinks, so a link inside an allowed folder that points elsewhere is still
checked. Every pass is appended to allow.log (time, tool, path; never content), as design.md's
"explicit overrides, always logged" asks.

Only the user edits the list: `harpocrates allow PATH`, or the file by hand. The hook refuses Bash
commands that change it or call `harpocrates allow` (read-only commands pass) and the Claude Code
settings deny Edit/Write on ~/.harpocrates.
ponytail: exact paths only, no globs; add `**/graphify-out`-style patterns if per-project entries get tedious.
"""
from __future__ import annotations

import re
import shlex
import time
from pathlib import Path
from typing import Optional

ALLOW_FILE = Path.home() / ".harpocrates" / "allow"
LOG_FILE = Path.home() / ".harpocrates" / "allow.log"

# ponytail: text match, best effort. A script that builds the path at runtime gets past it, but such a
# script could just as well open the secret file itself, a gap the hook already has (hooks.py docstring).
# The egress gate (M2) sees whatever reaches the provider either way.
# Rule: `harpocrates ... allow` anywhere in a segment is refused (only a mid-path `harpocrates/` is
# exempt), so a repo directory of that name over-blocks `git -C .../harpocrates log --grep allow`. A
# mention of the directory (`.harp`, also `~/.harp*`) passes only when it is a plain read: no output
# redirect, substitution or heredoc, and every command word in _READ_ONLY. Fail closed on anything else.
# A dot-name containing a glob or brace character (`~/.*`, `~/.h*`, `.[h]arp...`, `.{a,b}`, `.?arp...`) also
# counts as a mention, so it needs a plain read too. Known gap: a name built at runtime (variables, command
# substitution, printf) matches nothing; that is the script gap above. Matching runs on the raw command and on
# a normalised copy (see _normalise). Accepted over-block: `rm -rf .[a-z]*cache`.
_ALLOW_CMD = re.compile(
    r"\bharpocrates(?:\.cli)?\b(?!/)[^;&|\n]*\ballow\b"  # harpocrates / python -m Harpocrates.cli ... allow
    r"|(?:\$\([^)]*\)|`[^`]*`|\$\{?\w+\}?)\s*['\"]?allow\b"  # $(which harpocrates) allow
    r"|\ballowlist\s*\.\s*add\b", re.I)
_DIR = re.compile(r"""\.harp|(?:^|[/\s'"=~])\.[^/\s'";&|]*[*?\[{]""", re.I)
_UNSAFE_TEXT = re.compile(r">|\$\(|`|<\(|<<")
_READ_ONLY = frozenset(
    "cat ls head tail wc stat file du grep egrep fgrep diff cmp md5 md5sum shasum sha256sum nl cut echo printf "
    "test [ pwd realpath readlink basename dirname true cd".split())
_SEPARATORS = frozenset(";&|()")


def entries() -> list[Path]:
    if not ALLOW_FILE.exists():
        return []
    lines = (line.strip() for line in ALLOW_FILE.read_text(encoding="utf-8").splitlines())
    return [Path(line) for line in lines if line and not line.startswith("#")]


def printable(text: str) -> str:
    """Control characters (newlines, ANSI escapes) in a file name shown as '?', so a name can't fake
    lines in a prompt or the log."""
    return "".join(c if c.isprintable() else "?" for c in text)


def allowed(path: Path) -> bool:
    real = path.resolve()
    if real.is_file() and real.stat().st_nlink > 1:
        return False  # a hard link may be a secret file linked into an allowed folder: scan it
    return any(real == e or real.is_relative_to(e) for e in entries())


def add(path: Path) -> Path:
    real = path.resolve(strict=True)  # a typo fails here instead of allowing nothing
    if real not in entries():
        ALLOW_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ALLOW_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{real}\n")
    return real


def log_use(path: Path, tool: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{tool}\t{printable(str(path.resolve()))}\n")


def _segments(command: str) -> list[list[str]]:
    """Words of each simple command; newlines, ;&| and any token with `(` start a new one."""
    lex = shlex.shlex(re.sub(r"[\r\n]", ";", command), posix=True, punctuation_chars=True)
    lex.whitespace_split, lex.commenters = True, ""
    segments: list[list[str]] = [[]]
    for tok in lex:  # ValueError on unbalanced quotes: the caller fails closed
        if set(tok) <= _SEPARATORS or "(" in tok:
            if tok.strip(")") and segments[-1]:
                segments.append([])
        else:
            segments[-1].append(tok)
    return [seg for seg in segments if seg]


def _read_only(command: str) -> bool:
    """Fail closed: redirects, substitutions, heredocs, untokenisable text or any command word outside
    _READ_ONLY make it not read-only."""
    if _UNSAFE_TEXT.search(command):
        return False
    try:
        return all(seg[0] in _READ_ONLY for seg in _segments(command))
    except ValueError:
        return False


_ANSI_C = re.compile(r"\$'((?:[^'\\]|\\.)*)'", re.S)
_ESCAPE = re.compile(r"\\(x[0-9a-fA-F]{1,2}|u[0-9a-fA-F]{1,4}|[0-7]{1,3}|.)", re.S)
_SIMPLE_ESC = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "e": "\x1b", "f": "\f", "v": "\v"}
# On the normalised text: the allow file or the grants dir (FR-READ-06) by name, a glob char right before
# `arpocrates` (or `h?arpocrates`), or dotglob/globdots next to any glob, all count as a mention of the directory.
# `grants` must end the name: the module cli/Harpocrates/grants.py is no mention.
_MENTION_NORM = re.compile(r"arpocrates/(?:allow\b|grants(?![\w.-]))|[*?\]]h?arpocrates", re.I)
_DOTGLOB = re.compile(r"dotglob|globdots", re.I)


def _decode_escape(m: re.Match[str]) -> str:
    code = m.group(1)
    if code[0] in "xu" and len(code) > 1:
        return chr(int(code[1:], 16))
    if code[0] in "01234567":
        return chr(int(code, 8) & 0xFF)
    return _SIMPLE_ESC.get(code, code)


def _normalise(command: str) -> str:
    """Spelling tricks undone: $'...' escapes decoded, then backslashes, quotes, braces and commas removed,
    so `.h\\arpocrates`, `.ha""rpocrates`, `{.,}harpocrates` and `$'\\056harpocrates'` read as `.harpocrates`."""
    decoded = _ANSI_C.sub(lambda m: _ESCAPE.sub(_decode_escape, m.group(1)), command)
    return re.sub(r"[\\'\"{},]", "", decoded)


def _mentions_dir(command: str, norm: str) -> bool:
    return bool(_DIR.search(command) or _DIR.search(norm) or _MENTION_NORM.search(norm)
                or (_DOTGLOB.search(norm) and re.search(r"[*?\[]", norm)))


def _inside_state_dir(cwd: Optional[Path]) -> bool:
    """Whether the working directory is ~/.harpocrates or under it (links resolved): a command run there names
    its files without the directory, so the text alone shows nothing."""
    if cwd is None:
        return False
    state = (Path.home() / ".harpocrates").resolve()  # looked up per call so a test can move HOME
    real = cwd.resolve()
    return real == state or real.is_relative_to(state)


def touches(command: str, cwd: Optional[Path] = None) -> bool:
    """Whether a shell command could change the allowlist or the session grants: it runs `harpocrates allow`, or
    mentions the ~/.harpocrates directory (also in a quoted, escaped, braced or globbed spelling), or runs with
    that directory as its working directory (cwd), in anything but a provably read-only command."""
    norm = _normalise(command)
    if _ALLOW_CMD.search(command) or _ALLOW_CMD.search(norm):
        return True
    return (_mentions_dir(command, norm) or _inside_state_dir(cwd)) and not _read_only(command)
