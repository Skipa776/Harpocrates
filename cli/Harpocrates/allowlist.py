"""User-approved paths the read hook lets through even when they hold secrets (FR-READ-05).

One entry per line in ~/.harpo/allow (the state directory, see paths.py): an absolute path (a directory covers everything under it), or
`name:NAME`, which covers any path whose resolved real path has a component equal to NAME (exact, case-sensitive),
so `name:graphify-out` covers that folder in every project. `#` starts a comment. Everything is compared after
resolving symlinks: a link inside an allowed folder that points elsewhere is still checked, and a link NAMED like a
rule but pointing elsewhere does not match (its target's components count). Every pass is appended to allow.log
(time, tool, path; never content), as design.md's "explicit overrides, always logged" asks.

Only the user edits the list: `harpocrates allow` (add, --name, --remove, --list), or the file by hand. The hook
refuses Bash commands that change it or call `harpocrates allow` with any flags (read-only commands pass), refuses
Edit/Write-tool calls whose target is inside ~/.harpo or ~/.harpocrates (inside_state_dir, any mode, before any
grant or prompt), and the Claude Code settings deny Edit/Write on both as well.
Trade-off of name rules: they are broader than a path. add_name refuses names of secret stores, of system folders
and of anything in the home path, and asks for confirmation on the likes of `src` or `secrets`. Names are compared
NFKC-folded and case-folded (the default macOS file system treats `.AWS` as `.aws`). A name rule never covers a path
under any dot-directory of the home folder (~/.ssh, ~/.config, ~/.codex ...) or ~/Library, even if a component
matches: only a path entry can open those. "Under" is decided by file identity (device and inode of each ancestor),
not by spelling. Hand-edited `name:` lines that add_name would refuse match nothing.
"""
from __future__ import annotations

import fnmatch
import functools
import os
import re
import shlex
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Optional

from Harpocrates.harness import SENSITIVE_FILES, SENSITIVE_HOME_DIRS
from Harpocrates.paths import STATE_DIR_NAMES, state_dir

# None: the file sits in the state directory (paths.state_dir(), looked up per call so a test can move HOME). A test
# can set a Path here to point the module elsewhere.
ALLOW_FILE: Optional[Path] = None
LOG_FILE: Optional[Path] = None


def allow_file() -> Path:
    """
    Description: The allowlist file: ALLOW_FILE if a test set it, else allow in the state directory.
    Parameters:
        None: nothing is passed.
    Output: The path (it may not exist). May run the one-time ~/.harpocrates to ~/.harpo copy (paths.state_dir).
    """
    return ALLOW_FILE if ALLOW_FILE is not None else state_dir() / "allow"


def log_file() -> Path:
    """
    Description: The allow log: LOG_FILE if a test set it, else allow.log in the state directory.
    Parameters:
        None: nothing is passed.
    Output: The path (it may not exist). May run the one-time ~/.harpocrates to ~/.harpo copy (paths.state_dir).
    """
    return LOG_FILE if LOG_FILE is not None else state_dir() / "allow.log"


# ponytail: text match, best effort. A script that builds the path at runtime gets past it, but such a
# script could just as well open the secret file itself, a gap the hook already has (hooks.py docstring).
# The egress gate (M2) sees whatever reaches the provider either way.
# Rule: `harpocrates ... allow` anywhere in a segment is refused (only a mid-path `harpocrates/` is
# exempt), so a repo directory of that name over-blocks `git -C .../harpocrates log --grep allow`. A
# mention of the directory (`.harp`, also `~/.harp*`) passes only when it is a plain read: no output
# redirect, substitution or heredoc, and every command word in _READ_ONLY. Fail closed on anything else.
# Both state folders are covered (FR-NAME-04): `.harp` is the start of `.harpo` and of `.harpocrates`, so the first
# alternative of _DIR sees either, and the normalised patterns below take `arpo` with or without `crates`.
# A dot-name containing a glob or brace character (`~/.*`, `~/.h*`, `.[h]arp...`, `.{a,b}`, `.?arp...`) also
# counts as a mention, so it needs a plain read too. Known gap: a name built at runtime (variables, command
# substitution, printf) matches nothing; that is the script gap above. Matching runs on the raw command and on
# a normalised copy (see _normalise). Accepted over-block: `rm -rf .[a-z]*cache`.
_ALLOW_CMD = re.compile(
    r"\bharpocrates(?:\.cli)?\b(?!/)[^;&|\n]*\ballow\b"  # harpocrates / python -m Harpocrates.cli ... allow
    r"|(?:\$\([^)]*\)|`[^`]*`|\$\{?\w+\}?)\s*['\"]?allow\b"  # $(which harpocrates) allow
    r"", re.I)
_DIR = re.compile(r"""\.harp|(?:^|[/\s'"=~])\.[^/\s'";&|]*[*?\[{]""", re.I)
_UNSAFE_TEXT = re.compile(r">|\$\(|`|<\(|<<")
_READ_ONLY = frozenset(
    "cat ls head tail wc stat file du grep egrep fgrep diff cmp md5 md5sum shasum sha256sum nl cut echo printf "
    "test [ pwd realpath readlink basename dirname true cd".split())
_SEPARATORS = frozenset(";&|()")


_NAME_PREFIX = "name:"
_MAX_NAME = 255
_GLOB_CHARS = frozenset("*?[]{}")
# Names that cover whole trees in most projects, or sound like they hold secrets: allowed only after an explicit
# yes. Compared folded (_fold); the words are matched anywhere in the name.
_BROAD_NAMES = frozenset(
    "src lib app config configs settings .venv venv node_modules vendor build dist tests test scripts .git".split())
_SECRET_NAMES = frozenset(
    "secrets secret private keys key certs cert credentials credential token tokens vault auth passwords "
    "client_secret.json".split())
_SECRET_WORDS = ("secret", "credential", "password", "token")
# Top-level system and personal folders: far too broad for any yes (casefolded).
_SYSTEM_NAMES = frozenset(
    "private tmp var etc opt usr volumes system library applications home root mnt documents desktop downloads "
    "users".split())


class NeedsConfirmationError(ValueError):
    """A name broad enough to cover most of a project (or machine); add_name needs confirmed=True."""


def _lines() -> list[str]:
    path = allow_file()
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _write(lines: list[str]) -> None:
    """Atomic rewrite: temp file in the same folder (mkstemp gives 0o600), then os.replace. A reader never sees
    a half-written list, and a file hand-edited without a final newline can't glue the next entry onto its line."""
    target = allow_file()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".allow.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("".join(f"{line}\n" for line in lines))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def entries() -> list[Path]:
    """Path entries only; name rules are in names()."""
    stripped = (line.strip() for line in _lines())
    return [Path(line) for line in stripped if line and not line.startswith(("#", _NAME_PREFIX))]


def _fold(text: str) -> str:
    """NFKC + casefold: `.ſsh`, `．ssh` and `.SSH` all read as `.ssh`, as the file system may treat them."""
    return unicodedata.normalize("NFKC", text).casefold()


@functools.lru_cache(maxsize=8)
def _home_facts(home: str, stamp: int) -> tuple[frozenset[str], frozenset[str], frozenset[tuple[int, int]]]:
    """(folded components of the home path, folded names of the dot-directories directly in home, identities
    (device, inode) of the directories no name rule may cover: those dot-directories, ~/Library, and
    SENSITIVE_HOME_DIRS). Cached per home folder and its modification time, so a search over many files lists home
    once; a folder added to home changes the stamp."""
    base = Path(home)
    try:
        dots = [e.name for e in os.scandir(base) if e.name.startswith(".") and e.is_dir()]
    except OSError:
        dots = []
    protected = [base / d for d in dots] + [base / "Library"] + [Path(d.replace("~", home, 1)) for d in SENSITIVE_HOME_DIRS]
    ids = set()
    for d in protected:
        try:
            st = os.stat(d)
        except OSError:
            continue  # not there: nothing to protect
        ids.add((st.st_dev, st.st_ino))
    parts = frozenset(_fold(part) for part in base.resolve().parts[1:])
    return parts, frozenset(_fold(d) for d in dots), frozenset(ids)


def _facts() -> tuple[frozenset[str], frozenset[str], frozenset[tuple[int, int]]]:
    home = Path.home()  # looked up per call so a test can move HOME
    try:
        stamp = os.stat(home).st_mtime_ns
    except OSError:
        stamp = 0
    return _home_facts(str(home), stamp)


def _refusal(name: str) -> Optional[str]:
    """Why a name can never be a rule (None if it can). Secret-store and home checks compare _fold-ed text, since
    the default macOS file system folds case and Unicode forms: `.ENV` is the same file as `.env`."""
    if not name or name != name.strip():
        return "a name can't be empty or start or end with a space"
    if name in (".", ".."):
        return "a name can't be . or .."
    if "/" in name or "\\" in name:
        return "a name is one folder or file name, without / or \\"
    if not name.isprintable():
        return "a name can't hold control characters"
    if len(name) > _MAX_NAME:
        return f"a name can't be longer than {_MAX_NAME} characters"
    folded = _fold(name)
    if _GLOB_CHARS & (set(name) | set(folded)):
        return "a name is exact; globs (* ? [ ] { }) are not supported"
    secret_dirs = {_fold(Path(d).name) for d in SENSITIVE_HOME_DIRS} | {_fold(n) for n in STATE_DIR_NAMES}
    if folded in secret_dirs or any(fnmatch.fnmatchcase(folded, _fold(pat)) for pat in SENSITIVE_FILES):
        return f"{name} names a secret store, which no name rule may cover"
    parts, dots, _ = _facts()
    if folded in _SYSTEM_NAMES or folded in parts:
        return f"{name} is a system or home folder name: far too broad for a name rule"
    if folded in dots:
        return f"{name} is a folder in your home directory that tools keep private; allow a path instead"
    return None


def names() -> list[str]:
    """Name rules in effect: `name:` lines that add_name would accept (a hand-edited `name:/`, `name:.env` or the
    like matches nothing, so a typo can't open the whole disk)."""
    found = (line.strip()[len(_NAME_PREFIX):] for line in _lines() if line.strip().startswith(_NAME_PREFIX))
    return [name for name in found if _refusal(name) is None]


def listing() -> list[tuple[str, str]]:
    """(kind, value) for display: path entries first, then names."""
    return [("path", str(p)) for p in entries()] + [("name", n) for n in names()]


def printable(text: str) -> str:
    """Control characters (newlines, ANSI escapes) in a file name shown as '?', so a name can't fake
    lines in a prompt or the log."""
    return "".join(c if c.isprintable() else "?" for c in text)


def _in_protected(real: Path) -> bool:
    """Whether real is, or sits under, a folder no name rule may open (_home_facts), decided by identity: each
    ancestor is stat-ed and compared by (device, inode), so `~/.AWS` on a case-insensitive file system is the
    credential folder whatever the spelling."""
    ids = _facts()[2]
    for ancestor in (real, *real.parents):
        try:
            st = os.stat(ancestor)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) in ids:
            return True
    return False


def allowed(path: Path) -> bool:
    real = path.resolve()
    if real.is_file() and real.stat().st_nlink > 1:
        return False  # a hard link may be a secret file linked into an allowed folder: scan it
    if any(real == e or real.is_relative_to(e) for e in entries()):
        return True
    rules = set(names())  # components of the RESOLVED path: a link named like a rule but pointing away fails
    return bool(rules) and not rules.isdisjoint(real.parts) and not _in_protected(real)


def add(path: Path) -> Path:
    real = path.resolve(strict=True)  # a typo fails here instead of allowing nothing
    if real not in entries():
        _write([*_lines(), str(real)])
    return real


def add_name(name: str, confirmed: bool = False) -> str:
    """Allow a folder or file name in every project. ValueError if it can never be a rule; NeedsConfirmationError
    (a ValueError) if it is broad and not confirmed."""
    reason = _refusal(name)
    if reason:
        raise ValueError(reason)
    folded = _fold(name)
    broad = (folded in {_fold(b) for b in _BROAD_NAMES} or folded in _SECRET_NAMES
             or any(word in folded for word in _SECRET_WORDS))
    if broad and not confirmed:
        raise NeedsConfirmationError(
            f"{name} is a broad or secret-sounding name: it would cover every {name} folder on this machine")
    if name not in names():
        _write([*_lines(), f"{_NAME_PREFIX}{name}"])
    return name


def _drop(match) -> None:
    lines = _lines()
    kept = [line for line in lines if not match(line.strip())]
    if len(kept) == len(lines):
        raise ValueError("not in the allowlist")
    _write(kept)


def remove(path: Path) -> Path:
    """Drop a path entry (kept as typed or resolved: the entry may be for a folder that is gone). Comments and
    order stay."""
    real = path.resolve()
    forms = {real, Path(os.path.abspath(path))}
    _drop(lambda line: bool(line) and not line.startswith(("#", _NAME_PREFIX)) and Path(line) in forms)
    return real


def remove_name(name: str) -> str:
    """Drop a name rule, matched on the raw line so a hand-edited line add_name would refuse can still go."""
    _drop(lambda line: line == f"{_NAME_PREFIX}{name}")
    return name


def log_use(path: Path, tool: str) -> None:
    target = log_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
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
# `arpo` (or `h?arpo`; `arpocrates` starts with `arpo`), or dotglob/globdots next to any glob, all count as a mention
# of either state directory (`harpo` or `harpocrates`).
# `grants` must end the name: the module cli/Harpocrates/grants.py is no mention.
_MENTION_NORM = re.compile(r"arpo(?:crates)?/(?:allow\b|grants(?![\w.-]))|[*?\]]h?arpo", re.I)
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


def inside_state_dir(path: Optional[Path]) -> bool:
    """
    Description: Whether path is ~/.harpo or ~/.harpocrates, or lies under either: links resolved, names compared
        NFKC- and case-folded (the default macOS file system treats .HARPO as .harpo), and by file identity where
        the folder exists. A command run there names its files without the directory, so for a shell command the
        working directory is what shows it; for an Edit or Write tool call it is the target path.
    Parameters:
        path: an absolute path, or one relative to the current directory; it need not exist. None means no path.
    Output: True when inside either folder, False when not or when path is None. Both names are checked whichever
        one is in use, and the migration is not run. Raises OSError or RuntimeError when the path cannot be
        resolved; callers treat that as a refusal.
    """
    if path is None:
        return False
    real = path.resolve()
    home = Path.home()  # looked up per call so a test can move HOME
    folded = [_fold(part) for part in real.parts]
    ids = set()
    for name in STATE_DIR_NAMES:
        state = (home / name).resolve()
        wanted = [_fold(part) for part in state.parts]
        if folded[:len(wanted)] == wanted:
            return True
        try:
            st = os.stat(state)
        except OSError:
            continue  # not there (yet): the name comparison above is all there is to check
        ids.add((st.st_dev, st.st_ino))
    for ancestor in (real, *real.parents):
        try:
            st = os.stat(ancestor)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) in ids:
            return True
    return False


# Any command naming the allowlist module together with an add/remove word (add, add_name, remove_name ...) in any
# import form: `allowlist.add_name(..)`, `from Harpocrates.allowlist import add_name`, `import ...allowlist as al; al.add(..)`.
# `allowlist.py` as a file name is no mention, so `git add cli/Harpocrates/allowlist.py` still passes. Over-block: a
# command holding both words for another reason (`grep -n add allowlist`).
_MODULE = re.compile(r"(?<![A-Za-z0-9_])allowlist(?![A-Za-z0-9]|\.py\b)", re.I)
_EDIT_WORD = re.compile(r"(?<![A-Za-z0-9])(?:add|remove)(?![A-Za-z0-9])", re.I)


def _edits_module(text: str) -> bool:
    return bool(_MODULE.search(text) and _EDIT_WORD.search(text))


def touches(command: str, cwd: Optional[Path] = None) -> bool:
    """Whether a shell command could change the allowlist or the session grants: it runs `harpocrates allow`, or
    mentions the ~/.harpo or ~/.harpocrates directory (also in a quoted, escaped, braced or globbed spelling), or
    runs with either as its working directory (cwd), in anything but a provably read-only command."""
    norm = _normalise(command)
    if _ALLOW_CMD.search(command) or _ALLOW_CMD.search(norm) or _edits_module(command) or _edits_module(norm):
        return True
    return (_mentions_dir(command, norm) or inside_state_dir(cwd)) and not _read_only(command)
