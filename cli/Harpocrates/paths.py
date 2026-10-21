"""Where Harpo keeps the user's data (FR-NAME-04): ~/.harpo, copied once from ~/.harpocrates.

state_dir() is the one place that knows. The allowlist, the session grants and the install record all go through it,
so a process never mixes the two folders.

Migration (the first state_dir() call in a process finds ~/.harpo missing and ~/.harpocrates present):
  * It copies, never moves. ~/.harpocrates is only read: not changed, not deleted, no file added to it.
  * It copies only when ~/.harpo does not exist. A ~/.harpo that exists is the state, even if ~/.harpocrates holds
    more; the agent-cannot-edit guard (allowlist.touches for Bash, the hook's refusal of Edit/Write targets, the
    Edit deny rules) covers both folders for that reason.
  * It fails closed. Anything doubtful keeps ~/.harpocrates in use (never an empty allowlist that lost its grants) and
    prints one line to stderr, with no file contents: ~/.harpocrates or anything inside it is a symlink or not a
    plain file or folder, or a copy, fsync or rename step failed. After a failure the process does not retry or
    repeat the line. This fallback holds only while ~/.harpo does not exist at all.
  * Once ~/.harpo exists in any form, ~/.harpocrates is never used again (SEC-09). A ~/.harpo that is a symlink or a
    file is unusable: state_dir() still returns it, every reader treats it as empty (no allowlist entry, no grant, no
    install record) and every writer refuses with StateDirUnusableError rather than write through it, so removing an
    allowlist entry cannot be undone by swapping ~/.harpo for a link and letting an older list come back. One line
    on stderr says so, once per process. The same holds for a ~/.harpocrates that is a link, when it is the one in use.
  * The walk is made with directory descriptors: ~/.harpocrates is opened with O_NOFOLLOW, every entry is opened relative
    to its folder's descriptor with O_NOFOLLOW, and what was actually opened is judged by fstat and copied from its
    descriptor. A folder or file swapped for a link after it was looked at is refused, never followed (SEC-09).
  * The copy is made in a fresh temp folder inside the home folder (same file system), fsynced, then os.rename-d
    to ~/.harpo, which is atomic: other processes see no ~/.harpo or the complete one. Two processes starting at once
    each copy to their own temp folder; the rename of the first wins and the other, finding ~/.harpo there, discards
    its copy and carries on. The copy always holds MIGRATION_NOTE, so the winner's folder is never empty, and rename
    only replaces an empty folder; the loser can never replace the winner's.

No environment variable moves the state directory (SEC-08). The home folder is the one the password database names for
the user running the process (home()), so neither $HOME nor HARPO_DIR nor HARPOCRATES_DIR is read for it: a project's
settings can set environment variables, and a state directory the agent can write would replace the allowlist. The
hook's guards also protect the $HOME-derived folders, in case a tool expands `~` from the environment.
"""
from __future__ import annotations

import errno
import os
import shutil
import stat
import sys
import tempfile
import threading
from pathlib import Path

try:
    import pwd
except ImportError:  # pragma: no cover - no password database on Windows
    pwd = None

NEW_DIR = ".harpo"
OLD_DIR = ".harpocrates"
STATE_DIR_NAMES = (NEW_DIR, OLD_DIR)  # both stay guarded from the agent
MIGRATION_NOTE = "migrated-from-harpocrates.txt"
_NOTE_TEXT = "Copied from ~/.harpocrates when Harpo moved here. That folder was left in place and is no longer read.\n"

_lock = threading.Lock()
_failed: set[str] = set()  # home folders whose migration failed in this process: not retried, announced once
_unusable_announced: set[str] = set()  # home folders whose unusable ~/.harpo was announced in this process


class _RefusedError(Exception):
    """
    Description: The migration stopped on purpose because the source holds a symlink or something that is not a plain file or folder.
    Parameters:
        None: the message says what was refused.
    Output: Raised by _copy_tree and _assert_plain_tree; state_dir() turns it into one stderr line and keeps ~/.harpocrates.
    """


def home() -> Path:
    """
    Description: The user's home folder from the password database, never from $HOME (SEC-08): the environment of a hook
        can be set by a project's settings, and a home folder the agent chose would move the state directory.
    Parameters:
        None: looked up on every call, so a test can fake the password database.
    Output: The home folder's path. Raises RuntimeError when the password database has no entry for the user or names
        an empty or relative folder; there is no fall back to $HOME. Without a password database (Windows) it is
        Path.home().
    """
    if pwd is None:  # pragma: no cover - Windows
        return Path.home()
    try:
        folder = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError as exc:
        raise RuntimeError("no password database entry for this user, so no home folder") from exc
    if not isinstance(folder, str) or not os.path.isabs(folder):
        raise RuntimeError("the password database names no absolute home folder")
    return Path(folder)


class StateDirUnusableError(OSError, ValueError):
    """
    Description: A writer was asked to write into a state directory that is a symlink or a file (SEC-09).
    Parameters:
        None: the message says which folder and what to do.
    Output: Raised by require_usable and the writers that call it. It is both an OSError and a ValueError, so the CLI's
        existing handlers report it as a refusal with this message.
    """


def _plain_dir(path: Path) -> bool:
    """
    Description: Whether path is a real directory and not a symlink to one (nothing is followed).
    Parameters:
        path: the path to test.
    Output: True for a plain directory; False for anything else, including a missing or unreadable path. No side effects.
    """
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def unusable(state: Path) -> bool:
    """
    Description: Whether a state directory exists in a form Harpo will not use: a symlink (even a dangling one), a file or
        anything else that is not a plain folder (SEC-09).
    Parameters:
        state: the folder state_dir() returned.
    Output: True when something is there that is not a plain directory; False for a plain directory and for nothing at
        all (a state directory that does not exist yet is normal: writers create it). No side effects, nothing followed.
    """
    return os.path.lexists(state) and not _plain_dir(state)


def require_usable(state: Path) -> None:
    """
    Description: Refuse to write into a state directory that is a symlink or a file.
    Parameters:
        state: the folder state_dir() returned.
    Output: None for a plain folder or a missing one. Raises StateDirUnusableError, with a message that names the folder
        and says what to do, when unusable(state).
    """
    if unusable(state):
        raise StateDirUnusableError(
            f"~/{state.name} is not a plain folder (it is a symlink or a file), so Harpo reads its allowlist as empty "
            "and writes nothing there. Move or remove it, then try again.")


def _fsync(path: Path) -> None:
    """
    Description: Flush one file or directory to disk.
    Parameters:
        path: the file or directory to flush.
    Output: None. Raises OSError if it cannot be opened or flushed.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# Whether this platform can open a folder's entries relative to its descriptor and refuse links (decided at import).
_SAFE_WALK = (hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd
              and os.stat in os.supports_dir_fd and os.scandir in os.supports_fd)
_LINK_ERRNOS = (errno.ELOOP, errno.ENOTDIR, getattr(errno, "EMLINK", errno.ELOOP))  # what O_NOFOLLOW says about a link


def _copy_file_fd(src_fd: int, target: Path, st: os.stat_result) -> None:
    """
    Description: Copy the bytes of an already opened regular file into a new file, with its mode and times.
    Parameters:
        src_fd: an open read-only descriptor of the source file (left open for the caller to close).
        target: the new file's path, which must not exist (inside the migration's own temp folder).
        st: the fstat of src_fd, for the mode and times.
    Output: None. Raises OSError on a read, write or metadata failure. The source is only read, through src_fd.
    """
    out = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(os.dup(src_fd), "rb") as fsrc, os.fdopen(out, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
        fdst.flush()
        os.fchmod(fdst.fileno(), stat.S_IMODE(st.st_mode))
        os.utime(fdst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))


def _copy_dir_fd(dir_fd: int, dst: Path) -> None:
    """
    Description: Copy the plain files and folders of an open source folder into the existing folder dst, one at a time.
        Every entry is opened relative to the folder's descriptor with O_NOFOLLOW and judged by fstat of what was
        actually opened, so a folder or file swapped for a link after it was looked at is never followed (SEC-09).
    Parameters:
        dir_fd: an open descriptor of the source folder (left open for the caller to close).
        dst: an existing folder to fill.
    Output: None. Raises _RefusedError on a symlink or a special file in the source, OSError on a copy failure. Writes
        only under dst.
    """
    with os.scandir(dir_fd) as scan:
        names = sorted(entry.name for entry in scan)
    for name in names:
        target = dst / name
        looked = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if stat.S_ISLNK(looked.st_mode):
            raise _RefusedError(f"{name} is a symlink")
        if not (stat.S_ISDIR(looked.st_mode) or stat.S_ISREG(looked.st_mode)):
            raise _RefusedError(f"{name} is not a plain file or folder")
        try:  # O_NONBLOCK: a file swapped for a pipe must not block the open
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno in _LINK_ERRNOS:
                raise _RefusedError(f"{name} is a symlink") from exc
            raise
        try:
            opened = os.fstat(fd)
            if stat.S_ISDIR(opened.st_mode):
                target.mkdir(mode=0o700)
                _copy_dir_fd(fd, target)
                os.chmod(target, stat.S_IMODE(opened.st_mode))  # after the children: a read-only folder is filled first
                os.utime(target, ns=(opened.st_atime_ns, opened.st_mtime_ns))
            elif stat.S_ISREG(opened.st_mode):
                _copy_file_fd(fd, target, opened)
            else:
                raise _RefusedError(f"{name} is not a plain file or folder")
        finally:
            os.close(fd)


def _copy_tree(src: Path, dst: Path) -> None:
    """
    Description: Copy the plain files and folders under src into the existing folder dst, with modes and times, walking
        with directory descriptors and O_NOFOLLOW so that nothing swapped for a link while the copy runs is followed.
    Parameters:
        src: the folder to read (only read).
        dst: an existing folder to fill.
    Output: None. Raises _RefusedError on a symlink or a special file in src, or when the platform cannot walk this way
        (fail closed), OSError on a copy failure. Writes only under dst.
    """
    if not _SAFE_WALK:  # pragma: no cover - not POSIX
        raise _RefusedError("this platform cannot walk a folder without following links")
    try:
        fd = os.open(src, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in _LINK_ERRNOS:
            raise _RefusedError(f"{src.name} is a symlink or not a folder") from exc
        raise
    try:
        _copy_dir_fd(fd, dst)
    finally:
        os.close(fd)


def _assert_plain_tree(root: Path) -> None:
    """
    Description: Check that nothing under root is a symlink, so a link swapped in while copying is caught in the copy.
    Parameters:
        root: the folder to check.
    Output: None. Raises _RefusedError if any entry under root is a symlink.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            if os.path.islink(os.path.join(dirpath, name)):
                raise _RefusedError(f"{name} is a symlink")


def _fsync_tree(root: Path) -> None:
    """
    Description: Flush every file and folder under root, and root itself, to disk.
    Parameters:
        root: the folder to flush.
    Output: None. Raises OSError if any flush fails.
    """
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            _fsync(Path(dirpath) / name)
        _fsync(Path(dirpath))


def _migrate(old: Path, new: Path) -> None:
    """
    Description: Copy old to new through a temp folder next to new, then rename it into place atomically.
    Parameters:
        old: the existing ~/.harpocrates (a plain folder; only read).
        new: ~/.harpo, which must not exist yet.
    Output: None once new holds a complete copy, made here or by a concurrent caller that renamed first. Raises
        _RefusedError or OSError when the copy cannot be made; nothing is left behind and old is never changed.
    """
    if not _plain_dir(old):
        raise _RefusedError(f"{old.name} is a symlink or not a folder")
    tmp = Path(tempfile.mkdtemp(prefix=f"{new.name}.migrating.", dir=new.parent))  # 0o700, same file system as new
    try:
        _copy_tree(old, tmp)
        (tmp / MIGRATION_NOTE).write_text(_NOTE_TEXT, encoding="utf-8")
        os.chmod(tmp / MIGRATION_NOTE, 0o600)
        _assert_plain_tree(tmp)
        _fsync_tree(tmp)
        os.rename(tmp, new)  # atomic; fails if new now exists (never replaces it: the note keeps it non-empty)
    except BaseException as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        if isinstance(exc, OSError) and _plain_dir(new):
            return  # a concurrent migration got there first, and its copy is complete
        raise
    try:
        _fsync(new.parent)  # best effort: the rename is done and new is complete whatever this says
    except OSError:
        pass


def _announce(home: Path, reason: str) -> None:
    """
    Description: Remember that the migration did not happen in this process and print one line to stderr.
    Parameters:
        home: the home folder, as the key that stops a retry and a second line.
        reason: why, without file contents.
    Output: None. Adds home to the failed set and writes one line to stderr (a closed stderr is ignored).
    """
    _failed.add(str(home))
    try:
        print(f"harpo: could not move to ~/{NEW_DIR} ({reason}); keeping ~/{OLD_DIR}", file=sys.stderr)
    except OSError:
        pass


def _announce_unusable(home_folder: Path) -> None:
    """
    Description: Print one line to stderr, once per process, saying ~/.harpo is a symlink or a file and so counts as empty.
    Parameters:
        home_folder: the home folder, as the key that stops a second line.
    Output: None. Adds the folder to the announced set and writes one line to stderr (a closed stderr is ignored).
    """
    if str(home_folder) in _unusable_announced:
        return
    _unusable_announced.add(str(home_folder))
    try:
        print(f"harpo: ~/{NEW_DIR} is not a plain folder (a symlink or a file); its allowlist counts as empty "
              "and nothing is written there", file=sys.stderr)
    except OSError:
        pass


def peek_state_dir() -> Path:
    """
    Description: What state_dir() would return right now, without doing anything it does: no copy of the old folder,
        no line on stderr, nothing created (SEC-09 F8c; a dry run must have no side effects).
    Parameters:
        None: the home folder is looked up on every call (home(): the password database, not $HOME).
    Output: ~/.harpo when it exists in any form or when ~/.harpocrates is not there; ~/.harpocrates when only that exists
        (state_dir() would copy it to ~/.harpo first, so a dry run reads it where it is). Raises RuntimeError only when
        no home folder can be determined.
    """
    base = home()
    new, old = base / NEW_DIR, base / OLD_DIR
    return new if os.path.lexists(new) or not os.path.lexists(old) else old


def state_dir() -> Path:
    """
    Description: The folder that holds the user's allowlist, session grants and install record: ~/.harpo, after copying
        it once from ~/.harpocrates when only that exists. Fails closed to ~/.harpocrates (see the module docstring) only
        while ~/.harpo does not exist at all; a ~/.harpo that is a symlink or a file is returned too, as unusable().
    Parameters:
        None: the home folder is looked up on every call (home(): the password database, not $HOME).
    Output: The folder's path (it may not exist yet; writers create it). Side effects: the first call in a process may
        copy ~/.harpocrates to ~/.harpo; a refusal or failure prints one line to stderr. Raises nothing of its own;
        only home() can raise (RuntimeError), when no home folder can be determined.
    """
    base = home()
    new, old = base / NEW_DIR, base / OLD_DIR
    if os.path.lexists(new):  # in any form: ~/.harpocrates is never used again (SEC-09)
        if not _plain_dir(new):  # a symlink or a file: unusable, read as empty and never written through
            _announce_unusable(base)
        return new
    if str(base) in _failed:
        return old
    if not os.path.lexists(old):
        return new  # nothing to migrate
    with _lock:
        if os.path.lexists(new):  # another thread of this process (or process) got there while this one waited
            return state_dir()
        try:
            _migrate(old, new)
        except _RefusedError as exc:
            _announce(base, f"refused: {exc}")
            return old
        except Exception as exc:  # noqa: BLE001 - any failure keeps the old folder in use
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else type(exc).__name__
            _announce(base, f"copy failed: {reason}")
            return old
    return new
