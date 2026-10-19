"""Where Harpocrates keeps the user's data (FR-NAME-04): ~/.harpo, copied once from ~/.harpocrates.

state_dir() is the one place that knows. The allowlist, the session grants and the install record all go through it,
so a process never mixes the two folders.

Migration (the first state_dir() call in a process finds ~/.harpo missing and ~/.harpocrates present):
  * It copies, never moves. ~/.harpocrates is only read: not changed, not deleted, no file added to it.
  * It copies only when ~/.harpo does not exist. A ~/.harpo that exists is the state, even if ~/.harpocrates holds
    more; the agent-cannot-edit guard (allowlist.touches, the Edit deny rules) covers both folders for that reason.
  * It fails closed. Anything doubtful keeps ~/.harpocrates in use (never an empty allowlist that lost its grants) and
    prints one line to stderr, with no file contents: ~/.harpocrates or anything inside it is a symlink or not a
    plain file or folder, ~/.harpo exists but is a symlink or a file, or a copy, fsync or rename step failed. After a
    failure the process does not retry or repeat the line.
  * The copy is made in a fresh temp folder inside the home folder (same file system), fsynced, then os.rename-d
    to ~/.harpo, which is atomic: other processes see no ~/.harpo or the complete one. Two processes starting at once
    each copy to their own temp folder; the rename of the first wins and the other, finding ~/.harpo there, discards
    its copy and carries on. The copy always holds MIGRATION_NOTE, so the winner's folder is never empty, and rename
    only replaces an empty folder; the loser can never replace the winner's.

No environment variable moves the state directory. HARPO_DIR and HARPOCRATES_DIR are not read: a project's settings
can set environment variables, and a state directory the agent can write would replace the allowlist.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import threading
from pathlib import Path

NEW_DIR = ".harpo"
OLD_DIR = ".harpocrates"
STATE_DIR_NAMES = (NEW_DIR, OLD_DIR)  # both stay guarded from the agent
MIGRATION_NOTE = "migrated-from-harpocrates.txt"
_NOTE_TEXT = "Copied from ~/.harpocrates when Harpo moved here. That folder was left in place and is no longer read.\n"

_lock = threading.Lock()
_failed: set[str] = set()  # home folders whose migration failed in this process: not retried, announced once


class _RefusedError(Exception):
    """
    Description: The migration stopped on purpose because the source holds a symlink or something that is not a plain file or folder.
    Parameters:
        None: the message says what was refused.
    Output: Raised by _copy_tree and _assert_plain_tree; state_dir() turns it into one stderr line and keeps ~/.harpocrates.
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


def _copy_tree(src: Path, dst: Path) -> None:
    """
    Description: Copy the plain files and folders under src into the existing folder dst, with modes and times, one at a time.
    Parameters:
        src: the folder to read (only read).
        dst: an existing folder to fill.
    Output: None. Raises _RefusedError on a symlink or a special file in src, OSError on a copy failure. Writes only under dst.
    """
    with os.scandir(src) as scan:
        entries = sorted(scan, key=lambda e: e.name)
    for entry in entries:
        target = dst / entry.name
        if entry.is_symlink():
            raise _RefusedError(f"{entry.name} is a symlink")
        if entry.is_dir(follow_symlinks=False):
            target.mkdir(mode=0o700)
            _copy_tree(Path(entry.path), target)
            shutil.copystat(entry.path, target)
        elif entry.is_file(follow_symlinks=False):
            shutil.copy2(entry.path, target, follow_symlinks=False)
        else:
            raise _RefusedError(f"{entry.name} is not a plain file or folder")


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
        print(f"harpocrates: could not move to ~/{NEW_DIR} ({reason}); keeping ~/{OLD_DIR}", file=sys.stderr)
    except OSError:
        pass


def state_dir() -> Path:
    """
    Description: The folder that holds the user's allowlist, session grants and install record: ~/.harpo, after copying
        it once from ~/.harpocrates when only that exists. Fails closed to ~/.harpocrates (see the module docstring).
    Parameters:
        None: the home folder is looked up on every call, so a test can move HOME.
    Output: The folder's path (it may not exist yet; writers create it). Side effects: the first call in a process may
        copy ~/.harpocrates to ~/.harpo; a refusal or failure prints one line to stderr. Raises nothing of its own;
        only Path.home() can raise (RuntimeError), when no home folder can be determined.
    """
    home = Path.home()
    new, old = home / NEW_DIR, home / OLD_DIR
    if _plain_dir(new):
        return new
    if str(home) in _failed:
        return old
    if os.path.lexists(new):  # a symlink or a file: never trusted, never written through
        _announce(home, f"~/{NEW_DIR} is not a plain folder")
        return old
    if not os.path.lexists(old):
        return new  # nothing to migrate
    with _lock:
        if _plain_dir(new):  # another thread of this process finished while this one waited
            return new
        try:
            _migrate(old, new)
        except _RefusedError as exc:
            _announce(home, f"refused: {exc}")
            return old
        except Exception as exc:  # noqa: BLE001 - any failure keeps the old folder in use
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else type(exc).__name__
            _announce(home, f"copy failed: {reason}")
            return old
    return new
