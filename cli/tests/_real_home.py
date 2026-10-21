"""Read-only snapshot of the real home folder's Harpo state and agent config, for the test session's safety net.

SEC-08 made the state folder come from the password database, so a test that moves $HOME no longer moves it. The
session fixture in conftest.py takes snapshot() before the first test and after the last, and fails the run if anything
the tests could have written to changed: ~/.harpo, ~/.harpocrates (every file and folder under them), and the two agent
config files `harpo install` edits (~/.claude/settings.json, ~/.codex/config.toml).

Only metadata and a SHA-256 of each file are taken; nothing is printed from a file's content, and nothing is written.
Run as a script (`python cli/tests/_real_home.py`) it prints the listing, one line per entry, for a before/after check.
"""
from __future__ import annotations

import hashlib
import os
import pwd
import stat
import sys
from pathlib import Path
from typing import Optional

TREES = (".harpo", ".harpocrates")
FILES = (".claude/settings.json", ".codex/config.toml")


def real_home() -> Path:
    """
    Description: The home folder the password database names for this user, ignoring $HOME and any patch of paths.home.
    Parameters: None
    Output: The folder's path. Raises KeyError when the user has no password database entry.
    """
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _digest(path: Path) -> Optional[str]:
    """
    Description: SHA-256 of a regular file's content, read in chunks.
    Parameters:
        path: the file to hash.
    Output: The hex digest, or None when the file cannot be read (a changed permission still shows in the mode).
    """
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except OSError:
        return None


def _entry(path: Path) -> Optional[tuple]:
    """
    Description: One snapshot row for a path, taken without following a link.
    Parameters:
        path: the path to describe.
    Output: (kind, mode, size, mtime_ns, sha256) with kind d, f, l or o; sha256 only for a regular file, otherwise None;
        None when the path does not exist.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return None
    kind = "d" if stat.S_ISDIR(st.st_mode) else "f" if stat.S_ISREG(st.st_mode) else "l" if stat.S_ISLNK(st.st_mode) else "o"
    return (kind, stat.S_IMODE(st.st_mode), st.st_size, st.st_mtime_ns, _digest(path) if kind == "f" else None)


def snapshot(home: Optional[Path] = None) -> dict[str, Optional[tuple]]:
    """
    Description: Describe everything under the two state folders and the two agent config files of a home folder.
    Parameters:
        home: the home folder to describe, defaulting to the real one (real_home()).
    Output: A dict from path (relative to home) to the row of _entry, None for a path that is not there. The state
        folders are walked without following links; every file and folder under them gets a row. Read-only.
    """
    base = home if home is not None else real_home()
    rows: dict[str, Optional[tuple]] = {}
    for tree in TREES:
        root = base / tree
        rows[tree] = _entry(root)
        if rows[tree] is not None and rows[tree][0] == "d":
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                for name in sorted(dirnames) + sorted(filenames):
                    full = Path(dirpath) / name
                    rows[str(full.relative_to(base))] = _entry(full)
    for rel in FILES:
        rows[rel] = _entry(base / rel)
    return rows


def changed(before: dict, after: dict) -> list[str]:
    """
    Description: List the paths whose row differs between two snapshots.
    Parameters:
        before: the earlier snapshot.
        after: the later snapshot.
    Output: Sorted relative paths that were added, removed or changed (mode, size, mtime or content). Never a content.
    """
    return sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))


if __name__ == "__main__":  # pragma: no cover - the before/after listing
    listing = snapshot()
    for rel, row in sorted(listing.items()):
        print(rel, "-" if row is None else " ".join("-" if x is None else str(x) for x in row))
    sys.exit(0)
