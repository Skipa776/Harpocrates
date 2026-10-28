"""The gate token (SEC-03): a random value the agent sends in a header so no other caller can use the gate."""
from __future__ import annotations

import hmac
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Sequence

TOKEN_HEADER = "x-harpo-gate-token"
_TOKEN_SHAPE = re.compile(r"[A-Za-z0-9_-]{32,256}")
_FILE_READ_LIMIT = 1024


def new_token() -> str:
    """
    Description: Make a fresh gate token.
    Parameters: None
    Output: A random URL-safe string of 43 characters (256 bits). Different on every call.
    """
    return secrets.token_urlsafe(32)


def _check_shape(token: str) -> str:
    """
    Description: Refuse a token that is too short, too long or holds a character that is unsafe in a header.
    Parameters:
        token: the candidate token.
    Output: The token unchanged. Raises ValueError without repeating the token.
    """
    if not _TOKEN_SHAPE.fullmatch(token):
        raise ValueError("the gate token must be 32 to 256 characters from A-Z a-z 0-9 _ -")
    return token


def write_token_file(path: Path, token: str) -> None:
    """
    Description: Store a token where only this user can read it, replacing any earlier one in one step.
    Parameters:
        path: the token file.
        token: a token that passes the shape check (see new_token).
    Output: None. The file is mode 0600 from the moment it exists; a temporary file in the same folder is renamed over
        the target, and removed if anything fails. Raises ValueError for a bad token, OSError for a write failure.
    """
    _check_shape(token)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(token + "\n")
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def read_token_file(path: Path) -> str:
    """
    Description: Read the gate token, refusing a file another user could have read or written.
    Parameters:
        path: the token file.
    Output: The token without surrounding whitespace. Raises ValueError, never repeating the contents, when the file is
        missing, is a symbolic link or not a regular file, is owned by another user, has any group or other permission
        bit (it must be 0600 or tighter), or holds a token that fails the shape check.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise ValueError("the gate token file cannot be opened (missing, or a symbolic link)") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("the gate token file is not a regular file")
        if info.st_uid != os.geteuid():
            raise ValueError("the gate token file is not owned by the user running the gate")
        if info.st_mode & 0o077:
            raise ValueError("the gate token file must be mode 0600: others can reach it")
        raw = os.read(fd, _FILE_READ_LIMIT + 1)
    finally:
        os.close(fd)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("the gate token file holds non-ASCII bytes") from None
    return _check_shape(text.strip())


def check_token(supplied: Sequence[str], expected: str) -> bool:
    """
    Description: Decide whether a request carried the right token, comparing in constant time.
    Parameters:
        supplied: every value the request had for the token header, in order.
        expected: the token the gate runs with.
    Output: True only when there is exactly one value and it equals the token. A missing, repeated, empty or wrong
        value gives False; no input raises.
    """
    if len(supplied) != 1:
        return False
    return hmac.compare_digest(supplied[0].encode("utf-8", "surrogatepass"), expected.encode("utf-8"))
