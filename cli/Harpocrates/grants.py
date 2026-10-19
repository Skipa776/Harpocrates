"""Per-session edit grants (FR-READ-06): the one path the user said Yes to for a built-in edit of a secret-bearing file.

~/.harpo/grants/<session_id>.json (the state directory, see paths.py):
    {"pending": {"<tool_use_id>": {"path": "<realpath>", "tool": "Edit", "ts": 1760000000.0}},
     "granted": {"<realpath>": 1760000100.0}}
The PreToolUse hook records a `pending` entry when it asks. `python -m Harpocrates.hooks --post` moves it to `granted`
once the tool has run: PostToolUse and PostToolUseFailure fire only for a call that was allowed, so a refused or
declined prompt never reaches them. Subagents carry the parent's session_id, so they share the grant. A grant names
one real path, never a directory, and is ignored after TTL (a stale file is pruned on the next write).

Storage: the directory is 0o700 and each file 0o600, written to a temp file and renamed so a reader never sees half a
file. A file that is missing, unreadable or corrupt means no grants (fail closed). Parallel hook processes may lose
an update to each other (no lock); the cost is a repeated prompt, never a grant nobody gave.
RESIDUAL: ~/.harpo and ~/.harpocrates are protected from the agent by the Edit deny rule and the Bash `touches` guard, but a script
that writes the file itself could forge a grant. It is the same documented script gap as the allowlist (hooks.py
docstring); the egress gate (M2) covers what reaches the provider either way.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from Harpocrates.paths import state_dir

TTL = 24 * 3600  # seconds a pending or granted entry stays valid
_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # becomes a file name: no separator, no leading dot


def _dir() -> Path:
    """
    Description: Locate the session grant directory in the current state directory.
    Parameters: None
    Output: Path to the grants directory; state_dir may migrate legacy state and raise filesystem
        errors.
    """
    return state_dir() / "grants"  # looked up per call so a test can move HOME


def _file(session_id: object) -> Optional[Path]:
    """
    Description: Choose a grant file only when the session identifier is a safe file name.
    Parameters:
        session_id: Untrusted session identifier to validate.
    Output: Grant file path, or None for an invalid identifier; state directory lookup may migrate
        legacy state.
    Notes: The grant file of a session, None if the id could not be a safe file name.
    """
    if isinstance(session_id, str) and _SESSION.fullmatch(session_id):
        return _dir() / f"{session_id}.json"
    return None


def _fresh(ts: object, now: float) -> bool:
    """
    Description: Check that a timestamp is numeric and within the grant lifetime, rejecting booleans
        and future times.
    Parameters:
        ts: Timestamp to validate.
        now: Current time in seconds.
    Output: True for a timestamp whose age is between zero and TTL inclusive; otherwise False.
    """
    return isinstance(ts, (int, float)) and not isinstance(ts, bool) and 0 <= now - ts <= TTL


def _load(path: Path, now: float) -> dict:
    """
    Description: Load grant state and discard malformed or expired entries so unusable state
        provides no grants.
    Parameters:
        path: Session grant file to read.
        now: Current time in seconds used to check expiration.
    Output: Dictionary containing pending and granted mappings; missing, unreadable, or corrupt
        files yield empty mappings (fail closed).
    Notes: {"pending": ..., "granted": ...} with every malformed or expired entry dropped.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, unreadable or corrupt: no grants
        raw = None
    raw = raw if isinstance(raw, dict) else {}
    pending, granted = raw.get("pending"), raw.get("granted")
    return {
        "pending": {k: v for k, v in pending.items()
                    if isinstance(v, dict) and isinstance(v.get("path"), str) and _fresh(v.get("ts"), now)}
        if isinstance(pending, dict) else {},
        "granted": {k: v for k, v in granted.items() if _fresh(v, now)} if isinstance(granted, dict) else {},
    }


def _save(path: Path, data: dict) -> None:
    """
    Description: Save grant state atomically with a private directory and file to prevent partial
        reads.
    Parameters:
        path: Destination grant file.
        data: Grant state to serialize as JSON.
    Output: None; creates or chmods the parent to 0o700, writes a new 0o600 temporary file, and
        replaces the destination. On failure removes the temporary file and re-raises; filesystem
        and serialization errors propagate.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)  # also when it already existed, or the umask narrowed the mode
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _sweep(now: float) -> None:
    """
    Description: Remove stale session files to clean up sessions without a SessionEnd event.
    Parameters:
        now: Current time in seconds for comparing file modification times.
    Output: None; deletes files older than TTL and ignores directory, stat, and unlink OSError
        failures.
    Notes: Delete grant files nothing has touched for TTL (a session that ended without a
        SessionEnd).
    """
    try:
        for f in _dir().iterdir():
            try:
                if now - f.stat().st_mtime > TTL:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def add_pending(session_id: object, tool_use_id: object, path: Path, tool: str) -> bool:
    """
    Description: Record the edit awaiting user approval so only a recorded pending request can
        become a grant.
    Parameters:
        session_id: Session identifier used to select the grant file.
        tool_use_id: Nonempty identifier of the tool call awaiting approval.
        path: Target path to record as a real path.
        tool: Name of the tool requesting approval.
    Output: True after saving and sweeping stale grants; False for unusable identifiers or an
        OSError while saving. The caller must refuse instead of asking if recording fails.
    Notes: Record the edit the hook is about to ask about. False when it can't be recorded (no
        usable ids, an unwritable directory): the caller then refuses instead of asking, so no
        prompt can lead to a grant that was never recorded as pending.
    """
    file = _file(session_id)
    if file is None or not isinstance(tool_use_id, str) or not tool_use_id:
        return False
    now = time.time()
    try:
        data = _load(file, now)
        data["pending"][tool_use_id] = {"path": os.path.realpath(path), "tool": tool, "ts": now}
        _save(file, data)
    except OSError:
        return False
    _sweep(now)
    return True


def promote(session_id: object, tool_use_id: object, tool: object) -> None:
    """
    Description: Promote a matching pending edit after its tool runs to record user approval.
    Parameters:
        session_id: Session identifier selecting the grant file.
        tool_use_id: Identifier of the completed tool call.
        tool: Tool name reported by the Post event.
    Output: None; moves a fresh matching pending entry into granted and saves it. Invalid
        identifiers, unknown or expired calls, and tool mismatches do nothing; save errors
        propagate.
    Notes: The tool ran: its pending path becomes a grant. Nothing changes for an unknown or expired
        tool_use_id, or when `tool` (the tool the Post event names) is not the one the hook asked
        about.
    """
    file = _file(session_id)
    if file is None or not isinstance(tool_use_id, str):
        return
    now = time.time()
    data = _load(file, now)
    entry = data["pending"].get(tool_use_id)
    if entry is not None and entry.get("tool") == tool:
        del data["pending"][tool_use_id]
        data["granted"][entry["path"]] = now
        _save(file, data)


def granted(session_id: object) -> frozenset[str]:
    """
    Description: Read the real paths approved by the user for this session.
    Parameters:
        session_id: Session identifier selecting the grant file.
    Output: Frozen set of unexpired granted paths; invalid identifiers or missing, unreadable, or
        corrupt state yield an empty set.
    Notes: Real paths the user approved in this session.
    """
    file = _file(session_id)
    return frozenset(_load(file, time.time())["granted"]) if file is not None else frozenset()


def clear(session_id: object) -> None:
    """
    Description: Forget session grants when SessionEnd arrives.
    Parameters:
        session_id: Session identifier selecting the grant file.
    Output: None; deletes the grant file if present, ignores invalid identifiers, and propagates
        unlink errors other than a missing file.
    Notes: SessionEnd: forget the session's grants.
    """
    file = _file(session_id)
    if file is not None:
        file.unlink(missing_ok=True)
