"""Per-session edit grants (FR-READ-06): the one path the user said Yes to for a built-in edit of a secret-bearing file.

~/.harpocrates/grants/<session_id>.json:
    {"pending": {"<tool_use_id>": {"path": "<realpath>", "tool": "Edit", "ts": 1760000000.0}},
     "granted": {"<realpath>": 1760000100.0}}
The PreToolUse hook records a `pending` entry when it asks. `python -m Harpocrates.hooks --post` moves it to `granted`
once the tool has run: PostToolUse and PostToolUseFailure fire only for a call that was allowed, so a refused or
declined prompt never reaches them. Subagents carry the parent's session_id, so they share the grant. A grant names
one real path, never a directory, and is ignored after TTL (a stale file is pruned on the next write).

Storage: the directory is 0o700 and each file 0o600, written to a temp file and renamed so a reader never sees half a
file. A file that is missing, unreadable or corrupt means no grants (fail closed). Parallel hook processes may lose
an update to each other (no lock); the cost is a repeated prompt, never a grant nobody gave.
RESIDUAL: ~/.harpocrates is protected from the agent by the Edit deny rule and the Bash `touches` guard, but a script
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

TTL = 24 * 3600  # seconds a pending or granted entry stays valid
_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # becomes a file name: no separator, no leading dot


def _dir() -> Path:
    return Path.home() / ".harpocrates" / "grants"  # looked up per call so a test can move HOME


def _file(session_id: object) -> Optional[Path]:
    """The grant file of a session, None if the id could not be a safe file name."""
    if isinstance(session_id, str) and _SESSION.fullmatch(session_id):
        return _dir() / f"{session_id}.json"
    return None


def _fresh(ts: object, now: float) -> bool:
    return isinstance(ts, (int, float)) and not isinstance(ts, bool) and 0 <= now - ts <= TTL


def _load(path: Path, now: float) -> dict:
    """{"pending": ..., "granted": ...} with every malformed or expired entry dropped."""
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
    """Delete grant files nothing has touched for TTL (a session that ended without a SessionEnd)."""
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
    """Record the edit the hook is about to ask about. False when it can't be recorded (no usable ids, an
    unwritable directory): the caller then refuses instead of asking, so no prompt can lead to a grant that was
    never recorded as pending."""
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
    """The tool ran: its pending path becomes a grant. Nothing changes for an unknown or expired tool_use_id, or
    when `tool` (the tool the Post event names) is not the one the hook asked about."""
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
    """Real paths the user approved in this session."""
    file = _file(session_id)
    return frozenset(_load(file, time.time())["granted"]) if file is not None else frozenset()


def clear(session_id: object) -> None:
    """SessionEnd: forget the session's grants."""
    file = _file(session_id)
    if file is not None:
        file.unlink(missing_ok=True)
