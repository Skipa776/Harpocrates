"""G6: the findings log (FR-LOG-01) and the hash-chained proof log (FR-PROOF-01 to -03), both in ~/.harpo.

Neither ever holds a value. A finding names the secret type and its placeholder hash, an HMAC prefix under the session's
in-memory key (SEC-02). A proof entry holds the SHA-256 of the exact bytes the gate forwarded and placeholder counts by
type, never the bytes (Josh, Oct 21, 2026). Each file rotates at ROTATE_BYTES, keeping one previous file. The proof chain
continues across a rotation, and proof.head pins both of its ends, so an edited, inserted or deleted entry fails
verification, and the gate will not add to a chain that does not match proof.head (it blocks instead). Anyone who can
write ~/.harpo can rebuild a consistent chain: the hook keeps the agent out (SEC-09), and a malicious local user is out
of scope.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from Harpocrates import paths

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl; Harpo's hook and gate target macOS and Linux
    fcntl = None  # type: ignore[assignment]

FINDINGS = "findings.jsonl"
PROOF = "proof.jsonl"
HEAD = "proof.head"
ROTATE_BYTES = 10 * 1024 * 1024
GENESIS = "0" * 64
# A placeholder: the secret type, then four hex characters of the session-keyed hash (read.Redactor.placeholder).
PLACEHOLDER = re.compile(r"<<HARPO:(\w+):([0-9a-f]{4})>>")
_PLACEHOLDER_BYTES = re.compile(PLACEHOLDER.pattern.encode())
_ENTRY_KEYS = ("seq", "time", "provider", "route", "bytes", "sha256", "placeholders", "prev", "hash")
_TAIL_BYTES = 64 * 1024  # far more than one entry


def _now() -> str:
    """
    Description: The current time for a log line.
    Parameters: None
    Output: ISO 8601 in UTC, to the millisecond.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def placeholders(text: Union[bytes, str]) -> list[tuple[str, str]]:
    """
    Description: The placeholders in a text, in order.
    Parameters:
        text: the text.
    Output: (secret type, hash) for each one.
    """
    if isinstance(text, str):
        return PLACEHOLDER.findall(text)
    return [(kind.decode(), digest.decode()) for kind, digest in _PLACEHOLDER_BYTES.findall(text)]


def _usable(state: Path) -> Path:
    """
    Description: Refuse to write through a state folder that is a symlink or a file (SEC-09).
    Parameters:
        state: the state folder.
    Output: The folder. Raises OSError when it is unusable.
    """
    if paths.unusable(state):
        raise OSError(f"{state} is not a plain folder")
    return state


def rotated(path: Path) -> Path:
    """
    Description: The name a log takes when it is rotated.
    Parameters:
        path: the log, for example findings.jsonl.
    Output: findings.1.jsonl beside it.
    """
    return path.with_name(f"{path.stem}.1{path.suffix}")


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """
    Description: Hold an exclusive lock beside a log, so that two processes never rotate it or extend its chain at once.
    Parameters:
        path: the log.
    Output: A context manager; the lock file is <log>.lock (0600).
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path.with_name(path.name + ".lock"), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _append(path: Path, line: str) -> bool:
    """
    Description: Append one line to a log whose lock the caller holds, rotating it first if the line would pass
        ROTATE_BYTES.
    Parameters:
        path: the log.
        line: one JSON line, without the newline.
    Output: True when the log was rotated before this line. Raises OSError.
    """
    data = (line + "\n").encode("utf-8")
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    turned = bool(size) and size + len(data) > ROTATE_BYTES
    if turned:
        os.replace(path, rotated(path))
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, data)  # one write per line, so lines from concurrent writers never interleave
    finally:
        os.close(fd)
    return turned


def _last_line(path: Path) -> Optional[str]:
    """
    Description: Read a log's last line without reading the whole file.
    Parameters:
        path: the log.
    Output: The last non-empty line among its final _TAIL_BYTES, or None when it is missing or empty.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - _TAIL_BYTES))
            tail = handle.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return None
    lines = [line for line in tail.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _lines(path: Path) -> list[str]:
    """
    Description: Read a log's lines.
    Parameters:
        path: the log.
    Output: Its non-empty lines; [] when it does not exist.
    """
    try:
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return []


class FindingsLog:
    """
    Description: The findings log of one session of one layer (FR-LOG-01).
    Parameters:
        layer: gate, read or hook.
        session_id: the session the findings belong to.
        state: the state folder, or None for paths.state_dir() at each write.
    Output: An object whose methods append lines. A write failure is reported as False and never raised: logging
        must not change a decision.
    """

    def __init__(self, layer: str, session_id: str, state: Optional[Path] = None) -> None:
        """
        Description: Prepare the log; nothing is written yet.
        Parameters:
            layer: gate, read or hook.
            session_id: the session id.
            state: the state folder or None.
        Output: None.
        """
        self.layer, self.session_id, self.state = layer, session_id, state
        self._seen: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()

    def record(self, event_type: str, secret_type: Optional[str], hash_prefix: Optional[str], source: str,
               action: str, reason: Optional[str] = None) -> bool:
        """
        Description: Append one finding.
        Parameters:
            event_type: secret_redacted, request_blocked or read_refused.
            secret_type: the secret type, or None.
            hash_prefix: the placeholder's hash, or None.
            source: the gate route, or the file and tool.
            action: redact, block or refuse.
            reason: a block's reason code, why a file could not be checked, or None.
        Output: True when the line was written.
        """
        line = json.dumps({"time": _now(), "event_type": event_type, "layer": self.layer, "type": secret_type,
                           "hash": hash_prefix, "source": source, "action": action, "reason": reason,
                           "session_id": self.session_id})
        try:
            path = _usable(self.state if self.state is not None else paths.state_dir()) / FINDINGS
            with _locked(path):
                _append(path, line)
            return True
        except (OSError, RuntimeError):  # RuntimeError: no home folder
            return False

    def redactions(self, found: Iterable[tuple[str, str]], source: str) -> int:
        """
        Description: Log each placeholder this layer made that this session has not logged for this source yet.
        Parameters:
            found: (secret type, hash) pairs; see placeholders().
            source: the gate route, or the file.
        Output: How many new findings were logged.
        """
        with self._lock:
            new = [(kind, digest) for kind, digest in dict.fromkeys(found) if (kind, digest, source) not in self._seen]
            self._seen.update((kind, digest, source) for kind, digest in new)
        return sum(self.record("secret_redacted", kind, digest, source, "redact") for kind, digest in new)


def entry_hash(entry: dict) -> str:
    """
    Description: The hash that chains a proof entry: SHA-256 of its canonical JSON without the "hash" key.
    Parameters:
        entry: the entry.
    Output: Hex digest.
    """
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_head(path: Path) -> Optional[dict]:
    """
    Description: Read proof.head.
    Parameters:
        path: the head file.
    Output: {"seq", "hash", "first_prev", "base_prev"}, or None when it is missing or not of that shape.
    """
    try:
        head = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    keys = ("seq", "hash", "first_prev", "base_prev")
    return head if isinstance(head, dict) and all(k in head for k in keys) else None


def _write_head(path: Path, head: dict) -> None:
    """
    Description: Replace proof.head atomically (mode 0600).
    Parameters:
        path: the head file.
        head: its new contents.
    Output: None. Raises OSError.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(head, handle)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ProofChain:
    """
    Description: The gate's proof log: one hash-chained entry per forwarded request (FR-PROOF-01, FR-PROOF-02).
    Parameters:
        state: the state folder (~/.harpo).
    Output: An object whose append() the gate calls before it sends each request.
    """

    def __init__(self, state: Path) -> None:
        """
        Description: Prepare the chain; nothing is read or written yet.
        Parameters:
            state: the state folder.
        Output: None.
        """
        self.path, self.head_path = state / PROOF, state / HEAD
        self._lock = threading.Lock()

    def _head(self) -> dict:
        """
        Description: The chain's head, checked against the log's last entry before anything is added after it.
        Parameters: None
        Output: The head dict; a new chain's when there is neither a log nor proof.head. A last line that the head does
            not name yet, but that follows it (written just before a crash), moves the head on to it. Raises ValueError
            for anything else that does not fit: proof.head missing while entries exist, entries missing while
            proof.head exists, a damaged last line, or a last line that is not the head's. The gate then blocks with
            HARPO-BLOCK-AUDIT rather than go on from a chain it cannot vouch for.
        """
        head = _read_head(self.head_path)
        tail = _last_line(self.path) or _last_line(rotated(self.path))
        if tail is None:
            if head is not None:
                raise ValueError("the proof log is gone but proof.head is not")
            return {"seq": 0, "hash": GENESIS, "first_prev": GENESIS, "base_prev": GENESIS}
        if head is None:
            raise ValueError("proof.head is missing or damaged")
        try:
            last = json.loads(tail)
            if (last["seq"], last["hash"]) == (head["seq"], head["hash"]):
                return head
            if last["prev"] == head["hash"] and last["seq"] == head["seq"] + 1 and entry_hash(last) == last["hash"]:
                return {**head, "seq": last["seq"], "hash": last["hash"]}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError("the proof log's last line is damaged") from error
        raise ValueError("the proof log does not end where proof.head says")

    def append(self, provider: str, route: str, forwarded: bytes, plain: bytes) -> dict:
        """
        Description: Record one request the gate is about to forward.
        Parameters:
            provider: anthropic, openai or chatgpt.
            route: the gate route, for example anthropic.messages.
            forwarded: the exact bytes that will be sent (after re-encoding).
            plain: the same body uncompressed, where placeholders are counted.
        Output: The entry. Raises OSError (or ValueError for a damaged log) when it cannot be written: the gate then
            blocks the request with HARPO-BLOCK-AUDIT.
        """
        counts = Counter(kind for kind, _ in placeholders(plain))
        _usable(self.path.parent)
        with self._lock, _locked(self.path):
            head = self._head()
            entry = {"seq": head["seq"] + 1, "time": _now(), "provider": provider, "route": route,
                     "bytes": len(forwarded), "sha256": hashlib.sha256(forwarded).hexdigest(),
                     "placeholders": dict(sorted(counts.items())), "prev": head["hash"]}
            entry["hash"] = entry_hash(entry)
            if _append(self.path, json.dumps(entry)):
                head["base_prev"], head["first_prev"] = head["first_prev"], entry["prev"]
            _write_head(self.head_path, {**head, "seq": entry["seq"], "hash": entry["hash"]})
        return entry


@dataclass
class Verification:
    """
    Description: The result of checking the proof chain.
    Parameters:
        entries: every retained entry, oldest first.
        problem: what is wrong, or None when the chain verifies.
    Output: A record.
    """
    entries: list = field(default_factory=list)
    problem: Optional[str] = None


def verify(state: Path) -> Verification:
    """
    Description: Check the proof chain across the rotated and the current file (FR-PROOF-02, FR-PROOF-03).
    Parameters:
        state: the state folder.
    Output: A Verification. A problem names the first entry that does not fit: a damaged line, a hash that does not
        match the entry, a prev that is not the previous hash, a gap in seq, or ends that do not match proof.head.
    """
    head = _read_head(state / HEAD)
    found = Verification()
    prev, seq = None, None
    for path in (rotated(state / PROOF), state / PROOF):
        for number, line in enumerate(_lines(path), 1):
            where = f"{path.name} line {number}"
            try:
                entry = json.loads(line)
            except ValueError:
                return Verification(found.entries, f"{where} is not JSON")
            if not isinstance(entry, dict) or tuple(entry) != _ENTRY_KEYS:
                return Verification(found.entries, f"{where} is not a proof entry")
            if entry_hash(entry) != entry["hash"]:
                return Verification(found.entries, f"{where} was changed: its hash does not match")
            if prev is None:
                if head is not None and entry["prev"] != head["base_prev"]:
                    return Verification(found.entries, f"{where}: entries before it were removed")
            elif entry["prev"] != prev or entry["seq"] != seq + 1:
                return Verification(found.entries, f"{where} does not follow the entry before it")
            prev, seq = entry["hash"], entry["seq"]
            found.entries.append(entry)
    if not found.entries:
        return Verification([], None if head is None or head["seq"] == 0 else "the proof log is gone")
    if head is None:
        return Verification(found.entries, f"{HEAD} is missing or damaged")
    if head["hash"] != prev or head["seq"] != seq:
        return Verification(found.entries, "the newest entries were removed: the log ends before proof.head")
    return found
