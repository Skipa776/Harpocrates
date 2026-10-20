"""The per-session cache of findings (G4 spec, "Performance"; SEC-02).

The agent resends the whole history every turn, so most of what the scan step sees was seen a turn ago. Redaction is a
pure function of a leaf, so the spans found in it can be remembered. The cache is keyed by a hash of (field class, key
hint, text) and holds spans only, so it never holds a raw value; it is bounded in entries and bytes, evicts the least
recently used entry first, lives in memory and is dropped with the gate.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import NamedTuple, Optional

MAX_ENTRIES = 50_000
MAX_BYTES = 64 * 1024 * 1024
ENTRY_BASE_COST = 200  # an estimate of one entry's own bytes: the 32-byte key, the table slot, the tuple
SPAN_COST = 100  # an estimate of one Span


class Span(NamedTuple):
    """
    Description: One finding in a leaf: where it is and what kind of secret it is. Never the value.
    Parameters:
        start: index of the first character of the secret within the leaf text.
        end: index just past its last character.
        kind: the coarse secret type for the placeholder (for example "api_token").
    Output: An immutable record.
    """

    start: int
    end: int
    kind: str


def cache_key(field_class: str, hint: Optional[str], text: str) -> bytes:
    """
    Description: Hash a leaf into the key under which its findings are kept.
    Parameters:
        field_class: "R" (rewrite), "S" (scan only), "K" (object key) or "H" (header, path or query).
        hint: the name of the field the text sits in, or None.
        text: the leaf text.
    Output: A 32-byte SHA-256 digest of the three parts, each delimited so they cannot run into one another. The text
        itself is not recoverable from it.
    """
    digest = hashlib.sha256()
    digest.update(field_class.encode("ascii") + b"|")
    if hint is None:
        digest.update(b"-")
    else:
        raw_hint = hint.encode("utf-8", "surrogatepass")
        digest.update(b"+%d:" % len(raw_hint) + raw_hint)
    digest.update(b"|" + text.encode("utf-8", "surrogatepass"))
    return digest.digest()


class SpanCache:
    """
    Description: A thread-safe, bounded, least-recently-used map from leaf hashes to spans.
    Parameters:
        max_entries: the most entries kept (default 50,000).
        max_bytes: the most estimated bytes kept (default 64 MiB).
    Output: An object with get, put, clear, len() and size_bytes.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES, max_bytes: int = MAX_BYTES) -> None:
        """
        Description: Make an empty cache.
        Parameters:
            max_entries: the entry bound.
            max_bytes: the byte bound.
        Output: None.
        """
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries: OrderedDict[bytes, tuple[Span, ...]] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """
        Description: Count the entries.
        Parameters: None
        Output: The number of entries held.
        """
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        """
        Description: The estimated size of what is held.
        Parameters: None
        Output: The sum of ENTRY_BASE_COST plus SPAN_COST per span over all entries.
        """
        return self._size

    @staticmethod
    def _cost(spans: tuple[Span, ...]) -> int:
        """
        Description: Estimate one entry's size.
        Parameters:
            spans: the entry's spans.
        Output: ENTRY_BASE_COST plus SPAN_COST per span.
        """
        return ENTRY_BASE_COST + SPAN_COST * len(spans)

    def get(self, key: bytes) -> Optional[tuple[Span, ...]]:
        """
        Description: Look a leaf up and mark it as recently used.
        Parameters:
            key: a cache_key digest.
        Output: The spans (an empty tuple for a leaf with no findings), or None when the leaf is not cached.
        """
        with self._lock:
            spans = self._entries.get(key)
            if spans is not None:
                self._entries.move_to_end(key)
            return spans

    def put(self, key: bytes, spans: tuple[Span, ...]) -> None:
        """
        Description: Remember a leaf's findings, evicting the least recently used entries to stay within the bounds.
        Parameters:
            key: a cache_key digest.
            spans: the findings (an empty tuple for none).
        Output: None. An entry that alone is over the byte bound is not kept.
        """
        cost = self._cost(spans)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._size -= self._cost(old)
            if cost > self.max_bytes:
                return
            self._entries[key] = spans
            self._size += cost
            while len(self._entries) > self.max_entries or self._size > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._size -= self._cost(evicted)

    def clear(self) -> None:
        """
        Description: Forget everything.
        Parameters: None
        Output: None.
        """
        with self._lock:
            self._entries.clear()
            self._size = 0
