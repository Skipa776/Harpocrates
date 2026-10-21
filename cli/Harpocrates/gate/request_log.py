"""The request log (FR-GATE-01): one JSON line per request, built from a fixed list of fields so that no header value,
path, query or body content can reach it. The full audit log is FR-LOG-01 (G6)."""
from __future__ import annotations

import json
import math
import re
import threading
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

from Harpocrates.gate.errors import REASONS, is_block_code
from Harpocrates.gate.media import KINDS

_METHODS = frozenset({"GET", "POST", "HEAD", "PUT", "PATCH", "DELETE", "OPTIONS"})
ROUTE_NAMES = frozenset({"anthropic.messages", "anthropic.count_tokens", "openai.responses", "anthropic.models",
                         "openai.models", "websocket-upgrade", "unknown"})
_DECISIONS = frozenset({"forwarded", "blocked", "upgrade-refused", "upstream-error"})
_COUNTS = ("bytes_in", "bytes_out", "bytes_back")
_TIMINGS = ("ms_scan", "ms_connect", "ms_upstream", "ms_total")
_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")


def _number(value: object, integer: bool) -> Optional[object]:
    """
    Description: Accept a count or a duration, and nothing else.
    Parameters:
        value: the candidate.
        integer: True for a count (an int), False for a duration (an int or a finite float).
    Output: The value (a float rounded to 3 places), or None for a bool, a negative number, NaN, infinity or another type.
    """
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        return None
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(value, 3) if isinstance(value, float) else value


def format_entry(entry: Mapping[str, object]) -> str:
    """
    Description: Turn one request's facts into a log line, keeping only fields that cannot carry request content.
    Parameters:
        entry: facts about the request. Read: time, method, route, status, decision, reason, bytes_in, bytes_out,
            bytes_back, ms_scan, ms_connect, ms_upstream, ms_total, upstream_reused, media. Every other key is ignored.
    Output: One compact JSON object ending in a newline, fields in that order. A method outside the usual seven
        is logged as "OTHER". A route, decision or reason outside the gate's fixed vocabulary, a status outside 100 to 599,
        and a bad number are left out. media (FR-GATE-07) keeps only {"kind", "bytes"} pairs with a kind from the fixed list
        and a count, and is left out when none remain. upstream_reused is kept only as a bool. No side effects.
    """
    method, route = entry.get("method"), entry.get("route")
    status, decision, reason = entry.get("status"), entry.get("decision"), entry.get("reason")
    stamp = entry.get("time")
    fields: dict[str, object] = {}
    if isinstance(stamp, str) and _TIME.fullmatch(stamp):
        fields["time"] = stamp
    if "method" in entry:
        fields["method"] = method if method in _METHODS else "OTHER"
    if "route" in entry:
        fields["route"] = route if route in ROUTE_NAMES else "unknown"
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        fields["status"] = status
    if decision in _DECISIONS:
        fields["decision"] = decision
    if isinstance(reason, str) and (reason in REASONS or is_block_code(reason)):
        fields["reason"] = reason
    for names, integer in ((_COUNTS, True), (_TIMINGS, False)):
        for name in names:
            value = _number(entry.get(name), integer)
            if value is not None:
                fields[name] = value
    if isinstance(entry.get("upstream_reused"), bool):
        fields["upstream_reused"] = entry["upstream_reused"]
    media = entry.get("media")
    notes = [{"kind": item["kind"], "bytes": item["bytes"]} for item in media if isinstance(item, dict)
             and item.get("kind") in KINDS and _number(item.get("bytes"), True) is not None] if isinstance(media, list) else []
    if notes:
        fields["media"] = notes
    return json.dumps(fields, separators=(",", ":")) + "\n"


class RequestLog:
    """
    Description: Writes one log line per request to a sink, safely from many threads.
    Parameters:
        sink: a function that takes one line (ending in a newline), or None to log nothing.
    Output: An object whose write() stamps the time and sends the line to the sink.
    """

    def __init__(self, sink: Optional[Callable[[str], None]]) -> None:
        """
        Description: Keep the sink and make the lock that serializes writes.
        Parameters:
            sink: the line consumer, or None.
        Output: None.
        """
        self._sink = sink
        self._lock = threading.Lock()

    def write(self, entry: Mapping[str, object]) -> None:
        """
        Description: Log one request.
        Parameters:
            entry: the facts; see format_entry. Its "time" is replaced by the current UTC time.
        Output: None. Sends one line to the sink; does nothing when there is no sink. A failing sink raises.
        """
        if self._sink is None:
            return
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        line = format_entry({**entry, "time": stamp})
        with self._lock:
            self._sink(line)
