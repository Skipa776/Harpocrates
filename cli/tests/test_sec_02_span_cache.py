"""SEC-02 and FR-GATE-05: the per-session cache of findings. It is keyed by a hash, holds spans only, never grows past its
bounds and forgets the least recently used entry first. Unit tests of Harpocrates.gate.spancache."""
from __future__ import annotations

import threading

from Harpocrates.gate.spancache import (
    ENTRY_BASE_COST,
    MAX_BYTES,
    MAX_ENTRIES,
    Span,
    SpanCache,
    cache_key,
)


def test_fr_gate_05_the_default_bounds_are_the_specified_ones() -> None:
    assert (MAX_ENTRIES, MAX_BYTES) == (50_000, 64 * 1024 * 1024)
    cache = SpanCache()
    assert (cache.max_entries, cache.max_bytes, len(cache), cache.size_bytes) == (MAX_ENTRIES, MAX_BYTES, 0, 0)


def test_sec_02_cache_key_is_a_digest_of_class_and_text_that_tells_them_apart() -> None:
    key = cache_key("R", "echo hi")
    assert isinstance(key, bytes) and len(key) == 32
    assert key == cache_key("R", "echo hi")
    assert len({key, cache_key("S", "echo hi"), cache_key("R", "echo hi "), cache_key("R", "echo  hi"),
                cache_key("R", "")}) == 5
    assert b"echo hi" not in key
    assert len(cache_key("R", "a\ud800b")) == 32  # a lone surrogate (from a \ud800 escape) is still hashable


def test_fr_gate_05_a_miss_is_none_and_a_hit_returns_the_same_spans() -> None:
    cache = SpanCache()
    key = cache_key("R", "x")
    assert cache.get(key) is None
    spans = (Span(3, 9, "api_token"),)
    cache.put(key, spans)
    assert cache.get(key) == spans
    cache.put(cache_key("R", "clean"), ())  # a clean leaf is worth remembering too
    assert cache.get(cache_key("R", "clean")) == ()
    assert cache.get(cache_key("R", "clean")) is not None


def test_fr_gate_05_least_recently_used_entry_is_evicted_first() -> None:
    cache = SpanCache(max_entries=3)
    keys = [cache_key("R", str(i)) for i in range(4)]
    for key in keys[:3]:
        cache.put(key, ())
    cache.get(keys[0])  # keys[1] is now the oldest
    cache.put(keys[3], ())
    assert len(cache) == 3 and cache.get(keys[1]) is None
    assert all(cache.get(k) is not None for k in (keys[0], keys[2], keys[3]))


def test_fr_gate_05_the_byte_bound_evicts_too_and_an_entry_that_alone_is_too_big_is_not_kept() -> None:
    many = tuple(Span(i, i + 1, "generic") for i in range(10))
    cost = ENTRY_BASE_COST + 10 * 100
    cache = SpanCache(max_bytes=cost * 2 + 1)
    first, second, third = (cache_key("R", n) for n in "abc")
    for key in (first, second, third):
        cache.put(key, many)
    assert len(cache) == 2 and cache.get(first) is None and cache.size_bytes <= cost * 2 + 1
    tiny = SpanCache(max_bytes=ENTRY_BASE_COST)
    tiny.put(first, many)
    assert len(tiny) == 0 and tiny.size_bytes == 0


def test_fr_gate_05_replacing_an_entry_keeps_the_size_exact_and_clear_empties_the_cache() -> None:
    cache = SpanCache()
    key = cache_key("R", "k")
    cache.put(key, (Span(0, 1, "generic"),) * 3)
    cache.put(key, ())
    assert len(cache) == 1 and cache.size_bytes == ENTRY_BASE_COST
    cache.clear()
    assert len(cache) == 0 and cache.size_bytes == 0 and cache.get(key) is None


def test_fr_gate_05_the_cache_survives_many_threads() -> None:
    cache = SpanCache(max_entries=200)

    def work(offset: int) -> None:
        for i in range(2000):
            key = cache_key("R", str((i + offset) % 300))
            if cache.get(key) is None:
                cache.put(key, (Span(0, 1, "generic"),))

    threads = [threading.Thread(target=work, args=(n * 37,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(cache) <= 200 and cache.size_bytes == sum(ENTRY_BASE_COST + 100 for _ in range(len(cache)))
