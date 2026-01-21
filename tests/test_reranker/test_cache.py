"""Tests for reranker cache."""

from __future__ import annotations

from reranker.cache import InMemoryCache, build_cache_key


def test_cache_roundtrip() -> None:
    cache = InMemoryCache(ttl_seconds=60, max_size=10)
    key = build_cache_key("query", ["1", "2"])
    cache.set(key, [("1", 0.1), ("2", 0.2)])
    value = cache.get(key)
    assert value == [("1", 0.1), ("2", 0.2)]
