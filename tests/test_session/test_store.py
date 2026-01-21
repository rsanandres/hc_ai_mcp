"""Tests for session store."""

from __future__ import annotations

import time

from session.store import SessionStore


def test_session_append_and_get() -> None:
    store = SessionStore(provider="memory", max_recent=2, ttl_days=1)
    store.append_turn("s1", "user", "hello")
    store.append_turn("s1", "assistant", "hi")
    recent = store.get_recent("s1")
    assert len(recent) == 2
    assert recent[0]["role"] == "assistant"


def test_session_summary_update() -> None:
    store = SessionStore(provider="memory", max_recent=2, ttl_days=1)
    store.update_summary("s1", {"topic": "demo"})
    summary = store.get_summary("s1")
    assert summary["topic"] == "demo"


def test_session_cleanup_expired() -> None:
    store = SessionStore(provider="memory", max_recent=2, ttl_days=1)
    store.append_turn("s1", "user", "hello")
    store._last_access["s1"] = time.time() - (2 * 86400)
    store._last_cleanup = 0
    store._cleanup_if_needed()
    assert store.get_recent("s1") == []
