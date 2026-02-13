"""Tests for InMemorySessionStore.

Validates the in-memory session backend that serves as the default
when SESSION_PROVIDER=memory (no DynamoDB dependency).
"""

from __future__ import annotations

import pytest

from session.store import InMemorySessionStore, SessionTurn


@pytest.fixture
def store():
    """Create a fresh in-memory store for each test."""
    return InMemorySessionStore()


class TestAppendTurn:
    def test_returns_session_turn(self, store):
        turn = store.append_turn("s1", "user", "hello")
        assert isinstance(turn, SessionTurn)
        assert turn.session_id == "s1"
        assert turn.role == "user"
        assert turn.text == "hello"
        assert turn.turn_ts  # non-empty timestamp

    def test_stores_metadata(self, store):
        meta = {"source": "test", "score": 0.9}
        turn = store.append_turn("s1", "assistant", "hi", meta=meta)
        assert turn.meta == meta

    def test_stores_patient_id(self, store):
        turn = store.append_turn("s1", "user", "hello", patient_id="p-123")
        assert turn.patient_id == "p-123"

    def test_multiple_turns_accumulate(self, store):
        store.append_turn("s1", "user", "msg1")
        store.append_turn("s1", "assistant", "msg2")
        store.append_turn("s1", "user", "msg3")
        recent = store.get_recent("s1", limit=10)
        assert len(recent) == 3


class TestGetRecent:
    def test_empty_session(self, store):
        recent = store.get_recent("nonexistent")
        assert recent == []

    def test_returns_newest_first(self, store):
        store.append_turn("s1", "user", "first")
        store.append_turn("s1", "assistant", "second")
        store.append_turn("s1", "user", "third")
        recent = store.get_recent("s1", limit=10)
        assert recent[0]["text"] == "third"
        assert recent[-1]["text"] == "first"

    def test_respects_limit(self, store):
        for i in range(5):
            store.append_turn("s1", "user", f"msg{i}")
        recent = store.get_recent("s1", limit=2)
        assert len(recent) == 2

    def test_default_limit(self, store):
        for i in range(15):
            store.append_turn("s1", "user", f"msg{i}")
        recent = store.get_recent("s1")
        assert len(recent) == 10  # default limit

    def test_sessions_are_isolated(self, store):
        store.append_turn("s1", "user", "session1")
        store.append_turn("s2", "user", "session2")
        recent_s1 = store.get_recent("s1")
        recent_s2 = store.get_recent("s2")
        assert len(recent_s1) == 1
        assert len(recent_s2) == 1
        assert recent_s1[0]["text"] == "session1"
        assert recent_s2[0]["text"] == "session2"


class TestSummary:
    def test_empty_summary(self, store):
        summary = store.get_summary("nonexistent")
        assert summary == {}

    def test_update_and_get(self, store):
        store.update_summary("s1", {"topic": "diabetes", "entities": ["insulin"]})
        summary = store.get_summary("s1")
        assert summary["topic"] == "diabetes"
        assert summary["entities"] == ["insulin"]

    def test_update_merges(self, store):
        store.update_summary("s1", {"topic": "diabetes"})
        store.update_summary("s1", {"severity": "moderate"})
        summary = store.get_summary("s1")
        assert summary["topic"] == "diabetes"
        assert summary["severity"] == "moderate"

    def test_update_with_patient_id(self, store):
        store.update_summary("s1", {"topic": "test"}, patient_id="p-abc")
        summary = store.get_summary("s1")
        assert summary["patient_id"] == "p-abc"

    def test_update_with_user_id(self, store):
        store.update_summary("s1", {"topic": "test"}, user_id="u-xyz")
        summary = store.get_summary("s1")
        assert summary["user_id"] == "u-xyz"

    def test_update_sets_last_activity(self, store):
        store.update_summary("s1", {"topic": "test"})
        summary = store.get_summary("s1")
        assert "last_activity" in summary
        assert summary["last_activity"].endswith("Z")


class TestPatient:
    def test_set_and_get_patient(self, store):
        store.set_patient("s1", "patient-uuid-123")
        assert store.get_patient("s1") == "patient-uuid-123"

    def test_get_patient_no_session(self, store):
        assert store.get_patient("nonexistent") is None


class TestClearSession:
    def test_clears_turns(self, store):
        store.append_turn("s1", "user", "hello")
        store.append_turn("s1", "assistant", "hi")
        store.clear_session("s1")
        assert store.get_recent("s1") == []

    def test_clears_summary(self, store):
        store.update_summary("s1", {"topic": "test"})
        store.clear_session("s1")
        assert store.get_summary("s1") == {}

    def test_does_not_affect_other_sessions(self, store):
        store.append_turn("s1", "user", "keep")
        store.append_turn("s2", "user", "delete")
        store.clear_session("s2")
        assert len(store.get_recent("s1")) == 1
        assert len(store.get_recent("s2")) == 0

    def test_clear_nonexistent_session(self, store):
        # Should not raise
        store.clear_session("nonexistent")


class TestClearAll:
    def test_clears_everything(self, store):
        store.append_turn("s1", "user", "msg1")
        store.append_turn("s2", "user", "msg2")
        store.update_summary("s1", {"topic": "a"})
        store.update_summary("s2", {"topic": "b"})
        store.clear_all()
        assert store.get_recent("s1") == []
        assert store.get_recent("s2") == []
        assert store.get_summary("s1") == {}
        assert store.get_summary("s2") == {}


class TestGetSessionStore:
    def test_memory_provider(self):
        """get_session_store() returns InMemorySessionStore for memory provider."""
        import session.store as mod
        # Reset singleton
        mod._SESSION_STORE = None
        try:
            store = mod.get_session_store()
            assert isinstance(store, InMemorySessionStore)
        finally:
            mod._SESSION_STORE = None

    def test_singleton_returns_same_instance(self):
        """get_session_store() returns the same instance on repeated calls."""
        import session.store as mod
        mod._SESSION_STORE = None
        try:
            store1 = mod.get_session_store()
            store2 = mod.get_session_store()
            assert store1 is store2
        finally:
            mod._SESSION_STORE = None
