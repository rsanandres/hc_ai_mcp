"""Tests for session store (DynamoDB-only).

These tests mock the DynamoDB layer since the store no longer has
an in-memory fallback. They verify the public interface contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_store():
    """Create a SessionStore with a mocked DynamoDB resource."""
    with patch("session.store.BOTO3_AVAILABLE", True), \
         patch("session.store.boto3") as mock_boto3, \
         patch("session.store._warn_on_bad_endpoint"):
        mock_resource = MagicMock()
        mock_boto3.resource.return_value = mock_resource

        # Mock Table objects
        mock_turns_table = MagicMock()
        mock_summary_table = MagicMock()
        mock_resource.Table.side_effect = lambda name: (
            mock_turns_table if "turns" in name else mock_summary_table
        )

        from session.store import SessionStore
        store = SessionStore(
            region_name="us-east-1",
            turns_table="test-turns",
            summary_table="test-summary",
            endpoint_url="http://localhost:8001",
            ttl_days=7,
            max_recent=10,
        )
        return store, mock_turns_table, mock_summary_table


def test_session_store_creates() -> None:
    store, _, _ = _make_store()
    assert store is not None
    assert store.turns_table_name == "test-turns"
    assert store.summary_table_name == "test-summary"


def test_session_append_turn() -> None:
    store, mock_turns, _ = _make_store()
    store.append_turn("s1", "user", "hello")
    mock_turns.put_item.assert_called_once()
    item = mock_turns.put_item.call_args[1]["Item"]
    assert item["session_id"] == "s1"
    assert item["role"] == "user"
    assert item["text"] == "hello"


def test_session_get_recent() -> None:
    store, mock_turns, _ = _make_store()
    mock_turns.query.return_value = {
        "Items": [
            {"session_id": "s1", "timestamp": "2025-01-01T00:00:00Z", "role": "user", "text": "hi"},
            {"session_id": "s1", "timestamp": "2025-01-01T00:00:01Z", "role": "assistant", "text": "hello"},
        ]
    }
    recent = store.get_recent("s1", limit=10)
    assert len(recent) == 2
    mock_turns.query.assert_called_once()


def test_session_summary_update() -> None:
    store, _, mock_summary = _make_store()
    store.update_summary("s1", {"topic": "demo"})
    mock_summary.update_item.assert_called_once()


def test_session_get_summary() -> None:
    store, _, mock_summary = _make_store()
    mock_summary.get_item.return_value = {
        "Item": {"session_id": "s1", "topic": "demo"}
    }
    summary = store.get_summary("s1")
    assert summary["topic"] == "demo"


def test_session_get_summary_missing() -> None:
    store, _, mock_summary = _make_store()
    mock_summary.get_item.return_value = {}
    summary = store.get_summary("s1")
    assert summary == {}


def test_session_clear() -> None:
    store, mock_turns, mock_summary = _make_store()
    # Mock the query for turns to delete
    mock_turns.query.return_value = {
        "Items": [
            {"session_id": "s1", "turn_ts": "2025-01-01T00:00:00Z"},
        ]
    }
    store.clear_session("s1")
    # Should have queried turns and deleted summary
    mock_turns.query.assert_called()
    mock_summary.delete_item.assert_called_once()
