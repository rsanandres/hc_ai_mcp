"""Tests for embedding utilities."""

from __future__ import annotations

import importlib
import os


def test_unknown_provider_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "unknown")
    import embeddings.embedder as embedder
    importlib.reload(embedder)
    assert embedder.get_embeddings(["text"]) is None
