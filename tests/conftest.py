"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Ensure repo root is importable during test collection.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure required env vars are set for tests."""
    monkeypatch.setenv("DB_PASSWORD", "test-password")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("SESSION_PROVIDER", "memory")
    monkeypatch.setenv("SESSION_TTL_DAYS", "1")
    monkeypatch.setenv("SESSION_CLEANUP_INTERVAL_SECONDS", "0")
