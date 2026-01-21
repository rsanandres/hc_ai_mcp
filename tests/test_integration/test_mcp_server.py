"""Basic integration tests for server config validation."""

from __future__ import annotations

from config.loader import validate_env


def test_validate_env_ok() -> None:
    errors = validate_env()
    assert errors == []
