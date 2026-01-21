"""Tests for tools utilities."""

from __future__ import annotations

from tools.utils import error_response, validate_k, validate_metadata, validate_non_empty


def test_error_response_shape() -> None:
    payload = error_response("VALIDATION_ERROR", "bad input", {"field": "query"})
    assert payload["status"] == "error"
    assert payload["error_code"] == "VALIDATION_ERROR"
    assert payload["details"]["field"] == "query"


def test_validate_non_empty() -> None:
    assert validate_non_empty("query", "") is not None
    assert validate_non_empty("query", "ok") is None


def test_validate_k_bounds() -> None:
    assert validate_k("k", -1) is not None
    assert validate_k("k", 1) is None
    assert validate_k("k", 1001) is not None


def test_validate_metadata() -> None:
    assert validate_metadata(None) is None
    assert validate_metadata({"a": 1}) is None
    assert validate_metadata("bad") is not None
