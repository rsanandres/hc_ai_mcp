"""Tests for guardrails validators."""

from __future__ import annotations

from agent.guardrails.validators import validate_output, setup_guard


def test_validate_output_returns_tuple() -> None:
    is_valid, error = validate_output("This is safe text.")
    assert isinstance(is_valid, bool)
    assert isinstance(error, str)


def test_validate_output_passes_clean_text() -> None:
    # Without guardrails-ai installed, should always return True
    is_valid, error = validate_output("Patient has normal vitals.")
    assert is_valid is True
    assert error == ""


def test_validate_output_empty_text() -> None:
    is_valid, error = validate_output("")
    # Empty text should still pass (not an error condition)
    assert is_valid is True


def test_setup_guard_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("GUARDRAILS_ENABLED", "false")
    guard = setup_guard()
    assert guard is None
