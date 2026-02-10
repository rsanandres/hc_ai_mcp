"""Tests for patient context auto-injection."""

from __future__ import annotations

from agent.tools.context import (
    set_patient_context,
    get_patient_context,
    clear_patient_context,
)


def test_set_and_get_context() -> None:
    set_patient_context("patient-uuid-1234")
    assert get_patient_context() == "patient-uuid-1234"
    clear_patient_context()


def test_clear_context() -> None:
    set_patient_context("test-id")
    clear_patient_context()
    assert get_patient_context() is None


def test_default_context_is_none() -> None:
    clear_patient_context()
    assert get_patient_context() is None


def test_overwrite_context() -> None:
    set_patient_context("first")
    set_patient_context("second")
    assert get_patient_context() == "second"
    clear_patient_context()


def test_set_none_clears() -> None:
    set_patient_context("test")
    set_patient_context(None)
    assert get_patient_context() is None
