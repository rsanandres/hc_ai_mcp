"""Tests for argument validators."""

from __future__ import annotations

from agent.tools.argument_validators import (
    is_valid_uuid,
    is_valid_icd10,
    is_valid_fhir_resource_id,
    validate_patient_id,
    validate_icd10_code,
    get_argument_type_hint,
)


# --- UUID validation ---

def test_valid_uuid() -> None:
    assert is_valid_uuid("550e8400-e29b-41d4-a716-446655440000") is True


def test_invalid_uuid() -> None:
    assert is_valid_uuid("not-a-uuid") is False


def test_empty_uuid() -> None:
    assert is_valid_uuid("") is False


# --- ICD-10 validation ---

def test_valid_icd10_simple() -> None:
    assert is_valid_icd10("E11") is True


def test_valid_icd10_with_decimal() -> None:
    assert is_valid_icd10("E11.9") is True


def test_valid_icd10_hypertension() -> None:
    assert is_valid_icd10("I10") is True


def test_invalid_icd10() -> None:
    assert is_valid_icd10("hello") is False


def test_uuid_is_not_icd10() -> None:
    assert is_valid_icd10("550e8400-e29b-41d4-a716-446655440000") is False


# --- FHIR resource ID ---

def test_valid_fhir_resource_id() -> None:
    assert is_valid_fhir_resource_id("Patient/abc-123") is True


def test_invalid_fhir_resource_id() -> None:
    assert is_valid_fhir_resource_id("not-a-resource") is False


# --- Patient ID validation ---

def test_validate_patient_id_valid() -> None:
    valid, error = validate_patient_id("550e8400-e29b-41d4-a716-446655440000")
    assert valid is True
    assert error == ""


def test_validate_patient_id_with_icd10() -> None:
    valid, error = validate_patient_id("E11.9")
    assert valid is False
    assert "ICD-10" in error


def test_validate_patient_id_empty() -> None:
    valid, error = validate_patient_id("")
    assert valid is False


def test_validate_patient_id_with_fhir() -> None:
    valid, error = validate_patient_id("Patient/abc")
    assert valid is False
    assert "FHIR" in error


# --- ICD-10 code validation ---

def test_validate_icd10_code_valid() -> None:
    valid, error = validate_icd10_code("E11.9")
    assert valid is True
    assert error == ""


def test_validate_icd10_code_with_uuid() -> None:
    valid, error = validate_icd10_code("550e8400-e29b-41d4-a716-446655440000")
    assert valid is False
    assert "UUID" in error


def test_validate_icd10_code_garbage() -> None:
    valid, error = validate_icd10_code("xyz123")
    assert valid is False


# --- Type hints ---

def test_type_hint_uuid() -> None:
    assert "UUID" in get_argument_type_hint("550e8400-e29b-41d4-a716-446655440000")


def test_type_hint_icd10() -> None:
    assert "ICD-10" in get_argument_type_hint("E11.9")


def test_type_hint_fhir() -> None:
    assert "FHIR" in get_argument_type_hint("Observation/abc-123")


def test_type_hint_unknown() -> None:
    assert get_argument_type_hint("blah") == "unknown"


def test_type_hint_empty() -> None:
    assert get_argument_type_hint("") == "empty"
