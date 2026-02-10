"""Validation utilities for agent tool arguments."""

from __future__ import annotations

import re
from typing import Tuple


# Regex patterns for common medical/FHIR identifiers
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# ICD-10-CM format: Letter + 2 digits, optionally followed by a decimal and 1-4 more characters
ICD10_PATTERN = re.compile(
    r"^[A-TV-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$",
    re.IGNORECASE
)

# FHIR Resource ID format: ResourceType/UUID or ResourceType/ID
FHIR_RESOURCE_ID_PATTERN = re.compile(
    r"^[A-Z][a-zA-Z]+/[a-zA-Z0-9-]+$"
)


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID format."""
    if not value:
        return False
    return bool(UUID_PATTERN.match(value.strip()))


def is_valid_icd10(value: str) -> bool:
    """Check if a string looks like an ICD-10-CM code."""
    if not value:
        return False
    return bool(ICD10_PATTERN.match(value.strip()))


def is_valid_fhir_resource_id(value: str) -> bool:
    """Check if a string looks like a FHIR resource ID."""
    if not value:
        return False
    return bool(FHIR_RESOURCE_ID_PATTERN.match(value.strip()))


def validate_patient_id(value: str) -> Tuple[bool, str]:
    """
    Validate a patient_id argument.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not value or not value.strip():
        return False, "patient_id is required"

    value = value.strip()

    if is_valid_uuid(value):
        return True, ""

    if is_valid_icd10(value):
        return False, (
            f"'{value}' looks like an ICD-10 code, not a patient ID. "
            "Patient IDs are UUIDs (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx). "
            "Use search_icd10() for ICD-10 lookups."
        )

    if is_valid_fhir_resource_id(value):
        return False, (
            f"'{value}' looks like a FHIR resource ID, not a patient ID. "
            "Patient IDs are UUIDs (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)."
        )

    return False, (
        f"Invalid patient_id format: '{value}'. "
        "Must be a UUID (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)."
    )


def validate_icd10_code(value: str) -> Tuple[bool, str]:
    """
    Validate an ICD-10 code argument.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not value or not value.strip():
        return False, "ICD-10 code is required"

    value = value.strip()

    if is_valid_icd10(value):
        return True, ""

    if is_valid_uuid(value):
        return False, (
            f"'{value}' looks like a patient UUID, not an ICD-10 code. "
            "ICD-10 codes look like 'E11.9', 'I10', 'J06.9'. "
            "Use search_patient_records() with patient_id for patient data."
        )

    return False, (
        f"'{value}' doesn't look like a valid ICD-10 code. "
        "ICD-10 codes start with a letter followed by 2 digits, "
        "optionally with a decimal (e.g., 'E11.9', 'I10', 'M54.5')."
    )


def get_argument_type_hint(value: str) -> str:
    """Detect what type of identifier a value looks like."""
    if not value or not value.strip():
        return "empty"

    value = value.strip()

    if is_valid_uuid(value):
        return "UUID (patient_id)"
    if is_valid_icd10(value):
        return "ICD-10 code"
    if is_valid_fhir_resource_id(value):
        return "FHIR resource ID"

    return "unknown"
