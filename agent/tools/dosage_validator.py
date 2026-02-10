"""Dosage validation tool using openFDA labels."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool

from agent.tools.schemas import DosageValidationResponse

OPENFDA_URL = os.getenv("OPENFDA_LABEL_URL", "https://api.fda.gov/drug/label.json")


def _parse_dose_values(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", re.IGNORECASE)
    values: List[Dict[str, Any]] = []
    for match in pattern.finditer(text):
        values.append({"value": float(match.group(1)), "unit": match.group(2).lower()})
    return values


def _normalize_unit(unit: str) -> str:
    unit = unit.strip().lower()
    if unit in {"mcg", "ug"}:
        return "mcg"
    if unit in {"g", "gram", "grams"}:
        return "g"
    return "mg"


def _dose_in_unit(value: float, unit: str, target_unit: str) -> Optional[float]:
    unit = _normalize_unit(unit)
    target_unit = _normalize_unit(target_unit)
    if unit == target_unit:
        return value
    if unit == "g" and target_unit == "mg":
        return value * 1000
    if unit == "mg" and target_unit == "g":
        return value / 1000
    if unit == "mcg" and target_unit == "mg":
        return value / 1000
    if unit == "mg" and target_unit == "mcg":
        return value * 1000
    return None


@tool
async def validate_dosage(
    drug_name: str,
    dose_amount: float,
    dose_unit: str,
    frequency: str,
    patient_weight_kg: Optional[float] = None,
    patient_gfr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate dosage using openFDA labels.
    Returns dict with validity, warnings, and label excerpt.
    """
    if dose_amount <= 0:
        return DosageValidationResponse(
            success=False,
            error="dose_amount must be positive",
            medication=drug_name,
            dose=f"{dose_amount}{dose_unit}",
            is_valid=False,
            warnings=["dose_amount must be positive"],
            frequency=frequency,
            patient_weight_kg=patient_weight_kg,
        ).model_dump()

    if patient_gfr is not None and patient_gfr < 30:
        return DosageValidationResponse(
            medication=drug_name,
            dose=f"{dose_amount}{dose_unit}",
            is_valid=False,
            warnings=["Renal impairment detected (GFR < 30). Dose adjustment required."],
            reference_range=frequency,
            frequency=frequency,
            patient_weight_kg=patient_weight_kg,
        ).model_dump()

    query = f'(openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}")'
    params = {"search": query, "limit": 1}
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(OPENFDA_URL, params=params)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return DosageValidationResponse(
                success=False,
                error=f"openFDA request failed: {exc}",
                medication=drug_name,
                dose=f"{dose_amount}{dose_unit}",
                is_valid=False,
                warnings=[f"openFDA request failed: {exc}"],
                reference_range=frequency,
                frequency=frequency,
                patient_weight_kg=patient_weight_kg,
            ).model_dump()

    results = payload.get("results", [])
    if not results:
        return DosageValidationResponse(
            success=False,
            error="No openFDA label found for drug.",
            medication=drug_name,
            dose=f"{dose_amount}{dose_unit}",
            is_valid=False,
            warnings=["No openFDA label found for drug."],
            reference_range=frequency,
            frequency=frequency,
            patient_weight_kg=patient_weight_kg,
        ).model_dump()

    label = results[0]
    dosage_sections = label.get("dosage_and_administration", [])
    dosage_text = " ".join(dosage_sections) if isinstance(dosage_sections, list) else str(dosage_sections)
    dose_values = _parse_dose_values(dosage_text)

    normalized_unit = _normalize_unit(dose_unit)
    converted_values = []
    for item in dose_values:
        converted = _dose_in_unit(item["value"], item["unit"], normalized_unit)
        if converted is not None:
            converted_values.append(converted)

    valid = None
    warning = ""
    if converted_values:
        min_dose = min(converted_values)
        max_dose = max(converted_values)
        valid = min_dose <= dose_amount <= max_dose
        if not valid:
            warning = f"Dose outside label range ({min_dose}-{max_dose} {normalized_unit})."
    else:
        valid = True
        warning = "Unable to parse label dose range; manual review recommended."

    warnings = [warning] if warning else []
    return DosageValidationResponse(
        medication=drug_name,
        dose=f"{dose_amount}{dose_unit}",
        is_valid=bool(valid),
        warnings=warnings,
        reference_range=frequency,
        frequency=frequency,
        label_excerpt=dosage_text[:500],
        patient_weight_kg=patient_weight_kg,
    ).model_dump()
