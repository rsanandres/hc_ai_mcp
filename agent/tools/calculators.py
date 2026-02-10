"""Medical calculator tools."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agent.tools.schemas import CalculationResponse


def _egfr_stage(gfr: float) -> str:
    if gfr >= 90:
        return "G1"
    if gfr >= 60:
        return "G2"
    if gfr >= 45:
        return "G3a"
    if gfr >= 30:
        return "G3b"
    if gfr >= 15:
        return "G4"
    return "G5"


@tool
def calculate_gfr(
    age: int,
    sex: str,
    creatinine: float,
    race: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate eGFR using CKD-EPI 2021 equation (no race adjustment)."""
    sex_lower = sex.strip().lower()
    if sex_lower not in {"male", "female"}:
        return CalculationResponse(
            success=False, error="sex must be 'male' or 'female'.", result=None,
        ).model_dump()
    if age <= 0 or creatinine <= 0:
        return CalculationResponse(
            success=False, error="age and creatinine must be positive.", result=None,
        ).model_dump()

    k = 0.7 if sex_lower == "female" else 0.9
    alpha = -0.241 if sex_lower == "female" else -0.302
    min_ratio = min(creatinine / k, 1) ** alpha
    max_ratio = max(creatinine / k, 1) ** -1.200
    sex_factor = 1.012 if sex_lower == "female" else 1.0
    gfr = 142 * min_ratio * max_ratio * (0.9938 ** age) * sex_factor

    return CalculationResponse(
        result={
            "gfr": round(gfr, 1),
            "stage": _egfr_stage(gfr),
            "race_note": "Race adjustment not applied (CKD-EPI 2021).",
            "input_race": race,
        }
    ).model_dump()


@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> Dict[str, Any]:
    """Calculate body mass index and category."""
    if weight_kg <= 0 or height_cm <= 0:
        return CalculationResponse(
            success=False, error="weight_kg and height_cm must be positive.", result=None,
        ).model_dump()
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return CalculationResponse(
        result={"bmi": round(bmi, 1), "category": category},
    ).model_dump()


@tool
def calculate_bsa(weight_kg: float, height_cm: float) -> Dict[str, Any]:
    """Calculate body surface area using Mosteller formula."""
    if weight_kg <= 0 or height_cm <= 0:
        return CalculationResponse(
            success=False, error="weight_kg and height_cm must be positive.", result=None,
        ).model_dump()
    bsa = math.sqrt((height_cm * weight_kg) / 3600.0)
    return CalculationResponse(result={"bsa": round(bsa, 3)}).model_dump()


@tool
def calculate_creatinine_clearance(
    age: int,
    weight_kg: float,
    sex: str,
    creatinine: float,
) -> Dict[str, Any]:
    """Calculate creatinine clearance using Cockcroft-Gault formula."""
    sex_lower = sex.strip().lower()
    if sex_lower not in {"male", "female"}:
        return CalculationResponse(
            success=False, error="sex must be 'male' or 'female'.", result=None,
        ).model_dump()
    if age <= 0 or weight_kg <= 0 or creatinine <= 0:
        return CalculationResponse(
            success=False, error="age, weight_kg, and creatinine must be positive.", result=None,
        ).model_dump()

    base = ((140 - age) * weight_kg) / (72 * creatinine)
    if sex_lower == "female":
        base *= 0.85
    return CalculationResponse(
        result={"creatinine_clearance": round(base, 1)}
    ).model_dump()
