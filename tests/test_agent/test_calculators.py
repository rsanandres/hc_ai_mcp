"""Tests for medical calculator tools."""

from __future__ import annotations

from agent.tools.calculators import (
    calculate_bmi,
    calculate_bsa,
    calculate_gfr,
    calculate_creatinine_clearance,
)


def test_bmi_normal() -> None:
    result = calculate_bmi.invoke({"weight_kg": 70, "height_cm": 175})
    assert result["success"] is True
    assert result["result"]["category"] == "Normal"
    assert 20 < result["result"]["bmi"] < 25


def test_bmi_obese() -> None:
    result = calculate_bmi.invoke({"weight_kg": 120, "height_cm": 170})
    assert result["result"]["category"] == "Obese"


def test_bmi_invalid_input() -> None:
    result = calculate_bmi.invoke({"weight_kg": -1, "height_cm": 170})
    assert result["success"] is False


def test_bsa_normal() -> None:
    result = calculate_bsa.invoke({"weight_kg": 70, "height_cm": 175})
    assert result["success"] is True
    assert 1.5 < result["result"]["bsa"] < 2.5


def test_bsa_invalid() -> None:
    result = calculate_bsa.invoke({"weight_kg": 0, "height_cm": 175})
    assert result["success"] is False


def test_gfr_male() -> None:
    result = calculate_gfr.invoke({"age": 50, "sex": "male", "creatinine": 1.0})
    assert result["success"] is True
    assert result["result"]["gfr"] > 0
    assert result["result"]["stage"] in ("G1", "G2", "G3a", "G3b", "G4", "G5")


def test_gfr_female() -> None:
    result = calculate_gfr.invoke({"age": 50, "sex": "female", "creatinine": 0.8})
    assert result["success"] is True
    assert result["result"]["gfr"] > 0


def test_gfr_invalid_sex() -> None:
    result = calculate_gfr.invoke({"age": 50, "sex": "other", "creatinine": 1.0})
    assert result["success"] is False


def test_gfr_race_note() -> None:
    result = calculate_gfr.invoke({"age": 50, "sex": "male", "creatinine": 1.0, "race": "black"})
    assert "CKD-EPI 2021" in result["result"]["race_note"]


def test_creatinine_clearance_male() -> None:
    result = calculate_creatinine_clearance.invoke({
        "age": 50, "weight_kg": 80, "sex": "male", "creatinine": 1.2
    })
    assert result["success"] is True
    assert result["result"]["creatinine_clearance"] > 0


def test_creatinine_clearance_female() -> None:
    result = calculate_creatinine_clearance.invoke({
        "age": 50, "weight_kg": 60, "sex": "female", "creatinine": 0.9
    })
    assert result["success"] is True
    # Female should be lower (0.85 factor)
    male_result = calculate_creatinine_clearance.invoke({
        "age": 50, "weight_kg": 60, "sex": "male", "creatinine": 0.9
    })
    assert result["result"]["creatinine_clearance"] < male_result["result"]["creatinine_clearance"]


def test_creatinine_clearance_invalid() -> None:
    result = calculate_creatinine_clearance.invoke({
        "age": 0, "weight_kg": 80, "sex": "male", "creatinine": 1.0
    })
    assert result["success"] is False
