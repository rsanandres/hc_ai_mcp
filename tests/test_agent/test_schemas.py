"""Tests for tool response schemas."""

from __future__ import annotations

from agent.tools.schemas import (
    ToolResponse,
    ChunkResult,
    RetrievalResponse,
    CalculationResponse,
    FDAResponse,
    ResearchResponse,
    TerminologyResponse,
    MedicationCheckResponse,
    TimelineResponse,
    DosageValidationResponse,
    LOINCResponse,
)


def test_tool_response_defaults() -> None:
    r = ToolResponse()
    assert r.success is True
    assert r.error is None


def test_tool_response_error() -> None:
    r = ToolResponse(success=False, error="something broke")
    assert r.success is False
    assert r.error == "something broke"


def test_chunk_result() -> None:
    c = ChunkResult(id="1", content="test", score=0.9, metadata={"key": "val"})
    assert c.id == "1"
    assert c.score == 0.9


def test_retrieval_response() -> None:
    r = RetrievalResponse(
        query="test query",
        chunks=[ChunkResult(id="1", content="x", score=0.5, metadata={})],
        count=1,
    )
    assert r.count == 1
    assert len(r.chunks) == 1


def test_calculation_response() -> None:
    r = CalculationResponse(result={"bmi": 22.5})
    d = r.model_dump()
    assert d["result"]["bmi"] == 22.5
    assert d["success"] is True


def test_medication_check_response() -> None:
    r = MedicationCheckResponse(
        medications=["aspirin", "warfarin"],
        warnings=["bleeding risk"],
    )
    assert len(r.warnings) == 1


def test_dosage_validation_response() -> None:
    r = DosageValidationResponse(
        medication="metformin",
        dose="500mg",
        is_valid=True,
        warnings=[],
        frequency="BID",
    )
    assert r.is_valid is True


def test_loinc_response() -> None:
    r = LOINCResponse(code="2339-0", name="Glucose")
    d = r.model_dump()
    assert d["code"] == "2339-0"
    assert d["success"] is True


def test_timeline_response_defaults() -> None:
    r = TimelineResponse(patient_id="test-123")
    assert r.events == []
    assert r.success is True
