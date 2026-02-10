from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    success: bool = True
    error: Optional[str] = None


class ChunkResult(BaseModel):
    id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(ToolResponse):
    query: str  # Cleaned query used for search
    original_query: Optional[str] = None  # Original query before name stripping
    chunks: List[ChunkResult] = Field(default_factory=list)
    count: int = 0


class CalculationResponse(ToolResponse):
    result: Any
    unit: Optional[str] = None
    interpretation: Optional[str] = None


class FDAResponse(ToolResponse):
    results: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class ResearchResponse(ToolResponse):
    results: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class TerminologyResponse(ToolResponse):
    results: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class SessionContextResponse(ToolResponse):
    summary: Dict[str, Any] = Field(default_factory=dict)
    recent_turns: List[Dict[str, Any]] = Field(default_factory=list)


class MedicationCheckResponse(ToolResponse):
    medications: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TimelineResponse(ToolResponse):
    patient_id: str
    events: List[ChunkResult] = Field(default_factory=list)


class DosageValidationResponse(ToolResponse):
    medication: str
    dose: str
    is_valid: bool = False
    warnings: List[str] = Field(default_factory=list)
    reference_range: Optional[str] = None
    frequency: Optional[str] = None
    label_excerpt: Optional[str] = None
    patient_weight_kg: Optional[float] = None


class LOINCResponse(ToolResponse):
    code: str
    name: Optional[str] = None
    component: Optional[str] = None
    system: Optional[str] = None
