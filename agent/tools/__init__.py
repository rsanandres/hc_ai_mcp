"""Tool definitions for the medical agents."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from agent.pii_masker.factory import create_pii_masker
from session.store import get_session_store
from agent.tools.schemas import (
    CalculationResponse,
    ChunkResult,
    MedicationCheckResponse,
    RetrievalResponse,
    SessionContextResponse,
    TimelineResponse,
)
from agent.tools.retrieval import detect_resource_type_from_query
from agent.tools.context import get_patient_context

_pii_masker = create_pii_masker()


def _mask_content(text: str) -> str:
    masked, _ = _pii_masker.mask_pii(text)
    return masked


@tool
async def search_clinical_notes(
    query: str,
    patient_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search clinical notes for relevant context using hybrid search + reranking.

    Args:
        query: Search query string
        patient_id: Patient UUID (auto-injected from context)
    """
    from db.vector_store import hybrid_search
    from reranker.cross_encoder import Reranker

    # ALWAYS use patient_id from context when available
    context_patient_id = get_patient_context()
    if context_patient_id:
        if patient_id and patient_id != context_patient_id:
            print(f"[CLINICAL_NOTES] Overriding patient_id ({patient_id[:8]}...) with context ({context_patient_id[:8]}...)")
        patient_id = context_patient_id
    elif not patient_id:
        print("[CLINICAL_NOTES] Warning: No patient_id provided and none in context")

    # Auto-detect resource type from query keywords
    detected_resource_type = detect_resource_type_from_query(query)

    filter_metadata = {}
    if patient_id:
        try:
            uuid.UUID(patient_id)
        except (ValueError, TypeError):
            return RetrievalResponse(
                query=query,
                chunks=[],
                count=0,
                success=False,
                error=f"Invalid patient_id format. Must be a UUID, got: {patient_id}",
            ).model_dump()
        filter_metadata["patient_id"] = patient_id
    if detected_resource_type:
        filter_metadata["resource_type"] = detected_resource_type
        print(f"[CLINICAL_NOTES] Auto-detected resource_type: {detected_resource_type}")

    # Hybrid search (BM25 + semantic)
    docs = await hybrid_search(
        query=query,
        k=50,
        filter_metadata=filter_metadata if filter_metadata else None,
    )

    # Rerank results
    if docs:
        try:
            reranker = Reranker()
            reranked = reranker.rerank_with_scores(query, docs)
            docs = [doc for doc, _ in reranked[:10]]
        except Exception as e:
            print(f"[CLINICAL_NOTES] Reranking failed, using raw results: {e}")
            docs = docs[:10]

    # Mask PII in results
    results = []
    for doc in docs:
        results.append({
            "id": str(getattr(doc, "id", "")),
            "content": _mask_content(doc.page_content),
            "metadata": doc.metadata or {},
        })

    return RetrievalResponse(
        query=query,
        chunks=[
            ChunkResult(
                id=str(item.get("id", "")),
                content=str(item.get("content", "")),
                score=0.0,
                metadata=item.get("metadata", {}) or {},
            )
            for item in results
        ],
        count=len(results),
    ).model_dump()


@tool
async def get_patient_timeline(patient_id: Optional[str] = None, k_return: int = 50) -> Dict[str, Any]:
    """
    Return a chronological timeline for a patient based on retrieved notes.

    Queries database directly to get ALL patient chunks sorted by date.

    Args:
        patient_id: Patient UUID (auto-injected from context)
        k_return: Number of timeline events to return (max 100)
    """
    from db.vector_store import get_patient_timeline as db_get_timeline

    # ALWAYS use patient_id from context when available
    context_patient_id = get_patient_context()
    if context_patient_id:
        if patient_id and patient_id != context_patient_id:
            print(f"[TIMELINE] Overriding patient_id ({patient_id[:8]}...) with context ({context_patient_id[:8]}...)")
        patient_id = context_patient_id
    elif not patient_id:
        return TimelineResponse(
            patient_id="",
            events=[],
            success=False,
            error="No patient_id provided and none in context",
        ).model_dump()

    # Validate UUID format
    try:
        uuid.UUID(patient_id)
    except (ValueError, TypeError):
        return TimelineResponse(
            patient_id=patient_id,
            events=[],
            success=False,
            error=f"Invalid patient_id format. Must be a UUID, got: {patient_id}",
        ).model_dump()

    # Use direct DB query (not vector search) to get all patient chunks
    results = await db_get_timeline(patient_id, k=min(k_return, 100))

    return TimelineResponse(
        patient_id=patient_id,
        events=[
            ChunkResult(
                id=str(doc.id) if doc.id else "",
                content=_mask_content(doc.page_content),
                score=0.0,
                metadata=doc.metadata or {},
            )
            for doc in results
        ],
    ).model_dump()


@tool
def cross_reference_meds(medication_list: List[str]) -> Dict[str, Any]:
    """Check medication list for basic interaction warnings."""
    meds = {med.strip().lower() for med in medication_list}
    warnings = []
    if "warfarin" in meds and "aspirin" in meds:
        warnings.append("Potential interaction: warfarin with aspirin may increase bleeding risk.")
    if "metformin" in meds and "contrast dye" in meds:
        warnings.append("Potential interaction: metformin with contrast dye may increase lactic acidosis risk.")
    return MedicationCheckResponse(
        medications=sorted(meds),
        warnings=warnings,
    ).model_dump()


@tool
def get_session_context(session_id: str, limit: int = 10) -> Dict[str, Any]:
    """Retrieve session summary and recent turns."""
    store = get_session_store()
    recent = store.get_recent(session_id, limit=limit)
    summary = store.get_summary(session_id)
    return SessionContextResponse(
        summary=summary,
        recent_turns=recent,
    ).model_dump()


@tool
def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate simple arithmetic expressions."""
    allowed = set("0123456789+-*/(). ")
    if any(ch not in allowed for ch in expression):
        return CalculationResponse(
            success=False,
            error="Unsupported characters in expression.",
            result=None,
        ).model_dump()
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception:
        return CalculationResponse(
            success=False,
            error="Unable to evaluate expression.",
            result=None,
        ).model_dump()
    return CalculationResponse(result=str(result)).model_dump()


@tool
def get_current_date() -> Dict[str, Any]:
    """Return current date in ISO format."""
    return CalculationResponse(result=dt.datetime.utcnow().isoformat()).model_dump()


def summarize_tool_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for item in results:
        content = item.get("content", "")
        summaries.append(
            {
                "id": item.get("id", ""),
                "content_preview": content[:200].replace("\n", " "),
                "metadata": item.get("metadata", {}),
            }
        )
    return summaries


from .calculators import (
    calculate_bmi,
    calculate_bsa,
    calculate_creatinine_clearance,
    calculate_gfr,
)
from .dosage_validator import validate_dosage
from .fda_tools import get_drug_recalls, get_drug_shortages, get_faers_events, search_fda_drugs
from .loinc_lookup import lookup_loinc
from .research_tools import get_who_stats, search_clinical_trials, search_pubmed
from .retrieval import retrieve_patient_data, search_patient_records
from .terminology_tools import lookup_rxnorm, search_icd10, validate_icd10_code


__all__ = [
    "calculate",
    "calculate_bmi",
    "calculate_bsa",
    "calculate_creatinine_clearance",
    "calculate_gfr",
    "cross_reference_meds",
    "get_current_date",
    "get_drug_recalls",
    "get_drug_shortages",
    "get_patient_timeline",
    "get_session_context",
    "get_who_stats",
    "lookup_loinc",
    "lookup_rxnorm",
    "search_clinical_notes",
    "search_clinical_trials",
    "search_fda_drugs",
    "search_icd10",
    "search_patient_records",
    "retrieve_patient_data",
    "search_pubmed",
    "summarize_tool_results",
    "validate_icd10_code",
    "validate_dosage",
    "get_faers_events",
]
