"""Enhanced retrieval tool for patient records."""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agent.tools.schemas import ChunkResult, RetrievalResponse
from agent.tools.context import get_patient_context


def strip_patient_name_from_query(query: str, patient_id: Optional[str] = None) -> str:
    """Remove patient names from query when patient_id is provided.

    Safeguard: Old embeddings don't contain patient names, so including them
    in semantic search causes mismatches. Use patient_id filter instead.
    """
    if not patient_id or not query:
        return query

    original_query = query.strip()
    words = original_query.split()

    # Pattern 1: Possessive form - "Adam Abbott's conditions" → "conditions"
    possessive_match = re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+['\u2019]s?\s+(.+)$", original_query)
    if possessive_match:
        return possessive_match.group(1).strip()

    # Pattern 2: Query is ONLY a name (2 capitalized words, nothing else)
    if len(words) == 2:
        first, second = words[0], words[1]
        if (first[0].isupper() and first.isalpha() and len(first) > 1 and
            second[0].isupper() and second.isalpha() and len(second) > 1):
            return "Condition Observation MedicationRequest"

    # Pattern 3: Two capitalized words at start followed by lowercase
    if len(words) >= 3:
        first, second, third = words[0], words[1], words[2]
        if (first and second and third and
            first[0].isupper() and first.isalpha() and len(first) > 1 and
            second[0].isupper() and second.isalpha() and len(second) > 1 and
            not third[0].isupper()):
            cleaned = " ".join(words[2:]).strip()
            if cleaned:
                return cleaned

    return original_query


# Keyword-to-resource-type mapping for automatic query intent detection
RESOURCE_TYPE_KEYWORDS = {
    "Condition": [
        "condition", "conditions", "diagnosis", "diagnoses", "diagnosed",
        "disease", "diseases", "problem", "problems", "illness", "illnesses",
        "disorder", "disorders", "ailment", "ailments", "sickness",
    ],
    "Observation": [
        "observation", "observations", "lab", "labs", "laboratory",
        "test", "tests", "result", "results", "measurement", "measurements",
        "vital", "vitals", "blood pressure", "heart rate", "temperature",
        "weight", "height", "bmi", "glucose", "cholesterol", "hemoglobin",
    ],
    "MedicationRequest": [
        "medication", "medications", "medicine", "medicines", "drug", "drugs",
        "prescription", "prescriptions", "prescribed", "rx", "pharma",
    ],
    "Procedure": [
        "procedure", "procedures", "surgery", "surgeries", "surgical",
        "operation", "operations", "intervention", "interventions",
    ],
    "Immunization": [
        "immunization", "immunizations", "vaccine", "vaccines", "vaccination",
        "vaccinations", "immunized", "vaccinated", "shot", "shots",
    ],
    "Encounter": [
        "encounter", "encounters", "visit", "visits", "appointment",
        "appointments", "admission", "admissions", "hospitalization",
    ],
    "DiagnosticReport": [
        "report", "reports", "diagnostic", "diagnostics", "imaging",
        "radiology", "xray", "x-ray", "mri", "ct scan", "ultrasound",
    ],
}


def detect_resource_type_from_query(query: str) -> Optional[str]:
    """Detect FHIR resource type from keywords in the query."""
    if not query:
        return None

    query_lower = query.lower()

    for resource_type, keywords in RESOURCE_TYPE_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower):
                return resource_type

    return None


_RERANKER_INSTANCE = None


def _get_reranker():
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        from reranker.cross_encoder import Reranker
        model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        device = os.getenv("RERANKER_DEVICE", "auto")
        _RERANKER_INSTANCE = Reranker(model_name=model, device=device)
    return _RERANKER_INSTANCE


@tool
async def search_patient_records(
    query: str,
    patient_id: Optional[str] = None,
    k_chunks: int = 10,
    include_full_json: bool = False,
) -> Dict[str, Any]:
    """
    Search patient records using hybrid search (BM25 + semantic) with reranking.

    Args:
        query: Search query string
        patient_id: Patient UUID (NOT an ICD-10 code!) - use the patient_id from context
        k_chunks: Number of chunks to return (default: 10)
        include_full_json: Whether to include full document JSON
    """
    from agent.tools.argument_validators import validate_patient_id
    from db.vector_store import hybrid_search, search_similar_chunks

    # ALWAYS use patient_id from context when available
    context_patient_id = get_patient_context()
    if context_patient_id:
        if patient_id and patient_id != context_patient_id:
            print(f"[RETRIEVAL] Overriding LLM patient_id ({patient_id[:8]}...) with context ({context_patient_id[:8]}...)")
        patient_id = context_patient_id
    elif not patient_id:
        print("[RETRIEVAL] Warning: No patient_id provided and none in context")

    original_query = query

    if not query or not query.strip():
        return RetrievalResponse(
            query="",
            original_query=original_query,
            chunks=[],
            count=0,
            success=False,
            error="Query cannot be empty. Please provide a search term like 'Condition', 'Observation', or 'MedicationRequest'.",
        ).model_dump()

    if patient_id:
        is_valid, error_msg = validate_patient_id(patient_id)
        if not is_valid:
            return RetrievalResponse(
                query=query,
                original_query=original_query,
                chunks=[],
                count=0,
                success=False,
                error=error_msg,
            ).model_dump()

        cleaned_query = strip_patient_name_from_query(query, patient_id)
        if not cleaned_query or not cleaned_query.strip():
            cleaned_query = "Condition Observation MedicationRequest"

        detected_resource_type = detect_resource_type_from_query(original_query)
        filter_metadata = {"patient_id": patient_id}
        if detected_resource_type:
            filter_metadata["resource_type"] = detected_resource_type
    else:
        cleaned_query = query
        detected_resource_type = detect_resource_type_from_query(query)
        filter_metadata = {"resource_type": detected_resource_type} if detected_resource_type else None

    # Use hybrid search (BM25 + semantic) for better coverage
    k_retrieve = max(k_chunks * 4, 20)
    try:
        candidates = await hybrid_search(
            cleaned_query,
            k=k_retrieve,
            filter_metadata=filter_metadata,
        )
    except Exception:
        # Fallback to semantic-only search
        candidates = await search_similar_chunks(cleaned_query, k=k_retrieve, filter_metadata=filter_metadata)

    if not candidates:
        return RetrievalResponse(
            query=cleaned_query,
            original_query=original_query if original_query != cleaned_query else None,
            chunks=[],
            count=0,
        ).model_dump()

    # Rerank with cross-encoder
    reranker = _get_reranker()
    scored = reranker.rerank_with_scores(cleaned_query, candidates)
    top_docs = scored[:k_chunks]

    chunks = [
        ChunkResult(
            id=str(getattr(doc, "id", "")),
            content=doc.page_content,
            score=score,
            metadata=doc.metadata or {},
        )
        for doc, score in top_docs
    ]
    return RetrievalResponse(
        query=cleaned_query,
        original_query=original_query if original_query != cleaned_query else None,
        chunks=chunks,
        count=len(chunks),
    ).model_dump()


@tool
async def retrieve_patient_data(
    query: str,
    patient_id: Optional[str] = None,
    k_retrieve: int = 50,
    k_return: int = 10,
    use_hybrid: bool = True,
) -> Dict[str, Any]:
    """
    Direct retrieval from PostgreSQL with hybrid search and cross-encoder reranking.

    Args:
        query: Search query string
        patient_id: Patient UUID (NOT an ICD-10 code!) - use the patient_id from context
        k_retrieve: Number of candidates to retrieve before reranking
        k_return: Number of results to return after reranking
        use_hybrid: If True, use BM25+semantic hybrid search. If False, semantic only.
    """
    from agent.tools.argument_validators import validate_patient_id
    from db.vector_store import hybrid_search, search_similar_chunks

    context_patient_id = get_patient_context()
    if context_patient_id:
        if patient_id and patient_id != context_patient_id:
            print(f"[RETRIEVAL] Overriding LLM patient_id ({patient_id[:8]}...) with context ({context_patient_id[:8]}...)")
        patient_id = context_patient_id
    elif not patient_id:
        print("[RETRIEVAL] Warning: No patient_id provided and none in context")

    original_query = query

    if not query or not query.strip():
        return RetrievalResponse(
            query="",
            original_query=original_query,
            chunks=[],
            count=0,
            success=False,
            error="Query cannot be empty.",
        ).model_dump()

    if patient_id:
        is_valid, error_msg = validate_patient_id(patient_id)
        if not is_valid:
            return RetrievalResponse(
                query=query,
                original_query=original_query,
                chunks=[],
                count=0,
                success=False,
                error=error_msg,
            ).model_dump()

        cleaned_query = strip_patient_name_from_query(query, patient_id)
        if not cleaned_query or not cleaned_query.strip():
            cleaned_query = "Condition Observation MedicationRequest"

        detected_resource_type = detect_resource_type_from_query(original_query)
        filter_metadata = {"patient_id": patient_id}
        if detected_resource_type:
            filter_metadata["resource_type"] = detected_resource_type
    else:
        cleaned_query = query
        detected_resource_type = detect_resource_type_from_query(query)
        filter_metadata = {"resource_type": detected_resource_type} if detected_resource_type else None

    if use_hybrid:
        candidates = await hybrid_search(
            cleaned_query,
            k=k_retrieve,
            filter_metadata=filter_metadata,
            bm25_weight=0.5,
            semantic_weight=0.5,
        )
    else:
        candidates = await search_similar_chunks(cleaned_query, k=k_retrieve, filter_metadata=filter_metadata)

    if not candidates:
        return RetrievalResponse(
            query=cleaned_query,
            original_query=original_query if original_query != cleaned_query else None,
            chunks=[],
            count=0,
        ).model_dump()

    reranker = _get_reranker()
    scored = reranker.rerank_with_scores(cleaned_query, candidates)
    top_docs = scored[:k_return]

    chunks = [
        ChunkResult(
            id=str(getattr(doc, "id", "")),
            content=doc.page_content,
            score=score,
            metadata=doc.metadata or {},
        )
        for doc, score in top_docs
    ]
    return RetrievalResponse(
        query=cleaned_query,
        original_query=original_query if original_query != cleaned_query else None,
        chunks=chunks,
        count=len(chunks),
    ).model_dump()
