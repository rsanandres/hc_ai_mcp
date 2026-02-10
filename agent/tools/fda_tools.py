"""Tools for openFDA endpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool

from agent.tools.schemas import FDAResponse

_OPENFDA_BASE_URL = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov")
_OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY")


async def _openfda_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if _OPENFDA_API_KEY and "api_key" not in params:
        params["api_key"] = _OPENFDA_API_KEY
    url = f"{_OPENFDA_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return {"error": f"openFDA request failed: {exc}"}


@tool
async def search_fda_drugs(drug_name: str, limit: int = 5) -> Dict[str, Any]:
    """Search openFDA drug labels by generic or brand name."""
    if not drug_name.strip():
        return FDAResponse(
            success=False, error="drug_name is required", results=[], count=0,
        ).model_dump()
    query = f'(openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}")'
    data = await _openfda_get("/drug/label.json", {"search": query, "limit": limit})
    if "error" in data:
        return FDAResponse(
            success=False, error=str(data.get("error")), results=[], count=0,
        ).model_dump()
    results = data.get("results", [])
    return FDAResponse(results=results, count=len(results)).model_dump()


@tool
async def get_drug_recalls(drug_name: str, limit: int = 5) -> Dict[str, Any]:
    """Return recent FDA drug recalls by name."""
    if not drug_name.strip():
        return FDAResponse(
            success=False, error="drug_name is required", results=[], count=0,
        ).model_dump()
    query = f'(openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}")'
    data = await _openfda_get("/drug/enforcement.json", {"search": query, "limit": limit})
    if "error" in data:
        return FDAResponse(
            success=False, error=str(data.get("error")), results=[], count=0,
        ).model_dump()
    recalls = []
    for item in data.get("results", []):
        recalls.append({
            "reason": item.get("reason_for_recall", ""),
            "status": item.get("status", ""),
            "classification": item.get("classification", ""),
            "recall_initiation_date": item.get("recall_initiation_date", ""),
        })
    return FDAResponse(results=recalls, count=len(recalls)).model_dump()


@tool
async def get_drug_shortages(drug_name: str, limit: int = 5) -> Dict[str, Any]:
    """Return current FDA drug shortages by name."""
    if not drug_name.strip():
        return FDAResponse(
            success=False, error="drug_name is required", results=[], count=0,
        ).model_dump()
    escaped_name = drug_name.replace('"', '\\"')
    query = (
        f'(openfda.generic_name:"{escaped_name}" OR '
        f'openfda.brand_name:"{escaped_name}" OR '
        f'product_name:"{escaped_name}")'
    )
    data = await _openfda_get("/drug/shortages.json", {"search": query, "limit": limit})
    if "error" in data:
        error_msg = str(data.get("error", ""))
        if "404" in error_msg or "Not Found" in error_msg:
            return FDAResponse(success=True, results=[], count=0).model_dump()
        return FDAResponse(
            success=False, error=error_msg, results=[], count=0,
        ).model_dump()
    shortages = []
    for item in data.get("results", []):
        shortage_info = {
            "status": item.get("status", ""),
            "availability": item.get("availability", ""),
            "reason": item.get("reason", ""),
            "updated": item.get("updated", ""),
        }
        openfda = item.get("openfda", {})
        if openfda:
            generic_names = openfda.get("generic_name", [])
            brand_names = openfda.get("brand_name", [])
            if generic_names:
                shortage_info["generic_name"] = generic_names[0]
            if brand_names:
                shortage_info["brand_name"] = brand_names[0]
        if "product_name" in item and not shortage_info.get("generic_name"):
            shortage_info["product_name"] = item.get("product_name", "")
        shortages.append(shortage_info)
    return FDAResponse(results=shortages, count=len(shortages)).model_dump()


@tool
async def get_faers_events(drug_name: str, limit: int = 5) -> Dict[str, Any]:
    """Return adverse event summaries for a drug from FAERS."""
    if not drug_name.strip():
        return FDAResponse(
            success=False, error="drug_name is required", results=[], count=0,
        ).model_dump()
    query = f'patient.drug.medicinalproduct:"{drug_name}"'
    data = await _openfda_get("/drug/event.json", {"search": query, "limit": limit})
    if "error" in data:
        return FDAResponse(
            success=False, error=str(data.get("error")), results=[], count=0,
        ).model_dump()
    events = []
    for item in data.get("results", []):
        reactions = item.get("patient", {}).get("reaction", [])
        events.append({
            "safetyreportid": item.get("safetyreportid", ""),
            "receivedate": item.get("receivedate", ""),
            "reactions": [r.get("reactionmeddrapt", "") for r in reactions],
        })
    return FDAResponse(results=events, count=len(events)).model_dump()
