"""Load and manage agent prompts from external YAML file.

This module loads prompts from prompts.yaml and injects appropriate fragments
for each agent type. Fragments are reusable blocks that provide consistent
guidance across agents (HIPAA compliance, safety rules, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.yaml"

_prompts_cache: Optional[Dict[str, object]] = None


def load_prompts(reload: bool = False) -> Dict[str, object]:
    """Load prompts from YAML file with caching."""
    global _prompts_cache
    if _prompts_cache is not None and not reload:
        return _prompts_cache

    prompts_path = Path(os.getenv("AGENT_PROMPTS_FILE", str(PROMPTS_FILE)))
    with prompts_path.open("r", encoding="utf-8") as handle:
        _prompts_cache = yaml.safe_load(handle) or {}
    return _prompts_cache


def _get_fragment(fragments: Dict[str, Any], name: str) -> str:
    """Safely get a fragment, returning empty string if not found."""
    return str(fragments.get(name, "")).strip()


def get_researcher_prompt(patient_id: Optional[str] = None) -> str:
    """Get researcher system prompt with all required fragments.

    Injects the following fragments:
    - hipaa_compliance: PII handling rules
    - fhir_json_criteria: When to use include_full_json=True
    - confidence_scoring: HIGH/MEDIUM/LOW methodology
    - safety_reminder: Critical safety rules
    - citation_format: Source citation format
    - patient_context: Patient-specific context (if patient_id provided)
    """
    prompts = load_prompts()
    base = str(prompts.get("researcher", {}).get("system_prompt", "")).strip()
    fragments = prompts.get("fragments", {})

    # Core fragments for researcher
    base += "\n\n" + _get_fragment(fragments, "hipaa_compliance")
    base += "\n\n" + _get_fragment(fragments, "fhir_json_criteria")
    base += "\n\n" + _get_fragment(fragments, "confidence_scoring")
    base += "\n\n" + _get_fragment(fragments, "safety_reminder")
    base += "\n\n" + _get_fragment(fragments, "citation_format")

    # Patient context (with ID substitution)
    if patient_id:
        context = _get_fragment(fragments, "patient_context")
        if context:
            context = context.format(patient_id=patient_id)
            base += "\n\n" + context

    return base.strip()


def get_validator_prompt() -> str:
    """Get validator system prompt with required fragments.

    Injects the following fragments:
    - hipaa_compliance: PII handling rules
    - safety_reminder: Critical safety rules
    """
    prompts = load_prompts()
    base = str(prompts.get("validator", {}).get("system_prompt", "")).strip()
    fragments = prompts.get("fragments", {})

    # Core fragments for validator
    base += "\n\n" + _get_fragment(fragments, "hipaa_compliance")
    base += "\n\n" + _get_fragment(fragments, "safety_reminder")

    return base.strip()


def get_conversational_prompt() -> str:
    """Get conversational system prompt."""
    prompts = load_prompts()
    return str(prompts.get("conversational_responder", {}).get("system_prompt", "")).strip()


def get_response_prompt() -> str:
    """Get final response synthesis prompt."""
    prompts = load_prompts()
    return str(prompts.get("response", {}).get("system_prompt", "")).strip()


def get_metadata() -> Dict[str, Any]:
    """Get metadata from prompts file (version, tool_count, etc.)."""
    prompts = load_prompts()
    return dict(prompts.get("metadata", {}))


def reload_prompts() -> Dict[str, object]:
    """Force reload prompts from file (useful for hot-reloading in dev)."""
    return load_prompts(reload=True)
