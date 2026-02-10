"""Guardrails AI setup for validating agent output."""

from __future__ import annotations

import os
from typing import Optional, Tuple


def setup_guard() -> Optional[object]:
    """Initialize Guardrails validators if enabled."""

    if os.getenv("GUARDRAILS_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        return None

    try:
        from guardrails import Guard
        from guardrails.hub import DetectHallucination, DetectPII
    except Exception:
        return None

    guard = Guard().use(
        DetectPII(threshold=0.5),
        DetectHallucination(threshold=0.5),
    )
    return guard


def validate_output(text: str, context: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate text against guardrails, return (is_valid, error_message).

    Args:
        text: The text to validate
        context: Optional context for hallucination detection

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    guard = setup_guard()
    if not guard:
        return True, ""

    try:
        guard.validate(text)
        return True, ""
    except Exception as e:
        error_msg = str(e)
        if "hallucination" in error_msg.lower():
            return False, f"Potential hallucination detected: {error_msg}"
        if "pii" in error_msg.lower():
            return False, f"PII detected in output: {error_msg}"
        return False, f"Guardrails validation failed: {error_msg}"
